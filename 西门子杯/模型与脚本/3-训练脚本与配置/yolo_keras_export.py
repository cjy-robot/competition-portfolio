"""YOLO-FastestV2: Keras重写 + PT权重迁移 + TFLite导出"""
import numpy as np, tensorflow as tf, sys, torch, os
from tensorflow import keras
from tensorflow.keras import layers

NUM_ANCHORS = 3
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ==================== Helpers ====================
def cbr(x, f, k, s, name, bias=False):
    p = k//2
    if p: x = layers.ZeroPadding2D(p)(x)
    x = layers.Conv2D(f, k, s, 'valid', use_bias=bias, name=name+'_c')(x)
    x = layers.BatchNormalization(epsilon=1e-5, name=name+'_bn')(x)
    return layers.ReLU()(x)

def dw_bn(x, k, s, name):
    p = k//2
    if p: x = layers.ZeroPadding2D(p)(x)
    x = layers.DepthwiseConv2D(k, s, 'valid', use_bias=False, name=name+'_d')(x)
    return layers.BatchNormalization(epsilon=1e-5, name=name+'_bn')(x)

def block(x, half_inp, oup, mid, k, s, name):
    # IMPORTANT: build main first, then proj — matching PyTorch module registration order
    if s == 2:
        mi, mo_proj = x, half_inp
    else:
        C = x.shape[-1]; _, mi = x[:,:,:,0:C:2], x[:,:,:,1:C:2]
        mo_proj = half_inp
    mo_main = oup - half_inp

    m = cbr(mi, mid, 1, 1, name+'m1')
    m = dw_bn(m, k, s, name+'md')
    m = cbr(m, mo_main, 1, 1, name+'m2')

    if s == 2:
        p = dw_bn(x, k, 2, name+'p')
        p = cbr(p, mo_proj, 1, 1, name+'p1')
    else:
        C = x.shape[-1]; p, _ = x[:,:,:,0:C:2], x[:,:,:,1:C:2]

    return layers.Concatenate()([p, m])

def dwconv_block(x, f, size, name):
    p = size//2
    y = layers.ZeroPadding2D(p)(x)
    y = layers.DepthwiseConv2D(size, 1, 'valid', use_bias=False, name=name+'d1')(y)
    y = layers.BatchNormalization(epsilon=1e-5, name=name+'d1b')(y); y = layers.ReLU()(y)
    y = layers.Conv2D(f, 1, use_bias=False, name=name+'p1')(y)
    y = layers.BatchNormalization(epsilon=1e-5, name=name+'p1b')(y)
    y = layers.ZeroPadding2D(p)(y)
    y = layers.DepthwiseConv2D(size, 1, 'valid', use_bias=False, name=name+'d2')(y)
    y = layers.BatchNormalization(epsilon=1e-5, name=name+'d2b')(y); y = layers.ReLU()(y)
    y = layers.Conv2D(f, 1, use_bias=False, name=name+'p2')(y)
    y = layers.BatchNormalization(epsilon=1e-5, name=name+'p2b')(y)
    return y

# ==================== Full Model ====================
def build_model():
    inp = layers.Input((192,192,3), name='input')
    x = cbr(inp, 24, 3, 2, 'f')  # [B,96,96,24]
    x = layers.MaxPool2D(3, 2, 'same')(x)  # [B,48,48,24]

    for i in range(4):
        x = block(x, 24 if i==0 else 24, 48, 24, 3, 2 if i==0 else 1, f's2_{i}_')

    for i in range(8):
        x = block(x, 48 if i==0 else 48, 96, 48, 3, 2 if i==0 else 1, f's3_{i}_')
    C2 = x  # [B,12,12,96] ← stage3 output (matching PyTorch)

    for i in range(4):
        x = block(x, 96 if i==0 else 96, 192, 96, 3, 2 if i==0 else 1, f's4_{i}_')
    C3 = x  # [B,6,6,192] ← stage4 output

    # FPN (mirrors PyTorch LightFPN exactly)
    S3 = cbr(C3, 72, 1, 1, 'fpn3')  # [B,6,6,192]→[B,6,6,72]

    P2_up = layers.UpSampling2D(2, interpolation='bilinear')(C3)  # [B,12,12,192]
    P2_cat = layers.Concatenate()([P2_up, C2])  # [B,12,12,288]
    S2 = cbr(P2_cat, 72, 1, 1, 'fpn2')  # [B,12,12,288]→[B,12,12,72]

    # DWConv blocks (separate per head, like PyTorch)
    cls_h0 = dwconv_block(S2, 72, 5, 'h0c_')
    reg_h0 = dwconv_block(S2, 72, 5, 'h0r_')
    obj_h0 = cls_h0  # PyTorch: obj_2 = cls_2

    cls_h1 = dwconv_block(S3, 72, 5, 'h1c_')
    reg_h1 = dwconv_block(S3, 72, 5, 'h1r_')
    obj_h1 = cls_h1  # PyTorch: obj_3 = cls_3

    # Output layers (SHARED, same as PyTorch: one Conv for both heads)
    out_reg = layers.Conv2D(NUM_ANCHORS*4, 1, use_bias=True, name='oreg')
    out_obj = layers.Conv2D(NUM_ANCHORS, 1, use_bias=True, name='oobj')
    out_cls = layers.Conv2D(1, 1, use_bias=True, name='ocls')

    def make_head(r, o, c):
        r = layers.Activation('sigmoid')(out_reg(r))
        o = layers.Activation('sigmoid')(out_obj(o))
        c = layers.Activation('softmax')(out_cls(c))
        return layers.Concatenate()([r, o, c])

    head0 = make_head(reg_h0, obj_h0, cls_h0)  # [B,12,12,16]
    head1 = make_head(reg_h1, obj_h1, cls_h1)  # [B,6,6,16]

    return keras.Model(inputs=inp, outputs=[head0, head1], name='yolo_keras')

# ==================== Build & Check ====================
print("Building Keras model...")
model = build_model()
trainable = [l for l in model.layers if len(l.get_weights()) > 0]
print(f"Trainable layers: {len(trainable)}")

# Count weight slots
total_slots = sum(len(l.get_weights()) for l in trainable)
print(f"Total weight slots: {total_slots}")

# Quick forward test
h0, h1 = model.predict(np.random.randn(1,192,192,3).astype(np.float32), verbose=0)
print(f"Output: h0={h0.shape} h1={h1.shape}")

# ==================== Load PyTorch ====================
print("\nLoading PyTorch model...")
sys.path.insert(0, r'C:\Users\LX\gd32ai-modelzoo\object_detection\yolo_fastestv2\submodules\Yolo_FastestV2')
import model.detector as _detector
import utils.utils

cfg = utils.utils.load_datafile(r'C:\Users\LX\gd32ai-modelzoo\object_detection\yolo_fastestv2\configs\plate_best.data')
pt = _detector.Detector(cfg['classes'], cfg['anchor_num'], load_param=False, export_onnx=True)
pt.load_state_dict(torch.load(r'C:\Users\LX\gd32ai-modelzoo\object_detection\yolo_fastestv2\submodules\Yolo_FastestV2\weights\plate_best-120-epoch-0.997542ap-model.pth', map_location='cpu'), strict=False)
pt.eval()

# Collect PT state_dict (skip num_batches_tracked)
pt_sd = [(n, v.numpy()) for n, v in pt.state_dict().items() if 'num_batches_tracked' not in n]
print(f"PT state_dict entries: {len(pt_sd)}")

# ==================== Transfer Weights ====================
# PT BN: weight, bias, running_mean, running_var (order in state_dict)
# PT Conv: weight, [bias]
# Keras BN: gamma, beta, moving_mean, moving_variance
# Keras Conv: kernel, [bias]

def transpose_conv(w):
    """[O,I,Kh,Kw] → [Kh,Kw,I,O]"""
    if w.ndim == 4:
        return np.transpose(w, (2, 3, 1, 0))
    return w

def transpose_dw(w):
    """[C,1,Kh,Kw] → [Kh,Kw,C,1]"""
    if w.ndim == 4:
        return np.transpose(w, (2, 3, 0, 1))
    return w

pt_idx = 0
pt_arrs = [v for _, v in pt_sd]
keras_layers = [l for l in model.layers if len(l.get_weights()) > 0]

transferred = 0
mismatch = 0

for layer in keras_layers:
    w = layer.get_weights()
    new_w = list(w)  # copy original shapes

    for j in range(len(w)):
        if pt_idx >= len(pt_arrs):
            break

        pt_val = pt_arrs[pt_idx]
        expected_shape = w[j].shape

        # Determine conversion
        if isinstance(layer, layers.DepthwiseConv2D) and pt_val.ndim == 4:
            new_w[j] = transpose_dw(pt_val)
        elif isinstance(layer, layers.Conv2D) and j == 0 and pt_val.ndim == 4:
            new_w[j] = transpose_conv(pt_val)
        elif isinstance(layer, layers.Conv2D) and j == 1:
            new_w[j] = pt_val  # bias, same shape
        elif isinstance(layer, layers.BatchNormalization):
            new_w[j] = pt_val  # gamma/beta/mean/var, same shape
        else:
            new_w[j] = pt_val

        if new_w[j].shape != expected_shape:
            mismatch += 1
            # Don't set if shape doesn't match
            new_w[j] = w[j]
        else:
            pt_idx += 1
            transferred += 1

    layer.set_weights(new_w)

print(f"Transferred: {transferred}  Mismatches: {mismatch}")

# ==================== Verify ====================
print("\nVerifying...")
np.random.seed(42)
test_nchw = np.clip(np.random.randn(1,3,192,192).astype(np.float32)*0.3+0.5, 0, 1)
test_nhwc = np.transpose(test_nchw, (0,2,3,1))

h0_k, h1_k = model.predict(test_nhwc, verbose=0)

pt_in = torch.from_numpy(test_nchw)
with torch.no_grad():
    h0_p, h1_p = pt(pt_in)

d0 = np.abs(h0_k - h0_p.numpy()).max()
d1 = np.abs(h1_k - h1_p.numpy()).max()
print(f"head0 diff: {d0:.6f}  head1 diff: {d1:.6f}")
print("✅ PERFECT!" if max(d0,d1) < 0.01 else f"❌ Large diff: d0={d0:.4f} d1={d1:.4f}")

if max(d0,d1) < 0.01:
    model.save(r'D:\CCPD2019_YOLO\yolo_keras.h5')
    print("Saved: yolo_keras.h5")
