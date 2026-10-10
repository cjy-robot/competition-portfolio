"""PT → Keras 权重迁移: 完整名字映射"""
import numpy as np, tensorflow as tf, sys
from tensorflow import keras
from tensorflow.keras import layers

NUM_ANCHORS = 3

# ==================== Build (exact same as before) ====================
def cbr(x, f, k, s, name):
    p = k//2
    if p: x = layers.ZeroPadding2D(p)(x)
    x = layers.Conv2D(f, k, s, 'valid', use_bias=False, name=name+'_c')(x)
    x = layers.BatchNormalization(epsilon=1e-5, name=name+'_bn')(x)
    return layers.ReLU()(x)

def dw_bn(x, k, s, name):
    p = k//2
    if p: x = layers.ZeroPadding2D(p)(x)
    x = layers.DepthwiseConv2D(k, s, 'valid', use_bias=False, name=name+'_d')(x)
    return layers.BatchNormalization(epsilon=1e-5, name=name+'_bn')(x)

def block(x, half_inp, oup, mid, k, s, name):
    if s == 2: mi, mo_proj = x, half_inp
    else: mi, mo_proj = x[:,:,:,1::2], half_inp  # odd channels for main

    mo_main = oup - half_inp
    m = cbr(mi, mid, 1, 1, name+'m1')
    m = dw_bn(m, k, s, name+'md')
    m = cbr(m, mo_main, 1, 1, name+'m2')
    if s == 2:
        p = dw_bn(x, k, 2, name+'p')
        p = cbr(p, mo_proj, 1, 1, name+'p1')
    else:
        p = x[:,:,:,0::2]  # even channels for proj
    return layers.Concatenate()([p, m])

def dwb(x, f, size, name):
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

print("Building Keras model...")
inp = layers.Input((192,192,3), name='input')
x = cbr(inp, 24, 3, 2, 'f')
x = layers.MaxPool2D(3, 2, 'same')(x)
for i in range(4): x = block(x, 24 if i==0 else 24, 48, 24, 3, 2 if i==0 else 1, f's2_{i}_')
for i in range(8): x = block(x, 48 if i==0 else 48, 96, 48, 3, 2 if i==0 else 1, f's3_{i}_')
C2 = x
for i in range(4): x = block(x, 96 if i==0 else 96, 192, 96, 3, 2 if i==0 else 1, f's4_{i}_')
C3 = x
S3 = cbr(C3, 72, 1, 1, 'fpn3')
P2_up = layers.UpSampling2D(2, interpolation='bilinear')(C3)
S2 = cbr(layers.Concatenate()([P2_up, C2]), 72, 1, 1, 'fpn2')
cls_h0 = dwb(S2, 72, 5, 'h0c_'); reg_h0 = dwb(S2, 72, 5, 'h0r_')
cls_h1 = dwb(S3, 72, 5, 'h1c_'); reg_h1 = dwb(S3, 72, 5, 'h1r_')
out_reg = layers.Conv2D(12, 1, use_bias=True, name='oreg')
out_obj = layers.Conv2D(3, 1, use_bias=True, name='oobj')
out_cls = layers.Conv2D(1, 1, use_bias=True, name='ocls')
head0 = layers.Concatenate()([layers.Activation('sigmoid')(out_reg(reg_h0)),
                                layers.Activation('sigmoid')(out_obj(cls_h0)),
                                layers.Activation('softmax')(out_cls(cls_h0))])
head1 = layers.Concatenate()([layers.Activation('sigmoid')(out_reg(reg_h1)),
                                layers.Activation('sigmoid')(out_obj(cls_h1)),
                                layers.Activation('softmax')(out_cls(cls_h1))])
model = keras.Model(inputs=inp, outputs=[head0, head1])
# Build layer name map
lm = {l.name: l for l in model.layers}

# ==================== PT→Keras Mapping ====================
print("\nLoading PT weights...")
pt = np.load(r'D:\CCPD2019_YOLO\pt_weights.npz')

def bn_index(param):
    return {'weight':0,'bias':1,'running_mean':2,'running_var':3}.get(param, -1)

def set_weight(ks_name, wi, pt_key, transpose='conv'):
    """Set Keras layer weight from PT tensor"""
    if ks_name not in lm: return False
    layer = lm[ks_name]
    w = list(layer.get_weights())
    if wi >= len(w): return False
    val = pt[pt_key]
    if transpose == 'conv' and val.ndim == 4:
        val = np.transpose(val, (2,3,1,0))  # [O,I,Kh,Kw] → [Kh,Kw,I,O]
    elif transpose == 'dw' and val.ndim == 4:
        val = np.transpose(val, (2,3,0,1))  # [C,1,Kh,Kw] → [Kh,Kw,C,1]
    if val.shape != w[wi].shape:
        # Try without transpose
        if val.shape == w[wi].shape:
            pass
        else:
            return False  # shape mismatch
    w[wi] = val
    layer.set_weights(w)
    return True

# Process all PT keys
ok = 0; fail = 0; skipped = 0
for pt_key in pt.keys():
    p = pt_key.split('.')
    if 'num_batches_tracked' in pt_key:
        skipped += 1; continue

    res = None  # (ks_name, wi, transpose)

    # ── Backbone ──
    if p[0] == 'backbone' and p[1] == 'first_conv':
        idx, param = int(p[2]), p[3]
        layer = 'f'
        if idx == 0: res = (f'{layer}_c', 0, 'conv')
        else: res = (f'{layer}_bn', bn_index(param), 'bn')

    elif p[0] == 'backbone' and p[1].startswith('stage'):
        sn = int(p[1][-1]); bn = int(p[2]); sub = p[3]; idx = int(p[4])
        param = p[5] if len(p)>5 else 'weight'
        prefix = f's{sn}_{bn}_'

        if sub == 'branch_main':
            tbl = {0:(f'{prefix}m1_c',0,'conv'), 1:(f'{prefix}m1_bn',bn_index(param),'bn'),
                   3:(f'{prefix}md_d',0,'dw'),  4:(f'{prefix}md_bn',bn_index(param),'bn'),
                   5:(f'{prefix}m2_c',0,'conv'), 6:(f'{prefix}m2_bn',bn_index(param),'bn')}
            res = tbl.get(idx)
        elif sub == 'branch_proj':
            tbl = {0:(f'{prefix}p_d',0,'dw'),  1:(f'{prefix}p_bn',bn_index(param),'bn'),
                   2:(f'{prefix}p1_c',0,'conv'), 3:(f'{prefix}p1_bn',bn_index(param),'bn')}
            res = tbl.get(idx)

    # ── FPN conv1x1 ──
    elif p[0] == 'fpn' and p[1].startswith('conv1x1_'):
        num = p[1][-1]  # '2' or '3'
        idx, param = int(p[2]), p[3] if len(p)>3 else 'weight'
        layer = f'fpn{num}'
        if idx == 0: res = (f'{layer}_c', 0, 'conv')
        else: res = (f'{layer}_bn', bn_index(param), 'bn')

    # ── DWConv blocks (cls_head_2/3, reg_head_2/3) ──
    elif p[0] == 'fpn' and ('head' in p[1]):
        # p[1] = cls_head_2, reg_head_2, cls_head_3, reg_head_3
        head_type = p[1].split('_')[0]  # cls or reg
        head_num = p[1].split('_')[-1]  # 2 or 3
        block_idx = int(p[3])  # 0,1,3,4,5,6,8,9
        param = p[4] if len(p)>4 else 'weight'

        # Map: head_2 → h0, head_3 → h1
        prefix = f'h{int(head_num)-2}{head_type[0]}_'  # h0c_, h0r_, h1c_, h1r_

        idx_map = {0:('d1',0,'dw'), 1:('d1b',bn_index(param),'bn'),
                   3:('p1',0,'conv'), 4:('p1b',bn_index(param),'bn'),
                   5:('d2',0,'dw'), 6:('d2b',bn_index(param),'bn'),
                   8:('p2',0,'conv'), 9:('p2b',bn_index(param),'bn')}
        entry = idx_map.get(block_idx)
        if entry:
            res = (prefix + entry[0], entry[1], entry[2])

    # ── Output layers ──
    elif p[0] == 'output_reg_layers':
        wi = 0 if 'weight' in pt_key else 1
        res = ('oreg', wi, 'conv' if wi==0 else 'bias')
    elif p[0] == 'output_obj_layers':
        wi = 0 if 'weight' in pt_key else 1
        res = ('oobj', wi, 'conv' if wi==0 else 'bias')
    elif p[0] == 'output_cls_layers':
        wi = 0 if 'weight' in pt_key else 1
        res = ('ocls', wi, 'conv' if wi==0 else 'bias')

    if res:
        if set_weight(res[0], res[1], pt_key, res[2]):
            ok += 1
        else:
            fail += 1
    else:
        fail += 1

print(f"OK: {ok}  Failed: {fail}  Skipped: {skipped}")

# ==================== Save & Verify ====================
if fail == 0:
    model.save(r'D:\CCPD2019_YOLO\yolo_keras.h5')
    print(f"Saved yolo_keras.h5")

    # Verify against ONNX (faster and avoids PT CWD issue)
    print("\nVerifying vs ONNX...")
    import onnxruntime as ort
    sess = ort.InferenceSession(r'D:\CCPD2019_YOLO\plate_best_120.onnx')
    np.random.seed(42)
    nchw=np.clip(np.random.randn(1,3,192,192).astype(np.float32)*0.3+0.5,0,1)
    nhwc=np.transpose(nchw,(0,2,3,1))
    h0o,h1o=sess.run(None,{'input.1':nchw})
    h0k,h1k=model.predict(nhwc,verbose=0)
    d0=np.abs(h0k-h0o).max()
    d1=np.abs(h1k-h1o).max()
    print(f"Keras vs ONNX: head0 diff={d0:.6f}  head1 diff={d1:.6f}")
    if max(d0,d1) < 0.01:
        print("PERFECT MATCH!")
    else:
        print(f"WARNING: large difference!")
