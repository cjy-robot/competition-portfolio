"""
LPRNetTiny PyTorch → Keras → TFLite INT8 量化
直接权重迁移，避开ONNX转换依赖问题
"""
import numpy as np
import torch
import tensorflow as tf
import os, sys

sys.path.insert(0, r"D:\CCPD2019_YOLO")
from lprnet_model import LPRNetTiny
from prepare_lprnet_data import VOCAB

# ============ 配置 ============
PTH_PATH = r"D:\CCPD2019_YOLO\lprnet_weights\lprnet_tiny_best.pth"
CALIB_PATH = r"D:\CCPD2019_YOLO\lprnet_weights\lprnet_calib_data_500.npz"
OUT_DIR = r"D:\CCPD2019_YOLO\lprnet_weights\lprnet_int8"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASSES = len(VOCAB)  # 65
SEQ_LEN = 7
FEAT_DIM = 96

# ============ Step 1: 加载PyTorch模型权重 ============
print("=" * 60)
print("Step 1: Load PyTorch weights")
print("=" * 60)

pt_model = LPRNetTiny(num_classes=NUM_CLASSES, seq_len=SEQ_LEN)
pt_model.load_state_dict(torch.load(PTH_PATH, map_location='cpu'))
pt_model.eval()

pt_state = pt_model.state_dict()
print("PyTorch state dict keys:")
for k, v in pt_state.items():
    print("  %-40s %s" % (k, list(v.shape)))

# ============ Step 2: 构建Keras等价模型 ============
print()
print("=" * 60)
print("Step 2: Build Keras equivalent model")
print("=" * 60)

# Keras使用NHWC格式: [B, 24, 94, 1]
inputs = tf.keras.Input(shape=(24, 94, 1), name='input')

# ── Conv1 block (NHWC) ──
# PyTorch: Conv2d(1,24,3,padding=1) NCHW → weights [24,1,3,3]
# Keras: Conv2D(24,3,padding='same') NHWC → weights [3,3,1,24]
x = tf.keras.layers.Conv2D(24, 3, padding='same', use_bias=False, name='conv1_conv')(inputs)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='conv1_bn')(x)
x = tf.keras.layers.ReLU(name='conv1_relu')(x)
x = tf.keras.layers.MaxPool2D(2, 2, name='pool1')(x)  # → [B,12,47,24]

# ── Conv2 block ──
# PyTorch: Conv2d(24,48,3,padding=1) → weights [48,24,3,3]
x = tf.keras.layers.Conv2D(48, 3, padding='same', use_bias=False, name='conv2_conv')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='conv2_bn')(x)
x = tf.keras.layers.ReLU(name='conv2_relu')(x)
x = tf.keras.layers.MaxPool2D((2, 2), (2, 2), name='pool2')(x)  # → [B,6,23,48]

# ── Conv3 block ──
# PyTorch: Conv2d(48,96,3,padding=1) → weights [96,48,3,3]
x = tf.keras.layers.Conv2D(FEAT_DIM, 3, padding='same', use_bias=False, name='conv3_conv')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='conv3_bn')(x)
x = tf.keras.layers.ReLU(name='conv3_relu')(x)
x = tf.keras.layers.MaxPool2D((2, 2), (2, 2), name='pool3')(x)  # → [B,3,11,96]

# ── AdaptiveAvgPool2d((1, None)): pool H→1, keep W ──
# NHWC [B, H=3, W=11, C=96] → reduce_mean on axis=1 → [B, 1, 11, 96]
def adaptive_height_pool(x):
    return tf.reduce_mean(x, axis=1, keepdims=True)

x_pooled = tf.keras.layers.Lambda(adaptive_height_pool, name='pool_h')(x)  # → [B,1,11,96]

x_dropout = tf.keras.layers.Dropout(0.3, name='dropout')(x_pooled)

# Squeeze dim 1 → [B, 11, 96]
x_squeeze = tf.keras.layers.Reshape((11, FEAT_DIM), name='squeeze')(x_dropout)

# ── Conv1d: kernel=5, stride=1, padding=0 → 11→7 steps ──
# PyTorch Conv1d(96,96,5) → [B,96,7]
# Keras Conv1D: input [B, 11, 96], kernel 5 → [B, 7, 96]
x_conv1d = tf.keras.layers.Conv1D(FEAT_DIM, 5, strides=1, padding='valid',
                                   use_bias=True, name='conv1d')(x_squeeze)

# ── FC → [B, 7, 65] ──
x_fc = tf.keras.layers.Dense(NUM_CLASSES, use_bias=True, name='fc')(x_conv1d)

model = tf.keras.Model(inputs=inputs, outputs=x_fc, name='LPRNetTiny')
model.summary()

# ============ Step 3: 权重迁移 PyTorch → Keras ============
print()
print("=" * 60)
print("Step 3: Transfer weights PyTorch → Keras")
print("=" * 60)

# PyTorch Conv2d weight: [out_ch, in_ch, kh, kw] (NCHW)
# Keras Conv2D weight: [kh, kw, in_ch, out_ch] (NHWC)
def pt_conv_to_keras(pt_weight):
    """PyTorch [O,I,Kh,Kw] → Keras [Kh,Kw,I,O]"""
    return np.transpose(pt_weight, (2, 3, 1, 0))

# PyTorch BN: weight(out_ch), bias(out_ch), running_mean(out_ch), running_var(out_ch)
# Keras BN: gamma(out_ch), beta(out_ch), moving_mean(out_ch), moving_variance(out_ch)

# PyTorch Conv1d weight: [out_ch, in_ch, kw] (96, 96, 5)
# Keras Conv1D weight: [kw, in_ch, out_ch] (5, 96, 96)
def pt_conv1d_to_keras(pt_weight):
    """PyTorch [O,I,Kw] → Keras [Kw,I,O]"""
    return np.transpose(pt_weight, (2, 1, 0))

# PyTorch Linear weight: [out, in] (65, 96)
# Keras Dense weight: [in, out] (96, 65)
def pt_fc_to_keras(pt_weight):
    """PyTorch [out,in] → Keras [in,out]"""
    return np.transpose(pt_weight, (1, 0))

# conv1_conv
w = pt_conv_to_keras(pt_state['conv1.0.weight'].cpu().numpy())  # [24,1,3,3]→[3,3,1,24]
model.get_layer('conv1_conv').set_weights([w])
print("conv1_conv: PyTorch %s → Keras %s" % (str(pt_state['conv1.0.weight'].shape), str(w.shape)))

# conv1_bn
for bn_name, pt_name in [('conv1_bn', 'conv1'), ('conv2_bn', 'conv2'), ('conv3_bn', 'conv3')]:
    gamma = pt_state['%s.1.weight' % pt_name].cpu().numpy()
    beta = pt_state['%s.1.bias' % pt_name].cpu().numpy()
    mean = pt_state['%s.1.running_mean' % pt_name].cpu().numpy()
    var = pt_state['%s.1.running_var' % pt_name].cpu().numpy()
    model.get_layer(bn_name).set_weights([gamma, beta, mean, var])
    print("%s: gamma=%s beta=%s" % (bn_name, str(gamma.shape), str(beta.shape)))

# conv2_conv
w = pt_conv_to_keras(pt_state['conv2.0.weight'].cpu().numpy())  # [48,24,3,3]→[3,3,24,48]
model.get_layer('conv2_conv').set_weights([w])
print("conv2_conv: %s → %s" % (str(pt_state['conv2.0.weight'].shape), str(w.shape)))

# conv3_conv
w = pt_conv_to_keras(pt_state['conv3.0.weight'].cpu().numpy())  # [96,48,3,3]→[3,3,48,96]
model.get_layer('conv3_conv').set_weights([w])
print("conv3_conv: %s → %s" % (str(pt_state['conv3.0.weight'].shape), str(w.shape)))

# conv1d
w = pt_conv1d_to_keras(pt_state['conv1d.weight'].cpu().numpy())  # [96,96,5]→[5,96,96]
b = pt_state['conv1d.bias'].cpu().numpy()  # [96]
model.get_layer('conv1d').set_weights([w, b])
print("conv1d: weight %s bias %s" % (str(w.shape), str(b.shape)))

# fc
w = pt_fc_to_keras(pt_state['fc.weight'].cpu().numpy())  # [65,96]→[96,65]
b = pt_state['fc.bias'].cpu().numpy()
model.get_layer('fc').set_weights([w, b])
print("fc: weight %s bias %s" % (str(w.shape), str(b.shape)))

# ============ Step 4: 验证 Keras vs PyTorch 输出一致 ============
print()
print("=" * 60)
print("Step 4: Verify Keras vs PyTorch output")
print("=" * 60)

calib = np.load(CALIB_PATH)
test_data = calib['data'][:5]  # NHWC [5, 24, 94, 1]

# PyTorch: NCHW
test_nchw = np.transpose(test_data, (0, 3, 1, 2))  # [5, 1, 24, 94]
pt_tensor = torch.from_numpy(test_nchw.astype(np.float32))
with torch.no_grad():
    pt_out = pt_model(pt_tensor).cpu().numpy()  # [5, 7, 65]

# Keras: NHWC
tf_out = model.predict(test_data, verbose=0)  # [5, 7, 65]

max_diff = np.abs(pt_out - tf_out).max()
print("Max difference (PyTorch vs Keras): %.6f" % max_diff)
print("Outputs match: %s" % ("YES" if max_diff < 1e-4 else "NO (diff too large)"))

for i in range(3):
    pt_pred = "".join(VOCAB[p] for p in pt_out[i].argmax(axis=-1))
    tf_pred = "".join(VOCAB[p] for p in tf_out[i].argmax(axis=-1))
    print("  Sample %d: PyTorch=%s  Keras=%s  %s" % (i, pt_pred, tf_pred,
           "OK" if pt_pred == tf_pred else "DIFF"))

# ============ Step 5: TFLite INT8 量化 ============
print()
print("=" * 60)
print("Step 5: TFLite INT8 Quantization")
print("=" * 60)

# Save Keras model for TFLite conversion
saved_model_dir = os.path.join(OUT_DIR, "keras_saved_model")
model.export(saved_model_dir)
print("Keras SavedModel exported to: %s" % saved_model_dir)

def representative_dataset_gen():
    for i in range(min(200, len(calib['data']))):
        sample = calib['data'][i:i+1]  # [1, 24, 94, 1]
        yield [sample.astype(np.float32)]

# FP32 TFLite
converter_fp32 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_fp32 = converter_fp32.convert()
tflite_fp32_path = os.path.join(OUT_DIR, "lprnet_tiny_fp32.tflite")
with open(tflite_fp32_path, 'wb') as f:
    f.write(tflite_fp32)
fp32_kb = os.path.getsize(tflite_fp32_path) / 1024
print("FP32 TFLite: %.1f KB" % fp32_kb)

# INT8 TFLite
converter_int8 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset_gen
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

tflite_int8 = converter_int8.convert()
tflite_int8_path = os.path.join(OUT_DIR, "lprnet_tiny_int8.tflite")
with open(tflite_int8_path, 'wb') as f:
    f.write(tflite_int8)
int8_kb = os.path.getsize(tflite_int8_path) / 1024
print("INT8 TFLite: %.1f KB" % int8_kb)
print("Compression: %.1f%%" % (int8_kb / fp32_kb * 100))

# ============ Step 6: 验证 INT8 精度 ============
print()
print("=" * 60)
print("Step 6: Verify INT8 accuracy")
print("=" * 60)

interpreter = tf.lite.Interpreter(model_path=tflite_int8_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite INT8 input:  %s dtype=%s" % (input_details[0]['shape'], input_details[0]['dtype']))
print("TFLite INT8 output: %s dtype=%s" % (output_details[0]['shape'], output_details[0]['dtype']))

input_scale, input_zp = input_details[0]['quantization']
output_scale, output_zp = output_details[0]['quantization']

print("Input  quantization: scale=%.6f zp=%d" % (input_scale, input_zp))
print("Output quantization: scale=%.6f zp=%d" % (output_scale, output_zp))

int8_matched = 0
# Use samples 200-299 as test set (separate from calibration 0-199)
test_start = 200
test_end = min(300, len(calib['data']))
for i in range(test_start, test_end):
    sample_fp = calib['data'][i:i+1]  # [1,24,94,1], float32

    # Quantize input
    sample_int8 = np.clip(sample_fp / input_scale + input_zp, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], sample_int8)
    interpreter.invoke()
    int8_output = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output
    int8_output_fp = (int8_output.astype(np.float32) - output_zp) * output_scale

    # Compare with FP32 PyTorch output
    test_nchw_single = np.transpose(sample_fp, (0, 3, 1, 2))
    pt_tensor_single = torch.from_numpy(test_nchw_single.astype(np.float32))
    with torch.no_grad():
        pt_ref = pt_model(pt_tensor_single).cpu().numpy()

    pt_pred_idx = pt_ref[0].argmax(axis=-1)
    int8_pred_idx = int8_output_fp[0].argmax(axis=-1)

    if np.array_equal(pt_pred_idx, int8_pred_idx):
        int8_matched += 1

    if i < 3:
        pt_str = "".join(VOCAB[p] for p in pt_pred_idx)
        int8_str = "".join(VOCAB[p] for p in int8_pred_idx)
        print("  %d: PT=%s INT8=%s %s" % (i, pt_str, int8_str, "OK" if pt_str == int8_str else "DIFF"))

print()
print("INT8 Top-1 match: %d/%d = %.1f%%" % (int8_matched, test_end - test_start, int8_matched/(test_end - test_start)*100))
print()
print("=" * 60)
print("QUANTIZATION COMPLETE")
print("=" * 60)
print("FP32 model:  %s (%.1f KB)" % (tflite_fp32_path, fp32_kb))
print("INT8 model:  %s (%.1f KB)" % (tflite_int8_path, int8_kb))
print()
print("Next step: use GD32 Embedded AI GUI to convert TFLite → C code")
print("  Tool: D:\\桌面\\GD32EmbeddedAI_V1.2.0.39432\\GD32EmbeddedAI_V1.2.0.39432\\GD32EmbeddedAI.exe")
