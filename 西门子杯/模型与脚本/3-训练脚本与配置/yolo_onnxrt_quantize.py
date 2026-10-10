"""
YOLO-FastestV2 ONNX Runtime 静态 INT8 量化
基于 STM32 AI Model Zoo 验证过的 ShuffleNetV2 量化方案
使用 QDQ (Quantize-Dequantize) 格式
"""
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationMethod
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process
import cv2, os, glob, random, sys

sys.path.insert(0, r"D:\CCPD2019_YOLO")

# ==================== 配置 ====================
ONNX_PATH = r"D:\CCPD2019_YOLO\plate_best_120.onnx"
OUT_DIR = r"D:\CCPD2019_YOLO\qat_output"
os.makedirs(OUT_DIR, exist_ok=True)

CALIB_SAMPLES = 200
SRC_DIR = r"D:\CCPD2019_YOLO\images_mixed"

# ==================== Step 1: 预处理 ONNX ====================
print("=" * 60)
print("Step 1: Pre-process ONNX model")
print("=" * 60)

onnx_preprocessed = os.path.join(OUT_DIR, "plate_best_120_preproc.onnx")

try:
    quant_pre_process(ONNX_PATH, onnx_preprocessed, skip_optimization=False, skip_shape_inference=False)
    print(f"Pre-processed: {onnx_preprocessed}")
except Exception as e:
    print(f"Pre-process failed (may be OK): {e}")
    onnx_preprocessed = ONNX_PATH
    print("Using original ONNX directly")

# ==================== Step 2: 生成校准数据 ====================
print()
print("=" * 60)
print("Step 2: Generate calibration data")
print("=" * 60)

random.seed(42)
all_files = []
for cat in ['ccpd_base', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db',
            'ccpd_fn', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather']:
    cat_dir = os.path.join(SRC_DIR, cat)
    if os.path.isdir(cat_dir):
        files = glob.glob(os.path.join(cat_dir, "*.jpg"))
        n = max(30, int(CALIB_SAMPLES * len(files) / 200000))
        all_files.extend(random.sample(files, min(n, len(files))))
random.shuffle(all_files)
all_files = all_files[:CALIB_SAMPLES]

calib_data = []
for f in all_files:
    img = cv2.imread(f)
    if img is None: continue
    img192 = cv2.resize(img, (192, 192))
    img_rgb = cv2.cvtColor(img192, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_nchw = np.transpose(img_rgb, (2, 0, 1))  # [3,192,192]
    calib_data.append(img_nchw)

calib_data = np.stack(calib_data, axis=0)  # [N, 3, 192, 192]
print(f"Calibration data: {calib_data.shape}")

# ==================== Step 3: 校准数据读取器 ====================
class YoloCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data, input_name, batch_size=1):
        self.data = data
        self.input_name = input_name
        self.batch_size = batch_size
        self.iter = iter(range(0, len(data), batch_size))

    def get_next(self):
        try:
            idx = next(self.iter)
            batch = self.data[idx:idx + self.batch_size]
            return {self.input_name: batch}
        except StopIteration:
            return None

# Get input name from ONNX
temp_sess = ort.InferenceSession(ONNX_PATH)
input_name = temp_sess.get_inputs()[0].name
print(f"ONNX input name: {input_name}")

calib_reader = YoloCalibrationDataReader(calib_data, input_name)

# ==================== Step 4: 静态量化 ====================
print()
print("=" * 60)
print("Step 4: Static INT8 Quantization (QDQ)")
print("=" * 60)

onnx_int8 = os.path.join(OUT_DIR, "yolo_int8_qdq.onnx")

# Try multiple quantization configs
configs = [
    {
        'name': 'default_symmetric',
        'extra_options': {
            'ActivationSymmetric': True,
            'WeightSymmetric': True,
        }
    },
    {
        'name': 'asymmetric_activation',
        'extra_options': {
            'ActivationSymmetric': False,
            'WeightSymmetric': True,
        }
    },
]

best_config = None
for cfg in configs:
    out_path = os.path.join(OUT_DIR, f"yolo_int8_{cfg['name']}.onnx")
    print(f"\nTrying: {cfg['name']}...")
    try:
        quantize_static(
            model_input=onnx_preprocessed,
            model_output=out_path,
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=False,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options=cfg['extra_options'],
        )
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  SUCCESS: {size_kb:.0f} KB")
        best_config = out_path
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")

if best_config is None:
    print("\nAll configs failed! Using original model.")
    exit(1)

# ==================== Step 5: 验证精度 ====================
print()
print("=" * 60)
print("Step 5: Verify INT8 vs FP32 accuracy")
print("=" * 60)

# Load test data
test_data = np.load(r"D:\CCPD2019_YOLO\test_plate_data.npz")
test_imgs = test_data['data'][:15]

# FP32 session
sess_fp32 = ort.InferenceSession(ONNX_PATH)
# INT8 session
sess_int8 = ort.InferenceSession(best_config)

H0_A = np.array([[31.24,8.24],[54.07,13.34],[49.55,24.23]], dtype=np.float32)
H1_A = np.array([[77.18,18.28],[69.73,35.59],[114.20,31.10]], dtype=np.float32)

def decode_head(data, anchors, stride, thr=0.2):
    h, w = data.shape[0], data.shape[1]
    dets = []
    for gy in range(h):
        for gx in range(w):
            cell = data[gy, gx]
            for a in range(3):
                base = a * 4
                conf = float(cell[12+a]) * float(cell[15])
                if conf < thr: continue
                aw, ah = anchors[a]
                tx,ty,tw,th = float(cell[base]),float(cell[base+1]),float(cell[base+2]),float(cell[base+3])
                cx=(tx*2-0.5+gx)*stride; cy=(ty*2-0.5+gy)*stride
                bw=(tw*2)**2*aw; bh=(th*2)**2*ah
                x1,y1=cx-bw/2,cy-bh/2;x2,y2=cx+bw/2,cy+bh/2
                if x2<=x1 or y2<=y1: continue
                if 1.5<=(x2-x1)/(y2-y1+1e-6)<=6.0: dets.append([x1,y1,x2,y2,conf])
    return np.array(dets)

def nms(dets, thr=0.3):
    if len(dets)<=1: return dets
    o=np.argsort(-dets[:,4]); d=dets[o]; k=[]
    for i in range(len(d)):
        sup=False
        for j in k:
            ix1,iy1=max(d[i,0],d[j,0]),max(d[i,1],d[j,1])
            ix2,iy2=min(d[i,2],d[j,2]),min(d[i,3],d[j,3])
            if ix2<=ix1 or iy2<=iy1: continue
            inter=(ix2-ix1)*(iy2-iy1)
            ai=(d[i,2]-d[i,0])*(d[i,3]-d[i,1]); aj=(d[j,2]-d[j,0])*(d[j,3]-d[j,1])
            if inter/(ai+aj-inter+1e-6)>thr: sup=True; break
        if not sup: k.append(i)
    return d[k]

ok = 0
print(f"{'Img':>4}  {'FP32':^22}  {'INT8':^22}  {'Diff':>7}")
print("-" * 60)

for idx in range(15):
    img_nhwc = test_imgs[idx:idx+1].astype(np.float32)
    img_nchw = np.transpose(img_nhwc, (0, 3, 1, 2))

    fp32_out = sess_fp32.run(None, {input_name: img_nchw})
    int8_out = sess_int8.run(None, {input_name: img_nchw})

    d0=decode_head(fp32_out[0][0],H0_A,16,0.2); d1=decode_head(fp32_out[1][0],H1_A,32,0.2)
    fp32_d=np.vstack([d0,d1]) if len(d0)>0 and len(d1)>0 else (d0 if len(d0)>0 else d1)
    fp32_d=nms(fp32_d,0.3) if len(fp32_d)>0 else np.zeros((0,5))

    d0=decode_head(int8_out[0][0],H0_A,16,0.2); d1=decode_head(int8_out[1][0],H1_A,32,0.2)
    int8_d=np.vstack([d0,d1]) if len(d0)>0 and len(d1)>0 else (d0 if len(d0)>0 else d1)
    int8_d=nms(int8_d,0.3) if len(int8_d)>0 else np.zeros((0,5))

    fp_s = f'[{fp32_d[0][0]:.0f},{fp32_d[0][1]:.0f},{fp32_d[0][2]:.0f},{fp32_d[0][3]:.0f}]' if len(fp32_d)>0 else 'MISS'
    i8_s = f'[{int8_d[0][0]:.0f},{int8_d[0][1]:.0f},{int8_d[0][2]:.0f},{int8_d[0][3]:.0f}]' if len(int8_d)>0 else 'MISS'

    diff = ''
    if len(fp32_d)>0 and len(int8_d)>0:
        d = np.abs(fp32_d[0,:4]-int8_d[0,:4]).max()
        diff = f'{d:.1f}px'
        if d < 3: ok += 1
    elif len(fp32_d)==0 and len(int8_d)==0:
        diff = 'OK'; ok += 1
    print(f'{idx:>4}  {fp_s:^22}  {i8_s:^22}  {diff:>7}')

print(f"\nMatch: {ok}/15 = {ok/15*100:.0f}%")
print(f"Model: {best_config} ({os.path.getsize(best_config)/1024:.0f} KB)")
print(f"Acuracy loss: <3px match rate = {ok/15*100:.0f}%")
