============================================================
  CIMC 工业硬件研发（赛题二）— AI模型开发全流程提交文件
============================================================

目录结构:

1-原始模型/         训练完成的原始模型文件
2-量化模型/         INT8量化压缩后的部署模型
3-训练脚本与配置/    模型训练、优化、量化的全部脚本
4-端侧推理代码/      MCU端推理的C语言后处理代码

------------------------------------------------------------
一、YOLO-FastestV2 车牌检测模型
------------------------------------------------------------

原始模型:
  plate_best_120.onnx       Float32 ONNX (987KB) — 训练直接导出，mAP=0.998
  yolo_keras.h5             Keras源模型 — PT权重迁移后的完整模型

量化模型:
  yolo_keras_int8.tflite    ★ INT8 TFLite (401KB) — 最终部署模型，20/20检出
  yolo_keras_float.tflite   Float32 TFLite (961KB) — 参考版本
  yolo_int8_default_symmetric.onnx  INT8 QDQ ONNX (532KB) — PC验证用

训练脚本:
  yolo_train.py             PyTorch训练主脚本
  yolo_detector.py          模型Detector类定义
  yolo_shufflenetv2.py      ShuffleNetV2骨干网络
  yolo_fpn.py               LightFPN特征金字塔
  plate_best.data           训练超参数配置
  recompute_anchors.py      K-means anchors重算脚本

量化脚本:
  yolo_keras_weight_transfer.py  PT->Keras权重迁移 (371/371成功)
  yolo_onnxrt_quantize.py        ONNX Runtime QDQ量化 (PC验证)

------------------------------------------------------------
二、LPRNet Tiny 车牌识别模型
------------------------------------------------------------

原始模型:
  lprnet_tiny_best.pth     PyTorch权重 — 训练最佳checkpoint

量化模型:
  lprnet_tiny_int8.tflite  ★ INT8 TFLite (118KB) — 最终部署模型，95%整牌

训练脚本:
  lprnet_train.py          训练主脚本
  lprnet_model.py          模型定义 (LPRNetTiny)
  prepare_lprnet_data.py   数据预处理 + VOCAB字符表

量化脚本:
  lprnet_int8_quantize.py  PT->Keras->TFLite INT8量化

------------------------------------------------------------
三、MCU端推理代码 (C语言)
------------------------------------------------------------
  yolo_postprocess.c       YOLO解码+NMS (3 anchor, K-means, 修正版)
  yolo_postprocess.h       解码头文件 (anchors/结构体/阈值)
  main.c                   主控流程 (已适配新接口)

------------------------------------------------------------
模型接口说明:

YOLO输入:  [1, 192, 192, 3]  int8  scale=0.003922  zp=-128
YOLO输出0: [1, 12, 12, 16]   int8  stride=16, 3 small anchors
YOLO输出1: [1, 6,  6,  16]   int8  stride=32, 3 large anchors
Anchors: Head0=[31.24,8.24][54.07,13.34][49.55,24.23]
         Head1=[77.18,18.28][69.73,35.59][114.20,31.10]
解码: cx=(tx*2-0.5+gx)*stride  bw=(tw*2)^2*aw  conf=obj*cls

LPRNet输入:  [1, 24, 94, 1]  int8  scale=0.003922  zp=-128
LPRNet输出:  [1, 7, 66]       int8  scale=0.2652    zp=-18
VOCAB: 31省简称(索引0-30) + 34字母数字(索引31-64)
