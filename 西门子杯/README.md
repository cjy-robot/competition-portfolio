# CIMC"西门子杯" · 工业硬件研发赛项（边缘 AI 方向）

**国家级一等奖** ｜ 队伍编号 2026462686 ｜ 2026.03–2026.07 ｜ 5 人团队

**本人负责：边缘 AI 模型全链路** —— 模型选型、训练、INT8 量化压缩、端侧部署与 C 语言后处理实现。

---

## 项目一句话

在 **GD32H759（Cortex-M7，2 MB Flash / 1 MB SRAM）裸机平台**上，不依赖 PC 与 GPU，
完成"车牌检测 → 字符识别"双模型级联推理，作为云台式停车巡检系统的端侧视觉核心。

## 关键技术决策

| 问题 | 方案 | 结果 |
|---|---|---|
| M7 上跑什么检测网络 | 对比 6 种轻量网络，选定 YOLO-FastestV2（238K 参数 / 0.06G MACs） | 排除 YOLOv5-nano / YOLOv8-nano 的 Flash 与 SRAM 溢出风险 |
| 如何压缩到 2 MB Flash | INT8 训练后静态量化（PTQ），500 张 CCPD 样本 MinMax 校准 | 文件 1402 KB → 519 KB（2.70 倍）；Flash 实际占用 461 KB |
| ONNX → TFLite 算子不兼容 | 自研 Python 侧 PyTorch → Keras 权重迁移 | **371/371 层迁移成功**，替代 ONNX 直转路线 |
| 两模型内存叠加超限 | 串行推理共用峰值 334 KB 静态缓冲区，权重按链接脚本分布到 Flash / AXI SRAM / 外部 RAM | 片上 RAM 峰值 430 KB |
| 后处理如何上 MCU | 手写 YOLO 解码 + NMS 的 C 实现，K-means 重算 anchors | 与量化模型接口对齐，裸机运行 |

## 端侧实测（MCU 上真跑出来的数）

| 模型 | 精度 | 延迟 | 帧率 | Flash | RAM 峰值 |
|---|---|---|---|---|---|
| YOLO-FastestV2 检测 | 94%（500 样本） | 101 ms | 10 fps | 309 KB | 349 KB |
| LPRNet Tiny 识别 | 92%（500 样本） | 20 ms | 50 fps | 152 KB | 81 KB |

---

## 目录导航

### 📄 方案与答辩

- [`项目方案.pdf`](项目方案.pdf) — 完整工程设计方案，78 页
- [`答辩PPT.pdf`](答辩PPT.pdf) — 决赛答辩材料（原 PPT 内嵌 176 MB 演示视频，此处为 PDF 版）

### 🧠 模型与脚本/ — 可直接复现的全流程

| 目录 | 内容 |
|---|---|
| `1-原始模型/` | 训练产出：`plate_best_120.onnx`（987 KB，AP 99.98%）、`yolo_keras.h5`、`lprnet_tiny_best.pth` |
| `2-量化模型/` | 部署模型：`yolo_keras_int8.tflite`（401 KB，★最终部署）、`lprnet_tiny_int8.tflite`（118 KB，★最终部署） |
| `3-训练脚本与配置/` | 13 个脚本：训练、Detector 定义、ShuffleNetV2 骨干、LightFPN、anchor 重算、**PT→Keras 权重迁移**、INT8 量化 |
| `4-端侧推理代码/` | `yolo_postprocess.c/h`（YOLO 解码 + NMS）、`main.c` |
| `README.txt` | **模型接口说明**：int8 scale / zero-point、anchors、解码公式、Vocab 表 |

### 📊 量化对比样本/ — 量化前后 9 类场景对比

`base` `blur` `rotate` `tilt` `weather` `challenge` `np` `db` `fn` —— 覆盖模糊、旋转、倾斜、恶劣天气等边界工况，
用于验证量化是否引入精度退化，以及定位漏检样本。

### 📈 图表/

系统架构图、量化流程图、YOLO / LPRNet 网络结构图、通道混洗示意、时序分析、类别分布、消融实验、
精度-体积雷达图、训练曲线（PNG + PDF + LaTeX 源）。

### 📝 技术文档/

`全部工作梳理.md`、`完整流程梳理.md`、`交付总结.md`、`模型参数卡片.md` —— 开发过程的技术复盘。
