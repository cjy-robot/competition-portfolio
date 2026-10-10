/**
 * @file    yolo_postprocess.h
 * @brief   YOLO-FastestV2 后处理：解码输出 + NMS (修正版 v2)
 *
 * 针对 plate-best-sim.tflite (3-anchor, already sigmoid'd)
 *
 * 输入:  1×192×192×3 (NHWC, int8, scale=0.003922, zp=-128)
 * 输出0: 1×6×6×16   (int8, stride=32, 大anchor) — 顺序由 nn_model_configure 决定
 * 输出1: 1×12×12×16 (int8, stride=16, 小anchor)
 *
 * ═══════════ 重要: 每个 cell 的 16 通道布局 ═══════════
 *   ch 0-3:   anchor0 reg  [tx, ty, tw, th]   已 sigmoid, 值 ∈ [0,1]
 *   ch 4-7:   anchor1 reg  [tx, ty, tw, th]   已 sigmoid
 *   ch 8-11:  anchor2 reg  [tx, ty, tw, th]   已 sigmoid
 *   ch 12:    anchor0 obj  (objectness)        已 sigmoid
 *   ch 13:    anchor1 obj                      已 sigmoid
 *   ch 14:    anchor2 obj                      已 sigmoid
 *   ch 15:    shared cls                       已 sigmoid
 *
 * 解码公式 (值已经是 sigmoid 后的, 不再重复 sigmoid):
 *   cx = (tx*2 - 0.5 + gx) * stride
 *   cy = (ty*2 - 0.5 + gy) * stride
 *   bw = (tw*2)^2 * aw
 *   bh = (th*2)^2 * ah
 *   conf = obj * cls
 */

#ifndef YOLO_POSTPROCESS_H
#define YOLO_POSTPROCESS_H

#include <stdint.h>

/* ============== 模型参数 ============== */
#define MODEL_INPUT_W           192
#define MODEL_INPUT_H           192
#define NUM_CLASSES             1
#define MAX_DETECTIONS          20

/* 输出头参数 */
#define HEAD0_GRID             12    /* 192/16 = 12 (小 stride, 大网格) */
#define HEAD1_GRID             6     /* 192/32 = 6  (大 stride, 小网格) */
#define HEAD0_STRIDE           16
#define HEAD1_STRIDE           32

/* 每格 16 通道, 3 个 anchor, 1 个共享 cls */
#define NUM_ANCHORS_PER_SCALE  3
#define OUTPUT_CHANNELS        16
#define NUM_HEADS              2

/* 后处理阈值 */
#define CONF_THRESHOLD         0.2f
#define IOU_THRESHOLD          0.3f
#define ASPECT_RATIO_MIN       1.5f
#define ASPECT_RATIO_MAX       6.0f

/*
 * K-means Anchors (192×192, sorted by area)
 * 从 CCPD 数据集重新聚类得到，与 plate_best 训练一致
 *
 * Head0 (stride=16, 12×12 grid): 小面积 anchor, 检测小车牌
 * Head1 (stride=32, 6×6 grid):  大面积 anchor, 检测大车牌
 */
static const float ANCHORS[NUM_HEADS][NUM_ANCHORS_PER_SCALE][2] = {
    /* Head0: stride=16, 12×12 grid */
    { {31.24f, 8.24f}, {54.07f, 13.34f}, {49.55f, 24.23f} },
    /* Head1: stride=32, 6×6 grid */
    { {77.18f, 18.28f}, {69.73f, 35.59f}, {114.20f, 31.10f} }
};

/* 检测结果结构体 */
typedef struct {
    float x1;       /* 左上角 x (192×192 坐标) */
    float y1;       /* 左上角 y */
    float x2;       /* 右下角 x */
    float y2;       /* 右下角 y */
    float score;    /* 置信度 obj * cls */
    int   cls;      /* 类别 ID (0=plate) */
} detection_t;

/**
 * @brief  YOLO 后处理主函数
 *
 * @param  output_data0  模型输出0 (6×6×16, stride=32) int8
 * @param  output_data1  模型输出1 (12×12×16, stride=16) int8
 * @param  output_scale  量化 scale (0.003922 = 1/255)
 * @param  output_zp     量化 zero_point (-128)
 * @param  dets          输出检测结果数组
 * @param  max_dets      数组最大容量
 * @retval 实际检测数量
 *
 * @note   两个输出共用相同的 scale 和 zp
 * @note   如果 nn_model_configure 的输出顺序不同,
 *         交换 output_data0/output_data1 参数即可
 */
int yolo_postprocess(
    int8_t*       output_data0,    /* [1, 6,  6,  16] int8 */
    int8_t*       output_data1,    /* [1, 12, 12, 16] int8 */
    float         output_scale,
    int           output_zp,
    detection_t   dets[],
    int           max_dets
);

/**
 * @brief  NMS (Non-Maximum Suppression)
 */
int nms(detection_t dets[], int num_dets, float iou_thr);

/**
 * @brief  通过串口打印检测结果
 */
void yolo_print_detections(detection_t dets[], int num_dets);

#endif /* YOLO_POSTPROCESS_H */
