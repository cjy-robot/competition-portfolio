/**
 * @file    yolo_postprocess.c
 * @brief   YOLO-FastestV2 后处理实现 (修正版 v2)
 *
 * 将 INT8 模型输出解码为车牌检测框，并执行 NMS
 *
 * 关键修正:
 *   1. 3 anchor/scale (原代码误用 2)
 *   2. 正确 channel 布局: reg[a*4..a*4+3], obj[12+a], cls[15]
 *   3. 不再重复 sigmoid — 模型输出已经 sigmoid 过
 *   4. 正确 K-means anchors
 */

#include "yolo_postprocess.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/* ============== 内部辅助函数 ============== */

static inline float dequant_val(int8_t val, float scale, int zp)
{
    return ((float)val - (float)zp) * scale;
}

static float box_iou(detection_t *a, detection_t *b)
{
    float ix1 = a->x1 > b->x1 ? a->x1 : b->x1;
    float iy1 = a->y1 > b->y1 ? a->y1 : b->y1;
    float ix2 = a->x2 < b->x2 ? a->x2 : b->x2;
    float iy2 = a->y2 < b->y2 ? a->y2 : b->y2;

    float iw = ix2 - ix1;
    float ih = iy2 - iy1;

    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;

    float inter  = iw * ih;
    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    float union_area = area_a + area_b - inter;

    return inter / (union_area + 1e-6f);
}

/* ============== 解码单个输出头 ============== */

/**
 * @brief 解码一个特征图输出
 *
 * @param output     int8 输出数据 [grid_size * grid_size * 16]
 * @param scale      量化 scale (0.003922)
 * @param zp         量化 zero_point (-128)
 * @param grid_size  网格尺寸 (12 或 6)
 * @param stride     步长 (16 或 32)
 * @param anchors    该头的 anchor 数组 [num_anchors][2]
 * @param num_anchors anchor 数量 (3)
 * @param dets       输出检测结果
 * @param max_dets   最大容量
 * @param count      当前已检测数量 (in/out)
 */
static int decode_head(
    int8_t       *output,
    float         scale,
    int           zp,
    int           grid_size,
    int           stride,
    const float   anchors[][2],
    int           num_anchors,
    detection_t  *dets,
    int           max_dets,
    int          *count
)
{
    /*
     * 数据布局 (NHWC, C=16):
     *   cell_offset = (gy * grid_size + gx) * 16
     *
     *   对于每个 anchor a ∈ [0, 2]:
     *     tx  = dequant_val(output[cell_offset + a*4 + 0])
     *     ty  = dequant_val(output[cell_offset + a*4 + 1])
     *     tw  = dequant_val(output[cell_offset + a*4 + 2])
     *     th  = dequant_val(output[cell_offset + a*4 + 3])
     *     obj = dequant_val(output[cell_offset + 12 + a])
     *     cls = dequant_val(output[cell_offset + 15])
     */

    for (int gy = 0; gy < grid_size; gy++) {
        for (int gx = 0; gx < grid_size; gx++) {

            /* 当前 cell 的起始偏移 (16 channels) */
            int cell_base = (gy * grid_size + gx) * OUTPUT_CHANNELS;

            /* 共享 cls (channel 15) — 所有 anchor 共用 */
            float cls = dequant_val(output[cell_base + 15], scale, zp);
            /* cls 已经 sigmoid, 置信度 = obj * cls */

            for (int a = 0; a < num_anchors; a++) {
                /* reg: channels a*4 .. a*4+3 */
                float tx = dequant_val(output[cell_base + a*4 + 0], scale, zp);
                float ty = dequant_val(output[cell_base + a*4 + 1], scale, zp);
                float tw = dequant_val(output[cell_base + a*4 + 2], scale, zp);
                float th = dequant_val(output[cell_base + a*4 + 3], scale, zp);

                /* obj: channel 12 + a */
                float obj = dequant_val(output[cell_base + 12 + a], scale, zp);

                /*
                 * 关键: tx, ty, tw, th, obj, cls 都已经过 sigmoid
                 * 值在 [0, 1], 不再重复 sigmoid!
                 */
                float conf = obj * cls;

                if (conf < CONF_THRESHOLD) continue;

                /*
                 * YOLO-FastestV2 解码 (值已 sigmoid):
                 *   cx = (tx*2 - 0.5 + gx) * stride
                 *   cy = (ty*2 - 0.5 + gy) * stride
                 *   bw = (tw*2)^2 * aw
                 *   bh = (th*2)^2 * ah
                 */
                float cx = (tx * 2.0f - 0.5f + (float)gx) * (float)stride;
                float cy = (ty * 2.0f - 0.5f + (float)gy) * (float)stride;
                float bw = (tw * 2.0f) * (tw * 2.0f) * anchors[a][0];
                float bh = (th * 2.0f) * (th * 2.0f) * anchors[a][1];

                /* 转为 (x1, y1, x2, y2) */
                float x1 = cx - bw / 2.0f;
                float y1 = cy - bh / 2.0f;
                float x2 = cx + bw / 2.0f;
                float y2 = cy + bh / 2.0f;

                /* 有效性检查 */
                if (x2 <= x1 || y2 <= y1) continue;

                /* 长宽比过滤 (车牌: 1.5~6:1) */
                float aspect = (x2 - x1) / (y2 - y1 + 1e-6f);
                if (aspect < ASPECT_RATIO_MIN || aspect > ASPECT_RATIO_MAX) continue;

                /* 裁剪到图像范围 */
                if (x1 < 0.0f) x1 = 0.0f;
                if (y1 < 0.0f) y1 = 0.0f;
                if (x2 > (float)MODEL_INPUT_W) x2 = (float)MODEL_INPUT_W;
                if (y2 > (float)MODEL_INPUT_H) y2 = (float)MODEL_INPUT_H;

                if (*count >= max_dets) return *count;

                dets[*count].x1    = x1;
                dets[*count].y1    = y1;
                dets[*count].x2    = x2;
                dets[*count].y2    = y2;
                dets[*count].score = conf;
                dets[*count].cls   = 0;
                (*count)++;
            }
        }
    }

    return *count;
}

/* ============== 公共函数 ============== */

int yolo_postprocess(
    int8_t*       output_data0,
    int8_t*       output_data1,
    float         output_scale,
    int           output_zp,
    detection_t   dets[],
    int           max_dets
)
{
    int count = 0;

    /*
     * 注意输出顺序:
     *   output_data0 = [1, 6,  6,  16] → stride=32, HEAD1_GRID
     *   output_data1 = [1, 12, 12, 16] → stride=16, HEAD0_GRID
     *
     * 如果 nn_model_configure 的输出顺序相反, 把下面两行交换即可
     */

    /* Head1: 6×6 grid, stride=32, 大 anchor */
    decode_head(
        output_data0, output_scale, output_zp,
        HEAD1_GRID, HEAD1_STRIDE,
        ANCHORS[1], NUM_ANCHORS_PER_SCALE,
        dets, max_dets, &count
    );

    /* Head0: 12×12 grid, stride=16, 小 anchor */
    decode_head(
        output_data1, output_scale, output_zp,
        HEAD0_GRID, HEAD0_STRIDE,
        ANCHORS[0], NUM_ANCHORS_PER_SCALE,
        dets, max_dets, &count
    );

    /* NMS */
    count = nms(dets, count, IOU_THRESHOLD);

    return count;
}

/* ============== NMS ============== */

int nms(detection_t dets[], int num_dets, float iou_thr)
{
    if (num_dets <= 1) return num_dets;

    /* 按置信度降序排序 (冒泡) */
    for (int i = 0; i < num_dets - 1; i++) {
        for (int j = 0; j < num_dets - 1 - i; j++) {
            if (dets[j].score < dets[j + 1].score) {
                detection_t tmp = dets[j];
                dets[j] = dets[j + 1];
                dets[j + 1] = tmp;
            }
        }
    }

    /* 标记抑制 */
    int suppressed[MAX_DETECTIONS];
    memset(suppressed, 0, sizeof(suppressed));

    for (int i = 0; i < num_dets; i++) {
        if (suppressed[i]) continue;
        for (int j = i + 1; j < num_dets; j++) {
            if (suppressed[j]) continue;
            if (dets[i].cls == dets[j].cls) {
                if (box_iou(&dets[i], &dets[j]) > iou_thr) {
                    suppressed[j] = 1;
                }
            }
        }
    }

    /* 移除被抑制的检测 */
    int kept = 0;
    for (int i = 0; i < num_dets; i++) {
        if (!suppressed[i]) {
            if (kept != i) {
                dets[kept] = dets[i];
            }
            kept++;
        }
    }

    return kept;
}

/* ============== 调试输出 ============== */

void yolo_print_detections(detection_t dets[], int num_dets)
{
    printf("\r\n=== YOLO Detection Results: %d plate(s) ===\r\n", num_dets);
    for (int i = 0; i < num_dets; i++) {
        printf("  [#%d] box=(%d,%d,%d,%d) conf=%.3f\r\n",
            i,
            (int)dets[i].x1, (int)dets[i].y1,
            (int)dets[i].x2, (int)dets[i].y2,
            dets[i].score
        );
    }
    printf("==========================================\r\n");
}
