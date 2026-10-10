"""对混合数据集重新聚类计算 YOLO Anchor (192x192 输入)"""
import numpy as np
import os
from tqdm import tqdm

LABEL_DIR = r"D:\CCPD2019_YOLO\labels_mixed"
IMG_W, IMG_H = 192, 192

# 读取所有 label 文件，提取归一化的 w, h
boxes = []
label_files = []
for root, dirs, files in os.walk(LABEL_DIR):
    for f in files:
        if f.endswith('.txt'):
            label_files.append(os.path.join(root, f))
print(f"Found {len(label_files)} label files")

for fpath in tqdm(label_files):
    with open(fpath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class cx cy w h (normalized)
                w = float(parts[3])
                h = float(parts[4])
                if w > 0 and h > 0:
                    boxes.append([w * IMG_W, h * IMG_H])

boxes = np.array(boxes, dtype=np.float32)
print(f"Total boxes: {len(boxes)}")
print(f"Box stats (pixels at {IMG_W}x{IMG_H}):")
print(f"  Width:  min={boxes[:,0].min():.1f}  mean={boxes[:,0].mean():.1f}  max={boxes[:,0].max():.1f}")
print(f"  Height: min={boxes[:,1].min():.1f}  mean={boxes[:,1].mean():.1f}  max={boxes[:,1].max():.1f}")
print(f"  Aspect ratio (w/h): min={((boxes[:,0]+1e-6)/(boxes[:,1]+1e-6)).min():.2f}  mean={((boxes[:,0]+1e-6)/(boxes[:,1]+1e-6)).mean():.2f}  max={((boxes[:,0]+1e-6)/(boxes[:,1]+1e-6)).max():.2f}")

# K-means 聚类 (IoU 距离)
def iou_distance(boxes, centroids):
    """计算 boxes 和 centroids 之间的 IoU 距离 (1 - IoU)"""
    # boxes: [N, 2], centroids: [K, 2]
    inter_w = np.minimum(boxes[:, None, 0], centroids[None, :, 0])
    inter_h = np.minimum(boxes[:, None, 1], centroids[None, :, 1])
    inter_area = inter_w * inter_h
    box_area = boxes[:, 0] * boxes[:, 1]
    centroid_area = centroids[:, 0] * centroids[:, 1]
    union_area = box_area[:, None] + centroid_area[None, :] - inter_area
    iou = inter_area / (union_area + 1e-16)
    return 1 - iou

def kmeans_anchors(boxes, k, max_iter=100):
    """K-means 聚类 anchor boxes"""
    # 初始聚类中心
    indices = np.random.choice(len(boxes), k, replace=False)
    centroids = boxes[indices].copy()

    for _ in range(max_iter):
        distances = iou_distance(boxes, centroids)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([boxes[labels == i].mean(axis=0) if (labels == i).any() else centroids[i] for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids

# 多次运行取最优
best_centroids = None
best_avg_iou = 0
for run in range(10):
    centroids = kmeans_anchors(boxes, 6)
    distances = iou_distance(boxes, centroids)
    avg_iou = 1 - distances.min(axis=1).mean()
    print(f"Run {run+1}: avg IoU = {avg_iou:.4f}")
    if avg_iou > best_avg_iou:
        best_avg_iou = avg_iou
        best_centroids = centroids

# 按面积排序，分成两个 head
areas = best_centroids[:, 0] * best_centroids[:, 1]
idx = np.argsort(areas)
sorted_anchors = best_centroids[idx]
head0 = sorted_anchors[:3]  # stride=16, small anchors
head1 = sorted_anchors[3:]  # stride=32, large anchors

print(f"\nBest avg IoU: {best_avg_iou:.4f}")
print(f"Head0 (stride=16, 12x12 grid):")
for a in head0:
    print(f"  [{a[0]:.2f}, {a[1]:.2f}]")
print(f"Head1 (stride=32, 6x6 grid):")
for a in head1:
    print(f"  [{a[0]:.2f}, {a[1]:.2f}]")

# 对比当前 anchor
current = np.array([
    [47.30, 10.52], [59.75, 17.20], [62.45, 12.91],
    [70.81, 24.55], [79.95, 16.35], [99.27, 21.31]
])
d_curr = iou_distance(boxes, current)
avg_curr = 1 - d_curr.min(axis=1).mean()
print(f"\nCurrent anchors avg IoU: {avg_curr:.4f} (new: {best_avg_iou:.4f})")
print(f"Improvement: {(best_avg_iou - avg_curr) * 100:.2f}%")
