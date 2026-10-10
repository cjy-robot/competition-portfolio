"""
CCPD2019 → LPRNet 数据集准备
从 CCPD 文件名解析车牌号，裁剪车牌区域，生成 LPRNet 训练数据
"""
import os, cv2, random
import numpy as np
from pathlib import Path

# ============ 配置 ============
CCPD_ROOT = r"D:\CCPD2019"
SPLITS_DIR = os.path.join(CCPD_ROOT, "splits")
OUTPUT_DIR = r"D:\CCPD2019_LPR"
IMG_W, IMG_H = 94, 24  # LPRNet 标准输入尺寸

# ============ 字符映射表 ============
# 省份 (位置0): 31个
PROVINCES = [
    "皖","沪","津","渝","冀","晋","蒙","辽","吉","黑",
    "苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
    "桂","琼","川","贵","云","藏","陕","甘","青","宁","新"
]

# 字母+数字 (位置1-6): A-Z(去掉I,O) + 0-9 = 34
LETTERS_DIGITS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")

# 完整词表: 省份(31) + 字母数字(34) = 65
VOCAB = PROVINCES + LETTERS_DIGITS
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB) + 1  # +1 for CTC blank
BLANK_IDX = len(VOCAB)

print(f"词表大小: {len(VOCAB)} (省份{len(PROVINCES)} + 字母数字{len(LETTERS_DIGITS)})")
print(f"CTC类别数: {NUM_CLASSES} (含blank={BLANK_IDX})")


def parse_ccpd_filename(filename):
    """
    解析 CCPD 文件名，返回 (bbox, plate_code_indices, plate_str)
    文件名格式: [area]_[tilt]-[x1]&[y1]_[x2]&[y2]-[4 corners]-[plate_code]-[brightness]-[blur].jpg
    plate_code: 7个整数用_分隔
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split('-')

    if len(parts) < 5:
        return None

    try:
        # 解析 bbox: parts[2] = "x1&y1_x2&y2"
        coord_parts = parts[2].split('_')
        p1 = coord_parts[0].split('&')
        p2 = coord_parts[1].split('&')
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

        # 解析 plate_code: parts[4] = "0_0_28_29_16_29_32"
        code_str = parts[4]
        code_indices = [int(x) for x in code_str.split('_')]
        if len(code_indices) != 7:
            return None

        # 转换为实际字符
        plate_str = ""
        for i, idx in enumerate(code_indices):
            if i == 0:
                if 0 <= idx < len(PROVINCES):
                    plate_str += PROVINCES[idx]
                else:
                    return None
            else:
                if 0 <= idx < len(LETTERS_DIGITS):
                    plate_str += LETTERS_DIGITS[idx]
                else:
                    return None

        # 转到 VOCAB 索引
        vocab_indices = [CHAR2IDX[c] for c in plate_str]

        return bbox, vocab_indices, plate_str

    except (ValueError, IndexError):
        return None


def crop_plate(img, bbox, margin=0.1):
    """裁剪车牌区域，带边距"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    # 加margin
    mx = int(bw * margin)
    my = int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def process_split(split_name, max_samples=None):
    """处理一个数据集划分"""
    split_file = os.path.join(SPLITS_DIR, f"{split_name}.txt")
    img_out_dir = os.path.join(OUTPUT_DIR, "images", split_name)
    lbl_out_dir = os.path.join(OUTPUT_DIR, "labels", split_name)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    with open(split_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    random.shuffle(lines)
    if max_samples:
        lines = lines[:max_samples]

    entries = []
    success, skip = 0, 0
    for rel in lines:
        src = os.path.join(CCPD_ROOT, rel)
        if not os.path.exists(src):
            skip += 1
            continue

        result = parse_ccpd_filename(rel)
        if result is None:
            skip += 1
            continue
        bbox, label_indices, plate_str = result

        img = cv2.imread(src)
        if img is None:
            skip += 1
            continue

        # 裁剪车牌
        plate_img = crop_plate(img, bbox)
        if plate_img is None:
            skip += 1
            continue

        base = os.path.splitext(os.path.basename(rel))[0]
        img_path = os.path.join(img_out_dir, f"{base}.jpg")
        lbl_path = os.path.join(lbl_out_dir, f"{base}.txt")

        cv2.imwrite(img_path, plate_img)

        with open(lbl_path, 'w', encoding='utf-8') as f:
            # 格式: 字符列表
            f.write(plate_str + '\n')
            f.write(' '.join(map(str, label_indices)) + '\n')

        entries.append((img_path, plate_str))
        success += 1

        if (success + skip) % 1000 == 0:
            print(f"  {split_name}: {success} ok, {skip} skip")

    print(f"  {split_name} done: {success} ok, {skip} skip")
    return entries


if __name__ == "__main__":
    random.seed(42)
    print("=" * 60)

    # 训练集 (全量)
    train_entries = process_split("train")

    # 验证集 (取5000张)
    val_entries = process_split("val", max_samples=5000)

    print(f"\n总训练样本: {len(train_entries)}")
    print(f"总验证样本: {len(val_entries)}")

    # 保存词表
    vocab_path = os.path.join(OUTPUT_DIR, "vocabulary.txt")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for c in VOCAB:
            f.write(c + '\n')
    print(f"词表保存: {vocab_path}")
    print(f"\n输出目录: {OUTPUT_DIR}")
