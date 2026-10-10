"""
LPRNet 训练脚本 — 中国车牌字符识别
数据集: CCPD2019, 在线裁剪车牌区域
"""
import os, sys, random, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lprnet_model import LPRNetLite, LPRNetTiny, count_params
from prepare_lprnet_data import parse_ccpd_filename, crop_plate, VOCAB, NUM_CLASSES, BLANK_IDX

# ============ 配置 ============
CCPD_ROOT = r"D:\CCPD2019"
SPLITS_DIR = os.path.join(CCPD_ROOT, "splits")
IMG_W, IMG_H = 94, 24
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2  # Windows 下不宜太高
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "tiny"  # "tiny" or "lite"
SAVE_DIR = r"D:\CCPD2019_YOLO\lprnet_weights"

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Model: LPRNet{MODEL_TYPE.title()}")
print(f"Vocabulary: {len(VOCAB)} chars, {NUM_CLASSES} classes (incl blank)")


# ============ Dataset ============
class CCPDPlateDataset(Dataset):
    """从 CCPD 原始图片中在线裁剪车牌"""
    def __init__(self, split="train", max_samples=None, augment=False):
        split_file = os.path.join(SPLITS_DIR, f"{split}.txt")
        with open(split_file, 'r') as f:
            self.lines = [l.strip() for l in f if l.strip()]

        if max_samples:
            random.shuffle(self.lines)
            self.lines = self.lines[:max_samples]

        self.ccpd_root = CCPD_ROOT
        self.augment = augment

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        rel = self.lines[idx]
        src = os.path.join(self.ccpd_root, rel)

        # 解析车牌号
        result = parse_ccpd_filename(rel)
        if result is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        bbox, label_indices, plate_str = result
        img = cv2.imread(src)
        if img is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 裁剪车牌
        plate = crop_plate(img, bbox, margin=0.15)
        if plate is None or plate.size == 0:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 转灰度 → resize → normalize
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_W, IMG_H))

        # 数据增强
        if self.augment:
            gray = self._augment(gray)

        # Normalize to [0, 1]
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0)

        return tensor, torch.tensor(label_indices, dtype=torch.long), plate_str

    def _augment(self, img):
        """简单数据增强"""
        # 随机亮度调整
        if random.random() < 0.3:
            factor = 0.7 + random.random() * 0.6
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        # 随机高斯噪声
        if random.random() < 0.2:
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # 随机模糊
        if random.random() < 0.1:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return img


def collate_fn(batch):
    images, labels, texts = zip(*batch)
    images = torch.stack(images)

    if MODEL_TYPE == "lite":
        # CTC: label_lengths + input_lengths
        labels = torch.cat([l for l in labels])
        label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
        input_lengths = torch.full((len(images),), 47, dtype=torch.long)  # 47 time steps
        return images, labels, label_lengths, input_lengths, texts
    else:
        # 位置分类
        labels = torch.stack(labels)
        return images, labels, texts


# ============ 训练 ============
def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = total = 0

    for batch_idx, batch in enumerate(loader):
        images = batch[0].to(DEVICE)

        if MODEL_TYPE == "lite":
            labels = batch[1].to(DEVICE)
            label_lengths = batch[2].to(DEVICE)
            input_lengths = batch[3].to(DEVICE)

            optimizer.zero_grad()
            log_probs = model(images)  # [B, T, C]
            log_probs = log_probs.permute(1, 0, 2)  # [T, B, C] for CTCLoss
            loss = criterion(log_probs, labels, input_lengths, label_lengths)

        else:
            labels = batch[1].to(DEVICE)  # [B, 7]

            optimizer.zero_grad()
            outputs = model(images)  # [B, 7, C]
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            # 统计准确率
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            acc_str = f" acc={correct/total*100:.1f}%" if MODEL_TYPE == "tiny" and total > 0 else ""
            print(f"  Epoch {epoch:3d} [{batch_idx+1:4d}/{len(loader):4d}] loss={loss.item():.4f}{acc_str}")

    avg_loss = total_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(model, loader):
    model.eval()
    correct_chars = 0
    correct_plates = 0
    total_chars = 0
    total_plates = 0

    for batch in loader:
        images = batch[0].to(DEVICE)

        if MODEL_TYPE == "lite":
            labels = batch[1]
            label_lengths = batch[2]
            texts = batch[4]

            log_probs = model(images)  # [B, T, C]
            preds = log_probs.argmax(dim=-1)  # [B, T]

            for i, text in enumerate(texts):
                length = label_lengths[i].item()
                # CTC greedy decode
                decoded = []
                prev = -1
                for t in range(preds.size(1)):
                    p = preds[i, t].item()
                    if p != BLANK_IDX and p != prev:
                        decoded.append(p)
                    prev = p
                decoded = decoded[:length]

                total_chars += length
                total_plates += 1
                correct = sum(1 for j, d in enumerate(decoded) if j < length and d == labels[j].item())
                correct_chars += correct
                if correct == length:
                    correct_plates += 1
        else:
            labels_t = batch[1].to(DEVICE)
            texts = batch[2]

            outputs = model(images)  # [B, 7, C]
            preds = outputs.argmax(dim=-1)  # [B, 7]

            for i, text in enumerate(texts):
                length = len(text)
                total_chars += length
                total_plates += 1

                p = preds[i][:length]
                l = labels_t[i][:length]
                correct = (p == l).sum().item()
                correct_chars += correct
                if correct == length:
                    correct_plates += 1

    char_acc = correct_chars / total_chars * 100 if total_chars > 0 else 0
    plate_acc = correct_plates / total_plates * 100 if total_plates > 0 else 0
    return char_acc, plate_acc


def main():
    # 数据集
    train_ds = CCPDPlateDataset("train", max_samples=100000, augment=True)
    val_ds = CCPDPlateDataset("val", max_samples=5000, augment=False)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    # 模型
    if MODEL_TYPE == "lite":
        model = LPRNetLite(num_classes=NUM_CLASSES).to(DEVICE)
        criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    else:
        model = LPRNetTiny(num_classes=len(VOCAB), seq_len=7).to(DEVICE)
        criterion = nn.CrossEntropyLoss()

    print(f"Params: {count_params(model):,}")
    print(f"Model size: {count_params(model)/1024:.1f} KB INT8")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch)
        char_acc, plate_acc = validate(model, val_loader)
        scheduler.step()

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | char_acc={char_acc:.1f}% "
              f"| plate_acc={plate_acc:.1f}% | lr={lr:.6f} | {elapsed:.0f}s")

        # 保存最佳
        if char_acc > best_acc:
            best_acc = char_acc
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f"lprnet_{MODEL_TYPE}_best.pth"))
            print(f"  -> Saved best (char_acc={best_acc:.1f}%)")

        # 定期保存
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f"lprnet_{MODEL_TYPE}_epoch{epoch}.pth"))

    print(f"\nTraining done! Best char_acc: {best_acc:.1f}%")


if __name__ == "__main__":
    main()
