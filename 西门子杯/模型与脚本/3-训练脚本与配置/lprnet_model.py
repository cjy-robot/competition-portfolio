"""
轻量 LPRNet 模型 — 面向 GD32H759I MCU 部署
中国车牌: 7个字符 (省份+字母数字)
词表: 31省 + 34字母数字 = 65 + blank = 66
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallBasicBlock(nn.Module):
    """LPRNet 基础模块: 1x3 + 3x1 深度可分离替代"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = in_ch // 4
        self.conv1 = nn.Conv2d(in_ch, mid, (1, 3), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, out_ch, (3, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return F.relu(out + r)


class LPRNetLite(nn.Module):
    """
    轻量 LPRNet: ~0.1M 参数, 适合 MCU INT8 部署
    输入: [B, 1, 24, 94] 灰度车牌图
    输出: [B, T, num_classes] CTC logits
    """
    def __init__(self, num_classes=66, gru_hidden=64):
        super().__init__()
        self.num_classes = num_classes

        # Backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 94x24 → 47x12

        self.block1 = SmallBasicBlock(32, 64)
        self.block2 = SmallBasicBlock(64, 64)
        self.pool2 = nn.MaxPool2d((2, 1), (2, 1))  # 47x12 → 23x12

        self.block3 = SmallBasicBlock(64, 96)
        self.block4 = SmallBasicBlock(96, 96)
        self.pool3 = nn.MaxPool2d((2, 1), (2, 1))  # 23x12 → 11x12

        self.dropout = nn.Dropout(0.3)

        # Adaptive → height=1, keep width as sequence
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=96,
            hidden_size=gru_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, x):
        # x: [B, 1, 24, 94]
        x = self.conv1(x)        # [B, 32, 24, 94]
        x = self.conv2(x)        # [B, 32, 24, 94]
        x = self.pool1(x)        # [B, 32, 12, 47]

        x = self.block1(x)       # [B, 64, 12, 47]
        x = self.block2(x)       # [B, 64, 12, 47]
        x = self.pool2(x)        # [B, 64, 12, 23]

        x = self.block3(x)       # [B, 96, 12, 23]
        x = self.block4(x)       # [B, 96, 12, 23]
        x = self.pool3(x)        # [B, 96, 12, 11]

        x = self.dropout(x)
        x = self.adaptive_pool(x)  # [B, 96, 1, 11]

        # Reshape for GRU: [B, 96, 1, 11] → [B, 11, 96]
        x = x.squeeze(2).permute(0, 2, 1)  # [B, 11, 96]

        x, _ = self.gru(x)       # [B, 11, 128]
        x = self.fc(x)           # [B, 11, num_classes]

        # CTC loss expects log_softmax
        return F.log_softmax(x, dim=-1)


class LPRNetTiny(nn.Module):
    """
    轻量 LPRNet: ~80K 参数, 1D Conv 保留序列信息
    输入: [B, 1, 24, 94] 灰度车牌图
    输出: [B, 7, num_classes] 每个位置独立分类
    """
    def __init__(self, num_classes=65, seq_len=7, feat_dim=96):
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.feat_dim = feat_dim

        # Backbone — 保留宽度作为序列
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 94x24 → 47x12

        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))  # 47x12 → 23x6

        self.conv3 = nn.Sequential(
            nn.Conv2d(48, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d((2, 2), (2, 2))  # 23x6 → 11x3

        self.dropout = nn.Dropout(0.3)

        # 高度 → 1, 宽度保持
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))  # → [B, feat_dim, 1, 11]

        # Conv1d: 11步 → 7步 (kernel=5, stride=1)
        self.conv1d = nn.Conv1d(feat_dim, feat_dim, kernel_size=5, stride=1, padding=0)
        # 11 - 5 + 1 = 7 positions

        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # x: [B, 1, 24, 94]
        x = self.pool1(self.conv1(x))   # [B, 24, 12, 47]
        x = self.pool2(self.conv2(x))   # [B, 48, 6, 23]
        x = self.pool3(self.conv3(x))   # [B, feat_dim, 3, 11]
        x = self.pool_h(x)              # [B, feat_dim, 1, 11]
        x = self.dropout(x)
        x = x.squeeze(2)                # [B, feat_dim, 11]

        x = self.conv1d(x)              # [B, feat_dim, 7]
        x = x.permute(0, 2, 1)          # [B, 7, feat_dim]
        x = self.fc(x)                  # [B, 7, num_classes]
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # 测试
    for name, model_cls in [("LPRNetLite", LPRNetLite), ("LPRNetTiny", LPRNetTiny)]:
        if name == "LPRNetTiny":
            model = model_cls(num_classes=65, seq_len=7)
        else:
            model = model_cls(num_classes=66)
        n = count_params(model)
        print(f"{name}: {n:,} params ({n*4/1024:.1f} KB float32, ~{n/1024:.1f} KB int8)")

        x = torch.randn(1, 1, 24, 94)
        with torch.no_grad():
            y = model(x)
        print(f"  Input:  {x.shape}")
        print(f"  Output: {y.shape}\n")
