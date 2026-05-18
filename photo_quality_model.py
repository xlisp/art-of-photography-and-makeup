"""
照片美学评分模型 —— 基于 README.md 的"数学之美"思想
==================================================

核心原则映射到模型结构：
  - 黄金比 / 三分法     -> RuleOfThirdsHead   (主体放在 0.382 / 0.618 / 三分点)
  - 对称                -> SymmetryHead       (左右镜像差)
  - 反差（明暗 + 色彩）  -> ContrastHead       (亮度方差 + 互补色距离)
  - 统一（三色法则）     -> ColorUnityHead     (色相聚类数 ≤ 3)
  - 虚实反差            -> DepthHead          (背景虚化程度，用拉普拉斯方差近似)
  - 比例（不要天太多）   -> FramingHead        (上下/左右能量分布)

最终得分 = 加权融合 6 个分项 + 一个 CNN 全局美学头。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ----------------------------------------------------------------------
# 1. 手工数学特征（README 中明确写出的几何/色彩规则）
# ----------------------------------------------------------------------

def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (B,3,H,W) in [0,1] -> hsv: (B,3,H,W), H in [0,1]."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    cmax, argmax = rgb.max(dim=1)
    cmin, _ = rgb.min(dim=1)
    delta = cmax - cmin + 1e-8

    h = torch.zeros_like(cmax)
    mask_r = (argmax == 0)
    mask_g = (argmax == 1)
    mask_b = (argmax == 2)
    h[mask_r] = ((g - b) / delta)[mask_r] % 6
    h[mask_g] = ((b - r) / delta)[mask_g] + 2
    h[mask_b] = ((r - g) / delta)[mask_b] + 4
    h = (h / 6.0) % 1.0

    s = torch.where(cmax > 0, delta / (cmax + 1e-8), torch.zeros_like(cmax))
    v = cmax
    return torch.stack([h, s, v], dim=1)


def saliency_map(img: torch.Tensor) -> torch.Tensor:
    """简化显著性：梯度幅度（Sobel）—— 主体通常落在高频边缘聚集处。
    img: (B,3,H,W) -> sal: (B,1,H,W)
    """
    gray = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).unsqueeze(1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    # 归一化到每张图的概率分布
    B = mag.shape[0]
    flat = mag.view(B, -1)
    flat = flat / (flat.sum(dim=1, keepdim=True) + 1e-8)
    return flat.view_as(mag)


def rule_of_thirds_score(img: torch.Tensor) -> torch.Tensor:
    """主体距离三分点 / 黄金分割点越近，分数越高。
    返回 (B,) in [0,1]。
    """
    B, _, H, W = img.shape
    sal = saliency_map(img)  # (B,1,H,W)
    ys = torch.linspace(0, 1, H, device=img.device).view(1, 1, H, 1)
    xs = torch.linspace(0, 1, W, device=img.device).view(1, 1, 1, W)
    cy = (sal * ys).sum(dim=(1, 2, 3))  # (B,)
    cx = (sal * xs).sum(dim=(1, 2, 3))

    anchors = torch.tensor(
        [[1/3, 1/3], [1/3, 2/3], [2/3, 1/3], [2/3, 2/3],
         [0.382, 0.382], [0.382, 0.618], [0.618, 0.382], [0.618, 0.618]],
        device=img.device,
    )  # (8,2) -> (y,x)
    centers = torch.stack([cy, cx], dim=1).unsqueeze(1)        # (B,1,2)
    d = torch.linalg.norm(centers - anchors.unsqueeze(0), dim=-1)  # (B,8)
    d_min = d.min(dim=1).values                                # (B,)
    # 距离越小越好；最大可能距离 ~ sqrt(2)/2 ≈ 0.707
    return torch.clamp(1.0 - d_min / 0.5, 0.0, 1.0)


def symmetry_score(img: torch.Tensor) -> torch.Tensor:
    """左右镜像差越小越对称。返回 (B,) in [0,1]。
    完全对称是死的，所以最终我们不让它单独 = 1，而是和"破缺"配合 —— 此处只给原始信号。
    """
    flipped = torch.flip(img, dims=[-1])
    diff = (img - flipped).abs().mean(dim=(1, 2, 3))  # (B,)
    return torch.clamp(1.0 - diff * 2.0, 0.0, 1.0)


def contrast_score(img: torch.Tensor) -> torch.Tensor:
    """明暗反差：亮度标准差。0.25 左右最佳（≈ 视觉爆点而不过曝）。"""
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    std = gray.flatten(1).std(dim=1)
    # 用高斯峰，峰值在 0.25
    return torch.exp(-((std - 0.25) ** 2) / (2 * 0.12 ** 2))


def color_unity_score(img: torch.Tensor, n_bins: int = 12) -> torch.Tensor:
    """三色法则：主色调（色相直方图）峰值数 ≤ 3 得高分。"""
    hsv = rgb_to_hsv(img)
    h = hsv[:, 0]
    s = hsv[:, 1]
    # 只统计饱和度足够的像素（灰/白/黑不算颜色）
    mask = (s > 0.15).float()
    B = img.shape[0]
    scores = []
    for i in range(B):
        h_i = h[i][mask[i].bool()]
        if h_i.numel() < 32:
            scores.append(torch.tensor(0.5, device=img.device))
            continue
        hist = torch.histc(h_i, bins=n_bins, min=0.0, max=1.0)
        hist = hist / (hist.sum() + 1e-8)
        # 主色 = hist > 平均值 * 1.5 的 bin 数
        n_main = (hist > (1.0 / n_bins) * 1.5).sum().float()
        # 1~3 个主色最理想，越多越花
        score = torch.exp(-((n_main - 2.0) ** 2) / (2 * 1.2 ** 2))
        scores.append(score)
    return torch.stack(scores)


def complementary_score(img: torch.Tensor) -> torch.Tensor:
    """互补色反差：主前景色相 vs 主背景色相，差 ≈ 0.5 (180°) 最好。
    用显著性图加权区分前景/背景。
    """
    hsv = rgb_to_hsv(img)
    h = hsv[:, 0]  # (B,H,W)
    sal = saliency_map(img).squeeze(1)  # (B,H,W)
    B = img.shape[0]
    scores = []
    for i in range(B):
        s = sal[i]
        thr = s.flatten().quantile(0.7)
        fg = h[i][s > thr]
        bg = h[i][s <= thr]
        if fg.numel() < 16 or bg.numel() < 16:
            scores.append(torch.tensor(0.5, device=img.device))
            continue
        # 用环形均值（色相是周期量）
        def circ_mean(x):
            ang = x * 2 * math.pi
            return torch.atan2(torch.sin(ang).mean(), torch.cos(ang).mean()) / (2 * math.pi) % 1.0
        diff = (circ_mean(fg) - circ_mean(bg)).abs()
        diff = torch.minimum(diff, 1 - diff)  # 环形距离, in [0,0.5]
        # 0.5 = 互补色（180°），得 1 分
        scores.append(diff * 2.0)
    return torch.stack(scores)


def depth_score(img: torch.Tensor) -> torch.Tensor:
    """虚实反差：前景拉普拉斯方差高 + 背景拉普拉斯方差低 -> 大光圈虚化效果好。"""
    gray = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).unsqueeze(1)
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                       dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    edges = F.conv2d(gray, lap, padding=1).abs().squeeze(1)  # (B,H,W)
    sal = saliency_map(img).squeeze(1)

    B = img.shape[0]
    scores = []
    for i in range(B):
        s = sal[i]
        thr = s.flatten().quantile(0.7)
        fg_e = edges[i][s > thr].mean()
        bg_e = edges[i][s <= thr].mean()
        ratio = fg_e / (bg_e + 1e-6)
        # ratio > 2 视为明显虚化
        scores.append(torch.clamp(torch.log1p(ratio) / math.log(4), 0.0, 1.0))
    return torch.stack(scores)


def framing_score(img: torch.Tensor) -> torch.Tensor:
    """避免"天太多 / 地太多"：水平能量分布不应过于偏向上方或下方。"""
    sal = saliency_map(img).squeeze(1)  # (B,H,W)
    H = sal.shape[1]
    top = sal[:, : H // 3].sum(dim=(1, 2))
    mid = sal[:, H // 3 : 2 * H // 3].sum(dim=(1, 2))
    bot = sal[:, 2 * H // 3 :].sum(dim=(1, 2))
    total = top + mid + bot + 1e-8
    # 中段能量占比高 -> 好；同时上下不应失衡
    mid_ratio = mid / total
    imbalance = (top - bot).abs() / total
    return torch.clamp(mid_ratio * 1.5 - imbalance, 0.0, 1.0)


@dataclass
class HandCraftedScores:
    rule_of_thirds: torch.Tensor
    symmetry: torch.Tensor
    contrast: torch.Tensor
    color_unity: torch.Tensor
    complementary: torch.Tensor
    depth: torch.Tensor
    framing: torch.Tensor

    def stack(self) -> torch.Tensor:
        return torch.stack([
            self.rule_of_thirds,
            self.symmetry,
            self.contrast,
            self.color_unity,
            self.complementary,
            self.depth,
            self.framing,
        ], dim=1)  # (B,7)


def compute_handcrafted(img: torch.Tensor) -> HandCraftedScores:
    return HandCraftedScores(
        rule_of_thirds=rule_of_thirds_score(img),
        symmetry=symmetry_score(img),
        contrast=contrast_score(img),
        color_unity=color_unity_score(img),
        complementary=complementary_score(img),
        depth=depth_score(img),
        framing=framing_score(img),
    )


# ----------------------------------------------------------------------
# 2. CNN 全局美学头（学习 README 之外没写出来的"感觉"）
# ----------------------------------------------------------------------

class AestheticBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.feature = nn.Sequential(*list(net.children())[:-1])  # -> (B,512,1,1)
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x).flatten(1)


# ----------------------------------------------------------------------
# 3. 整合模型
# ----------------------------------------------------------------------

class PhotoQualityModel(nn.Module):
    """输入一张 RGB 图（[0,1] 区间），输出：
      - score: 总体美学分 [0,1]
      - breakdown: 7 维分项分（与 README 对应）
      - is_good: 阈值化的布尔判断
    """

    PRINCIPLES = [
        "三分法/黄金比构图",
        "对称性",
        "明暗反差",
        "三色法则（色彩统一）",
        "互补色反差",
        "虚实反差（背景虚化）",
        "比例（避免天/地过多）",
    ]

    def __init__(self, threshold: float = 0.6, pretrained: bool = True):
        super().__init__()
        self.threshold = threshold
        self.backbone = AestheticBackbone(pretrained=pretrained)

        # CNN 路：把 ResNet 特征压成 1 维美学分
        self.cnn_head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # 融合权重：7 个手工特征 + 1 个 CNN 头 -> 总分
        # 初始权重按 README 中的优先级（构图 > 反差 > 色彩 > 对称）
        init_w = torch.tensor([0.20, 0.08, 0.15, 0.15, 0.12, 0.10, 0.10, 0.10])
        self.fusion_weight = nn.Parameter(init_w)

        # ImageNet 归一化（仅用于 backbone）
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img: torch.Tensor) -> dict:
        """img: (B,3,H,W) in [0,1]."""
        hand = compute_handcrafted(img)               # 7 项
        hand_vec = hand.stack()                        # (B,7)

        feat = self.backbone((img - self.mean) / self.std)
        cnn_score = self.cnn_head(feat)                # (B,1)

        all_scores = torch.cat([hand_vec, cnn_score], dim=1)  # (B,8)
        w = F.softmax(self.fusion_weight, dim=0)              # 归一化权重
        total = (all_scores * w).sum(dim=1)                   # (B,)

        return {
            "score": total,
            "breakdown": {name: hand_vec[:, i] for i, name in enumerate(self.PRINCIPLES)},
            "cnn_aesthetic": cnn_score.squeeze(1),
            "is_good": total > self.threshold,
            "fusion_weights": w.detach(),
        }


# ----------------------------------------------------------------------
# 4. 训练损失（如果有人工标注的好/坏照片）
# ----------------------------------------------------------------------

class PhotoQualityLoss(nn.Module):
    """二分类 BCE + 可选的 7 项分项监督（如果数据集有细分标注）。"""

    def __init__(self, breakdown_weight: float = 0.3):
        super().__init__()
        self.bw = breakdown_weight

    def forward(self, output: dict, label: torch.Tensor,
                breakdown_label: torch.Tensor | None = None) -> torch.Tensor:
        loss = F.binary_cross_entropy(output["score"].clamp(1e-6, 1 - 1e-6), label.float())
        if breakdown_label is not None:
            preds = torch.stack(list(output["breakdown"].values()), dim=1)
            loss = loss + self.bw * F.mse_loss(preds, breakdown_label)
        return loss


# ----------------------------------------------------------------------
# 5. 推理便捷接口
# ----------------------------------------------------------------------

def load_image(path: str, size: int = 384) -> torch.Tensor:
    from PIL import Image
    tfm = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),  # -> [0,1]
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)


@torch.no_grad()
def judge(path: str, model: PhotoQualityModel | None = None) -> dict:
    if model is None:
        model = PhotoQualityModel().eval()
    img = load_image(path)
    out = model(img)
    return {
        "总分": float(out["score"][0]),
        "是否好照片": bool(out["is_good"][0]),
        "CNN美学": float(out["cnn_aesthetic"][0]),
        "分项": {k: float(v[0]) for k, v in out["breakdown"].items()},
    }


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "note.jpg"
    result = judge(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
