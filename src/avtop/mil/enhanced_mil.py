# src/avtop/mil/enhanced_mil.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedMIL(nn.Module):
    """结合您报告中的Top-K supervision和动态路由"""

    def __init__(self, d_in, num_classes=2, topk_ratio=0.2):
        super().__init__()
        self.topk_ratio = topk_ratio

        # 1. 异常评分网络（报告3.2节）
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_in, 1)
        )

        # 2. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_in // 2, num_classes)
        )

    def forward(self, z, is_training=True):
        # z: [B, T, D]
        B, T, D = z.shape

        # 1. 计算异常分数
        scores = self.anomaly_scorer(z).squeeze(-1)  # [B, T]

        # 2. Top-K selection (训练时)
        if is_training:
            k = max(1, int(T * self.topk_ratio))
            topk_scores, topk_idx = torch.topk(scores, k, dim=1)

            # 创建稀疏注意力mask
            weights = torch.zeros_like(scores)
            weights.scatter_(1, topk_idx, 1.0)
            weights = weights / k  # 归一化
        else:
            # 测试时使用软注意力
            weights = torch.softmax(scores, dim=1)

        # 3. 加权聚合
        clip_feat = (z * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        # 4. 分类
        clip_logits = self.classifier(clip_feat)
        seg_logits = self.classifier(z)

        return {
            'clip_logits': clip_logits,
            'segment_logits': seg_logits,
            'weights': weights,
            'scores': scores
        }