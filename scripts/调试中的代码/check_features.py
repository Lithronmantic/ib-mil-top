#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征质量诊断：检查音视频特征是否有区分能力
"""
import torch
import yaml
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader
from avtop.models.enhanced_detector import EnhancedAVTopDetector
from avtop.data.csv_dataset import BinaryAVCSVDataset, collate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def check_feature_quality(config_path):
    """检查特征质量"""

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = EnhancedAVTopDetector(cfg).to(device)
    model.eval()

    # 加载数据
    val_csv = cfg["data"]["val_csv"]
    ds = BinaryAVCSVDataset(
        val_csv,
        root=cfg["data"].get("root", ""),
        T_v=cfg["data"]["T_v"],
        T_a=cfg["data"]["T_a"],
        mel_bins=cfg["data"]["mel"],
        sample_rate=cfg["data"]["sr"]
    )
    loader = DataLoader(ds, batch_size=8, collate_fn=collate)

    # 收集特征
    video_features = []
    audio_features = []
    fusion_features = []
    labels = []

    print("\n" + "=" * 60)
    print("Extracting Features...")
    print("=" * 60)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            label = batch["label_idx"]

            # 提取backbone特征
            v_feat = model.video_backbone(video)  # (B, T, D)
            a_feat = model.audio_backbone(audio)  # (B, T, D)

            # 全局池化得到样本级特征
            v_feat_pooled = v_feat.mean(dim=1)  # (B, D)
            a_feat_pooled = a_feat.mean(dim=1)  # (B, D)

            # 融合特征
            # 注意：需要先对齐时间维度
            from torch.nn import functional as F
            if a_feat.shape[1] != v_feat.shape[1]:
                a_feat = F.interpolate(
                    a_feat.transpose(1, 2),
                    size=v_feat.shape[1],
                    mode='linear'
                ).transpose(1, 2)

            # 简单拼接作为融合特征
            fused = torch.cat([v_feat, a_feat], dim=-1).mean(dim=1)  # (B, D)

            video_features.append(v_feat_pooled.cpu().numpy())
            audio_features.append(a_feat_pooled.cpu().numpy())
            fusion_features.append(fused.cpu().numpy())
            labels.append(label.numpy())

            if batch_idx == 0:
                print(f"First batch:")
                print(f"  Video feature shape: {v_feat_pooled.shape}")
                print(f"  Audio feature shape: {a_feat_pooled.shape}")
                print(f"  Fusion feature shape: {fused.shape}")

    # 拼接
    video_features = np.concatenate(video_features, axis=0)
    audio_features = np.concatenate(audio_features, axis=0)
    fusion_features = np.concatenate(fusion_features, axis=0)
    labels = np.concatenate(labels, axis=0)

    print(f"\nTotal samples: {len(labels)}")
    print(f"  Class 0: {(labels == 0).sum()}")
    print(f"  Class 1: {(labels == 1).sum()}")

    # 分析特征质量
    print("\n" + "=" * 60)
    print("Feature Quality Analysis")
    print("=" * 60)

    def analyze_separability(features, labels, name):
        """分析特征的可分性"""
        # 计算类内和类间距离
        class_0_feat = features[labels == 0]
        class_1_feat = features[labels == 1]

        if len(class_0_feat) == 0 or len(class_1_feat) == 0:
            print(f"\n{name}: Insufficient data")
            return None

        # 类中心
        center_0 = class_0_feat.mean(axis=0)
        center_1 = class_1_feat.mean(axis=0)

        # 类间距离
        inter_class_dist = np.linalg.norm(center_0 - center_1)

        # 类内距离
        intra_class_dist_0 = np.mean([np.linalg.norm(f - center_0) for f in class_0_feat])
        intra_class_dist_1 = np.mean([np.linalg.norm(f - center_1) for f in class_1_feat])
        avg_intra_class_dist = (intra_class_dist_0 + intra_class_dist_1) / 2

        # Fisher比率：类间距离 / 类内距离（越大越好）
        fisher_ratio = inter_class_dist / avg_intra_class_dist if avg_intra_class_dist > 0 else 0

        print(f"\n{name}:")
        print(f"  Inter-class distance: {inter_class_dist:.4f}")
        print(f"  Intra-class distance: {avg_intra_class_dist:.4f}")
        print(f"  Fisher ratio: {fisher_ratio:.4f}")

        if fisher_ratio < 0.5:
            print(f"  ❌ POOR: Features are NOT separable")
        elif fisher_ratio < 1.0:
            print(f"  ⚠️  WEAK: Features are weakly separable")
        else:
            print(f"  ✅ GOOD: Features are separable")

        return fisher_ratio

    v_ratio = analyze_separability(video_features, labels, "Video Features")
    a_ratio = analyze_separability(audio_features, labels, "Audio Features")
    f_ratio = analyze_separability(fusion_features, labels, "Fusion Features")

    # PCA可视化
    print("\n" + "=" * 60)
    print("Generating Visualization...")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (feat, name) in enumerate([
        (video_features, 'Video'),
        (audio_features, 'Audio'),
        (fusion_features, 'Fusion')
    ]):
        # PCA降到2维
        pca = PCA(n_components=2)
        feat_2d = pca.fit_transform(feat)

        # 绘制
        for label in [0, 1]:
            mask = labels == label
            axes[idx].scatter(
                feat_2d[mask, 0],
                feat_2d[mask, 1],
                label=f'Class {label}',
                alpha=0.6,
                s=20
            )

        axes[idx].set_xlabel('PC1')
        axes[idx].set_ylabel('PC2')
        axes[idx].set_title(f'{name} Features (PCA)')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    output_dir = Path("diagnostic_output")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "feature_separability.png", dpi=150)
    print(f"✅ Saved plot to: {output_dir / 'feature_separability.png'}")

    # 诊断建议
    print("\n" + "=" * 60)
    print("Diagnostic Recommendations")
    print("=" * 60)

    best_ratio = max(v_ratio or 0, a_ratio or 0, f_ratio or 0)

    if best_ratio < 0.3:
        print("\n❌ CRITICAL: Features are NOT discriminative")
        print("\n   Possible causes:")
        print("   1. Audio/Video loading is broken (all zeros?)")
        print("   2. Labels are incorrect or random")
        print("   3. Task is inherently too difficult with current features")
        print("\n   Urgent actions:")
        print("   ✅ Check if audio/video files can be opened")
        print("   ✅ Print raw audio/video tensors to verify non-zero")
        print("   ✅ Manually inspect 5 samples to verify labels")
        print("   ✅ Try a simpler task (e.g., binary regression)")

    elif best_ratio < 0.7:
        print("\n⚠️  Features are WEAKLY discriminative")
        print("\n   This explains why training is not improving")
        print("\n   Possible solutions:")
        print("   1. Use stronger pretrained models")
        print("   2. Add data augmentation")
        print("   3. Increase model capacity")
        print("   4. Check if labels have noise")
        print("\n   Next steps:")
        if v_ratio and v_ratio > a_ratio:
            print("   ✅ Video features are better - focus on video")
        elif a_ratio and a_ratio > v_ratio:
            print("   ✅ Audio features are better - focus on audio")
        else:
            print("   ✅ Try training on single modality first")

    else:
        print("\n✅ Features ARE discriminative!")
        print("\n   The problem is likely in training, not features")
        print("\n   Solutions:")
        print("   ✅ Increase learning rate (try 2e-4 or 5e-4)")
        print("   ✅ Increase class weight further")
        print("   ✅ Try different optimizer (SGD with momentum)")
        print("   ✅ Add learning rate warmup")

    return {
        'video_ratio': v_ratio,
        'audio_ratio': a_ratio,
        'fusion_ratio': f_ratio
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/real_binary.yaml')
    args = parser.parse_args()

    check_feature_quality(args.config)