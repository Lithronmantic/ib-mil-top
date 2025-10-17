#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查PCA可视化是否正确
找出为什么音频只显示几个点
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path


def check_pca_visualization(features_path, labels_path=None):
    """
    检查PCA可视化

    Args:
        features_path: .npy文件路径，形状应该是 [N, D] 或 [N, T, D]
        labels_path: 标签文件路径（可选）
    """

    print("=" * 70)
    print("🔍 CHECKING PCA VISUALIZATION")
    print("=" * 70)

    # 1. 加载特征
    print(f"\n[1] Loading features from: {features_path}")

    try:
        features = np.load(features_path)
        print(f"  ✅ Loaded successfully")
        print(f"  Shape: {features.shape}")
        print(f"  Dtype: {features.dtype}")
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return

    # 2. 检查形状
    print(f"\n[2] Analyzing feature shape...")

    if features.ndim == 2:
        N, D = features.shape
        print(f"  Format: [N={N}, D={D}]")
        print(f"  ✅ Already aggregated (one vector per sample)")
        features_flat = features

    elif features.ndim == 3:
        N, T, D = features.shape
        print(f"  Format: [N={N}, T={T}, D={D}]")
        print(f"  ⚠️  Temporal features detected")

        # 需要聚合时间维度
        print(f"\n  Aggregating temporal dimension...")
        features_flat = features.mean(axis=1)  # [N, D]
        print(f"  After aggregation: {features_flat.shape}")

    else:
        print(f"  ❌ Unexpected shape: {features.shape}")
        return

    # 3. 检查数据统计
    print(f"\n[3] Feature statistics:")
    print(f"  Number of samples: {len(features_flat)}")
    print(f"  Feature dimension: {features_flat.shape[1]}")
    print(f"  Mean: {features_flat.mean():.6f}")
    print(f"  Std: {features_flat.std():.6f}")
    print(f"  Min: {features_flat.min():.6f}")
    print(f"  Max: {features_flat.max():.6f}")

    # 检查是否有退化
    per_sample_std = features_flat.std(axis=1)
    print(f"\n  Per-sample std:")
    print(f"    Mean: {per_sample_std.mean():.6f}")
    print(f"    Min: {per_sample_std.min():.6f}")
    print(f"    Max: {per_sample_std.max():.6f}")

    if features_flat.std() < 0.01:
        print(f"\n  ❌ CRITICAL: Features are nearly constant!")
        print(f"     All samples have almost identical features")
        print(f"     This explains why PCA shows only a few points")
        return

    # 4. 检查重复样本
    print(f"\n[4] Checking for duplicate samples...")

    # 计算唯一特征向量
    unique_features = np.unique(features_flat, axis=0)
    n_unique = len(unique_features)

    print(f"  Total samples: {len(features_flat)}")
    print(f"  Unique samples: {n_unique}")
    print(f"  Duplicate ratio: {(1 - n_unique / len(features_flat)) * 100:.1f}%")

    if n_unique < len(features_flat) * 0.1:
        print(f"\n  ❌ PROBLEM: >90% of samples are duplicates!")
        print(f"     This explains the few points in PCA")

        # 找出最常见的特征
        from collections import Counter
        feature_strings = [tuple(f) for f in features_flat]
        counter = Counter(feature_strings)
        most_common = counter.most_common(5)

        print(f"\n  Most common features:")
        for i, (feat, count) in enumerate(most_common):
            print(f"    #{i + 1}: {count} samples ({count / len(features_flat) * 100:.1f}%)")

    # 5. PCA分析
    print(f"\n[5] Performing PCA...")

    try:
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_flat)

        print(f"  ✅ PCA successful")
        print(f"  Explained variance: {pca.explained_variance_ratio_}")
        print(f"  PC1 range: [{features_pca[:, 0].min():.6f}, {features_pca[:, 0].max():.6f}]")
        print(f"  PC2 range: [{features_pca[:, 1].min():.6f}, {features_pca[:, 1].max():.6f}]")

        # 检查PC1的范围
        pc1_range = features_pca[:, 0].max() - features_pca[:, 0].min()

        if pc1_range < 0.01:
            print(f"\n  ❌ PROBLEM: PC1 range is tiny ({pc1_range:.6f})")
            print(f"     All points collapse to a small region")
            print(f"     This matches your PCA plot issue!")

    except Exception as e:
        print(f"  ❌ PCA failed: {e}")
        return

    # 6. 可视化
    print(f"\n[6] Creating visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: PCA scatter
    ax = axes[0]
    if labels_path and Path(labels_path).exists():
        labels = np.load(labels_path)
        for label_val in np.unique(labels):
            mask = labels == label_val
            ax.scatter(features_pca[mask, 0], features_pca[mask, 1],
                       label=f'Class {label_val}', alpha=0.6, s=10)
        ax.legend()
    else:
        ax.scatter(features_pca[:, 0], features_pca[:, 1], alpha=0.6, s=10)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    ax.set_title(f'PCA Plot (N={len(features_flat)})')
    ax.grid(True, alpha=0.3)

    # Plot 2: 特征均值分布
    ax = axes[1]
    ax.hist(features_flat.mean(axis=1), bins=50, alpha=0.7)
    ax.set_xlabel('Mean feature value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Sample Means')
    ax.grid(True, alpha=0.3)

    # Plot 3: 方差分布
    ax = axes[2]
    ax.hist(features_flat.std(axis=1), bins=50, alpha=0.7)
    ax.set_xlabel('Std feature value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Sample Stds')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = 'pca_check.png'
    plt.savefig(output_path, dpi=150)
    print(f"  ✅ Saved to: {output_path}")

    # 7. 总结
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")

    issues = []

    if features_flat.std() < 0.01:
        issues.append("Features are nearly constant")

    if n_unique < len(features_flat) * 0.1:
        issues.append(f"Only {n_unique}/{len(features_flat)} unique samples")

    if pc1_range < 0.01:
        issues.append(f"PCA range is tiny ({pc1_range:.6f})")

    if issues:
        print(f"\n❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print(f"\n🔧 LIKELY CAUSES:")
        print(f"  1. VGGish only extracted 1 frame per sample (0.96s window)")
        print(f"  2. All audio windows are identical")
        print(f"  3. VGGish model not properly loaded")
        print(f"  4. Feature extraction has a bug")

        print(f"\n💡 SOLUTIONS:")
        print(f"  1. Increase window size to 1.5s+ (get more VGGish frames)")
        print(f"  2. Check if audio files are actually different")
        print(f"  3. Re-extract features with correct VGGish")

    else:
        print(f"\n✅ Features look healthy")
        print(f"  Check your PCA plotting code for sampling issues")


def check_dataset_loading():
    """检查数据集加载是否正确"""

    print("\n" + "=" * 70)
    print("🔍 CHECKING DATASET LOADING")
    print("=" * 70)

    try:
        from src.avtop.data.csv_dataset import AVTopDataset

        dataset = AVTopDataset(
            windows_csv="data/train_windows.csv",
            ann_csv="data/annotations.csv",
            classes=['normal', 'defect'],
            use_audio=True,
            use_video=False
        )

        print(f"\n[1] Dataset info:")
        print(f"  Total samples: {len(dataset)}")

        # 检查前10个样本
        print(f"\n[2] Checking first 10 samples...")

        audio_shapes = []
        audio_hashes = []

        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            audio = sample['audio']

            audio_shapes.append(audio.shape)
            # 计算hash来检查是否相同
            audio_hash = hash(audio.numpy().tobytes())
            audio_hashes.append(audio_hash)

            print(f"  Sample {i}: shape={audio.shape}, hash={audio_hash}")

        # 检查是否所有样本相同
        unique_hashes = len(set(audio_hashes))

        if unique_hashes == 1:
            print(f"\n  ❌ PROBLEM: All samples have identical audio!")
            print(f"     This is the root cause!")
        elif unique_hashes < len(audio_hashes):
            print(f"\n  ⚠️  {len(audio_hashes) - unique_hashes} duplicate samples")
        else:
            print(f"\n  ✅ All samples are unique")

        # 检查shape
        if len(set(audio_shapes)) > 1:
            print(f"\n  ⚠️  Audio shapes are inconsistent:")
            for shape in set(audio_shapes):
                print(f"    - {shape}")
        else:
            print(f"\n  ✅ All audio shapes are consistent: {audio_shapes[0]}")

            # 如果T=1，说明只有1帧
            if audio_shapes[0][0] == 1:
                print(f"\n  ⚠️  Audio has only 1 frame!")
                print(f"     This is because window size is exactly 0.96s")
                print(f"     Solution: Increase window size to 1.5s+")

    except Exception as e:
        print(f"\n❌ Failed to load dataset: {e}")


def main():
    """主函数"""

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_features', type=str,
                        default='experiments/features/audio_features.npy',
                        help='Path to audio features .npy file')
    parser.add_argument('--labels', type=str,
                        default='experiments/features/labels.npy',
                        help='Path to labels .npy file')

    args = parser.parse_args()

    # 检查PCA可视化
    if Path(args.audio_features).exists():
        check_pca_visualization(args.audio_features, args.labels)
    else:
        print(f"Audio features not found at: {args.audio_features}")
        print(f"Skipping PCA check")

    # 检查数据集加载
    check_dataset_loading()

    print("\n" + "=" * 70)
    print("✅ DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()