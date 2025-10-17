#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本：检查模型输出的概率分布
用于发现训练问题的根源
"""
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader
from avtop.models.enhanced_detector import EnhancedAVTopDetector
from avtop.data.csv_dataset import BinaryAVCSVDataset, collate


def diagnose_model_output(config_path, checkpoint_path=None):
    """诊断模型输出"""

    # 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = EnhancedAVTopDetector(cfg).to(device)

    # 加载checkpoint（如果有）
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
    else:
        print("Using randomly initialized model")

    model.eval()

    # 加载验证集
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

    # 收集输出
    all_logits = []
    all_probs = []
    all_labels = []

    print("\n" + "=" * 60)
    print("Collecting model outputs...")
    print("=" * 60)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label_idx"]

            # 前向传播
            out = model(video, audio)
            logits = out['clip_logits']  # (B, 2)
            probs = torch.softmax(logits, dim=-1)  # (B, 2)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

            if batch_idx == 0:
                print(f"\nFirst batch analysis:")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                print(f"  Probs[:, 0] (class 0): {probs[:, 0].cpu().numpy()}")
                print(f"  Probs[:, 1] (class 1): {probs[:, 1].cpu().numpy()}")

    # 拼接
    all_logits = np.concatenate(all_logits, axis=0)  # (N, 2)
    all_probs = np.concatenate(all_probs, axis=0)  # (N, 2)
    all_labels = np.concatenate(all_labels, axis=0)  # (N,)

    # 分析
    print("\n" + "=" * 60)
    print("Output Distribution Analysis")
    print("=" * 60)

    # 1. Logits统计
    print(f"\n1. Logits Statistics:")
    print(f"   Class 0 logits: mean={all_logits[:, 0].mean():.3f}, std={all_logits[:, 0].std():.3f}")
    print(f"   Class 1 logits: mean={all_logits[:, 1].mean():.3f}, std={all_logits[:, 1].std():.3f}")
    print(f"   Logit diff (1-0): mean={np.mean(all_logits[:, 1] - all_logits[:, 0]):.3f}")

    # 2. Probs统计
    print(f"\n2. Probability Statistics:")
    print(f"   P(class 0): mean={all_probs[:, 0].mean():.3f}, std={all_probs[:, 0].std():.3f}")
    print(f"   P(class 1): mean={all_probs[:, 1].mean():.3f}, std={all_probs[:, 1].std():.3f}")

    # 3. 按真实标签分析
    print(f"\n3. By True Label:")
    for label in [0, 1]:
        mask = all_labels == label
        if mask.sum() > 0:
            print(f"   True label={label} (n={mask.sum()}):")
            print(f"     P(pred=0): mean={all_probs[mask, 0].mean():.3f}, std={all_probs[mask, 0].std():.3f}")
            print(f"     P(pred=1): mean={all_probs[mask, 1].mean():.3f}, std={all_probs[mask, 1].std():.3f}")

    # 4. 预测分布
    print(f"\n4. Prediction Distribution:")
    pred_labels = np.argmax(all_probs, axis=1)
    print(f"   Predicted as 0: {(pred_labels == 0).sum()} ({(pred_labels == 0).mean():.2%})")
    print(f"   Predicted as 1: {(pred_labels == 1).sum()} ({(pred_labels == 1).mean():.2%})")

    # 5. 检查是否有极端输出
    print(f"\n5. Extreme Values Check:")
    extreme_confident_0 = (all_probs[:, 0] > 0.99).sum()
    extreme_confident_1 = (all_probs[:, 1] > 0.99).sum()
    print(f"   P(class 0) > 0.99: {extreme_confident_0} samples")
    print(f"   P(class 1) > 0.99: {extreme_confident_1} samples")

    if extreme_confident_0 + extreme_confident_1 > len(all_probs) * 0.8:
        print(
            f"   ⚠️  Warning: {(extreme_confident_0 + extreme_confident_1) / len(all_probs):.1%} of predictions are extremely confident!")
        print(f"   This suggests the model is too certain (possible overconfidence)")

    # 6. 绘制分布图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 子图1: 按标签的概率分布
    for label in [0, 1]:
        mask = all_labels == label
        if mask.sum() > 0:
            axes[0, 0].hist(all_probs[mask, 1], bins=50, alpha=0.5, label=f'True={label}')
    axes[0, 0].set_xlabel('P(class 1)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Probability Distribution by True Label')
    axes[0, 0].legend()
    axes[0, 0].axvline(0.5, color='r', linestyle='--', label='threshold=0.5')

    # 子图2: Logit差异
    logit_diff = all_logits[:, 1] - all_logits[:, 0]
    axes[0, 1].hist(logit_diff, bins=50, alpha=0.7)
    axes[0, 1].set_xlabel('Logit(class 1) - Logit(class 0)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Logit Difference Distribution')
    axes[0, 1].axvline(0, color='r', linestyle='--')

    # 子图3: 概率散点图
    axes[1, 0].scatter(all_probs[:, 0], all_probs[:, 1],
                       c=all_labels, cmap='coolwarm', alpha=0.5, s=10)
    axes[1, 0].set_xlabel('P(class 0)')
    axes[1, 0].set_ylabel('P(class 1)')
    axes[1, 0].set_title('Probability Space')
    axes[1, 0].plot([0, 1], [1, 0], 'k--', alpha=0.3)

    # 子图4: 混淆矩阵（阈值=0.5）
    from sklearn.metrics import confusion_matrix
    pred_05 = (all_probs[:, 1] >= 0.5).astype(int)
    cm = confusion_matrix(all_labels, pred_05)

    im = axes[1, 1].imshow(cm, cmap='Blues')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('True')
    axes[1, 1].set_title('Confusion Matrix (threshold=0.5)')

    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, str(cm[i, j]),
                            ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")

    plt.tight_layout()

    # 保存
    output_dir = Path("diagnostic_output")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "model_output_analysis.png", dpi=150)
    print(f"\n✅ Saved analysis plot to: {output_dir / 'model_output_analysis.png'}")

    # 7. 给出诊断建议
    print("\n" + "=" * 60)
    print("Diagnostic Recommendations")
    print("=" * 60)

    # 检查是否完全偏向一边
    if (all_probs[:, 1] > 0.5).sum() > len(all_probs) * 0.95:
        print("❌ CRITICAL: Model predicts class 1 for >95% of samples")
        print("   Possible causes:")
        print("   1. Loss function is too biased toward class 1")
        print("   2. Weighted sampling is too aggressive")
        print("   3. Model initialization issue")
        print("\n   Suggestions:")
        print("   ✅ Try: Disable weighted_sampling")
        print("   ✅ Try: Use standard CrossEntropy first")
        print("   ✅ Try: Reduce focal_gamma to 0.5 or 1.0")

    elif (all_probs[:, 0] > 0.5).sum() > len(all_probs) * 0.95:
        print("❌ CRITICAL: Model predicts class 0 for >95% of samples")
        print("   (Same diagnosis as above)")

    elif logit_diff.std() < 0.1:
        print("⚠️  WARNING: Logit outputs have very low variance")
        print("   Model is not learning meaningful features")
        print("\n   Suggestions:")
        print("   ✅ Check if data is loaded correctly")
        print("   ✅ Increase learning rate")
        print("   ✅ Check for gradient vanishing")

    else:
        print("✅ Model outputs look reasonable")
        print("   Continue training and monitor metrics")

    return {
        'logits': all_logits,
        'probs': all_probs,
        'labels': all_labels
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/real_binary.yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (optional, will use random init if not provided)')

    args = parser.parse_args()

    diagnose_model_output(args.config, args.checkpoint)