# scripts/train_baseline.py
"""
最保守的baseline训练脚本
用标准CrossEntropy + 简单的pos_weight，不使用任何高级技巧
目的：验证数据和模型是否能学到基本模式
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader
from collections import defaultdict

from avtop.models.enhanced_detector import EnhancedAVTopDetector
from avtop.data.csv_dataset import BinaryAVCSVDataset, collate as csv_collate
from avtop.eval.enhanced_metrics import validate_model, ImbalancedMetricsCalculator
from avtop.utils.experiment import ExperimentManager


def compute_class_stats(csv_path):
    """计算类别统计"""
    import csv
    n_pos = n_neg = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["label"]) == 1:
                n_pos += 1
            else:
                n_neg += 1

    minority_class = 0 if n_neg < n_pos else 1
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    return n_pos, n_neg, minority_class, pos_weight


def make_simple_loader(csv_path, cfg, shuffle=True):
    """创建简单的数据加载器（无加权采样）"""
    ds = BinaryAVCSVDataset(
        csv_path,
        root=cfg["data"].get("root", ""),
        T_v=cfg["data"]["T_v"],
        T_a=cfg["data"]["T_a"],
        mel_bins=cfg["data"]["mel"],
        sample_rate=cfg["data"]["sr"]
    )

    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        collate_fn=csv_collate,
        num_workers=cfg["train"].get("workers", 4),
        pin_memory=True
    )


def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实验管理
    exp_name = cfg["experiment"]["name"] + "_baseline"
    mgr = ExperimentManager(cfg["experiment"]["workdir"], exp_name)
    mgr.save_config(cfg)

    # 数据准备
    train_csv = cfg["data"]["train_csv"]
    val_csv = cfg["data"]["val_csv"]

    n_pos, n_neg, minority_class, pos_weight = compute_class_stats(train_csv)

    print(f"\n{'=' * 60}")
    print(f"📊 Baseline Training Configuration")
    print(f"{'=' * 60}")
    print(f"Minority class: label={minority_class}")
    print(f"  label=0: {n_neg} samples ({n_neg / (n_pos + n_neg):.2%})")
    print(f"  label=1: {n_pos} samples ({n_pos / (n_pos + n_neg):.2%})")
    print(f"pos_weight: {pos_weight:.3f}")
    print(f"\n🎯 Training strategy:")
    print(f"  Loss: Weighted CrossEntropy (simplest)")
    print(f"  Sampling: Random (no weighted sampling)")
    print(f"  LR: Constant (no scheduler)")
    print(f"  Goal: Verify if model can learn basic patterns")
    print(f"{'=' * 60}\n")

    # 数据加载器（随机采样）
    train_loader = make_simple_loader(train_csv, cfg, shuffle=True)
    val_loader = make_simple_loader(val_csv, cfg, shuffle=False)

    # 创建模型
    model = EnhancedAVTopDetector(cfg).to(device)

    # ⭐ 最简单的损失：Weighted CrossEntropy
    # 只对正类（label=1）加权
    class_weights = torch.tensor([1.0, pos_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Loss function: CrossEntropy with weights = {class_weights.cpu().numpy()}")

    # 优化器：AdamW，单一学习率
    lr = 1e-4  # 较小的学习率，更保守
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 训练配置
    epochs = cfg["train"].get("epochs", 50)
    patience = cfg["train"].get("patience", 15)

    best_auprc = 0.0
    bad_epochs = 0

    print(f"Starting baseline training...")
    print(f"  LR: {lr:.2e}")
    print(f"  Epochs: {epochs}")
    print(f"  Patience: {patience}\n")

    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # 记录预测分布（用于诊断）
        train_preds_pos = 0
        train_preds_neg = 0

        for batch_idx, batch in enumerate(train_loader):
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label_idx"].to(device)

            # 前向传播
            out = model(video, audio)
            logits = out['clip_logits']  # (B, 2)

            # 计算损失（最简单）
            loss = criterion(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录
            epoch_loss += loss.item()
            n_batches += 1

            # 统计预测分布
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                train_preds_pos += (preds == 1).sum().item()
                train_preds_neg += (preds == 0).sum().item()

            if batch_idx % 10 == 0:
                print(f'[epoch {epoch}][{batch_idx}/{len(train_loader)}] loss: {loss.item():.4f}')

        epoch_loss /= n_batches

        # 打印训练集预测分布（诊断用）
        total_train = train_preds_pos + train_preds_neg
        print(f"\n[epoch {epoch}] Train predictions:")
        print(f"  Pred as 0: {train_preds_neg} ({train_preds_neg / total_train:.2%})")
        print(f"  Pred as 1: {train_preds_pos} ({train_preds_pos / total_train:.2%})")
        print(f"  Avg loss: {epoch_loss:.4f}")

        # 验证
        val_metrics, val_probs, val_labels = validate_model(
            model, val_loader, device, minority_class=minority_class
        )

        # 简化报告
        print(f"\n[epoch {epoch}] Validation:")
        print(
            f"  AUPRC: {val_metrics['auprc_minority']:.4f} (baseline: {val_metrics['auprc_baseline']:.4f}, gain: {val_metrics['auprc_gain']:.4f})")
        print(f"  MCC: {val_metrics['mcc']:.4f}")
        print(f"  Precision: {val_metrics['precision_minority']:.4f}, Recall: {val_metrics['recall_minority']:.4f}")
        print(f"  Specificity: {val_metrics['specificity']:.4f}")

        # 打印混淆矩阵
        print(f"  Confusion Matrix:")
        print(f"    TN={val_metrics['tn']}, FP={val_metrics['fp']}")
        print(f"    FN={val_metrics['fn']}, TP={val_metrics['tp']}")

        # 保存指标
        mgr.save_metrics({
            'epoch': epoch,
            **val_metrics,
            'train_loss': epoch_loss
        }, epoch)

        # 早停（基于AUPRC）
        current_auprc = val_metrics['auprc_minority']

        if current_auprc > best_auprc:
            best_auprc = current_auprc
            bad_epochs = 0

            mgr.save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_auprc': best_auprc,
                'minority_class': minority_class,
                'config': cfg
            }, epoch, fname="best_model.pth")

            print(f"  ⭐ New best AUPRC: {best_auprc:.4f}")
        else:
            bad_epochs += 1
            print(f"  No improvement ({bad_epochs}/{patience})")

        print()

        if bad_epochs >= patience:
            print(f"🛑 Early stopping at epoch {epoch}")
            break

    print(f"\n🎉 Baseline training finished!")
    print(f"Best AUPRC: {best_auprc:.4f}")

    # 最终诊断建议
    print(f"\n{'=' * 60}")
    print("Diagnostic Summary")
    print(f"{'=' * 60}")

    if best_auprc < 0.25:  # 低于baseline（假设baseline≈0.20）
        print("❌ Model failed to learn")
        print("   AUPRC is not better than baseline")
        print("\n   Possible issues:")
        print("   1. Data quality: labels may be incorrect")
        print("   2. Features insufficient: audio/video don't contain discriminative info")
        print("   3. Model capacity: architecture may be inadequate")
        print("\n   Next steps:")
        print("   ✅ Run diagnostic script to check model outputs")
        print("   ✅ Manually inspect some samples to verify labels")
        print("   ✅ Try simpler features (e.g., only video or only audio)")

    elif best_auprc < 0.40:
        print("⚠️  Model shows weak learning")
        print(f"   AUPRC gain: {best_auprc - 0.20:.4f}")
        print("\n   Next steps:")
        print("   ✅ Try Focal Loss with gamma=1.0 (mild)")
        print("   ✅ Add data augmentation")
        print("   ✅ Increase model capacity")

    else:
        print("✅ Model shows good learning!")
        print(f"   AUPRC gain: {best_auprc - 0.20:.4f}")
        print("\n   Next steps:")
        print("   ✅ Now try Focal Loss / CB Loss for better performance")
        print("   ✅ Add weighted sampling")
        print("   ✅ Fine-tune hyperparameters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/real_binary.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)