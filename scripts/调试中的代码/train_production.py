# scripts/train_production.py (最终修复版)
"""
生产级训练脚本：修复所有关键问题
⭐ 修复：1) 正确的Focal Loss alpha  2) Scheduler兼容解冻  3) 预测偏向问题
"""
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import defaultdict

from avtop.models.enhanced_detector import EnhancedAVTopDetector
from avtop.data.csv_dataset import BinaryAVCSVDataset, collate as csv_collate
from avtop.losses.advanced_losses import create_loss_function
from avtop.losses.enhanced_loss import RankingMILLoss
from avtop.eval.enhanced_metrics import validate_model, ImbalancedMetricsCalculator
from avtop.utils.experiment import ExperimentManager
from avtop.utils.repro import ReproducibilityManager


def compute_class_weights(csv_path):
    """计算类别权重并自动检测少数类"""
    import csv
    n_pos = n_neg = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["label"]) == 1:
                n_pos += 1
            else:
                n_neg += 1

    total = n_pos + n_neg
    if n_pos == 0 or n_neg == 0:
        return 1.0, 1.0, n_pos, n_neg, 1

    # 检测少数类
    if n_pos < n_neg:
        minority_class = 1
        n_minority = n_pos
        n_majority = n_neg
    else:
        minority_class = 0
        n_minority = n_neg
        n_majority = n_pos

    # ⭐ 关键修复：Focal Loss的alpha应该是少数类的权重
    # alpha > 0.5 表示更关注正类（class 1）
    # alpha < 0.5 表示更关注负类（class 0）
    if minority_class == 1:
        # 少数类是class 1，应该给它更高的权重
        focal_alpha = 0.75  # 更关注class 1
    else:
        # 少数类是class 0，应该给它更高的权重
        focal_alpha = 0.25  # 更关注class 0

    # BCE的pos_weight：永远是 n_neg/n_pos（不管谁是minority）
    pos_weight = n_neg / n_pos

    return pos_weight, focal_alpha, n_pos, n_neg, minority_class


def safe_float(value, default=1e-3):
    """安全转换为float"""
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        try:
            return float(value)
        except:
            return default
    else:
        return default


def make_loader(csv_path, cfg, shuffle=True, weighted_sampling=False):
    """创建数据加载器"""
    ds = BinaryAVCSVDataset(
        csv_path,
        root=cfg["data"].get("root", ""),
        T_v=cfg["data"]["T_v"],
        T_a=cfg["data"]["T_a"],
        mel_bins=cfg["data"]["mel"],
        sample_rate=cfg["data"]["sr"]
    )

    if weighted_sampling:
        import csv as _csv
        labels = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                labels.append(int(row["label"]))

        from collections import Counter
        cnt = Counter(labels)

        # ⭐ 修复：确保采样权重正确
        # 给少数类更高的采样概率，但不要过度
        minority_label = 0 if cnt[0] < cnt[1] else 1
        majority_label = 1 - minority_label

        # 使用平方根重加权（比线性更温和）
        total = cnt[0] + cnt[1]
        weight_minority = np.sqrt(total / cnt[minority_label])
        weight_majority = np.sqrt(total / cnt[majority_label])

        # 归一化
        weight_sum = cnt[minority_label] * weight_minority + cnt[majority_label] * weight_majority
        weight_minority = weight_minority / weight_sum * total
        weight_majority = weight_majority / weight_sum * total

        weight_per_class = {
            minority_label: weight_minority,
            majority_label: weight_majority
        }
        weights = [weight_per_class[label] for label in labels]

        print(f"⚖️  Sampling weights:")
        print(f"  Minority (label={minority_label}): {weight_minority:.3f}")
        print(f"  Majority (label={majority_label}): {weight_majority:.3f}")

        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True
        )

        return DataLoader(
            ds,
            batch_size=cfg["train"]["batch_size"],
            sampler=sampler,
            collate_fn=csv_collate,
            num_workers=cfg["train"].get("workers", 4),
            pin_memory=True
        )
    else:
        return DataLoader(
            ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=shuffle,
            collate_fn=csv_collate,
            num_workers=cfg["train"].get("workers", 4),
            pin_memory=True
        )


def freeze_backbone(model, freeze: bool = True):
    """冻结/解冻backbone"""
    for param in model.video_backbone.parameters():
        param.requires_grad = not freeze
    for param in model.audio_backbone.parameters():
        param.requires_grad = not freeze

    status = "frozen" if freeze else "unfrozen"
    print(f"🔒 Backbone {status}")


def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实验管理
    mgr = ExperimentManager(cfg["experiment"]["workdir"], cfg["experiment"]["name"])
    mgr.save_config(cfg)

    # 可重现性
    repro = ReproducibilityManager(exp_dir=(mgr.root / "snapshot").as_posix())
    repro.set_deterministic(seed=cfg.get("seed", 3407))
    repro.snapshot()

    # 数据准备
    train_csv = cfg["data"]["train_csv"]
    val_csv = cfg["data"]["val_csv"]

    # 计算类别统计
    pos_weight, focal_alpha, n_pos, n_neg, minority_class = compute_class_weights(train_csv)

    minority_label = f"label={minority_class}"
    minority_count = n_pos if minority_class == 1 else n_neg
    majority_count = n_neg if minority_class == 1 else n_pos

    print(f"\n{'=' * 60}")
    print(f"📊 Class Distribution (Training Set)")
    print(f"{'=' * 60}")
    print(f"⭐ Minority class: {minority_label} ({minority_count} samples, {minority_count / (n_pos + n_neg):.2%})")
    print(
        f"   Majority class: {'label=0' if minority_class == 1 else 'label=1'} ({majority_count} samples, {majority_count / (n_pos + n_neg):.2%})")
    print(f"\n🎯 Loss configuration:")
    print(f"  pos_weight (for BCE): {pos_weight:.3f}")
    print(f"  focal_alpha: {focal_alpha:.3f} ({'focus on class 1' if focal_alpha > 0.5 else 'focus on class 0'})")
    print(f"{'=' * 60}\n")

    # 数据加载器
    use_weighted = cfg["train"].get("weighted_sampling", True)
    train_loader = make_loader(train_csv, cfg, shuffle=False, weighted_sampling=use_weighted)
    val_loader = make_loader(val_csv, cfg, shuffle=False, weighted_sampling=False)

    # 创建模型
    model = EnhancedAVTopDetector(cfg).to(device)

    # ⭐ 损失函数（使用修正后的alpha）
    loss_type = cfg["train"].get("loss_type", "focal")

    if loss_type == "cb":
        if minority_class == 1:
            samples_per_class = [n_neg, n_pos]
        else:
            samples_per_class = [n_pos, n_neg]

        loss_fn = create_loss_function(
            "cb",
            samples_per_class=samples_per_class,
            beta=safe_float(cfg["train"].get("cb_beta", 0.9999)),
            gamma=safe_float(cfg["train"].get("focal_gamma", 2.0)),
            loss_type='focal'
        )
    elif loss_type == "focal":
        loss_fn = create_loss_function(
            "focal",
            alpha=focal_alpha,  # ⭐ 使用修正后的alpha
            gamma=safe_float(cfg["train"].get("focal_gamma", 2.0))
        )
    else:
        loss_fn = create_loss_function(
            "weighted_bce",
            pos_weight=pos_weight
        )

    ranking_loss_fn = RankingMILLoss(margin=0.5, topk=4)

    # ⭐ 优化器配置（修复scheduler问题）
    freeze_epochs = cfg["train"].get("freeze_backbone_epochs", 0)  # 改为0，不冻结
    base_lr = safe_float(cfg["train"]["lr"], 1e-3)
    weight_decay = safe_float(cfg["train"].get("weight_decay", 1e-4))

    # 不冻结backbone，直接使用所有参数
    freeze_backbone(model, freeze=False)

    optimizer = torch.optim.AdamW([
        {'params': model.video_backbone.parameters(), 'lr': base_lr * 0.01},
        {'params': model.audio_backbone.parameters(), 'lr': base_lr * 0.01},
        {'params': model.fusion.parameters(), 'lr': base_lr * 0.1},
        {'params': model.temporal.parameters(), 'lr': base_lr},
        {'params': model.mil.parameters(), 'lr': base_lr}
    ], weight_decay=weight_decay)

    # ⭐ 使用CosineAnnealing替代OneCycle（更稳定）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"],
        eta_min=base_lr * 0.01
    )

    # 训练配置
    epochs = cfg["train"]["epochs"]
    patience = cfg["train"].get("patience", 15)
    early_stop_metric = cfg["train"].get("early_stop_metric", "auprc_minority")

    best_metric = 0.0
    bad_epochs = 0

    print(f"\n🚀 Training Configuration:")
    print(f"  Loss: {loss_type} (alpha={focal_alpha:.3f})")
    print(f"  Base LR: {base_lr:.2e}")
    print(f"  Scheduler: CosineAnnealing")
    print(f"  Weighted sampling: {use_weighted}")
    print(f"  Early stop metric: {early_stop_metric}")
    print(f"  Patience: {patience}")
    print(f"  Minority class: {minority_label}")
    print(f"\n")

    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_losses = defaultdict(float)

        for batch_idx, batch in enumerate(train_loader):
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label_idx"].to(device)

            # 前向传播
            out = model(video, audio)

            # 计算损失
            cls_loss = loss_fn(out['clip_logits'], labels)
            ranking_loss = ranking_loss_fn(out['scores'], labels)

            total_loss = cls_loss + 0.3 * ranking_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录
            epoch_losses['total'] += total_loss.item()
            epoch_losses['cls'] += cls_loss.item()
            epoch_losses['ranking'] += ranking_loss.item()

            if batch_idx % 10 == 0:
                print(f'[epoch {epoch}][{batch_idx}/{len(train_loader)}] '
                      f'loss: {total_loss.item():.4f}')

        # 更新学习率（每个epoch一次）
        scheduler.step()

        # 平均损失
        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)

        # 验证
        val_metrics, val_probs, val_labels = validate_model(
            model, val_loader, device, minority_class=minority_class
        )

        # 打印报告
        calc = ImbalancedMetricsCalculator(minority_class=minority_class)
        calc.print_report(val_metrics, name=f"Validation (Epoch {epoch})")

        # 保存指标
        mgr.save_metrics({
            'epoch': epoch,
            **val_metrics,
            **epoch_losses
        }, epoch)

        # 早停
        current_metric = val_metrics[early_stop_metric]

        if current_metric > best_metric:
            best_metric = current_metric
            bad_epochs = 0

            mgr.save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_metric': best_metric,
                'metric_name': early_stop_metric,
                'minority_class': minority_class,
                'config': cfg
            }, epoch, fname="best_model.pth")

            print(f"\n⭐ New best {early_stop_metric}: {best_metric:.4f}\n")
        else:
            bad_epochs += 1
            print(f"\n📊 {early_stop_metric}: {current_metric:.4f} "
                  f"(best: {best_metric:.4f}, no improve: {bad_epochs}/{patience})\n")

        if bad_epochs >= patience:
            print(f"🛑 Early stopping at epoch {epoch}")
            break

    print(f"\n🎉 Training finished!")
    print(f"Best {early_stop_metric}: {best_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/real_binary.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg["train"].setdefault("loss_type", "focal")
    cfg["train"].setdefault("focal_gamma", 2.0)
    cfg["train"].setdefault("weighted_sampling", True)
    cfg["train"].setdefault("early_stop_metric", "auprc_minority")
    cfg["train"].setdefault("patience", 15)
    cfg["train"].setdefault("weight_decay", 1e-4)
    cfg["train"].setdefault("freeze_backbone_epochs", 0)  # 禁用冻结

    main(cfg)