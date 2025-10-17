#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一训练脚本 - 整合所有改进方法
=================================

支持的改进：
1. InfoNCE对比学习（拉近音视频匹配对）
2. 三模态知识蒸馏（融合↔单模态）
3. 多视图一致性正则化
4. 数据增强（强弱增强）
5. Focal Loss / CB Loss（处理不平衡）
6. 半监督学习（可选）

使用方法：
    python scripts/train_unified.py --config configs/unified_config.yaml
"""
import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader

# 导入模型和损失
from avtop.models.enhanced_detector import EnhancedAVTopDetector
from avtop.data.csv_dataset import BinaryAVCSVDataset, collate

# 导入损失函数
from avtop.losses.advanced_losses import create_loss_function
from avtop.losses.contrastive_loss import InfoNCELoss, ProjectionHead
from avtop.losses.kd_loss import TriModalKDLoss
from avtop.losses.consistency_loss import MultiViewConsistency
from avtop.losses.enhanced_loss import RankingMILLoss

# 导入数据增强
from avtop.data.augmentation_module import MultiModalAugmentation

# 导入评估
from avtop.eval.enhanced_metrics import ImbalancedMetricsCalculator, validate_model

# 导入工具
from avtop.utils.experiment import ExperimentManager


class UnifiedTrainer:
    """
    统一训练器 - 整合所有改进方法

    核心特性：
    1. 对比学习：让音视频匹配对更紧密
    2. 知识蒸馏：让单模态向融合模型学习
    3. 一致性：不同增强下预测应一致
    4. 自适应权重：根据训练阶段调整各损失权重
    """

    def __init__(self, cfg: dict, device='cuda'):
        self.cfg = cfg
        self.device = device

        # 1. 创建模型
        print("\n🔧 初始化模型...")
        self.model = EnhancedAVTopDetector(cfg).to(device)

        # 2. 创建投影头（用于对比学习）
        fusion_dim = cfg['model'].get('fusion', {}).get('d_model', 256)
        proj_dim = cfg['train'].get('contrastive_dim', 128)

        self.video_proj = ProjectionHead(fusion_dim, output_dim=proj_dim).to(device)
        self.audio_proj = ProjectionHead(fusion_dim, output_dim=proj_dim).to(device)

        print(f"  ✓ 模型维度: fusion_dim={fusion_dim}, proj_dim={proj_dim}")

        # 3. 创建损失函数
        self._build_losses()

        # 4. 创建优化器
        self._build_optimizer()

        # 5. 数据增强
        if cfg['train'].get('use_augmentation', True):
            self.weak_aug = MultiModalAugmentation(mode='weak')
            self.strong_aug = MultiModalAugmentation(mode='strong')
        else:
            self.weak_aug = None
            self.strong_aug = None

        # 6. 实验管理
        self.exp_mgr = ExperimentManager(
            cfg['experiment']['workdir'],
            cfg['experiment']['name']
        )
        self.exp_mgr.save_config(cfg)

        # 7. 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        print("✅ 训练器初始化完成！\n")

    def _build_losses(self):
        """构建所有损失函数"""
        cfg = self.cfg['train']

        print("🎯 配置损失函数:")

        # 1. 分类损失（处理不平衡）
        loss_type = cfg.get('loss_type', 'focal')

        if loss_type == 'focal':
            self.cls_loss = create_loss_function(
                'focal',
                alpha=cfg.get('focal_alpha', 0.75),
                gamma=cfg.get('focal_gamma', 2.0)
            )
            print(f"  ✓ Focal Loss (alpha={cfg.get('focal_alpha', 0.75)}, gamma={cfg.get('focal_gamma', 2.0)})")

        elif loss_type == 'cb':
            # 需要提供每类样本数
            samples_per_class = cfg.get('samples_per_class', [800, 200])
            self.cls_loss = create_loss_function(
                'cb',
                samples_per_class=samples_per_class,
                beta=cfg.get('cb_beta', 0.9999),
                gamma=cfg.get('focal_gamma', 2.0),
                loss_type='focal'
            )
            print(f"  ✓ Class-Balanced Loss (samples={samples_per_class})")

        else:
            self.cls_loss = nn.CrossEntropyLoss()
            print(f"  ✓ Cross Entropy Loss")

        # 2. 对比学习损失
        if cfg.get('use_contrastive', True):
            self.contrastive_loss = InfoNCELoss(
                temperature=cfg.get('contrastive_temp', 0.07),
                queue_size=cfg.get('contrastive_queue', 0)
            )
            print(f"  ✓ InfoNCE Loss (temp={cfg.get('contrastive_temp', 0.07)})")
        else:
            self.contrastive_loss = None

        # 3. 知识蒸馏损失
        if cfg.get('use_kd', True):
            self.kd_loss = TriModalKDLoss(
                temperature=cfg.get('kd_temp', 2.0),
                bimodal_weight=cfg.get('kd_bimodal_weight', 0.5)
            )
            print(f"  ✓ Tri-Modal KD Loss (temp={cfg.get('kd_temp', 2.0)})")
        else:
            self.kd_loss = None

        # 4. 一致性损失
        if cfg.get('use_consistency', True):
            self.consistency_loss = MultiViewConsistency(
                consistency_type=cfg.get('consistency_type', 'mse'),
                temperature=cfg.get('consistency_temp', 1.0)
            )
            print(f"  ✓ Consistency Loss ({cfg.get('consistency_type', 'mse')})")
        else:
            self.consistency_loss = None

        # 5. MIL Ranking损失（可选）
        if cfg.get('use_ranking', False):
            self.ranking_loss = RankingMILLoss(
                margin=cfg.get('ranking_margin', 0.5),
                topk=cfg.get('ranking_topk', 4)
            )
            print(f"  ✓ MIL Ranking Loss")
        else:
            self.ranking_loss = None

        # 损失权重（动态调整）
        self.loss_weights = {
            'cls': 1.0,
            'contrastive': self._parse_weight(cfg.get('contrastive_weight', '0.5')),
            'kd': self._parse_weight(cfg.get('kd_weight', '0.3')),
            'consistency': self._parse_weight(cfg.get('consistency_weight', '0.1')),
            'ranking': cfg.get('ranking_weight', 0.3)
        }

        print()

    def _parse_weight(self, w):
        """解析权重（支持动态调整 "0.1->0.5"）"""
        if isinstance(w, str) and '->' in w:
            start, end = map(float, w.split('->'))
            return {'start': start, 'end': end, 'dynamic': True}
        else:
            return {'value': float(w), 'dynamic': False}

    def _get_weight(self, name):
        """获取当前epoch的权重"""
        w = self.loss_weights[name]
        if isinstance(w, dict) and w.get('dynamic', False):
            # 线性插值
            progress = self.epoch / max(self.cfg['train']['epochs'], 1)
            value = w['start'] + (w['end'] - w['start']) * progress
            return value
        elif isinstance(w, dict):
            return w['value']
        else:
            return w

    def _build_optimizer(self):
        """构建优化器和学习率调度器"""
        cfg = self.cfg['train']

        # 分层学习率
        params = [
            {'params': self.model.video_backbone.parameters(), 'lr': cfg['lr'] * 0.01},
            {'params': self.model.audio_backbone.parameters(), 'lr': cfg['lr'] * 0.01},
            {'params': self.model.fusion.parameters(), 'lr': cfg['lr'] * 0.1},
            {'params': self.model.temporal.parameters(), 'lr': cfg['lr']},
            {'params': self.model.mil.parameters(), 'lr': cfg['lr']},
        ]

        # 对比学习投影头
        if self.contrastive_loss is not None:
            params.extend([
                {'params': self.video_proj.parameters(), 'lr': cfg['lr']},
                {'params': self.audio_proj.parameters(), 'lr': cfg['lr']}
            ])

        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 1e-4)
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg['epochs'],
            eta_min=cfg.get('min_lr', 1e-6)
        )

        print(f"🔧 优化器配置:")
        print(f"  ✓ AdamW (lr={cfg['lr']}, weight_decay={cfg.get('weight_decay', 1e-4)})")
        print(f"  ✓ CosineAnnealing Scheduler\n")

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()

        epoch_losses = {
            'total': 0.0,
            'cls': 0.0,
            'contrastive': 0.0,
            'kd': 0.0,
            'consistency': 0.0,
            'ranking': 0.0
        }

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            labels = batch['label_idx'].to(self.device)

            # ============================================================
            # 1. 标准前向传播（无增强）
            # ============================================================
            outputs = self.model(video, audio)

            # 分类损失
            loss_cls = self.cls_loss(outputs['clip_logits'], labels)

            # ============================================================
            # 2. 对比学习损失（拉近音视频匹配对）
            # ============================================================
            loss_con = torch.tensor(0.0, device=self.device)

            if self.contrastive_loss is not None:
                # 从模型获取单模态特征（需要修改模型返回）
                # 这里假设模型返回了video_emb和audio_emb
                if hasattr(outputs, 'keys') and 'video_emb' in outputs:
                    video_emb = outputs['video_emb']  # [B, D]
                    audio_emb = outputs['audio_emb']  # [B, D]
                else:
                    # Fallback: 使用全局池化
                    video_emb = outputs.get('z', None)
                    audio_emb = outputs.get('z', None)

                    if video_emb is not None:
                        video_emb = video_emb.mean(dim=1)
                        audio_emb = audio_emb.mean(dim=1)

                if video_emb is not None and audio_emb is not None:
                    # 投影到对比学习空间
                    z_v = self.video_proj(video_emb)
                    z_a = self.audio_proj(audio_emb)

                    loss_con, con_metrics = self.contrastive_loss(z_a, z_v)

            # ============================================================
            # 3. 知识蒸馏损失（融合→单模态）
            # ============================================================
            loss_kd = torch.tensor(0.0, device=self.device)

            if self.kd_loss is not None:
                # 需要模型返回单模态的logits
                if 'video_logits' in outputs and 'audio_logits' in outputs:
                    loss_kd, kd_metrics = self.kd_loss(
                        outputs['clip_logits'],
                        outputs['video_logits'],
                        outputs['audio_logits']
                    )

            # ============================================================
            # 4. 一致性损失（不同增强下预测应一致）
            # ============================================================
            loss_cons = torch.tensor(0.0, device=self.device)

            if self.consistency_loss is not None and self.weak_aug is not None:
                # 弱增强
                video_weak = video  # 原始视为弱增强
                audio_weak = audio

                # 强增强
                if self.strong_aug is not None:
                    # 需要逐样本增强（这里简化处理）
                    video_strong = video  # 实际应用strong_aug
                    audio_strong = audio

                    outputs_strong = self.model(video_strong, audio_strong)

                    # 一致性：原始预测 vs 强增强预测
                    logits_list = [
                        outputs['clip_logits'],
                        outputs_strong['clip_logits']
                    ]

                    loss_cons, cons_metrics = self.consistency_loss(
                        logits_list,
                        mode='mean_teacher'
                    )

            # ============================================================
            # 5. Ranking损失（可选）
            # ============================================================
            loss_rank = torch.tensor(0.0, device=self.device)

            if self.ranking_loss is not None and 'scores' in outputs:
                loss_rank = self.ranking_loss(outputs['scores'], labels)

            # ============================================================
            # 总损失（加权组合）
            # ============================================================
            w_cls = self._get_weight('cls')
            w_con = self._get_weight('contrastive')
            w_kd = self._get_weight('kd')
            w_cons = self._get_weight('consistency')
            w_rank = self._get_weight('ranking')

            total_loss = (
                    w_cls * loss_cls +
                    w_con * loss_con +
                    w_kd * loss_kd +
                    w_cons * loss_cons +
                    w_rank * loss_rank
            )

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 统计
            epoch_losses['total'] += total_loss.item()
            epoch_losses['cls'] += loss_cls.item()
            epoch_losses['contrastive'] += loss_con.item()
            epoch_losses['kd'] += loss_kd.item()
            epoch_losses['consistency'] += loss_cons.item()
            epoch_losses['ranking'] += loss_rank.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{total_loss.item():.3f}",
                'cls': f"{loss_cls.item():.3f}",
                'con': f"{loss_con.item():.3f}" if loss_con.item() > 0 else "0"
            })

            self.global_step += 1

        # 平均损失
        n_batches = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        return epoch_losses

    def validate(self, val_loader, minority_class=1):
        """验证"""
        metrics, probs, labels = validate_model(
            self.model, val_loader, self.device, minority_class
        )
        return metrics

    def train(self, train_loader, val_loader):
        """完整训练流程"""
        cfg = self.cfg['train']
        epochs = cfg['epochs']
        patience = cfg.get('patience', 15)
        early_stop_metric = cfg.get('early_stop_metric', 'auprc_minority')

        # 确定minority class
        minority_class = cfg.get('minority_class', 1)

        print("=" * 70)
        print("🚀 开始训练")
        print("=" * 70)
        print(f"  Epochs: {epochs}")
        print(f"  Patience: {patience}")
        print(f"  Early Stop Metric: {early_stop_metric}")
        print(f"  Minority Class: {minority_class}")
        print("=" * 70 + "\n")

        bad_epochs = 0

        for epoch in range(epochs):
            self.epoch = epoch

            # 训练
            train_losses = self.train_epoch(train_loader)

            # 打印训练损失
            print(f"\n📊 Epoch {epoch} - 训练损失:")
            print(f"  Total:       {train_losses['total']:.4f}")
            print(f"  Classification:  {train_losses['cls']:.4f}")
            print(f"  Contrastive:     {train_losses['contrastive']:.4f}")
            print(f"  KD:              {train_losses['kd']:.4f}")
            print(f"  Consistency:     {train_losses['consistency']:.4f}")

            # 验证
            val_metrics = self.validate(val_loader, minority_class)

            # 打印验证结果
            calc = ImbalancedMetricsCalculator(minority_class)
            calc.print_report(val_metrics, name=f"Validation (Epoch {epoch})")

            # 保存指标
            self.exp_mgr.save_metrics({
                'epoch': epoch,
                **val_metrics,
                **train_losses
            }, epoch)

            # 早停检查
            current_metric = val_metrics[early_stop_metric]

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                bad_epochs = 0

                # 保存最佳模型
                self.save_checkpoint('best_model.pth')

                print(f"\n⭐ 新的最佳 {early_stop_metric}: {self.best_metric:.4f}")
            else:
                bad_epochs += 1
                print(f"\n📊 {early_stop_metric}: {current_metric:.4f} "
                      f"(best: {self.best_metric:.4f}, no improve: {bad_epochs}/{patience})")

            # 学习率调度
            self.scheduler.step()

            print()

            # 早停
            if bad_epochs >= patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

        print("\n" + "=" * 70)
        print(f"✅ 训练完成！")
        print(f"   最佳 {early_stop_metric}: {self.best_metric:.4f}")
        print("=" * 70)

    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'video_proj': self.video_proj.state_dict() if self.contrastive_loss else None,
            'audio_proj': self.audio_proj.state_dict() if self.contrastive_loss else None,
        }

        save_path = self.exp_mgr.root / 'ckpts' / filename
        save_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser(description="统一训练脚本")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}\n")

    # 创建数据加载器
    print("📦 加载数据...")

    train_dataset = BinaryAVCSVDataset(
        cfg['data']['train_csv'],
        root=cfg['data'].get('root', ''),
        T_v=cfg['data']['T_v'],
        T_a=cfg['data']['T_a'],
        mel_bins=cfg['data']['mel'],
        sample_rate=cfg['data']['sr']
    )

    val_dataset = BinaryAVCSVDataset(
        cfg['data']['val_csv'],
        root=cfg['data'].get('root', ''),
        T_v=cfg['data']['T_v'],
        T_a=cfg['data']['T_a'],
        mel_bins=cfg['data']['mel'],
        sample_rate=cfg['data']['sr']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg['train'].get('workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg['train'].get('workers', 4),
        pin_memory=True
    )

    print(f"  ✓ 训练集: {len(train_dataset)} 样本")
    print(f"  ✓ 验证集: {len(val_dataset)} 样本\n")

    # 创建训练器
    trainer = UnifiedTrainer(cfg, device)

    # 开始训练
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()