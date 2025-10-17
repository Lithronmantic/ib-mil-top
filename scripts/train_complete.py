#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_complete.py - 完整训练脚本（无省略）
集成所有SOTA方法的端到端训练流程

功能：
1. Co-Attention融合
2. GRAM对比学习
3. 双向KD
4. 一致性正则
5. FixMatch半监督
6. 完整checkpoint管理
7. Wandb/TensorBoard日志
"""
from torch.amp import GradScaler
from contextlib import nullcontext
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os

import argparse
import yaml
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 尝试导入wandb（可选）
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not installed, logging disabled")

os.environ['WANDB_API_KEY'] = '5348ec832d279c723ddbf774a64d7b1b9d4fa407'

# 项目模块导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.avtop.models.enhanced_detector import EnhancedAVDetector
from src.avtop.data.window_dataset import WindowDataset, collate_fn
from src.avtop.losses.gram_contrastive import CompleteLossFunction


class Trainer:
    """完整训练器"""

    def __init__(self, config: Dict, output_dir: str):
        """
        Args:
            config: 配置字典
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # 设备
        self.device = self._setup_device(config)

        # 设置随机种子
        self._set_seed(config.get('seed', 42))

        # 模型
        print("\n" + "=" * 70)
        print("🗿 初始化模型")
        print("=" * 70)
        self.model = self._build_model(config)
        self.model = self.model.to(self.device)

        # 数据加载器
        print("\n" + "=" * 70)
        print("📂 加载数据集")
        print("=" * 70)
        self.train_loader, self.val_loader = self._build_dataloaders(config)

        # 损失函数
        print("\n" + "=" * 70)
        print("🎯 设置损失函数")
        print("=" * 70)
        self.criterion = self._build_criterion(config)

        # 优化器和调度器
        print("\n" + "=" * 70)
        print("⚙️ 设置优化器")
        print("=" * 70)
        self.optimizer, self.scheduler = self._build_optimizer(config)

        # 混合精度训练
        self.use_amp = config.get('hardware', {}).get('mixed_precision', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ 启用混合精度训练")
        else:
            self.scaler = None

        # 梯度累积
        self.grad_accum_steps = config.get('hardware', {}).get('gradient_accumulation_steps', 1)
        if self.grad_accum_steps > 1:
            print(f"✅ 启用梯度累积 (steps={self.grad_accum_steps})")

        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.global_step = 0

        # Wandb
        self.use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb(config)

    def _setup_device(self, config: Dict) -> torch.device:
        """设置设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            print("⚠️ 使用CPU（训练会很慢）")
        return device

    def _set_seed(self, seed: int):
        """设置随机种子（可复现）"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"✅ 随机种子设置为: {seed}")

    def _build_model(self, config: Dict) -> nn.Module:
        """构建模型"""
        model = EnhancedAVDetector(config)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"模型: EnhancedAVDetector")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练: {trainable_params:,}")
        print(f"  融合类型: {model.fusion_type}")

        return model

    def _build_dataloaders(self, config: Dict):
        """构建数据加载器"""
        data_config = config['data']
        training_config = config['training']

        # 训练集（包含有标签和无标签）
        train_dataset = WindowDataset(
            csv_path=data_config['train_csv'],
            target_sr=data_config.get('audio_sr', 16000),
            target_video_size=tuple(data_config.get('video_size', [224, 224])),
            max_audio_length=data_config.get('max_audio_length', 0.3),
            max_video_frames=data_config.get('max_video_frames', 16),
            cache_mode=data_config.get('cache_mode', 'none')
        )

        # 验证集
        val_dataset = WindowDataset(
            csv_path=data_config['val_csv'],
            target_sr=data_config.get('audio_sr', 16000),
            target_video_size=tuple(data_config.get('video_size', [224, 224])),
            max_audio_length=data_config.get('max_audio_length', 0.3),
            max_video_frames=data_config.get('max_video_frames', 16),
            cache_mode='none'
        )

        print(f"训练集: {len(train_dataset)} 窗口对")
        print(f"  - 有标签: {sum(train_dataset.data['is_labeled'])}")
        print(f"  - 无标签: {sum(~train_dataset.data['is_labeled'].astype(bool))}")
        print(f"验证集: {len(val_dataset)} 窗口对")

        # DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.get('batch_size', 8),
            shuffle=True,
            num_workers=config.get('hardware', {}).get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.get('batch_size', 8),
            shuffle=False,
            num_workers=config.get('hardware', {}).get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )

        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")

        return train_loader, val_loader

    def _build_criterion(self, config: Dict):
        """构建损失函数"""
        loss_config = config.get('loss', {})

        criterion = CompleteLossFunction(
            num_classes=config['model']['num_classes'],
            lambda_contrastive=loss_config.get('lambda_contrastive', 0.3),
            lambda_kd=loss_config.get('lambda_kd', 0.2),
            lambda_consistency=loss_config.get('lambda_consistency', 0.1),
            temperature=loss_config.get('temperature', 0.07),
            kd_temperature=loss_config.get('kd_temperature', 4.0)
        )

        print("损失函数: CompleteLossFunction")
        print(f"  - 对比学习权重: {loss_config.get('lambda_contrastive', 0.3)}")
        print(f"  - KD权重: {loss_config.get('lambda_kd', 0.2)}")
        print(f"  - 一致性权重: {loss_config.get('lambda_consistency', 0.1)}")

        return criterion.to(self.device)

    def _build_optimizer(self, config: Dict):
        """构建优化器和学习率调度器"""
        training_config = config['training']

        # 优化器
        optimizer_type = training_config.get('optimizer', 'adamw').lower()
        lr = training_config.get('learning_rate', 1e-5)  # 🔧 默认降低到1e-5
        weight_decay = training_config.get('weight_decay', 1e-4)

        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        print(f"优化器: {optimizer_type.upper()}")
        print(f"  - 学习率: {lr}")
        print(f"  - 权重衰减: {weight_decay}")

        # 学习率调度器
        scheduler_type = training_config.get('scheduler', 'cosine').lower()
        num_epochs = training_config.get('num_epochs', 100)

        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=lr * 0.01
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=training_config.get('step_size', 30),
                gamma=training_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            scheduler = None

        if scheduler:
            print(f"学习率调度: {scheduler_type}")

        return optimizer, scheduler

    def _init_wandb(self, config: Dict):
        """初始化Wandb"""
        wandb.init(
            project=config.get('wandb_project', 'avtop-training'),
            name=config.get('wandb_name', f'run_{int(time.time())}'),
            config=config,
            dir=str(self.logs_dir)
        )
        wandb.watch(self.model, log='all', log_freq=100)
        print("✅ Wandb已初始化")

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()

        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_ctr_loss = 0.0
        epoch_kd_loss = 0.0
        epoch_cons_loss = 0.0

        num_batches = len(self.train_loader)

        # 进度条
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}",
            ncols=100
        )

        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            is_labeled = batch['is_labeled'].to(self.device)

            # 🔧 检查输入是否有nan
            if torch.isnan(video).any() or torch.isnan(audio).any():
                print(f"⚠️ 输入包含nan，跳过此batch")
                continue

            # 前向传播（混合精度）
            if self.use_amp:
                amp_ctx = torch.amp.autocast('cuda') if self.device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    outputs = self.model(video, audio, return_aux=True)

                    # 🔧 检查输出是否有nan
                    if torch.isnan(outputs['clip_logits']).any():
                        print(f"⚠️ 模型输出包含nan，跳过此batch")
                        continue

                    loss_dict = self.criterion(outputs, labels, is_labeled)
                    loss = loss_dict['total_loss']

                    # 🔧 检查损失是否有nan
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        print(f"⚠️ 损失为nan/inf: {loss.item()}")
                        continue

                    loss = loss / self.grad_accum_steps

                # 反向传播（混合精度）
                self.scaler.scale(loss).backward()

                # 梯度累积
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    # 🔧 梯度裁剪（防止梯度爆炸）
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

            else:
                # 标准训练
                outputs = self.model(video, audio, return_aux=True)

                # 🔧 检查输出是否有nan
                if torch.isnan(outputs['clip_logits']).any():
                    print(f"⚠️ 模型输出包含nan，跳过此batch")
                    continue

                loss_dict = self.criterion(outputs, labels, is_labeled)
                loss = loss_dict['total_loss']

                # 🔧 检查损失是否有nan
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"⚠️ 损失为nan/inf: {loss.item()}")
                    continue

                loss = loss / self.grad_accum_steps

                # 反向传播
                loss.backward()

                # 梯度累积
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    # 🔧 梯度裁剪（防止梯度爆炸）
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # 累积损失
            epoch_loss += loss_dict['total_loss'].item()
            epoch_cls_loss += loss_dict['classification_loss'].item()
            epoch_ctr_loss += loss_dict['contrastive_loss'].item()
            epoch_kd_loss += loss_dict['kd_loss'].item()
            epoch_cons_loss += loss_dict['consistency_loss'].item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'cls': f"{loss_dict['classification_loss'].item():.4f}",
                'ctr': f"{loss_dict['contrastive_loss'].item():.4f}"
            })

            # 记录到Wandb
            self.global_step += 1
            if self.use_wandb and self.global_step % 10 == 0:
                wandb.log({
                    'train/loss': loss_dict['total_loss'].item(),
                    'train/cls_loss': loss_dict['classification_loss'].item(),
                    'train/ctr_loss': loss_dict['contrastive_loss'].item(),
                    'train/kd_loss': loss_dict['kd_loss'].item(),
                    'train/cons_loss': loss_dict['consistency_loss'].item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })

        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        avg_cls_loss = epoch_cls_loss / num_batches
        avg_ctr_loss = epoch_ctr_loss / num_batches
        avg_kd_loss = epoch_kd_loss / num_batches
        avg_cons_loss = epoch_cons_loss / num_batches

        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'ctr_loss': avg_ctr_loss,
            'kd_loss': avg_kd_loss,
            'cons_loss': avg_cons_loss
        }

    def validate(self, epoch: int):
        """验证"""
        self.model.eval()

        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", ncols=100):
                video = batch['video'].to(self.device)
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                is_labeled = batch['is_labeled'].to(self.device)

                # 前向传播
                outputs = self.model(video, audio, return_aux=True)
                loss_dict = self.criterion(outputs, labels, is_labeled)

                val_loss += loss_dict['total_loss'].item()

                # 🔧 使用正确的key获取logits
                clip_logits = outputs['clip_logits']
                preds = torch.argmax(clip_logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        avg_val_loss = val_loss / len(self.val_loader)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()

        # 每个类别的准确率
        class_acc = {}
        for c in range(self.config['model']['num_classes']):
            mask = all_labels == c
            if mask.sum() > 0:
                class_acc[f'class_{c}'] = (all_preds[mask] == all_labels[mask]).mean()

        metrics = {
            'val_loss': avg_val_loss,
            'val_accuracy': accuracy,
            **class_acc
        }

        # 记录到Wandb
        if self.use_wandb:
            wandb.log({
                'val/loss': avg_val_loss,
                'val/accuracy': accuracy,
                **{f'val/{k}': v for k, v in class_acc.items()},
                'epoch': epoch
            })

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存最新checkpoint
        latest_path = self.checkpoints_dir / f"checkpoint_epoch{epoch}.pth"
        torch.save(checkpoint, latest_path)
        print(f"💾 Checkpoint已保存: {latest_path}")

        # 保存最佳模型
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"🏆 最佳模型已保存: {best_path}")

        # 删除旧的checkpoint（保留最近5个）
        checkpoints = sorted(self.checkpoints_dir.glob("checkpoint_epoch*.pth"))
        if len(checkpoints) > 5:
            for ckpt in checkpoints[:-5]:
                ckpt.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"✅ Checkpoint已加载: {checkpoint_path}")
        print(f"   恢复到Epoch {self.current_epoch}")
        print(f"   最佳验证准确率: {self.best_val_acc:.4f}")

    def train(self):
        """完整训练流程"""
        num_epochs = self.config['training']['num_epochs']
        start_epoch = self.current_epoch + 1

        print("\n" + "=" * 70)
        print(f"🚀 开始训练 (Epoch {start_epoch} - {num_epochs})")
        print("=" * 70)

        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch(epoch)

            # 验证
            val_metrics = self.validate(epoch)

            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            # 打印结果
            print(f"\n" + "=" * 70)
            print(f"Epoch {epoch}/{num_epochs} 总结")
            print("=" * 70)
            print(f"训练损失: {train_metrics['loss']:.4f}")
            print(f"  - 分类: {train_metrics['cls_loss']:.4f}")
            print(f"  - 对比: {train_metrics['ctr_loss']:.4f}")
            print(f"  - KD: {train_metrics['kd_loss']:.4f}")
            print(f"  - 一致性: {train_metrics['cons_loss']:.4f}")
            print(f"验证损失: {val_metrics['val_loss']:.4f}")
            print(f"验证准确率: {val_metrics['val_accuracy']:.4f}")
            for k, v in val_metrics.items():
                if k.startswith('class_'):
                    print(f"  - {k}: {v:.4f}")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存checkpoint
            is_best = val_metrics['val_accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['val_accuracy']
                self.best_val_loss = val_metrics['val_loss']
                print(f"🎉 新的最佳准确率: {self.best_val_acc:.4f}")

            if epoch % self.config['training'].get('save_every', 5) == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)

            print("=" * 70 + "\n")

        print("✅ 训练完成！")
        print(f"🏆 最佳验证准确率: {self.best_val_acc:.4f}")

        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="完整训练脚本")
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--resume', default=None, help='恢复训练的checkpoint路径')
    parser.add_argument('--output_dir', default='outputs/sota_training', help='输出目录')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)

    # 创建训练器
    trainer = Trainer(config, args.output_dir)

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⛔ 训练被中断")
        # 保存最后的checkpoint
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
        print("💾 已保存当前checkpoint")


if __name__ == "__main__":
    main()