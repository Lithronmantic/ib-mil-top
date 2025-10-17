#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
半监督学习训练器
实现FixMatch、伪标签、一致性正则化等方法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import wandb


class SemiSupTrainer:
    """
    半监督学习训练器
    
    支持的方法：
    - FixMatch: 强弱增强 + 伪标签
    - Pseudo Label: 高置信度伪标签
    - Consistency Regularization: 一致性损失
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 num_classes: int,
                 method: str = 'fixmatch',
                 # 损失权重
                 unsup_loss_weight: float = 1.0,
                 # 伪标签参数
                 confidence_threshold: float = 0.95,
                 # FixMatch参数
                 use_ema: bool = True,
                 ema_decay: float = 0.999,
                 # 学习率调度
                 use_scheduler: bool = True,
                 # 日志
                 use_wandb: bool = False):
        """
        Args:
            model: 分类模型
            optimizer: 优化器
            device: 设备
            num_classes: 类别数
            method: 半监督方法 ('fixmatch', 'pseudolabel', 'consistency')
            unsup_loss_weight: 无监督损失权重
            confidence_threshold: 伪标签置信度阈值
            use_ema: 是否使用EMA模型
            ema_decay: EMA衰减率
            use_scheduler: 是否使用学习率调度
            use_wandb: 是否使用wandb记录
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.method = method
        
        # 损失权重
        self.unsup_loss_weight = unsup_loss_weight
        self.confidence_threshold = confidence_threshold
        
        # EMA模型（用于伪标签生成）
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = self._create_ema_model()
            self.ema_decay = ema_decay
        else:
            self.ema_model = None
        
        # 学习率调度
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=1e-6
            )
        else:
            self.scheduler = None
        
        # 日志
        self.use_wandb = use_wandb
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 统计
        self.global_step = 0
        self.epoch = 0
    
    def _create_ema_model(self) -> nn.Module:
        """创建EMA模型（参数的指数移动平均）"""
        ema_model = type(self.model)(
            **{k: v for k, v in self.model.__dict__.items() 
               if not k.startswith('_')}
        ).to(self.device)
        
        # 复制参数
        for param_q, param_k in zip(self.model.parameters(), ema_model.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        return ema_model
    
    def _update_ema_model(self):
        """更新EMA模型参数"""
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for param_q, param_k in zip(self.model.parameters(), 
                                       self.ema_model.parameters()):
                param_k.data = param_k.data * self.ema_decay + \
                               param_q.data * (1 - self.ema_decay)
    
    def train_epoch(self, 
                   loader: DataLoader,
                   weak_aug_fn=None,
                   strong_aug_fn=None) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            loader: 数据加载器（使用MixedBatchSampler）
            weak_aug_fn: 弱增强函数 (audio, video) -> (audio_aug, video_aug)
            strong_aug_fn: 强增强函数
        
        Returns:
            损失字典
        """
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'supervised': 0.0,
            'unsupervised': 0.0,
            'mask_ratio': 0.0,  # 使用的伪标签比例
        }
        
        pbar = tqdm(loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 解包batch
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)
            is_labeled = batch['is_labeled'].to(self.device)
            
            # 分离有标注和无标注数据
            labeled_mask = (is_labeled == 1)
            unlabeled_mask = (is_labeled == 0)
            
            # ================================================================
            # 有监督损失（在有标注数据上）
            # ================================================================
            sup_loss = torch.tensor(0.0).to(self.device)
            
            if labeled_mask.sum() > 0:
                audio_labeled = audio[labeled_mask]
                video_labeled = video[labeled_mask]
                labels_labeled = labels[labeled_mask]
                
                # 可选：对有标注数据也做弱增强
                if weak_aug_fn is not None:
                    audio_labeled, video_labeled = weak_aug_fn(audio_labeled, video_labeled)
                
                # 前向传播
                logits_labeled = self.model(audio_labeled, video_labeled)
                sup_loss = self.criterion(logits_labeled, labels_labeled)
            
            # ================================================================
            # 无监督损失（在无标注数据上）
            # ================================================================
            unsup_loss = torch.tensor(0.0).to(self.device)
            mask_ratio = 0.0
            
            if unlabeled_mask.sum() > 0:
                audio_unlabeled = audio[unlabeled_mask]
                video_unlabeled = video[unlabeled_mask]
                
                if self.method == 'fixmatch':
                    unsup_loss, mask_ratio = self._fixmatch_loss(
                        audio_unlabeled, video_unlabeled,
                        weak_aug_fn, strong_aug_fn
                    )
                
                elif self.method == 'pseudolabel':
                    unsup_loss, mask_ratio = self._pseudolabel_loss(
                        audio_unlabeled, video_unlabeled
                    )
                
                elif self.method == 'consistency':
                    unsup_loss = self._consistency_loss(
                        audio_unlabeled, video_unlabeled,
                        weak_aug_fn, strong_aug_fn
                    )
                    mask_ratio = 1.0  # 所有样本都用于一致性
            
            # ================================================================
            # 总损失
            # ================================================================
            total_loss = sup_loss + self.unsup_loss_weight * unsup_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # 更新EMA模型
            self._update_ema_model()
            
            # 统计
            epoch_losses['total'] += total_loss.item()
            epoch_losses['supervised'] += sup_loss.item()
            epoch_losses['unsupervised'] += unsup_loss.item()
            epoch_losses['mask_ratio'] += mask_ratio
            
            # 更新进度条
            pbar.set_postfix({
                'sup': f"{sup_loss.item():.3f}",
                'unsup': f"{unsup_loss.item():.3f}",
                'mask': f"{mask_ratio:.2f}"
            })
            
            # Wandb日志
            if self.use_wandb:
                wandb.log({
                    'train/sup_loss': sup_loss.item(),
                    'train/unsup_loss': unsup_loss.item(),
                    'train/total_loss': total_loss.item(),
                    'train/mask_ratio': mask_ratio,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'step': self.global_step
                })
            
            self.global_step += 1
        
        # 平均损失
        n_batches = len(loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        
        # 学习率调度
        if self.use_scheduler:
            self.scheduler.step()
        
        self.epoch += 1
        
        return epoch_losses
    
    def _fixmatch_loss(self,
                       audio: torch.Tensor,
                       video: torch.Tensor,
                       weak_aug_fn,
                       strong_aug_fn) -> Tuple[torch.Tensor, float]:
        """
        FixMatch损失
        
        流程：
        1. 弱增强 -> 生成伪标签
        2. 强增强 -> 预测
        3. 高置信度伪标签用于训练
        """
        batch_size = audio.shape[0]
        
        # 1. 弱增强 + 伪标签生成
        with torch.no_grad():
            if weak_aug_fn is not None:
                audio_weak, video_weak = weak_aug_fn(audio, video)
            else:
                audio_weak, video_weak = audio, video
            
            # 使用EMA模型或当前模型生成伪标签
            model_for_pseudo = self.ema_model if self.use_ema else self.model
            logits_weak = model_for_pseudo(audio_weak, video_weak)
            probs_weak = F.softmax(logits_weak, dim=1)
            
            # 选择高置信度样本
            max_probs, pseudo_labels = torch.max(probs_weak, dim=1)
            mask = (max_probs >= self.confidence_threshold).float()
        
        # 2. 强增强 + 预测
        if strong_aug_fn is not None:
            audio_strong, video_strong = strong_aug_fn(audio, video)
        else:
            audio_strong, video_strong = audio, video
        
        logits_strong = self.model(audio_strong, video_strong)
        
        # 3. 计算损失（只在高置信度样本上）
        loss = (F.cross_entropy(logits_strong, pseudo_labels, reduction='none') * mask).mean()
        
        # 统计掩码比例
        mask_ratio = mask.sum().item() / batch_size
        
        return loss, mask_ratio
    
    def _pseudolabel_loss(self,
                         audio: torch.Tensor,
                         video: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        伪标签损失（不使用增强）
        """
        batch_size = audio.shape[0]
        
        # 生成伪标签
        with torch.no_grad():
            model_for_pseudo = self.ema_model if self.use_ema else self.model
            logits = model_for_pseudo(audio, video)
            probs = F.softmax(logits, dim=1)
            
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            mask = (max_probs >= self.confidence_threshold).float()
        
        # 预测
        logits_pred = self.model(audio, video)
        
        # 损失
        loss = (F.cross_entropy(logits_pred, pseudo_labels, reduction='none') * mask).mean()
        
        mask_ratio = mask.sum().item() / batch_size
        
        return loss, mask_ratio
    
    def _consistency_loss(self,
                         audio: torch.Tensor,
                         video: torch.Tensor,
                         weak_aug_fn,
                         strong_aug_fn) -> torch.Tensor:
        """
        一致性正则化损失
        
        思想：同一样本的不同增强应有相似的预测
        """
        # 弱增强
        if weak_aug_fn is not None:
            audio_weak, video_weak = weak_aug_fn(audio, video)
        else:
            audio_weak, video_weak = audio, video
        
        logits_weak = self.model(audio_weak, video_weak)
        probs_weak = F.softmax(logits_weak, dim=1)
        
        # 强增强
        if strong_aug_fn is not None:
            audio_strong, video_strong = strong_aug_fn(audio, video)
        else:
            audio_strong, video_strong = audio, video
        
        logits_strong = self.model(audio_strong, video_strong)
        probs_strong = F.softmax(logits_strong, dim=1)
        
        # KL散度损失（鼓励一致性）
        loss = F.kl_div(
            F.log_softmax(logits_strong, dim=1),
            probs_weak,
            reduction='batchmean'
        )
        
        return loss
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            loader: 验证数据加载器（只包含有标注数据）
        
        Returns:
            评估指标字典
        """
        # 使用EMA模型或当前模型
        eval_model = self.ema_model if self.use_ema else self.model
        eval_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm(loader, desc="Evaluating"):
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 只评估有标注数据
            is_labeled = batch['is_labeled']
            if (is_labeled == 0).all():
                continue
            
            # 前向传播
            logits = eval_model(audio, video)
            loss = self.criterion(logits, labels)
            
            # 预测
            _, preds = torch.max(logits, 1)
            
            # 统计
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
        }
        
        # 每类准确率
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        for cls in range(self.num_classes):
            mask = (all_labels == cls)
            if mask.sum() > 0:
                cls_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
                metrics[f'acc_class_{cls}'] = cls_acc
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        
        if self.use_ema:
            checkpoint['ema_model_state'] = self.ema_model.state_dict()
        
        if self.use_scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if self.use_ema and 'ema_model_state' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_model_state'])
        
        if self.use_scheduler and 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])


# ============================================================================
# 完整训练循环示例
# ============================================================================
def train_semisupervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 100,
    method: str = 'fixmatch',
    save_dir: str = './checkpoints'
):
    """
    完整的半监督训练流程
    
    Args:
        model: 模型
        train_loader: 训练数据（包含有标注和无标注）
        val_loader: 验证数据（只有标注数据）
        device: 设备
        num_epochs: 训练轮数
        method: 半监督方法
        save_dir: 检查点保存目录
    """
    from pathlib import Path
    from avtop.data.augmentation import MultiModalAugmentation
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 创建训练器
    trainer = SemiSupTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        num_classes=2,
        method=method,
        unsup_loss_weight=1.0,
        confidence_threshold=0.95,
        use_ema=True,
        use_scheduler=True,
        use_wandb=False
    )
    
    # 创建增强器
    weak_aug = MultiModalAugmentation(mode='weak')
    strong_aug = MultiModalAugmentation(mode='strong')
    
    # 训练循环
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        
        # 训练
        train_losses = trainer.train_epoch(
            train_loader,
            weak_aug_fn=weak_aug,
            strong_aug_fn=strong_aug
        )
        
        print(f"\n训练损失:")
        print(f"  Total: {train_losses['total']:.4f}")
        print(f"  Supervised: {train_losses['supervised']:.4f}")
        print(f"  Unsupervised: {train_losses['unsupervised']:.4f}")
        print(f"  Mask Ratio: {train_losses['mask_ratio']:.2%}")
        
        # 验证
        if (epoch + 1) % 5 == 0:
            val_metrics = trainer.evaluate(val_loader)
            
            print(f"\n验证指标:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.2%}")
            
            # 保存最佳模型
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                trainer.save_checkpoint(save_dir / 'best_model.pth')
                print(f"  ✅ 保存最佳模型 (acc={best_acc:.2%})")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(save_dir / f'checkpoint_epoch{epoch+1}.pth')
    
    print(f"\n{'='*70}")
    print(f"训练完成！最佳验证准确率: {best_acc:.2%}")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("半监督训练器模块 - 使用示例请参考 train_semisupervised() 函数")
