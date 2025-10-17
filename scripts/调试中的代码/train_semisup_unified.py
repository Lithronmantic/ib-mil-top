#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一半监督训练脚本
支持两阶段训练：
1. Pretrain: 对比学习预训练（可选）
2. SemiSup: 半监督微调（CE + KD + Consistency + Contrastive）

使用方法：
# Stage 1: 对比学习预训练
python scripts/train_semisup.py \
    --config configs/real_binary_pretrain.yaml \
    --stage pretrain

# Stage 2: 半监督微调
python scripts/train_semisup.py \
    --config configs/real_binary_semisup.yaml \
    --stage semisup \
    --load_pretrain experiments/pretrain/best_model.pth
"""
import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入模型和损失
from src.avtop.models.enhanced_detector import EnhancedAVTopDetector
from src.avtop.losses.contrastive import InfoNCELoss, ProjectionHead
from src.avtop.losses.kd import BidirectionalKDLoss, TriModalKDLoss
from src.avtop.losses.consistency import ConsistencyLoss, MultiViewConsistency
from src.avtop.losses.classification import FocalLoss

# 导入数据和工具
from src.avtop.data.csv_dataset import AVTopDataset
from datasets.mix_sampler import create_mixed_dataloader
from hooks.pseudo_label_miner import PseudoLabelMiner
from src.avtop.utils.experiment_manager import ExperimentManager


class SemiSupervisedTrainer:
    """
    半监督训练器
    
    整合：
    - 对比学习（InfoNCE）
    - 知识蒸馏（Tri-Modal KD）
    - 一致性正则化
    - 伪标签挖掘
    """
    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.device = device
        
        # 1. 创建模型
        self.model = EnhancedAVTopDetector(cfg).to(device)
        
        # 2. 创建投影头（用于对比学习）
        if cfg.get('contrastive', {}).get('enable', False):
            proj_dim = cfg['contrastive'].get('proj_dim', 256)
            self.video_proj = ProjectionHead(
                cfg['model']['fusion_dim'], 
                output_dim=proj_dim
            ).to(device)
            self.audio_proj = ProjectionHead(
                cfg['model']['fusion_dim'], 
                output_dim=proj_dim
            ).to(device)
        else:
            self.video_proj = None
            self.audio_proj = None
        
        # 3. 创建损失函数
        self._build_losses()
        
        # 4. 创建优化器
        self._build_optimizer()
        
        # 5. 伪标签挖掘器
        if cfg['train']['stage'] == 'semisup':
            self.pseudo_miner = PseudoLabelMiner(
                confidence_threshold=cfg['train'].get('pseudo_threshold', 0.9),
                selection_strategy=cfg['train'].get('pseudo_strategy', 'confidence')
            )
        else:
            self.pseudo_miner = None
        
        # 6. 实验管理器
        self.exp_mgr = ExperimentManager(
            exp_name=cfg['exp_name'],
            base_dir=Path(cfg['output_dir'])
        )
        self.exp_mgr.save_config(cfg)
    
    def _build_losses(self):
        """构建所有损失函数"""
        cfg = self.cfg
        
        # 1. 监督分类损失
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(
            alpha=cfg['train'].get('focal_alpha', 0.25),
            gamma=cfg['train'].get('focal_gamma', 2.0)
        )
        
        # 2. 对比学习损失
        if cfg.get('contrastive', {}).get('enable', False):
            self.contrastive_loss = InfoNCELoss(
                temperature=cfg['contrastive'].get('temp', 0.07),
                queue_size=cfg['contrastive'].get('queue', 0)
            )
        else:
            self.contrastive_loss = None
        
        # 3. 知识蒸馏损失
        if cfg.get('kd', {}).get('enable', False):
            self.kd_loss = TriModalKDLoss(
                temperature=cfg['kd'].get('temperature', 2.0),
                bimodal_weight=cfg['kd'].get('bimodal_weight', 0.5)
            )
        else:
            self.kd_loss = None
        
        # 4. 一致性损失
        if cfg.get('consistency', {}).get('enable', False):
            self.consistency_loss = MultiViewConsistency(
                consistency_type=cfg['consistency'].get('type', 'mse')
            )
        else:
            self.consistency_loss = None
    
    def _build_optimizer(self):
        """构建优化器和学习率调度器"""
        cfg = self.cfg
        
        # 参数分组（backbone用小学习率）
        params = [
            {'params': self.model.fusion.parameters(), 'lr': cfg['train']['lr']},
            {'params': self.model.mil_head.parameters(), 'lr': cfg['train']['lr']}
        ]
        
        if self.video_proj is not None:
            params.append({'params': self.video_proj.parameters(), 'lr': cfg['train']['lr']})
            params.append({'params': self.audio_proj.parameters(), 'lr': cfg['train']['lr']})
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg['train']['lr'],
            weight_decay=cfg['train'].get('weight_decay', 1e-4)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg['train']['epochs'],
            eta_min=cfg['train'].get('min_lr', 1e-6)
        )
    
    def train_epoch_pretrain(self, loader, epoch):
        """
        预训练阶段（对比学习）
        只用InfoNCE损失
        """
        self.model.train()
        total_loss = 0
        total_acc = 0
        
        pbar = tqdm(loader, desc=f"Pretrain Epoch {epoch}")
        
        for batch in pbar:
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            
            # 前向传播
            outputs = self.model(video, audio, return_aux=True)
            
            # 获取全局嵌入
            video_emb = outputs['video_emb']  # [B, D]
            audio_emb = outputs['audio_emb']  # [B, D]
            
            # 投影到对比学习空间
            z_v = self.video_proj(video_emb)  # [B, proj_dim]
            z_a = self.audio_proj(audio_emb)  # [B, proj_dim]
            
            # 对比学习损失
            loss, metrics = self.contrastive_loss(z_a, z_v, bidirectional=True)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_acc += metrics['contrastive_acc']
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['contrastive_acc']*100:.1f}%"
            })
        
        avg_loss = total_loss / len(loader)
        avg_acc = total_acc / len(loader)
        
        return {
            'loss': avg_loss,
            'acc': avg_acc
        }
    
    def train_epoch_semisup(self, labeled_loader, unlabeled_loader, epoch):
        """
        半监督微调阶段
        整合：CE + KD + Consistency + InfoNCE + Pseudo Label
        """
        self.model.train()
        
        # 损失权重（动态调整）
        lambda_con = self._get_weight('contrastive', epoch)
        lambda_kd = self._get_weight('kd', epoch)
        lambda_cons = self._get_weight('consistency', epoch)
        
        total_losses = {
            'total': 0, 'ce': 0, 'kd': 0, 'cons': 0, 'con': 0
        }
        
        # 标注数据迭代器
        labeled_iter = iter(labeled_loader)
        
        pbar = tqdm(unlabeled_loader, desc=f"SemiSup Epoch {epoch}")
        
        for unlabeled_batch in pbar:
            # 1. 获取标注batch
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_batch = next(labeled_iter)
            
            # === 标注数据：监督学习 ===
            video_l = labeled_batch['video'].to(self.device)
            audio_l = labeled_batch['audio'].to(self.device)
            labels_l = labeled_batch['label'].to(self.device)
            
            outputs_l = self.model(video_l, audio_l, return_aux=True)
            
            # (1) 分类损失（融合模型）
            loss_ce = self.focal_loss(outputs_l['clip_logits'], labels_l)
            
            # (2) 知识蒸馏损失（融合↔单模态）
            if self.kd_loss is not None and 'video_logits' in outputs_l:
                loss_kd, kd_metrics = self.kd_loss(
                    outputs_l['clip_logits'],
                    outputs_l['video_logits'],
                    outputs_l['audio_logits']
                )
            else:
                loss_kd = torch.tensor(0.0, device=self.device)
            
            # === 未标注数据：一致性正则化 ===
            video_u = unlabeled_batch['video'].to(self.device)
            audio_u = unlabeled_batch['audio'].to(self.device)
            
            outputs_u = self.model(video_u, audio_u, return_aux=True)
            
            # (3) 一致性损失（单模态与融合模型）
            if self.consistency_loss is not None and 'video_logits' in outputs_u:
                logits_list = [
                    outputs_u['clip_logits'],
                    outputs_u['video_logits'],
                    outputs_u['audio_logits']
                ]
                loss_cons, cons_metrics = self.consistency_loss(
                    logits_list, mode='mean_teacher'
                )
            else:
                loss_cons = torch.tensor(0.0, device=self.device)
            
            # (4) 对比学习损失（标注+未标注一起）
            if self.contrastive_loss is not None:
                # 合并标注和未标注的嵌入
                video_emb = torch.cat([outputs_l['video_emb'], outputs_u['video_emb']], dim=0)
                audio_emb = torch.cat([outputs_l['audio_emb'], outputs_u['audio_emb']], dim=0)
                
                z_v = self.video_proj(video_emb)
                z_a = self.audio_proj(audio_emb)
                
                loss_con, con_metrics = self.contrastive_loss(z_a, z_v)
            else:
                loss_con = torch.tensor(0.0, device=self.device)
            
            # === 总损失 ===
            loss = loss_ce + \
                   lambda_kd * loss_kd + \
                   lambda_cons * loss_cons + \
                   lambda_con * loss_con
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 记录
            total_losses['total'] += loss.item()
            total_losses['ce'] += loss_ce.item()
            total_losses['kd'] += loss_kd.item() if isinstance(loss_kd, torch.Tensor) else 0
            total_losses['cons'] += loss_cons.item() if isinstance(loss_cons, torch.Tensor) else 0
            total_losses['con'] += loss_con.item() if isinstance(loss_con, torch.Tensor) else 0
            
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'ce': f"{loss_ce.item():.3f}",
                'kd': f"{loss_kd.item():.3f}" if isinstance(loss_kd, torch.Tensor) else "0",
            })
        
        # 平均损失
        for key in total_losses:
            total_losses[key] /= len(unlabeled_loader)
        
        return total_losses
    
    def _get_weight(self, loss_name, epoch):
        """获取动态损失权重"""
        cfg = self.cfg.get(loss_name, {})
        
        if not cfg.get('enable', False):
            return 0.0
        
        weight = cfg.get('weight', 1.0)
        
        # 支持线性调度 "0.1->0.5"
        if isinstance(weight, str) and '->' in weight:
            start, end = map(float, weight.split('->'))
            total_epochs = self.cfg['train']['epochs']
            alpha = epoch / max(total_epochs, 1)
            weight = start + (end - start) * alpha
        
        return float(weight)
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(self.device)
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(video, audio, return_aux=False)
                
                loss = self.ce_loss(outputs['clip_logits'], labels)
                total_loss += loss.item()
                
                preds = outputs['clip_logits'].argmax(dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        # AUC（需要概率）
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.0
        
        return {
            'loss': total_loss / len(val_loader),
            'acc': acc,
            'f1': f1,
            'auc': auc
        }


def main(args):
    """主训练函数"""
    
    # 加载配置
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 命令行参数覆盖
    if args.stage:
        cfg['train']['stage'] = args.stage
    
    print("\n" + "="*70)
    print(f"半监督训练 - {cfg['train']['stage'].upper()} 阶段")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建训练器
    trainer = SemiSupervisedTrainer(cfg, device)
    
    # 加载预训练权重（如果有）
    if args.load_pretrain and Path(args.load_pretrain).exists():
        print(f"\n加载预训练权重: {args.load_pretrain}")
        checkpoint = torch.load(args.load_pretrain)
        trainer.model.load_state_dict(checkpoint['model'], strict=False)
    
    # 加载数据
    # TODO: 根据实际情况加载数据集
    # train_loader = ...
    # val_loader = ...
    
    print("\n⚠️  数据加载部分需要根据实际数据集实现")
    print("   请参考 src/avtop/data/csv_dataset.py")
    
    print("\n✅ 训练器创建成功！")
    print(f"   融合策略: {trainer.model.fusion_type}")
    print(f"   训练阶段: {cfg['train']['stage']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--stage', type=str, choices=['pretrain', 'semisup'],
                       help='训练阶段')
    parser.add_argument('--load_pretrain', type=str, default=None,
                       help='预训练权重路径')
    
    args = parser.parse_args()
    
    main(args)
