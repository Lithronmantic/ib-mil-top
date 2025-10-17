#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EnhancedAVTopDetector - 更新版
支持多种融合策略：CFA, IB, CoAttention

修改内容：
1. 添加fusion.type配置支持
2. 集成CoAttentionFusion
3. 保持向后兼容
4. 返回格式统一
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

# 导入融合模块
from src.avtop.fusion.cfa_fusion import CFAFusion  # 原有
from src.avtop.fusion.ib_fusion import InformationBottleneckFusion  # 原有
from src.avtop.fusion.coattention import CoAttentionFusion  # 新增

# 导入其他模块
from src.avtop.models.backbones import VideoBackbone, AudioBackbone
from src.avtop.models.temporal_encoder import SimpleTemporalEncoder
from src.avtop.mil.enhanced_mil import EnhancedMIL


class EnhancedAVTopDetector(nn.Module):
    """
    增强的多模态焊接缺陷检测器
    
    新增功能：
    - 支持多种融合策略（通过配置选择）
    - 返回单模态分支输出（用于KD和一致性）
    - 返回全局嵌入（用于对比学习）
    
    配置示例：
    cfg = {
        'fusion': {
            'type': 'coattn',  # 'cfa', 'ib', 'coattn'
            'd_model': 256,
            'num_layers': 2,
            'num_heads': 8
        },
        ...
    }
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        
        self.cfg = cfg
        
        # 1. 特征提取器
        self.video_backbone = self._build_video_backbone(cfg)
        self.audio_backbone = self._build_audio_backbone(cfg)
        
        # 2. 时序编码器（可选）
        if cfg.get('use_temporal_encoder', True):
            self.video_temporal = SimpleTemporalEncoder(
                input_dim=cfg['model']['video_dim'],
                hidden_dim=cfg['model'].get('hidden_dim', 256)
            )
            self.audio_temporal = SimpleTemporalEncoder(
                input_dim=cfg['model']['audio_dim'],
                hidden_dim=cfg['model'].get('hidden_dim', 256)
            )
        else:
            self.video_temporal = None
            self.audio_temporal = None
        
        # 3. 融合模块（根据配置选择）
        fusion_cfg = cfg.get('fusion', {'type': 'cfa'})
        fusion_type = fusion_cfg.get('type', 'cfa')
        
        video_dim = cfg['model']['video_dim']
        audio_dim = cfg['model']['audio_dim']
        fusion_dim = cfg['model'].get('fusion_dim', 256)
        
        if fusion_type == 'coattn':
            # 新的协同注意力融合
            self.fusion = CoAttentionFusion(
                video_dim=video_dim,
                audio_dim=audio_dim,
                d_model=fusion_cfg.get('d_model', fusion_dim),
                num_layers=fusion_cfg.get('num_layers', 2),
                num_heads=fusion_cfg.get('num_heads', 8),
                dropout=fusion_cfg.get('dropout', 0.1)
            )
            self.fusion_type = 'coattn'
            
        elif fusion_type == 'ib':
            # Information Bottleneck融合
            self.fusion = InformationBottleneckFusion(
                video_dim=video_dim,
                audio_dim=audio_dim,
                fusion_dim=fusion_dim,
                beta=fusion_cfg.get('beta', 0.1)
            )
            self.fusion_type = 'ib'
            
        else:  # 默认使用CFA
            # Cross-modal Feature Alignment融合
            self.fusion = CFAFusion(
                video_dim=video_dim,
                audio_dim=audio_dim,
                fusion_dim=fusion_dim
            )
            self.fusion_type = 'cfa'
        
        print(f"[EnhancedDetector] 使用融合策略: {self.fusion_type}")
        
        # 4. MIL分类头
        self.mil_head = EnhancedMILHead(
            input_dim=fusion_dim,
            num_classes=cfg['model']['num_classes']
        )
        
        # 5. 单模态辅助分类头（用于KD和一致性）
        if cfg.get('use_aux_heads', True):
            self.video_head = nn.Sequential(
                nn.Linear(video_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, cfg['model']['num_classes'])
            )
            
            self.audio_head = nn.Sequential(
                nn.Linear(audio_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, cfg['model']['num_classes'])
            )
        else:
            self.video_head = None
            self.audio_head = None
    
    def _build_video_backbone(self, cfg):
        """构建视频backbone"""
        backbone_type = cfg['model'].get('video_backbone', 'resnet')
        return VideoBackbone(
            backbone_type=backbone_type,
            pretrained=cfg['model'].get('pretrained', True)
        )
    
    def _build_audio_backbone(self, cfg):
        """构建音频backbone"""
        backbone_type = cfg['model'].get('audio_backbone', 'cnn')
        return AudioBackbone(
            backbone_type=backbone_type,
            input_channels=cfg['model'].get('audio_channels', 1)
        )
    
    def forward(self, video, audio, return_aux=True):
        """
        前向传播
        
        Args:
            video: [B, T, C, H, W] 或 [B, T, D] (已提取特征)
            audio: [B, T, freq] 或 [B, T, D] (已提取特征)
            return_aux: 是否返回辅助信息（单模态输出、嵌入等）
        
        Returns:
            outputs: dict
                - clip_logits: [B, num_classes] - 视频级分类
                - seg_logits: [B, T, num_classes] - 帧级分类
                - weights: [B, T] - MIL注意力权重
                
                如果return_aux=True，还包括：
                - video_logits: [B, num_classes] - 视频单模态输出
                - audio_logits: [B, num_classes] - 音频单模态输出
                - video_emb: [B, D] - 视频全局嵌入（用于对比学习）
                - audio_emb: [B, D] - 音频全局嵌入
                - video_seq: [B, T, D] - 视频序列特征
                - audio_seq: [B, T, D] - 音频序列特征
        """
        # 1. 特征提取
        video_feat = self.video_backbone(video)  # [B, T, D_v]
        audio_feat = self.audio_backbone(audio)  # [B, T, D_a]
        
        # 2. 时序编码（可选）
        if self.video_temporal is not None:
            video_feat = self.video_temporal(video_feat)
            audio_feat = self.audio_temporal(audio_feat)
        
        # 3. 融合
        if self.fusion_type == 'coattn':
            # CoAttention返回融合特征和辅助信息
            fused, aux_info = self.fusion(video_feat, audio_feat)
            
            # 提取全局嵌入和序列特征
            video_emb = aux_info.get('video_emb')  # [B, D]
            audio_emb = aux_info.get('audio_emb')  # [B, D]
            video_seq = aux_info.get('video_seq')  # [B, T, D]
            audio_seq = aux_info.get('audio_seq')  # [B, T, D]
            
        else:
            # 其他融合方式
            fused = self.fusion(video_feat, audio_feat)
            
            # 使用平均池化生成全局嵌入
            video_emb = video_feat.mean(dim=1)  # [B, D]
            audio_emb = audio_feat.mean(dim=1)
            video_seq = video_feat
            audio_seq = audio_feat
        
        # 4. MIL分类（融合特征）
        mil_outputs = self.mil_head(fused)
        
        # 5. 构建输出
        outputs = {
            'clip_logits': mil_outputs['clip_logits'],
            'seg_logits': mil_outputs['seg_logits'],
            'weights': mil_outputs['weights']
        }
        
        # 6. 添加辅助输出（用于KD、一致性、对比学习）
        if return_aux:
            # 单模态分类
            if self.video_head is not None:
                video_pooled = video_seq.mean(dim=1)  # [B, D]
                audio_pooled = audio_seq.mean(dim=1)
                
                outputs['video_logits'] = self.video_head(video_pooled)
                outputs['audio_logits'] = self.audio_head(audio_pooled)
            
            # 全局嵌入（用于对比学习）
            outputs['video_emb'] = video_emb
            outputs['audio_emb'] = audio_emb
            
            # 序列特征（用于segment-level KD）
            outputs['video_seq'] = video_seq
            outputs['audio_seq'] = audio_seq
        
        return outputs


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("Enhanced Detector 测试")
    print("="*70)
    
    # 配置
    cfg = {
        'model': {
            'video_dim': 512,
            'audio_dim': 256,
            'fusion_dim': 256,
            'num_classes': 2,
            'video_backbone': 'resnet',
            'audio_backbone': 'cnn',
            'pretrained': False
        },
        'fusion': {
            'type': 'coattn',  # 测试CoAttention
            'd_model': 256,
            'num_layers': 2,
            'num_heads': 8
        },
        'use_temporal_encoder': False,
        'use_aux_heads': True
    }
    
    # 创建模型
    model = EnhancedAVTopDetector(cfg)
    
    # 测试数据（已提取特征）
    B, T = 4, 16
    video = torch.randn(B, T, cfg['model']['video_dim'])
    audio = torch.randn(B, T, cfg['model']['audio_dim'])
    
    # 前向传播
    outputs = model(video, audio, return_aux=True)
    
    print(f"\n输出:")
    print(f"  clip_logits: {outputs['clip_logits'].shape}")
    print(f"  seg_logits: {outputs['seg_logits'].shape}")
    print(f"  weights: {outputs['weights'].shape}")
    
    if 'video_logits' in outputs:
        print(f"\n辅助输出（单模态）:")
        print(f"  video_logits: {outputs['video_logits'].shape}")
        print(f"  audio_logits: {outputs['audio_logits'].shape}")
    
    if 'video_emb' in outputs:
        print(f"\n全局嵌入（对比学习）:")
        print(f"  video_emb: {outputs['video_emb'].shape}")
        print(f"  audio_emb: {outputs['audio_emb'].shape}")
    
    # 测试其他融合策略
    print(f"\n{'='*70}")
    print("测试其他融合策略")
    print(f"{'='*70}")
    
    for fusion_type in ['cfa', 'ib']:
        print(f"\n[{fusion_type.upper()}]")
        cfg['fusion']['type'] = fusion_type
        model_test = EnhancedAVTopDetector(cfg)
        
        outputs_test = model_test(video, audio, return_aux=True)
        print(f"  ✅ {fusion_type} 融合正常工作")
        print(f"     clip_logits: {outputs_test['clip_logits'].shape}")
    
    print(f"\n✅ 所有测试通过!")
