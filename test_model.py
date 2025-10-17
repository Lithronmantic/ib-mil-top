#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
维度诊断脚本
检查 audio/video backbone 输出维度是否与 fusion 模块匹配
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import yaml


def diagnose_dimensions(config_path):
    """诊断模型维度配置"""

    print("=" * 70)
    print("🔍 模型维度诊断")
    print("=" * 70)

    # 1. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print("\n📋 配置文件信息:")
    print(f"  video_backbone: {cfg['model'].get('video_backbone', 'unknown')}")
    print(f"  audio_backbone: {cfg['model'].get('audio_backbone', 'unknown')}")
    print(f"  fusion_type: {cfg['model'].get('fusion_type', 'unknown')}")

    # 2. 创建模型
    from src.avtop.models.enhanced_detector import EnhancedAVDetector

    try:
        model = EnhancedAVDetector(cfg)
        print("\n✅ 模型创建成功")
    except Exception as e:
        print(f"\n❌ 模型创建失败: {e}")
        return

    # 3. 检查 backbone 输出维度
    print("\n🔧 Backbone 输出维度测试:")

    # 测试视频
    try:
        video_input = torch.randn(2, 16, 3, 224, 224)  # [B, T, C, H, W]
        video_feat = model.video_backbone(video_input)
        print(f"  ✅ Video Backbone:")
        print(f"     输入: {list(video_input.shape)}")
        print(f"     输出: {list(video_feat.shape)}")
        video_dim = video_feat.shape[-1]
    except Exception as e:
        print(f"  ❌ Video Backbone 错误: {e}")
        video_dim = None

    # 测试音频
    try:
        audio_input = torch.randn(2, 3200)  # [B, T]
        audio_feat = model.audio_backbone(audio_input)
        print(f"  ✅ Audio Backbone:")
        print(f"     输入: {list(audio_input.shape)}")
        print(f"     输出: {list(audio_feat.shape)}")
        audio_dim = audio_feat.shape[-1]
    except Exception as e:
        print(f"  ❌ Audio Backbone 错误: {e}")
        audio_dim = None

    # 4. 检查 fusion 期望维度
    print("\n🔧 Fusion 模块期望维度:")

    fusion = model.fusion

    # 检查 audio_proj
    if hasattr(fusion, 'audio_proj'):
        audio_proj = fusion.audio_proj
        if hasattr(audio_proj, '0'):  # Sequential
            first_layer = audio_proj[0]
            if hasattr(first_layer, 'in_features'):
                fusion_audio_dim = first_layer.in_features
                print(f"  Audio Proj 输入维度: {fusion_audio_dim}")
            else:
                fusion_audio_dim = None
        else:
            fusion_audio_dim = None
    else:
        fusion_audio_dim = None

    # 检查 video_proj
    if hasattr(fusion, 'video_proj'):
        video_proj = fusion.video_proj
        if hasattr(video_proj, '0'):
            first_layer = video_proj[0]
            if hasattr(first_layer, 'in_features'):
                fusion_video_dim = first_layer.in_features
                print(f"  Video Proj 输入维度: {fusion_video_dim}")
            else:
                fusion_video_dim = None
        else:
            fusion_video_dim = None
    else:
        fusion_video_dim = None

    # 5. 对比结果
    print("\n" + "=" * 70)
    print("📊 维度对比结果:")
    print("=" * 70)

    problems = []

    if audio_dim and fusion_audio_dim:
        if audio_dim != fusion_audio_dim:
            problems.append(f"❌ Audio 维度不匹配: Backbone输出 {audio_dim}, Fusion期望 {fusion_audio_dim}")
        else:
            print(f"✅ Audio 维度匹配: {audio_dim}")

    if video_dim and fusion_video_dim:
        if video_dim != fusion_video_dim:
            problems.append(f"❌ Video 维度不匹配: Backbone输出 {video_dim}, Fusion期望 {fusion_video_dim}")
        else:
            print(f"✅ Video 维度匹配: {video_dim}")

    if problems:
        print("\n⚠️ 发现问题:")
        for p in problems:
            print(f"  {p}")

        print("\n💡 修复建议:")
        if audio_dim and fusion_audio_dim and audio_dim != fusion_audio_dim:
            print(f"  方案1: 修改配置文件中的 model.audio_dim 为 {audio_dim}")
            print(f"  方案2: 修改 audio_backbone 的输出维度为 {fusion_audio_dim}")

        if video_dim and fusion_video_dim and video_dim != fusion_video_dim:
            print(f"  方案1: 修改配置文件中的 model.video_dim 为 {video_dim}")
            print(f"  方案2: 修改 video_backbone 的输出维度为 {fusion_video_dim}")
    else:
        print("\n🎉 所有维度匹配，模型配置正确！")

    return {
        'audio_backbone_dim': audio_dim,
        'video_backbone_dim': video_dim,
        'fusion_audio_dim': fusion_audio_dim,
        'fusion_video_dim': fusion_video_dim,
        'has_problems': len(problems) > 0
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/real_binary_sota.yaml"

    print(f"使用配置文件: {config_path}\n")

    result = diagnose_dimensions(config_path)

    print("\n" + "=" * 70)
    if result['has_problems']:
        print("❌ 诊断完成：发现维度不匹配问题")
        sys.exit(1)
    else:
        print("✅ 诊断完成：所有维度正确")
        sys.exit(0)