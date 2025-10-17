#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试音频backbone修复是否有效
"""

import torch
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avtop.models.backbones import AudioBackbone


def test_mel_spectrogram_cnn():
    """测试MelSpectrogramCNN backbone"""
    print("=" * 60)
    print("测试 1: MelSpectrogramCNN Backbone")
    print("=" * 60)

    # 创建backbone
    backbone = AudioBackbone(
        backbone_type="mel_spectrogram_cnn",
        mel_bins=64
    )

    # 模拟dataset输出 (batch中的mel spectrogram)
    batch_size = 8
    T_a = 32
    mel_bins = 64
    mel_input = torch.randn(batch_size, T_a, mel_bins)

    print(f"输入形状: {tuple(mel_input.shape)} (Batch, Time, MelBins)")

    # 前向传播
    try:
        output = backbone(mel_input)
        print(f"✅ 输出形状: {tuple(output.shape)} (Batch, Time, Features)")

        # 验证形状
        expected_shape = (batch_size, T_a, 128)
        if output.shape == expected_shape:
            print(f"✅ 形状正确: {expected_shape}")
            return True
        else:
            print(f"❌ 形状错误! 期望 {expected_shape}, 得到 {output.shape}")
            return False
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vggish_with_waveform():
    """测试VGGish backbone（使用波形输入）"""
    print("\n" + "=" * 60)
    print("测试 2: VGGish Backbone (波形输入)")
    print("=" * 60)

    # 创建backbone
    try:
        backbone = AudioBackbone(
            backbone_type="vggish",
            sample_rate=16000
        )
    except Exception as e:
        print(f"⚠️  VGGish不可用: {e}")
        return None

    # 模拟原始波形输入
    batch_size = 8
    duration = 3.0  # 3秒
    sample_rate = 16000
    wave_input = torch.randn(batch_size, int(duration * sample_rate))

    print(f"输入形状: {tuple(wave_input.shape)} (Batch, Samples)")

    # 前向传播
    try:
        output = backbone(wave_input)
        print(f"✅ 输出形状: {tuple(output.shape)} (Batch, Time, Features)")

        # VGGish应该输出多个时间步
        if output.shape[0] == batch_size and output.shape[1] > 1 and output.shape[2] == 128:
            print(f"✅ VGGish输出正确（batch维度保持，时间步>1）")
            return True
        else:
            print(f"⚠️  VGGish输出异常: batch={output.shape[0]}, time={output.shape[1]}")
            return False
    except Exception as e:
        print(f"❌ VGGish前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dimension_mismatch_scenario():
    """测试之前导致错误的场景"""
    print("\n" + "=" * 60)
    print("测试 3: 模拟训练场景（视听对齐）")
    print("=" * 60)

    from avtop.models.backbones import VideoBackbone

    batch_size = 8
    T_v = 16
    T_a = 32
    mel_bins = 64

    # 视频输入 (B, T, C, H, W)
    video = torch.randn(batch_size, T_v, 3, 112, 112)

    # 音频输入 (B, T_a, mel_bins) - Dataset提供的mel spectrogram
    audio = torch.randn(batch_size, T_a, mel_bins)

    print(f"视频输入: {tuple(video.shape)}")
    print(f"音频输入: {tuple(audio.shape)}")

    # 创建backbones
    try:
        video_backbone = VideoBackbone(backbone_type="resnet18_2d", pretrained=False)
        audio_backbone = AudioBackbone(backbone_type="mel_spectrogram_cnn", mel_bins=mel_bins)
    except Exception as e:
        print(f"❌ Backbone创建失败: {e}")
        return False

    # 前向传播
    try:
        v_feat = video_backbone(video)
        a_feat = audio_backbone(audio)

        print(f"视频特征: {tuple(v_feat.shape)}")
        print(f"音频特征: {tuple(a_feat.shape)}")

        # 检查batch维度
        if v_feat.shape[0] == batch_size and a_feat.shape[0] == batch_size:
            print("✅ Batch维度正确匹配")

            # 时间维度对齐（可以用插值）
            if v_feat.shape[1] == T_v and a_feat.shape[1] == T_a:
                print(f"✅ 时间维度正确: video={T_v}, audio={T_a}")
                print("   (可用插值对齐到相同时间步)")
                return True
            else:
                print(f"⚠️  时间维度异常")
                return False
        else:
            print(f"❌ Batch维度不匹配! video={v_feat.shape[0]}, audio={a_feat.shape[0]}")
            return False

    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "🔍 AVTOP Audio Backbone 测试")
    print("=" * 60 + "\n")

    results = []

    # 测试1: MelSpectrogramCNN
    results.append(("MelSpectrogramCNN", test_mel_spectrogram_cnn()))

    # 测试2: VGGish (可选)
    vggish_result = test_vggish_with_waveform()
    if vggish_result is not None:
        results.append(("VGGish", vggish_result))

    # 测试3: 完整场景
    results.append(("视听对齐场景", test_dimension_mismatch_scenario()))

    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")

    all_passed = all(r for _, r in results)

    if all_passed:
        print("\n🎉 所有测试通过！可以开始训练了。")
        print("\n下一步:")
        print("1. 检查配置文件:")
        print("   python check_config.py --config configs/real_binary.yaml --fix")
        print("\n2. 开始训练:")
        print("   python scripts/train_real.py --config configs/real_binary.yaml")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
        return 1


if __name__ == '__main__':
    import sys

    sys.exit(main())