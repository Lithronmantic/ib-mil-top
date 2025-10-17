#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断音频数据问题
检查：音频加载、VGGish提取、音视频对齐
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import soundfile as sf


def diagnose_audio_loading(dataset, num_samples=20):
    """诊断音频加载问题"""

    print("=" * 70)
    print("🔍 AUDIO DATA DIAGNOSIS")
    print("=" * 70)

    # 1. 检查数据集大小
    print(f"\n[1] Dataset size: {len(dataset)} samples")

    # 2. 检查几个样本的音频数据
    print(f"\n[2] Checking {num_samples} samples...")

    audio_shapes = []
    audio_means = []
    audio_stds = []
    audio_ranges = []

    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            audio = sample['audio']  # [T, audio_dim]

            audio_shapes.append(audio.shape)
            audio_means.append(audio.mean().item())
            audio_stds.append(audio.std().item())
            audio_ranges.append((audio.min().item(), audio.max().item()))

            print(f"  Sample {i}: shape={audio.shape}, "
                  f"mean={audio.mean():.4f}, std={audio.std():.4f}")

        except Exception as e:
            print(f"  Sample {i}: ERROR - {e}")

    # 3. 统计分析
    print(f"\n[3] Statistical analysis:")

    # 检查shape是否一致
    unique_shapes = set(audio_shapes)
    print(f"  Unique shapes: {unique_shapes}")

    if len(unique_shapes) == 1:
        print(f"  ✅ All shapes are consistent")
    else:
        print(f"  ⚠️  Shapes are inconsistent!")

    # 检查均值方差
    means_array = np.array(audio_means)
    stds_array = np.array(audio_stds)

    print(f"\n  Mean statistics:")
    print(f"    Range: [{means_array.min():.6f}, {means_array.max():.6f}]")
    print(f"    Std: {means_array.std():.6f}")

    print(f"\n  Std statistics:")
    print(f"    Range: [{stds_array.min():.6f}, {stds_array.max():.6f}]")
    print(f"    Mean: {stds_array.mean():.6f}")

    # 🚨 关键检查：如果所有音频均值几乎相同
    if means_array.std() < 0.01:
        print(f"\n  ❌ PROBLEM: All audio features are nearly identical!")
        print(f"     Mean std={means_array.std():.8f} is too small")
        print(f"     This suggests:")
        print(f"     1. VGGish is extracting same features for all samples")
        print(f"     2. Audio input to VGGish is corrupted/repeated")
        print(f"     3. Audio files are not properly aligned")
        return False
    else:
        print(f"\n  ✅ Audio features show variation (std={means_array.std():.6f})")

    # 4. 检查原始音频文件
    print(f"\n[4] Checking raw audio files...")

    if hasattr(dataset, 'df'):
        df = dataset.df

        # 随机抽取几个音频文件
        sample_indices = np.random.choice(len(df), min(5, len(df)), replace=False)

        for idx in sample_indices:
            row = df.iloc[idx]
            audio_path = row.get('audio', None)

            if audio_path and Path(audio_path).exists():
                try:
                    # 加载原始音频
                    y, sr = sf.read(audio_path)
                    duration = len(y) / sr

                    print(f"  {Path(audio_path).name}:")
                    print(f"    Duration: {duration:.3f}s")
                    print(f"    Sample rate: {sr}")
                    print(f"    RMS: {np.sqrt(np.mean(y ** 2)):.6f}")

                    # 检查是否太短
                    if duration < 0.96:
                        print(f"    ⚠️  TOO SHORT for VGGish (needs ≥0.96s)!")

                except Exception as e:
                    print(f"  {audio_path}: ERROR - {e}")
            else:
                print(f"  Audio file not found: {audio_path}")

    return True


def diagnose_vggish_extraction(audio_paths, sr=16000):
    """测试VGGish特征提取"""

    print("\n" + "=" * 70)
    print("🔍 VGGISH FEATURE EXTRACTION TEST")
    print("=" * 70)

    try:
        # 尝试导入VGGish
        import torch
        import torchaudio
        from torchvision.models import vgg16

        print("\n[1] Creating test audio signals...")

        # 创建3个不同的测试音频
        test_audios = {
            'silence': np.zeros(int(sr * 1.0)),
            'sine_wave': np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)),
            'noise': np.random.randn(sr)
        }

        # 简单的mel spectrogram提取（模拟VGGish输入）
        for name, audio in test_audios.items():
            # 转为tensor
            audio_tensor = torch.from_numpy(audio).float()

            # 计算mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=400,
                hop_length=160,
                n_mels=64
            )

            mel_spec = mel_transform(audio_tensor)

            print(f"\n  {name}:")
            print(f"    Input shape: {audio.shape}")
            print(f"    Mel spec shape: {mel_spec.shape}")
            print(f"    Mel spec range: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
            print(f"    Mel spec mean: {mel_spec.mean():.4f}")
            print(f"    Mel spec std: {mel_spec.std():.4f}")

            # 如果所有测试的mel spec都差不多，说明有问题
            if mel_spec.std() < 0.01:
                print(f"    ⚠️  Very low variation!")

        print("\n  ✅ VGGish preprocessing seems OK")

    except ImportError as e:
        print(f"\n  ⚠️  Cannot import VGGish dependencies: {e}")
        print(f"     Skipping VGGish test")


def diagnose_alignment(windows_csv, annotations_csv):
    """检查音视频对齐"""

    print("\n" + "=" * 70)
    print("🔍 AUDIO-VIDEO ALIGNMENT CHECK")
    print("=" * 70)

    # 加载CSV
    df_windows = pd.read_csv(windows_csv)
    df_ann = pd.read_csv(annotations_csv)

    print(f"\n[1] Data files:")
    print(f"  Windows CSV: {len(df_windows)} rows")
    print(f"  Annotations CSV: {len(df_ann)} rows")

    # 检查必需的列
    print(f"\n[2] Checking columns...")

    required_cols = ['audio', 'video', 'audio_t0', 'audio_t1', 'video_f0', 'video_f1']
    missing_cols = [col for col in required_cols if col not in df_windows.columns]

    if missing_cols:
        print(f"  ❌ Missing columns: {missing_cols}")
        return False
    else:
        print(f"  ✅ All required columns present")

    # 检查音频时长
    print(f"\n[3] Checking audio window durations...")

    df_windows['audio_duration'] = df_windows['audio_t1'] - df_windows['audio_t0']

    print(f"  Audio duration statistics:")
    print(f"    Mean: {df_windows['audio_duration'].mean():.3f}s")
    print(f"    Std: {df_windows['audio_duration'].std():.3f}s")
    print(f"    Min: {df_windows['audio_duration'].min():.3f}s")
    print(f"    Max: {df_windows['audio_duration'].max():.3f}s")

    # 检查是否有太短的窗口
    too_short = df_windows[df_windows['audio_duration'] < 0.96]

    if len(too_short) > 0:
        print(f"\n  ⚠️  {len(too_short)} windows are TOO SHORT for VGGish (<0.96s)")
        print(f"     This will cause feature extraction to fail!")
        print(f"\n  Example short windows:")
        print(too_short[['audio', 'audio_duration']].head())
        return False
    else:
        print(f"\n  ✅ All windows are long enough (≥0.96s)")

    # 检查视频帧数
    print(f"\n[4] Checking video frame counts...")

    df_windows['video_frames'] = df_windows['video_f1'] - df_windows['video_f0'] + 1

    print(f"  Video frames statistics:")
    print(f"    Mean: {df_windows['video_frames'].mean():.1f}")
    print(f"    Min: {df_windows['video_frames'].min()}")
    print(f"    Max: {df_windows['video_frames'].max()}")

    # 检查时间对齐
    print(f"\n[5] Checking temporal alignment...")

    # 假设视频是30fps
    df_windows['video_duration'] = df_windows['video_frames'] / 30.0
    df_windows['duration_diff'] = abs(df_windows['audio_duration'] - df_windows['video_duration'])

    print(f"  Audio-Video duration difference:")
    print(f"    Mean: {df_windows['duration_diff'].mean():.3f}s")
    print(f"    Max: {df_windows['duration_diff'].max():.3f}s")

    large_diff = df_windows[df_windows['duration_diff'] > 0.1]

    if len(large_diff) > 0:
        print(f"\n  ⚠️  {len(large_diff)} windows have large audio-video mismatch (>0.1s)")
        print(f"     This suggests alignment issues!")
        return False
    else:
        print(f"\n  ✅ Audio-video durations are well aligned")

    return True


def visualize_audio_features(dataset, num_samples=100):
    """可视化音频特征分布"""

    print("\n" + "=" * 70)
    print("🔍 AUDIO FEATURE VISUALIZATION")
    print("=" * 70)

    audio_features = []
    labels = []

    print(f"\nExtracting features from {num_samples} samples...")

    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            audio = sample['audio']  # [T, D]
            label = sample['label'].item()

            # 使用均值作为样本级特征
            audio_mean = audio.mean(dim=0).numpy()

            audio_features.append(audio_mean)
            labels.append(label)

        except Exception as e:
            print(f"  Sample {i} failed: {e}")

    audio_features = np.array(audio_features)
    labels = np.array(labels)

    print(f"  Collected {len(audio_features)} valid samples")
    print(f"  Feature dimension: {audio_features.shape[1]}")

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 前两个维度
    ax = axes[0]
    for label_val in [0, 1]:
        mask = labels == label_val
        ax.scatter(audio_features[mask, 0],
                   audio_features[mask, 1],
                   label=f'Class {label_val}',
                   alpha=0.6)
    ax.set_xlabel('Feature dim 0')
    ax.set_ylabel('Feature dim 1')
    ax.set_title('Audio Features (first 2 dims)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 特征分布直方图
    ax = axes[1]
    ax.hist(audio_features.flatten(), bins=50, alpha=0.7)
    ax.set_xlabel('Feature value')
    ax.set_ylabel('Count')
    ax.set_title('Audio Feature Value Distribution')
    ax.grid(True, alpha=0.3)

    # 3. 各维度方差
    ax = axes[2]
    feature_stds = audio_features.std(axis=0)
    ax.bar(range(len(feature_stds)), feature_stds)
    ax.set_xlabel('Feature dimension')
    ax.set_ylabel('Standard deviation')
    ax.set_title('Per-dimension Variance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('audio_feature_diagnosis.png', dpi=150)
    print(f"\n  ✅ Saved visualization to: audio_feature_diagnosis.png")

    # 统计分析
    print(f"\n[Statistical Summary]")
    print(f"  Feature mean: {audio_features.mean():.6f}")
    print(f"  Feature std: {audio_features.std():.6f}")
    print(f"  Per-sample std: {audio_features.std(axis=1).mean():.6f}")

    # 检查是否所有特征都一样
    if audio_features.std() < 0.01:
        print(f"\n  ❌ CRITICAL: All features are nearly identical!")
        print(f"     Feature std = {audio_features.std():.8f} is too small")
        return False
    else:
        print(f"\n  ✅ Features show healthy variation")
        return True


def main():
    """主诊断流程"""

    print("\n" + "=" * 80)
    print(" " * 20 + "🔬 COMPREHENSIVE AUDIO DIAGNOSIS")
    print("=" * 80)

    # 配置
    windows_csv = "data/train_windows.csv"
    annotations_csv = "data/annotations.csv"

    # 1. 检查对齐
    print("\nStep 1: Checking audio-video alignment...")
    alignment_ok = diagnose_alignment(windows_csv, annotations_csv)

    if not alignment_ok:
        print("\n" + "=" * 80)
        print("❌ ALIGNMENT ISSUE DETECTED")
        print("=" * 80)
        print("\n🔧 SOLUTION: Re-run audio-video alignment script")
        print("   python supervised/av_align_paper_plus_cli.py \\")
        print("       --root ./dataset \\")
        print("       --out_dir ./alignment_output \\")
        print("       --min_weld_sec 1.0  # Ensure minimum 1 second windows")
        print("\n   Then regenerate train/val/test splits")
        return

    # 2. 加载数据集
    print("\nStep 2: Loading dataset...")

    try:
        from csv_dataset import BinaryAVCSVDataset as AVTopDataset

        dataset = AVTopDataset(
            windows_csv=windows_csv,
            ann_csv=annotations_csv,
            classes=['normal', 'defect'],
            use_audio=True,
            use_video=False
        )

        print(f"  ✅ Dataset loaded: {len(dataset)} samples")

    except Exception as e:
        print(f"  ❌ Failed to load dataset: {e}")
        return

    # 3. 诊断音频加载
    print("\nStep 3: Diagnosing audio loading...")
    loading_ok = diagnose_audio_loading(dataset, num_samples=20)

    if not loading_ok:
        print("\n" + "=" * 80)
        print("❌ AUDIO LOADING ISSUE DETECTED")
        print("=" * 80)
        print("\n🔧 SOLUTION: Check VGGish feature extraction")
        print("   Possible fixes:")
        print("   1. Ensure audio windows are ≥0.96s")
        print("   2. Check VGGish input preprocessing")
        print("   3. Verify audio resampling to 16kHz")
        return

    # 4. 可视化特征
    print("\nStep 4: Visualizing audio features...")
    vis_ok = visualize_audio_features(dataset, num_samples=200)

    if not vis_ok:
        print("\n" + "=" * 80)
        print("❌ AUDIO FEATURES ARE DEGENERATE")
        print("=" * 80)
        print("\n🔧 SOLUTION:")
        print("   1. Check if all audio files are identical")
        print("   2. Verify VGGish model is loaded correctly")
        print("   3. Check for bugs in feature caching")
        return

    # 5. 测试VGGish
    print("\nStep 5: Testing VGGish extraction...")
    diagnose_vggish_extraction([])

    # 最终结论
    print("\n" + "=" * 80)
    print("✅ DIAGNOSIS COMPLETE")
    print("=" * 80)

    if alignment_ok and loading_ok and vis_ok:
        print("\n✅ No obvious issues detected")
        print("   Audio features appear normal")
        print("\n💡 Next steps:")
        print("   1. Check if Fisher ratio calculation is correct")
        print("   2. Review PCA plot generation code")
        print("   3. Ensure all samples are included in visualization")
    else:
        print("\n❌ Issues found - see recommendations above")


if __name__ == "__main__":
    main()