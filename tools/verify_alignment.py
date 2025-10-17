#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单验证脚本：检查对齐结果是否正确
"""
import pandas as pd
import numpy as np
from pathlib import Path


def verify_alignment(data_dir="./test_alignment"):
    """验证对齐结果"""

    data_dir = Path(data_dir)

    print("=" * 70)
    print("🔍 对齐结果验证")
    print("=" * 70)

    # 1. 检查文件存在
    print("\n[1] 检查文件...")

    files = {
        'results': data_dir / 'results.csv',
        'windows': data_dir / 'windows.csv',
        'aligned': data_dir / 'windows_aligned.csv'
    }

    for name, path in files.items():
        if path.exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: 文件不存在 - {path}")
            return False

    # 2. 检查results.csv
    print("\n[2] 检查results.csv...")

    try:
        df_results = pd.read_csv(files['results'])
        print(f"  样本数: {len(df_results)}")

        if 'pair_pass_rate' in df_results.columns:
            avg_pass_rate = df_results['pair_pass_rate'].mean() * 100
            print(f"  平均通过率: {avg_pass_rate:.1f}%")

            if avg_pass_rate < 50:
                print(f"  ⚠️  通过率过低！")

        if 'keep_sample' in df_results.columns:
            keep_rate = df_results['keep_sample'].mean() * 100
            print(f"  样本保留率: {keep_rate:.1f}%")

        if 'audio_duration' in df_results.columns:
            durations = df_results['audio_duration']
            print(f"\n  音频主段时长统计:")
            print(f"    均值: {durations.mean():.3f}s")
            print(f"    最小: {durations.min():.3f}s")
            print(f"    最大: {durations.max():.3f}s")

        if 'n_audio_windows' in df_results.columns:
            print(f"\n  音频窗口数统计:")
            print(f"    均值: {df_results['n_audio_windows'].mean():.1f}")
            print(f"    最小: {df_results['n_audio_windows'].min()}")
            print(f"    最大: {df_results['n_audio_windows'].max()}")

    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        return False

    # 3. 检查windows.csv - 关键！
    print("\n[3] 检查windows.csv（音频窗口长度）...")

    try:
        df_windows = pd.read_csv(files['windows'])
        print(f"  总行数: {len(df_windows)}")

        # 分离音频和视频窗口
        if 'modality' in df_windows.columns:
            audio_wins = df_windows[df_windows['modality'] == 'audio'].copy()
            video_wins = df_windows[df_windows['modality'] == 'video'].copy()

            print(f"  音频窗口数: {len(audio_wins)}")
            print(f"  视频窗口数: {len(video_wins)}")

            # 关键检查：音频窗口长度
            if 'audio_start_s' in audio_wins.columns and 'audio_end_s' in audio_wins.columns:
                audio_wins['duration'] = audio_wins['audio_end_s'] - audio_wins['audio_start_s']

                print(f"\n  ✅ 音频窗口时长统计:")
                print(f"    均值: {audio_wins['duration'].mean():.3f}s")
                print(f"    标准差: {audio_wins['duration'].std():.6f}s")
                print(f"    最小: {audio_wins['duration'].min():.3f}s")
                print(f"    最大: {audio_wins['duration'].max():.3f}s")

                # 检查是否还是0.96s
                if audio_wins['duration'].mean() < 1.0:
                    print(f"\n  ❌ 警告: 音频窗口仍然太短（<1s）！")
                    print(f"     VGGish可能仍然只输出1帧")
                    return False
                elif audio_wins['duration'].mean() > 1.4:
                    print(f"\n  ✅ 成功: 音频窗口长度正常（>1.4s）")
                    print(f"     VGGish应该能输出2-3帧")
                else:
                    print(f"\n  ⚠️  音频窗口长度介于1.0-1.4s之间")

                # 显示前几个窗口
                print(f"\n  前5个音频窗口示例:")
                print(audio_wins[['sample', 'win_idx', 'audio_start_s', 'audio_end_s', 'duration']].head())

            # 视频窗口检查
            if 'video_start_frame' in video_wins.columns and 'video_end_frame' in video_wins.columns:
                video_wins['n_frames'] = video_wins['video_end_frame'] - video_wins['video_start_frame'] + 1

                print(f"\n  视频窗口帧数统计:")
                print(f"    均值: {video_wins['n_frames'].mean():.1f}")
                print(f"    最小: {video_wins['n_frames'].min()}")
                print(f"    最大: {video_wins['n_frames'].max()}")

                # 假设30fps
                fps = 30
                video_wins['duration'] = video_wins['n_frames'] / fps
                print(f"  视频窗口时长（@30fps）:")
                print(f"    均值: {video_wins['duration'].mean():.3f}s")
        else:
            print(f"  ⚠️  未找到'modality'列，尝试其他方式...")

            # 备选方案：根据列名判断
            if 'audio_start_s' in df_windows.columns:
                mask = df_windows['audio_start_s'].notna()
                audio_wins = df_windows[mask].copy()

                if len(audio_wins) > 0:
                    audio_wins['duration'] = audio_wins['audio_end_s'] - audio_wins['audio_start_s']
                    print(f"\n  音频窗口数: {len(audio_wins)}")
                    print(f"  平均时长: {audio_wins['duration'].mean():.3f}s")

    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 检查aligned.csv
    print("\n[4] 检查windows_aligned.csv（配对情况）...")

    try:
        df_aligned = pd.read_csv(files['aligned'])
        print(f"  配对记录数: {len(df_aligned)}")

        if 'score' in df_aligned.columns:
            scores = df_aligned['score']
            print(f"\n  配对分数统计:")
            print(f"    均值: {scores.mean():.3f}")
            print(f"    最小: {scores.min():.3f}")
            print(f"    最大: {scores.max():.3f}")

        if 'keep' in df_aligned.columns:
            keep_rate = df_aligned['keep'].mean() * 100
            print(f"\n  配对保留率: {keep_rate:.1f}%")

            if keep_rate < 50:
                print(f"  ⚠️  保留率偏低")
            else:
                print(f"  ✅ 保留率良好")

        # 显示前几条
        print(f"\n  前5条配对记录:")
        cols = ['a_idx', 'a_start_s', 'a_end_s', 'v_idx', 'score', 'keep']
        display_cols = [c for c in cols if c in df_aligned.columns]
        print(df_aligned[display_cols].head())

    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        return False

    # 5. 最终判断
    print("\n" + "=" * 70)
    print("📊 验证总结")
    print("=" * 70)

    checks = []

    # 检查1：通过率
    if avg_pass_rate >= 80:
        print("✅ 配对通过率良好 (≥80%)")
        checks.append(True)
    else:
        print(f"⚠️  配对通过率偏低 ({avg_pass_rate:.1f}%)")
        checks.append(False)

    # 检查2：窗口长度
    if audio_wins['duration'].mean() >= 1.4:
        print("✅ 音频窗口长度正常 (≥1.4s)")
        checks.append(True)
    else:
        print(f"❌ 音频窗口长度不足 ({audio_wins['duration'].mean():.3f}s)")
        checks.append(False)

    # 检查3：VGGish帧数估计
    # 1.5s音频 -> 约2-3帧VGGish输出
    expected_vggish_frames = int(audio_wins['duration'].mean() / 0.96) + 1
    print(f"\n💡 预计VGGish输出: 约{expected_vggish_frames}帧/窗口")

    if expected_vggish_frames >= 2:
        print("✅ 应该能获得多帧VGGish特征")
        checks.append(True)
    else:
        print("❌ 可能仍然只有1帧VGGish特征")
        checks.append(False)

    if all(checks):
        print("\n" + "=" * 70)
        print("🎉 验证通过！数据对齐成功！")
        print("=" * 70)
        print("\n下一步:")
        print("  1. 处理全部数据（如果还没有）")
        print("  2. 重新训练模型")
        print("  3. 应该能看到性能显著提升")
        return True
    else:
        print("\n" + "=" * 70)
        print("⚠️  存在问题，请检查")
        print("=" * 70)
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./test_alignment',
                        help='对齐结果目录')
    args = parser.parse_args()

    verify_alignment(args.data_dir)