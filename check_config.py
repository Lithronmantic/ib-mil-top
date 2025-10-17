#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置检查和自动修复脚本
用于检测和修复 AVTOP 配置中的常见问题
"""

import yaml
from pathlib import Path
import argparse
import sys


def check_and_fix_config(config_path: str, auto_fix: bool = False):
    """检查配置文件并报告问题"""

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False

    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print(f"📋 检查配置文件: {config_path}")
    print("=" * 60)

    issues = []
    fixes = []

    # 检查1: 音频backbone配置
    audio_backbone = None
    try:
        audio_backbone = cfg.get('model', {}).get('audio', {}).get('backbone')
    except:
        pass

    if audio_backbone is None:
        issues.append("⚠️  未配置 model.audio.backbone")
        fixes.append(("添加", "model.audio.backbone", "mel_spectrogram_cnn"))
    elif audio_backbone in ['vggish', 'vggish_ta']:
        issues.append(f"❌ 音频backbone使用 '{audio_backbone}' (需要波形输入)")
        issues.append("   但Dataset提供的是mel spectrogram，会导致维度错误！")
        fixes.append(("修改", "model.audio.backbone", "mel_spectrogram_cnn"))
    else:
        print(f"✅ 音频backbone: {audio_backbone}")

    # 检查2: mel bins配置
    mel_bins = cfg.get('data', {}).get('mel', 64)
    print(f"✅ Mel bins: {mel_bins}")

    # 检查3: 模型结构配置
    if 'model' not in cfg:
        issues.append("⚠️  缺少 'model' 配置节")
        fixes.append(("添加", "model", {
            'video': {'backbone': 'resnet18_2d'},
            'audio': {'backbone': 'mel_spectrogram_cnn'},
            'fusion': {'d_model': 256},
            'temporal': {'d_model': 256, 'n_layers': 2},
            'num_classes': 2
        }))

    # 检查4: 数据配置
    required_data_keys = ['train_csv', 'val_csv', 'test_csv', 'T_v', 'T_a']
    for key in required_data_keys:
        if key not in cfg.get('data', {}):
            issues.append(f"⚠️  缺少 data.{key}")

    # 打印问题
    print("\n📊 检查结果:")
    print("=" * 60)

    if not issues:
        print("✅ 配置文件完全正确！")
        return True

    print(f"发现 {len(issues)} 个问题:\n")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")

    # 自动修复
    if auto_fix and fixes:
        print("\n🔧 应用自动修复...")
        print("=" * 60)

        for action, key, value in fixes:
            if action == "添加":
                # 解析嵌套key
                keys = key.split('.')
                current = cfg
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                print(f"✅ {action}: {key} = {value}")
            elif action == "修改":
                keys = key.split('.')
                current = cfg
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                old_value = current.get(keys[-1], 'N/A')
                current[keys[-1]] = value
                print(f"✅ {action}: {key} 从 '{old_value}' 改为 '{value}'")

        # 保存修复后的配置
        backup_path = config_path.with_suffix('.yaml.backup')
        print(f"\n💾 备份原配置到: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

        print(f"💾 保存修复后的配置到: {config_path}")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

        print("\n✅ 配置文件已修复！")
        return True

    elif fixes:
        print("\n💡 建议修复:")
        print("=" * 60)
        print("运行以下命令自动修复:")
        print(f"python {sys.argv[0]} --config {config_path} --fix")
        return False

    return False


def main():
    parser = argparse.ArgumentParser(
        description="AVTOP配置文件检查和修复工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查配置
  python check_config.py --config configs/real_binary.yaml

  # 检查并自动修复
  python check_config.py --config configs/real_binary.yaml --fix
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--fix', action='store_true',
                        help='自动修复发现的问题')

    args = parser.parse_args()

    success = check_and_fix_config(args.config, auto_fix=args.fix)

    if success:
        print("\n" + "=" * 60)
        print("🎉 配置检查完成！可以开始训练了:")
        print(f"   python scripts/train_real.py --config {args.config}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()