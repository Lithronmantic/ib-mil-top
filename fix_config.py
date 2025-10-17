#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件修复工具：确保所有数值类型正确
"""
import yaml
from pathlib import Path
import argparse


def fix_numeric_values(cfg):
    """修复配置中的数值类型"""
    fixed = {}

    # 需要转换为float的键
    float_keys = ['lr', 'weight_decay', 'focal_alpha', 'focal_gamma',
                  'cb_beta', 'ib_beta', 'ib_warmup_ratio', 'pos_sampling_weight']

    # 需要转换为int的键
    int_keys = ['epochs', 'batch_size', 'workers', 'patience',
                'freeze_backbone_epochs', 'T_v', 'T_a', 'mel', 'sr', 'num_classes']

    # 需要转换为bool的键
    bool_keys = ['weighted_sampling', 'spec_augment']

    def convert_value(key, value):
        """转换单个值"""
        if key in float_keys:
            if isinstance(value, str):
                try:
                    # 支持科学计数法
                    return float(value)
                except:
                    return value
            elif isinstance(value, (int, float)):
                return float(value)
        elif key in int_keys:
            if isinstance(value, str):
                try:
                    return int(value)
                except:
                    return value
            elif isinstance(value, (int, float)):
                return int(value)
        elif key in bool_keys:
            if isinstance(value, str):
                return value.lower() in ['true', 'yes', '1']
            elif isinstance(value, bool):
                return value
        return value

    def fix_dict(d):
        """递归修复字典"""
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = fix_dict(v)
            else:
                result[k] = convert_value(k, v)
        return result

    return fix_dict(cfg)


def validate_config(cfg):
    """验证配置的有效性"""
    issues = []

    # 检查必需的顶层键
    required_top = ['experiment', 'data', 'train']
    for key in required_top:
        if key not in cfg:
            issues.append(f"❌ 缺少顶层键: {key}")

    # 检查data配置
    if 'data' in cfg:
        data_required = ['train_csv', 'val_csv', 'T_v', 'T_a', 'mel', 'sr']
        for key in data_required:
            if key not in cfg['data']:
                issues.append(f"⚠️  缺少 data.{key}")

    # 检查train配置
    if 'train' in cfg:
        train_required = ['epochs', 'batch_size', 'lr']
        for key in train_required:
            if key not in cfg['train']:
                issues.append(f"⚠️  缺少 train.{key}")

        # 检查lr类型
        if 'lr' in cfg['train']:
            lr = cfg['train']['lr']
            if isinstance(lr, str):
                issues.append(f"⚠️  train.lr 是字符串 '{lr}'，应该是数字")

    # 检查model配置
    if 'model' not in cfg:
        issues.append(f"⚠️  建议添加 'model' 配置节")
    elif 'audio' in cfg['model']:
        if 'backbone' not in cfg['model']['audio']:
            issues.append(f"⚠️  缺少 model.audio.backbone")

    return issues


def main():
    parser = argparse.ArgumentParser(description="配置文件检查和修复工具")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--fix', action='store_true', help='自动修复问题')
    parser.add_argument('--output', type=str, help='输出文件路径（默认覆盖原文件）')

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return 1

    # 读取配置
    print(f"📋 读取配置: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 验证
    print(f"\n{'=' * 60}")
    print("🔍 配置验证")
    print(f"{'=' * 60}")

    issues = validate_config(cfg)

    if not issues:
        print("✅ 配置文件没有发现问题")
    else:
        print(f"发现 {len(issues)} 个问题:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")

    # 修复
    if args.fix:
        print(f"\n{'=' * 60}")
        print("🔧 自动修复")
        print(f"{'=' * 60}")

        # 备份
        backup_path = config_path.with_suffix('.yaml.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        print(f"💾 备份原文件到: {backup_path}")

        # 修复数值类型
        cfg_fixed = fix_numeric_values(cfg)

        # 添加推荐的默认值
        if 'model' not in cfg_fixed:
            cfg_fixed['model'] = {}

        if 'audio' not in cfg_fixed['model']:
            cfg_fixed['model']['audio'] = {}

        if 'backbone' not in cfg_fixed['model']['audio']:
            cfg_fixed['model']['audio']['backbone'] = 'mel_spectrogram_cnn'
            print("✅ 添加 model.audio.backbone = mel_spectrogram_cnn")

        # 确保train配置完整
        train_defaults = {
            'loss_type': 'focal',
            'focal_gamma': 2.0,
            'weighted_sampling': True,
            'early_stop_metric': 'auprc_minority',
            'patience': 15,
            'freeze_backbone_epochs': 3,
            'weight_decay': 1e-4
        }

        for key, default_value in train_defaults.items():
            if key not in cfg_fixed['train']:
                cfg_fixed['train'][key] = default_value
                print(f"✅ 添加 train.{key} = {default_value}")

        # 保存
        output_path = Path(args.output) if args.output else config_path
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg_fixed, f, default_flow_style=False, allow_unicode=True)

        print(f"\n💾 保存修复后的配置到: {output_path}")

        # 再次验证
        print(f"\n{'=' * 60}")
        print("🔍 验证修复后的配置")
        print(f"{'=' * 60}")

        issues_after = validate_config(cfg_fixed)
        if not issues_after:
            print("✅ 配置文件已修复，可以开始训练了")
            print(f"\n运行命令:")
            print(f"  python scripts/train_production.py --config {output_path}")
            return 0
        else:
            print(f"⚠️  仍有 {len(issues_after)} 个问题需要手动处理")
            for issue in issues_after:
                print(f"  {issue}")
            return 1
    else:
        print(f"\n💡 提示: 运行 '{' '.join(sys.argv)} --fix' 自动修复")
        return 1 if issues else 0


if __name__ == '__main__':
    import sys

    sys.exit(main())