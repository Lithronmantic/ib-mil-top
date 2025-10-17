#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AVTOP 音频处理错误一键修复脚本
自动应用所有必要的代码修改
"""

import re
from pathlib import Path
import sys


def patch_enhanced_detector():
    """修改 enhanced_detector.py，添加 mel_bins 参数"""
    file_path = Path("src/avtop/models/enhanced_detector.py")

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False

    print(f"📝 修改文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 备份
    backup_path = file_path.with_suffix('.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 备份到: {backup_path}")

    # 检查是否已经修复
    if 'mel_bins=mel_bins' in content:
        print("✅ 文件已经包含修复，跳过")
        return True

    # 修改1: 添加 mel_bins 变量
    pattern1 = r'(a_sr\s*=\s*_safe_get\(cfg,.*?\))'
    replacement1 = r'\1\n        mel_bins = _safe_get(cfg, "data.mel", 64)  # ⭐ Auto-added by fix script'

    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        print("✅ 添加 mel_bins 变量")
    else:
        print("⚠️  找不到插入点，手动添加...")
        # 尝试在 a_sr 行后面添加
        content = content.replace(
            'a_sr   = _safe_get(cfg, "data.audio.sample_rate"',
            'a_sr   = _safe_get(cfg, "data.audio.sample_rate"'
        )
        # 然后在下一行添加
        lines = content.split('\n')
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            if 'a_sr' in line and '_safe_get' in line and 'sample_rate' in line:
                # 找到下一行并在其后插入
                if i + 1 < len(lines):
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + 'mel_bins = _safe_get(cfg, "data.mel", 64)  # ⭐ Auto-added')
        content = '\n'.join(new_lines)

    # 修改2: 更新 AudioBackbone 初始化
    pattern2 = r'self\.audio_backbone\s*=\s*AudioBackbone\(\s*backbone_type=a_type,\s*pretrained=True,\s*sample_rate=a_sr,\s*freeze=False\s*\)'
    replacement2 = '''self.audio_backbone = AudioBackbone(
            backbone_type=a_type, 
            pretrained=True, 
            sample_rate=a_sr, 
            mel_bins=mel_bins,  # ⭐ Auto-added by fix script
            freeze=False
        )'''

    if re.search(pattern2, content, re.DOTALL):
        content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)
        print("✅ 更新 AudioBackbone 初始化")
    else:
        # 尝试更宽松的匹配
        pattern2_loose = r'self\.audio_backbone\s*=\s*AudioBackbone\([^)]*\)'
        if re.search(pattern2_loose, content):
            # 找到该行，在参数中添加 mel_bins
            def replacer(match):
                s = match.group(0)
                if 'mel_bins=' not in s:
                    # 在 freeze=False 之前插入
                    s = s.replace('freeze=False', 'mel_bins=mel_bins, freeze=False')
                return s

            content = re.sub(pattern2_loose, replacer, content)
            print("✅ 更新 AudioBackbone 初始化（宽松匹配）")
        else:
            print("⚠️  找不到 AudioBackbone 初始化，请手动添加 mel_bins 参数")

    # 保存修改
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ 文件已修改并保存")
    return True


def verify_backbones():
    """验证 backbones.py 是否包含新的 MelSpectrogramCNN"""
    file_path = Path("src/avtop/models/backbones.py")

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False

    print(f"📝 检查文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查关键方法
    checks = [
        ('_detect_input_type', '输入类型检测'),
        ('_waveform_to_mel', '波形转mel'),
        ('class MelSpectrogramCNN', 'MelSpectrogramCNN类')
    ]

    all_good = True
    for keyword, desc in checks:
        if keyword in content:
            print(f"✅ 包含: {desc}")
        else:
            print(f"❌ 缺失: {desc}")
            all_good = False

    if not all_good:
        print("\n⚠️  backbones.py 需要更新！")
        print("请用 artifacts 中的 fixed_backbones.py 替换当前文件")
        return False

    return True


def check_config():
    """检查配置文件"""
    config_path = Path("configs/real_binary.yaml")

    if not config_path.exists():
        print(f"⚠️  配置文件不存在: {config_path}")
        return False

    print(f"📝 检查配置: {config_path}")

    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 检查必要字段
    checks = [
        (['data', 'sr'], '采样率'),
        (['data', 'mel'], 'Mel bins'),
        (['model', 'audio', 'backbone'], '音频backbone')
    ]

    all_good = True
    for keys, desc in checks:
        current = cfg
        for key in keys:
            if key in current:
                current = current[key]
            else:
                print(f"⚠️  缺失配置: {'.'.join(keys)} ({desc})")
                all_good = False
                break
        else:
            print(f"✅ 配置正确: {'.'.join(keys)} = {current}")

    # 检查 backbone 类型
    backbone = cfg.get('model', {}).get('audio', {}).get('backbone', 'N/A')
    if backbone == 'mel_spectrogram_cnn':
        print(f"✅ 使用推荐的智能检测 backbone")
    else:
        print(f"⚠️  当前使用: {backbone}")
        print(f"   建议改为: mel_spectrogram_cnn")

    return all_good


def main():
    print("=" * 70)
    print("🔧 AVTOP 音频处理错误一键修复")
    print("=" * 70)

    steps = [
        ("检查 backbones.py", verify_backbones),
        ("修复 enhanced_detector.py", patch_enhanced_detector),
        ("检查配置文件", check_config)
    ]

    results = []
    for step_name, step_func in steps:
        print(f"\n{'=' * 70}")
        print(f"步骤: {step_name}")
        print("=" * 70)
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            results.append((step_name, False))

    # 总结
    print("\n" + "=" * 70)
    print("📊 修复总结")
    print("=" * 70)

    for step_name, result in results:
        status = "✅ 成功" if result else "❌ 失败"
        print(f"{status} - {step_name}")

    all_success = all(r for _, r in results)

    if all_success:
        print("\n🎉 所有修复完成！可以开始训练了。")
        print("\n下一步:")
        print("  python scripts/train_real.py --config configs/real_binary.yaml")
        return 0
    else:
        print("\n⚠️  部分步骤失败，请查看上述错误信息。")
        print("\n手动修复指南:")
        print("1. 用 fixed_backbones.py 替换 src/avtop/models/backbones.py")
        print("2. 在 enhanced_detector.py 中添加 mel_bins 参数（见错误输出）")
        print("3. 确保配置文件包含 model.audio.backbone: mel_spectrogram_cnn")
        return 1


if __name__ == '__main__':
    sys.exit(main())