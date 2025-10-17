#!/usr/bin/env python3
"""深度调试：逐层检查nan来源"""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.avtop.models.enhanced_detector import EnhancedAVDetector
import torch
import yaml
from torch.utils.data import DataLoader
from src.avtop.data.window_dataset import WindowDataset, collate_fn

# 加载配置
with open('configs/real_binary_sota.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# 创建数据集
print("🔍 加载数据...")
dataset = WindowDataset(
    csv_path=cfg['data']['train_csv'],
    target_sr=16000,
    max_audio_length=0.3,
    max_video_frames=16
)

loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)

# 创建模型
print("🔍 加载模型...")
model = EnhancedAVDetector(cfg).cuda()
model.eval()

# 注册钩子检查每层输出
nan_info = {'layer': None, 'has_nan': False}


def check_nan_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                nan_info['layer'] = name
                nan_info['has_nan'] = True
                print(f"❌ NaN detected in: {name}")
                print(f"   Output shape: {output.shape}")
                print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
        elif isinstance(output, (tuple, list)):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor) and torch.isnan(o).any():
                    nan_info['layer'] = f"{name}[{i}]"
                    nan_info['has_nan'] = True
                    print(f"❌ NaN detected in: {name}[{i}]")

    return hook


# 注册钩子到所有层
print("🔍 注册调试钩子...")
for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # 叶子节点
        module.register_forward_hook(check_nan_hook(name))

# 测试多个batch
print("\n🔍 测试前100个batch...")
for batch_idx, batch in enumerate(loader):
    if batch_idx >= 100:
        break

    video = batch['video'].cuda()
    audio = batch['audio'].cuda()

    # 检查输入
    if torch.isnan(video).any() or torch.isnan(audio).any():
        print(f"\n⚠️ Batch {batch_idx}: 输入包含nan")
        continue

    # 前向传播
    nan_info['has_nan'] = False
    nan_info['layer'] = None

    with torch.no_grad():
        try:
            outputs = model(video, audio, return_aux=True)

            if nan_info['has_nan']:
                print(f"\n❌ Batch {batch_idx}: 模型内部产生nan")
                print(f"   问题层: {nan_info['layer']}")
                print(f"   视频范围: [{video.min():.4f}, {video.max():.4f}]")
                print(f"   音频范围: [{audio.min():.4f}, {audio.max():.4f}]")
                break

            # 检查输出
            if torch.isnan(outputs['clip_logits']).any():
                print(f"\n❌ Batch {batch_idx}: 最终输出包含nan")
                print(f"   clip_logits: {outputs['clip_logits']}")
                break

        except Exception as e:
            print(f"\n❌ Batch {batch_idx}: 异常 - {e}")
            break

    if batch_idx % 10 == 0:
        print(f"✅ Batch {batch_idx} 通过")

print("\n✅ 调试完成")