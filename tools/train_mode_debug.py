#!/usr/bin/env python3
"""训练模式调试：包含梯度+损失"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch
import yaml
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.avtop.models.enhanced_detector import EnhancedAVDetector
from src.avtop.data.window_dataset import WindowDataset, collate_fn
from src.avtop.losses.gram_contrastive import CompleteLossFunction

# 加载配置
with open('configs/real_binary_sota.yaml', 'r', encoding='utf-8-sig') as f:
    cfg = yaml.safe_load(f)

# 创建数据集
dataset = WindowDataset(
    csv_path=cfg['data']['train_csv'],
    target_sr=16000,
    max_audio_length=0.3,
    max_video_frames=16
)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)

# 创建模型和损失
model = EnhancedAVDetector(cfg).cuda()
model.train()  # 🔧 训练模式

criterion = CompleteLossFunction(
    num_classes=2,
    lambda_contrastive=0.3,
    lambda_kd=0.2,
    lambda_consistency=0.1
).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

print("🔍 模拟真实训练...")

for batch_idx, batch in enumerate(loader):
    if batch_idx >= 100:
        break

    video = batch['video'].cuda()
    audio = batch['audio'].cuda()
    labels = batch['label'].cuda()
    is_labeled = batch['is_labeled'].cuda()

    optimizer.zero_grad()

    # 前向传播
    outputs = model(video, audio, return_aux=True)

    # 🔧 逐个检查输出
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor) and torch.isnan(val).any():
            print(f"\n❌ Batch {batch_idx}: {key} 包含nan")
            print(f"   Shape: {val.shape}")
            print(f"   Values: {val}")
            exit(1)

    # 计算损失
    loss_dict = criterion(outputs, labels, is_labeled)

    # 🔧 检查损失
    for key, val in loss_dict.items():
        if torch.isnan(val).any() or torch.isinf(val).any():
            print(f"\n❌ Batch {batch_idx}: {key} 异常")
            print(f"   Value: {val.item()}")

            # 打印所有损失
            print(f"\n   所有损失:")
            for k, v in loss_dict.items():
                print(f"     {k}: {v.item()}")

            # 打印模型输出
            print(f"\n   模型输出:")
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    print(
                        f"     {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}], nan={torch.isnan(v).any()}")

            exit(1)

    loss = loss_dict['total_loss']

    # 反向传播
    loss.backward()

    # 🔧 检查梯度
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"\n❌ Batch {batch_idx}: 梯度包含nan")
            print(f"   参数: {name}")
            print(f"   梯度范围: [{param.grad.min():.4f}, {param.grad.max():.4f}]")
            exit(1)

    optimizer.step()

    if batch_idx % 10 == 0:
        print(
            f"✅ Batch {batch_idx}: loss={loss.item():.4f}, cls={loss_dict['classification_loss'].item():.4f}, ctr={loss_dict['contrastive_loss'].item():.4f}")

print("\n✅ 100个batch训练成功，未检测到nan")