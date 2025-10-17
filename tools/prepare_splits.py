#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avtop.data.grouped_split import prepare_grouped_splits

# 对训练集执行分组划分
train_path, val_path, stats = prepare_grouped_splits(
    csv_path="train.csv",
    output_dir="./splits",
    n_splits=5,
    val_fold=0,
    use_stratified=True  # 保持类别比例
)

print(f"\n✅ Split完成！")
print(f"训练集: {train_path}")
print(f"验证集: {val_path}")