#!/bin/bash
# pipeline_example.sh
# 从原始数据到训练就绪的完整流程

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "🚀 音视频异常检测完整数据处理流程"
echo "======================================================================"

# ============================================================================
# 阶段1: 音视频对齐
# ============================================================================
echo ""
echo "【阶段1/3】音视频对齐与质检..."
echo "----------------------------------------------------------------------"

# 输入：原始音视频文件
DATA_ROOT="./intel_robotic_welding_dataset"

# 输出：对齐结果
MANIFEST_DIR="manifest_out_plus"

python av_align_paper_plus_cli.py \
    --root "$DATA_ROOT" \
    --out_dir "$MANIFEST_DIR" \
    --video_backend auto \
    --save_plots \
    \
    --a_win_s 0.20 \
    --a_hop_s 0.10 \
    --v_frames 16 \
    --v_stride 1 \
    \
    --mapping_mode nominal_offset \
    --pair_strategy coverage_audio \
    --overlap_thr 0.70 \
    \
    --max_resid_ms 40.0 \
    --center_dev_thr 0.30 \
    --corr_thr 0.00 \
    --max_drift_pct 15.0

echo "✅ 对齐完成！结果保存在: $MANIFEST_DIR/"
echo "   - results.csv: 样本级质量统计"
echo "   - windows_aligned.csv: 窗口配对结果"

# ============================================================================
# 阶段2: 生成训练数据集
# ============================================================================
echo ""
echo "【阶段2/3】生成训练数据集..."
echo "----------------------------------------------------------------------"

# 输入：对齐结果
ALIGNED_CSV="$MANIFEST_DIR/windows_aligned.csv"
RESULTS_CSV="$MANIFEST_DIR/results.csv"

# 标签文件（可选）
# 如果有单独的标签文件，格式如下：
# sample,label
# video001,0
# video002,1
# ...
LABELS_CSV="data/labels.csv"  # 可选

# 输出：训练数据
OUTPUT_DIR="data"

python generate_training_dataset.py \
    --aligned_csv "$ALIGNED_CSV" \
    --results_csv "$RESULTS_CSV" \
    --labels_csv "$LABELS_CSV" \
    --output_dir "$OUTPUT_DIR" \
    \
    --quality_threshold 0.7 \
    --min_score 0.6 \
    --train_ratio 0.7 \
    --seed 42

echo "✅ 数据集生成完成！"
echo "   - data/train.csv"
echo "   - data/val.csv"
echo "   - data/unlabeled.csv"
echo "   - data/full_dataset.csv"
echo "   - data/dataset_report.txt"

# ============================================================================
# 阶段3: 验证数据集
# ============================================================================
echo ""
echo "【阶段3/3】验证数据集..."
echo "----------------------------------------------------------------------"

python -c "
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, 'src')
from avtop.data.window_dataset import WindowDataset, collate_fn

# 加载数据集
train_ds = WindowDataset('data/train.csv', max_video_frames=16)
val_ds = WindowDataset('data/val.csv', max_video_frames=16)

print(f'\n✅ 数据集加载成功！')
print(f'训练集: {len(train_ds)} 窗口对')
print(f'验证集: {len(val_ds)} 窗口对')

# 测试一个样本
sample = train_ds[0]
print(f'\n样本检查:')
print(f'  音频shape: {sample[\"audio\"].shape}')
print(f'  视频shape: {sample[\"video\"].shape}')
print(f'  标签: {sample[\"label\"]}')
print(f'  有标签: {sample[\"is_labeled\"]}')

# 测试DataLoader
loader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn)
batch = next(iter(loader))
print(f'\nBatch检查:')
print(f'  音频batch: {batch[\"audio\"].shape}')
print(f'  视频batch: {batch[\"video\"].shape}')
print(f'  标签batch: {batch[\"label\"].shape}')

print('\n🎉 数据集验证通过！可以开始训练了。')
"

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "======================================================================"
echo "🎉 数据处理流程完成！"
echo "======================================================================"
echo ""
echo "📊 查看统计报告:"
echo "   cat data/dataset_report.txt"
echo ""
echo "🚀 开始训练:"
echo "   python scripts/train_complete.py --config configs/real_binary_sota.yaml"
echo ""
echo "📈 可视化对齐结果:"
echo "   ls $MANIFEST_DIR/*_diag.png"
echo ""
