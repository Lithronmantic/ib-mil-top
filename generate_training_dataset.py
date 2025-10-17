#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_training_dataset.py
将对齐结果转换为训练所需的CSV格式

输入：
  - manifest_out_plus/windows_aligned.csv  （窗口配对）
  - manifest_out_plus/results.csv          （样本质量）
  - 原始标签文件（可选）

输出：
  - data/train.csv  （70% 有标签）
  - data/val.csv    （30% 有标签）
  - data/unlabeled.csv （无标签样本，用于半监督）

字段：
  video_path, audio_path, label, is_labeled, 
  video_start_frame, video_end_frame, audio_start_s, audio_end_s,
  sample_quality, confidence_score
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict


class DatasetGenerator:
    """数据集生成器"""
    
    def __init__(self, 
                 aligned_csv: str,
                 results_csv: str,
                 labels_csv: str = None,
                 quality_threshold: float = 0.7,
                 min_score: float = 0.6,
                 train_ratio: float = 0.7):
        """
        Args:
            aligned_csv: windows_aligned.csv路径
            results_csv: results.csv路径  
            labels_csv: 原始标签文件（可选，格式: sample,label）
            quality_threshold: 样本质量阈值（用于决定is_labeled）
            min_score: 最小配对分数（过滤低质量窗口）
            train_ratio: 训练集比例
        """
        self.aligned_csv = Path(aligned_csv)
        self.results_csv = Path(results_csv)
        self.labels_csv = Path(labels_csv) if labels_csv else None
        
        self.quality_threshold = quality_threshold
        self.min_score = min_score
        self.train_ratio = train_ratio
        
        # 加载数据
        self.aligned_df = pd.read_csv(aligned_csv)
        self.results_df = pd.read_csv(results_csv)
        
        # 加载标签（如果有）
        self.labels = {}
        if self.labels_csv and self.labels_csv.exists():
            labels_df = pd.read_csv(self.labels_csv)
            self.labels = dict(zip(labels_df['sample'], labels_df['label']))
            print(f"✅ 加载了 {len(self.labels)} 个标签")
        else:
            print("⚠️  未提供标签文件，将根据文件名推断标签")
    
    def _infer_label_from_path(self, sample_name: str) -> int:
        """
        从样本名推断标签（根据Intel焊接数据集命名规则）
        
        Intel数据集规则：
        - normal_xxx.mp4 -> 0 (正常)
        - defect_xxx.mp4 -> 1 (缺陷)
        - anomaly_xxx.mp4 -> 1 (异常)
        """
        name_lower = sample_name.lower()
        
        if 'normal' in name_lower or 'good' in name_lower:
            return 0
        elif 'defect' in name_lower or 'anomaly' in name_lower or 'bad' in name_lower:
            return 1
        else:
            # 默认返回-1表示未知（作为无标签样本）
            return -1
    
    def _compute_sample_quality(self, sample_stats: Dict) -> float:
        """
        计算样本综合质量分数 [0-1]
        
        考虑因素：
        - pair_pass_rate: 配对通过率（权重0.4）
        - corr_med: 音视频相关性（权重0.3）
        - 1 - center_dev_med: 中心对齐度（权重0.2）
        - 1 - (resid_med_ms/100): 映射残差（权重0.1）
        """
        try:
            # 归一化各指标
            pair_rate = float(sample_stats.get('pair_pass_rate', 0.0))
            corr = max(0.0, float(sample_stats.get('corr_med', 0.0)))  # 相关性可能为负
            
            # 中心偏差：越小越好，假设>1s就是很差
            dev = float(sample_stats.get('center_dev_med', 1.0))
            dev_score = max(0.0, 1.0 - dev)
            
            # 映射残差：假设>100ms就是很差
            resid = float(sample_stats.get('resid_med_ms', 100.0))
            resid_score = max(0.0, 1.0 - resid / 100.0)
            
            # 加权平均
            quality = (
                0.4 * pair_rate +
                0.3 * corr +
                0.2 * dev_score +
                0.1 * resid_score
            )
            
            return float(np.clip(quality, 0.0, 1.0))
            
        except Exception as e:
            print(f"⚠️  质量计算失败: {e}")
            return 0.0
    
    def generate_dataset(self, output_dir: str = "data"):
        """生成训练/验证/无标签数据集"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("🚀 开始生成训练数据集")
        print("="*70)
        
        # 1. 按样本分组窗口对
        sample_windows = defaultdict(list)
        
        for _, row in self.aligned_df.iterrows():
            sample = row['sample']
            
            # 过滤低质量窗口
            score = float(row.get('score', 0.0))
            keep = int(row.get('keep', 0))
            
            if score < self.min_score or not keep:
                continue
            
            sample_windows[sample].append(row)
        
        print(f"\n📊 统计信息:")
        print(f"  - 总样本数: {len(sample_windows)}")
        print(f"  - 总窗口对数: {sum(len(wins) for wins in sample_windows.values())}")
        
        # 2. 为每个样本获取质量统计和标签
        samples_data = []
        
        for sample, windows in sample_windows.items():
            # 获取样本质量统计
            sample_stats = self.results_df[self.results_df['sample'] == sample]
            
            if sample_stats.empty:
                print(f"⚠️  样本 {sample} 无质量统计，跳过")
                continue
            
            stats_dict = sample_stats.iloc[0].to_dict()
            
            # 计算综合质量
            quality = self._compute_sample_quality(stats_dict)
            
            # 获取标签
            if sample in self.labels:
                label = self.labels[sample]
            else:
                label = self._infer_label_from_path(sample)
            
            # 决定是否作为有标签样本
            # 高质量样本 -> 有标签
            # 低质量样本 -> 无标签（用于半监督）
            is_labeled = (quality >= self.quality_threshold) and (label != -1)
            
            # 获取路径
            video_path = stats_dict.get('video', '')
            audio_path = stats_dict.get('audio', '')
            
            # 为每个窗口对创建一条记录
            for win_row in windows:
                record = {
                    'sample': sample,
                    'video_path': video_path,
                    'audio_path': audio_path,
                    'label': int(label),
                    'is_labeled': int(is_labeled),
                    
                    # 窗口信息（关键！）
                    'video_start_frame': int(win_row['v_start_frame']),
                    'video_end_frame': int(win_row['v_end_frame']),
                    'audio_start_s': float(win_row['a_start_s']),
                    'audio_end_s': float(win_row['a_end_s']),
                    
                    # 质量指标
                    'sample_quality': round(quality, 4),
                    'window_score': float(win_row['score']),
                    
                    # 额外元数据
                    'pair_strategy': win_row.get('strategy', ''),
                    'temporal_iou': float(win_row.get('temporal_iou', 0.0))
                }
                
                samples_data.append(record)
        
        # 转为DataFrame
        df = pd.DataFrame(samples_data)
        
        print(f"\n📈 生成的样本统计:")
        print(f"  - 总记录数: {len(df)}")
        print(f"  - 有标签: {df['is_labeled'].sum()} ({df['is_labeled'].mean()*100:.1f}%)")
        print(f"  - 无标签: {(~df['is_labeled'].astype(bool)).sum()}")
        
        if len(df[df['is_labeled'] == 1]) > 0:
            print(f"\n  标签分布（有标签样本）:")
            label_counts = df[df['is_labeled'] == 1]['label'].value_counts()
            for label, count in label_counts.items():
                label_name = "正常" if label == 0 else "缺陷" if label == 1 else "未知"
                print(f"    - {label_name} (label={label}): {count}")
        
        # 3. 分割数据集
        # 只在有标签数据上做train/val划分
        labeled_df = df[df['is_labeled'] == 1].copy()
        unlabeled_df = df[df['is_labeled'] == 0].copy()
        
        if len(labeled_df) == 0:
            print("❌ 错误：没有有标签样本！请检查quality_threshold设置")
            return
        
        # 按样本名划分（确保同一样本的所有窗口在同一集合）
        unique_samples = labeled_df['sample'].unique()
        np.random.shuffle(unique_samples)
        
        n_train = int(len(unique_samples) * self.train_ratio)
        train_samples = unique_samples[:n_train]
        val_samples = unique_samples[n_train:]
        
        train_df = labeled_df[labeled_df['sample'].isin(train_samples)]
        val_df = labeled_df[labeled_df['sample'].isin(val_samples)]
        
        print(f"\n📂 数据集划分:")
        print(f"  - 训练集: {len(train_df)} 窗口对 ({len(train_samples)} 样本)")
        print(f"  - 验证集: {len(val_df)} 窗口对 ({len(val_samples)} 样本)")
        print(f"  - 无标签: {len(unlabeled_df)} 窗口对")
        
        # 4. 保存CSV
        train_csv = output_dir / "train.csv"
        val_csv = output_dir / "val.csv"
        unlabeled_csv = output_dir / "unlabeled.csv"
        full_csv = output_dir / "full_dataset.csv"
        
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        unlabeled_df.to_csv(unlabeled_csv, index=False)
        df.to_csv(full_csv, index=False)
        
        print(f"\n✅ 数据集已保存:")
        print(f"  - {train_csv}")
        print(f"  - {val_csv}")
        print(f"  - {unlabeled_csv}")
        print(f"  - {full_csv}")
        
        # 5. 生成数据集统计报告
        self._generate_report(train_df, val_df, unlabeled_df, output_dir)
        
        return train_df, val_df, unlabeled_df
    
    def _generate_report(self, train_df, val_df, unlabeled_df, output_dir):
        """生成数据集统计报告"""
        report_path = output_dir / "dataset_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("数据集生成报告\n")
            f.write("="*70 + "\n\n")
            
            # 基本统计
            f.write("【基本统计】\n")
            f.write(f"训练集: {len(train_df)} 窗口对\n")
            f.write(f"验证集: {len(val_df)} 窗口对\n")
            f.write(f"无标签: {len(unlabeled_df)} 窗口对\n\n")
            
            # 质量分布
            f.write("【质量分布】\n")
            for name, df in [("训练集", train_df), ("验证集", val_df)]:
                if len(df) > 0:
                    f.write(f"\n{name}:\n")
                    f.write(f"  平均质量: {df['sample_quality'].mean():.3f}\n")
                    f.write(f"  质量中位数: {df['sample_quality'].median():.3f}\n")
                    f.write(f"  质量范围: [{df['sample_quality'].min():.3f}, {df['sample_quality'].max():.3f}]\n")
            
            # 标签分布
            f.write("\n【标签分布】\n")
            for name, df in [("训练集", train_df), ("验证集", val_df)]:
                if len(df) > 0:
                    f.write(f"\n{name}:\n")
                    for label, count in df['label'].value_counts().items():
                        label_name = "正常" if label == 0 else "缺陷"
                        ratio = count / len(df) * 100
                        f.write(f"  {label_name}: {count} ({ratio:.1f}%)\n")
            
            # 窗口分数分布
            f.write("\n【窗口配对质量】\n")
            for name, df in [("训练集", train_df), ("验证集", val_df)]:
                if len(df) > 0:
                    f.write(f"\n{name}:\n")
                    f.write(f"  平均分数: {df['window_score'].mean():.3f}\n")
                    f.write(f"  分数>0.8: {(df['window_score'] > 0.8).sum()} ({(df['window_score'] > 0.8).mean()*100:.1f}%)\n")
        
        print(f"  - {report_path}")


def main():
    parser = argparse.ArgumentParser(description="生成训练数据集")
    
    # 输入文件
    parser.add_argument("--aligned_csv", required=True,
                        help="windows_aligned.csv路径")
    parser.add_argument("--results_csv", required=True,
                        help="results.csv路径")
    parser.add_argument("--labels_csv", default=None,
                        help="标签文件（可选，格式: sample,label）")
    
    # 输出目录
    parser.add_argument("--output_dir", default="data",
                        help="输出目录")
    
    # 参数
    parser.add_argument("--quality_threshold", type=float, default=0.7,
                        help="样本质量阈值（高于此值才作为有标签样本）")
    parser.add_argument("--min_score", type=float, default=0.6,
                        help="最小窗口配对分数（低于此值的窗口被过滤）")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="训练集比例")
    
    # 随机种子
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建生成器
    generator = DatasetGenerator(
        aligned_csv=args.aligned_csv,
        results_csv=args.results_csv,
        labels_csv=args.labels_csv,
        quality_threshold=args.quality_threshold,
        min_score=args.min_score,
        train_ratio=args.train_ratio
    )
    
    # 生成数据集
    generator.generate_dataset(output_dir=args.output_dir)
    
    print("\n🎉 数据集生成完成！")
    print("\n下一步：")
    print("  python scripts/train_complete.py --config configs/real_binary_sota.yaml")


if __name__ == "__main__":
    main()
