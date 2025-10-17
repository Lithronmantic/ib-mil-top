# -*- coding: utf-8 -*-
"""
把 av_align_paper_plus_cli.py 的三份 CSV 汇总成训练清单：
training_manifest.csv

字段：
 sample, video_path, audio_path,
 start_frame, end_frame,          # 视频窗口（含端，来自 windows.csv 的 video 行）
 start_time, end_time,            # 音频窗口（秒，来自 windows.csv 的 audio 行）
 score,                           # 配对得分
 label,                           # 二分类标签：good=1 / bad=0（可改）
 group_id                         # 同一原始样本的分组键，便于按视频做划分
"""

import argparse
import os
from pathlib import Path
import pandas as pd

def infer_label_from_path(p: str, default=1):
    low = p.lower()
    if "good" in low or "ok" in low or "pass" in low:
        return 1
    if "bad" in low or "ng" in low or "fail" in low or "defect" in low:
        return 0
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="包含 results.csv / windows.csv / windows_aligned.csv 的目录")
    ap.add_argument("--out_csv", default="training_manifest.csv")
    ap.add_argument("--label_by", choices=["auto","good","bad","one","zero"], default="auto",
                    help="标签来源：auto=从路径推断；good=全1；bad=全0；one=1；zero=0")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    df_res = pd.read_csv(in_dir/"results.csv")
    df_win = pd.read_csv(in_dir/"windows.csv")
    df_align = pd.read_csv(in_dir/"windows_aligned.csv")

    # 只保留成功样本
    keep_samples = set(df_res[df_res.get("keep_sample", 1)==1]["sample"].tolist())
    df_align = df_align[df_align.get("keep",1)==1]
    df_align = df_align[df_align["sample"].isin(keep_samples)]

    # 拆出 audio / video 窗口（windows.csv 里各占一行）
    a_win = df_win[df_win["modality"]=="audio"][["sample","win_idx","audio_start_s","audio_end_s"]]
    v_win = df_win[df_win["modality"]=="video"][["sample","win_idx","video_start_frame","video_end_frame"]]

    # 连接得到窗的时间与帧区间
    m = df_align.merge(a_win, left_on=["sample","a_idx"], right_on=["sample","win_idx"], how="left", suffixes=("","_a"))
    m = m.merge(v_win, left_on=["sample","v_idx"], right_on=["sample","win_idx"], how="left", suffixes=("","_v"))

    # 合回原始路径
    m = m.merge(df_res[["sample","video","audio"]], on="sample", how="left")

    # 标签
    if args.label_by == "good":
        m["label"] = 1
    elif args.label_by in ("bad","zero"):
        m["label"] = 0
    elif args.label_by == "one":
        m["label"] = 1
    else:
        # auto 从 video 路径推断
        m["label"] = m["video"].apply(lambda p: infer_label_from_path(str(p), default=1))

    # 整理输出列
    out = m.rename(columns={
        "video":"video_path",
        "audio":"audio_path",
        "audio_start_s":"start_time",
        "audio_end_s":"end_time",
        "video_start_frame":"start_frame",
        "video_end_frame":"end_frame",
        "score":"score"
    })[
        ["sample","video_path","audio_path",
         "start_frame","end_frame","start_time","end_time",
         "score","label"]
    ].copy()

    # 组 id：按原始样本分组（便于后续按视频做划分）
    out["group_id"] = out["sample"]

    # 基本清洗
    out = out.dropna(subset=["start_time","end_time","start_frame","end_frame"])
    out = out[out["end_time"] > out["start_time"]]
    out = out[out["end_frame"] >= out["start_frame"]]

    out_path = in_dir/args.out_csv
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✓ Wrote {len(out)} rows -> {out_path}")

if __name__ == "__main__":
    main()
