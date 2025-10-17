# -*- coding: utf-8 -*-
# 作用：在 data/labels.csv 基础上，随机给“good_weld* / *lack_of_fusion*”的一部分样本打标签
#  - good_weld 开头 → Good(0)
#  - 包含 lack_of_fusion → Defect(1)
#  - 其他样本保持无标签(-1,false)
# 用法举例见文末。

import argparse, re
from pathlib import Path
import pandas as pd  # 读取/写入CSV与抽样，见 pandas.read_csv / DataFrame.sample 文档

def make_mask(series, pattern, startswith=False, ignore_case=True):
    if startswith:
        # 匹配路径分隔后紧跟 good_weld：(^|/|\\)good_weld
        pat = r'(^|[/\\])' + re.escape(pattern)
        return series.str.contains(pat, case=not ignore_case, regex=True, na=False)
    else:
        return series.str.contains(pattern, case=not ignore_case, regex=False, na=False)

def main():
    ap = argparse.ArgumentParser("随机给一部分样本加标签：good_weld*→0，*lack_of_fusion*→1")
    ap.add_argument("--labels", required=True, help="输入 labels.csv（包含 video_path,audio_path,label,is_labeled）")
    ap.add_argument("--out", default=None, help="输出 CSV（默认就地覆盖输入）")
    ap.add_argument("--frac_good", type=float, default=0.5, help="good_weld* 中标注为有标签的比例 [0,1]")
    ap.add_argument("--frac_lof", type=float, default=0.5, help="*lack_of_fusion* 中标注为有标签的比例 [0,1]")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，保证抽样可复现")
    args = ap.parse_args()

    in_csv = Path(args.labels)
    out_csv = Path(args.out) if args.out else in_csv

    df = pd.read_csv(in_csv)  # 读取 CSV 【pandas.read_csv】:contentReference[oaicite:1]{index=1}
    required_cols = {"video_path","audio_path","label","is_labeled"}
    miss = required_cols - set(map(str.lower, df.columns))
    # 兼容大小写列名
    cols = {c.lower(): c for c in df.columns}
    if miss:
        raise ValueError(f"CSV 缺少列：{required_cols}，当前列={list(df.columns)}")

    # 统一列引用
    vp = cols.get("video_path")
    lab = cols.get("label")
    isl = cols.get("is_labeled")

    # 先把现有无标签样本统一成规范（-1,false），避免历史脏值
    df.loc[df[isl].astype(str).str.lower().isin(["false","0","no","n",""]), lab] = -1
    df.loc[df[isl].astype(str).str.lower().isin(["false","0","no","n",""]), isl] = "false"

    # 1) good_weld 开头（文件名或任一父目录段以 good_weld 起始）
    mask_good = make_mask(df[vp], pattern="good_weld", startswith=True, ignore_case=True)

    # 2) 含 lack_of_fusion（任意位置）
    mask_lof = make_mask(df[vp], pattern="lack_of_fusion", startswith=False, ignore_case=True)

    # 在各自掩码下“随机抽一部分”打标签（DataFrame.sample 支持 frac+random_state）【pandas.DataFrame.sample】:contentReference[oaicite:2]{index=2}
    good_idx = df[mask_good].sample(frac=max(0.0, min(1.0, args.frac_good)), random_state=args.seed).index
    lof_idx  = df[mask_lof ].sample(frac=max(0.0, min(1.0, args.frac_lof )),  random_state=args.seed).index

    # 打标签：Good=0 / Defect=1，且 is_labeled=true
    df.loc[good_idx, lab] = 0
    df.loc[lof_idx,  lab] = 1
    df.loc[good_idx.union(lof_idx), isl] = "true"

    # 其余保持无标签（-1,false），不动

    # 统计
    def cnt(mask): return int(mask.sum())
    print(f"[INFO] good_weld 匹配总数={cnt(mask_good)}，本次标注={len(good_idx)} (frac_good={args.frac_good})")
    print(f"[INFO] lack_of_fusion 匹配总数={cnt(mask_lof)}，本次标注={len(lof_idx)} (frac_lof={args.frac_lof})")
    print(f"[INFO] 标注后计数：is_labeled=true -> {int((df[isl].str.lower()=='true').sum())} / {len(df)}")

    df.to_csv(out_csv, index=False)
    print(f"[OK] 已写出 {out_csv}")

if __name__ == "__main__":
    main()
