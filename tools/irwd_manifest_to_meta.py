# -*- coding: utf-8 -*-
# 把 IRWD 的 manifest.csv 变成 2 列对照表：stem,label_name
import csv, argparse
from pathlib import Path

def norm_cat(name: str) -> str:
    # 把 manifest 里的下划线类别名转成我们常用的人类读法
    name = name.strip()
    # 统一几种常见写法
    return (name.replace("_w_", " w/ ")
                .replace("_", " ")
                .replace("  ", " ")
                .strip())

def main():
    ap = argparse.ArgumentParser("IRWD manifest -> meta_stem_labels.csv")
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--root", required=True, help="IRWD 数据根目录")
    ap.add_argument("--out_csv", default="data/meta_stem_labels.csv")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    with open(args.manifest_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cat = norm_cat(row.get("CATEGORY",""))
            base = row.get("DIRECTORY","")   # 会话目录，如 08-11-22
            subs = row.get("SUBDIRS","")     # 子目录或样本名列表；不同版本分隔符可能不同
            if not base or not subs:
                continue
            # 兼容 ; , | 三种最常见分隔
            for sub in [s for s in subs.replace("|",";").replace(",",";").split(";") if s]:
                d = root / base / sub
                if not d.exists():
                    # 有的清单直接给的就是样本“stem”，此时 d 不存在，但文件会是 root/base/sub.ext
                    stems = [sub]
                else:
                    # 读取该子目录下的 avi/flac，取 stem
                    stems = sorted({p.stem for p in d.glob("*.avi")} | {p.stem for p in d.glob("*.mp4")})
                    if not stems:
                        stems = sorted({p.stem for p in d.glob("*.flac")})
                    if not stems:
                        # 兜底：该子目录本身名也可能就是 stem
                        stems = [d.name]
                for stem in stems:
                    rows.append({"stem": stem, "label_name": cat})

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem","label_name"])
        w.writeheader(); w.writerows(rows)
    print(f"[OK] 写出 {args.out_csv} 共 {len(rows)} 行")

if __name__ == "__main__":
    main()
