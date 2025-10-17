# tools/make_labels_from_irwd.py
import argparse, csv, os, re, sys
from pathlib import Path
from typing import Dict, Optional, Tuple

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}

# 标准 12 类（来自你项目/论文常用命名），兼容常见写法
CLASS_ALIASES = {
    "good": "Good",
    "excessivepenetration": "Excessive Penetration",
    "porositywexcessivepenetration": "Porosity w/Excessive Penetration",
    "porosity": "Porosity",
    "spatter": "Spatter",
    "lackoffusion": "Lack of Fusion",
    "warping": "Warping",
    "burnthrough": "Burnthrough",
    "excessiveconvexity": "Excessive Convexity",
    "undercut": "Undercut",
    "overlap": "Overlap",
    "cratercracks": "Crater Cracks",
    # 容错：很多人把各种缺陷统称 defect / defective
    "defect": "Defect",
    "defective": "Defect",
}

MULTI_LABEL_MAP = {
    "Good": 0,
    "Excessive Penetration": 1,
    "Porosity w/Excessive Penetration": 2,
    "Porosity": 3,
    "Spatter": 4,
    "Lack of Fusion": 5,
    "Warping": 6,
    "Burnthrough": 7,
    "Excessive Convexity": 8,
    "Undercut": 9,
    "Overlap": 10,
    "Crater Cracks": 11,
    "Defect": 1,  # 二分类兜底
}

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def guess_label_from_tokens(tokens) -> Optional[str]:
    # 从 tokens（文件名与各级父目录名）中猜类别（优先“Good”精确匹配）
    for t in tokens:
        if norm(t) == "good":
            return "Good"
    for t in tokens:
        k = norm(t)
        if k in CLASS_ALIASES:
            return CLASS_ALIASES[k]
    return None

def find_audio_for_video(vid: Path, root: Path) -> Optional[Path]:
    stem = vid.stem
    # 1) 同目录同名
    for ext in AUDIO_EXTS:
        cand = vid.with_suffix(ext)
        if cand.exists():
            return cand
    # 2) 同目录模糊匹配（包含 stem）
    for p in vid.parent.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and stem in p.stem:
            return p
    # 3) 向上两级以内搜索 “audio/ audios/ sound/” 等常见子目录
    up = [vid.parent, vid.parent.parent if vid.parent.parent else None]
    for base in [b for b in up if b]:
        for sub in ("audio","audios","sound","sounds","audio_files"):
            d = base / sub
            if d.exists():
                for p in d.rglob("*"):
                    if p.is_file() and p.suffix.lower() in AUDIO_EXTS and p.stem == stem:
                        return p
    # 4) 最后手段：在 root 下全局匹配同名 stem（可能慢，但稳）
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and p.stem == stem:
            return p
    return None

def collect_pairs(root: Path):
    video_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    for v in sorted(video_files):
        a = find_audio_for_video(v, root)
        yield v, a

def infer_label(v: Path, binarize: bool) -> Tuple[int, bool]:
    # tokens：文件名 + 父/祖父目录名
    parts = [v.stem] + [p.name for p in v.parents if p != v.anchor]
    label_name = guess_label_from_tokens(parts)
    if label_name is None:
        return -1, False  # 半监督无标签
    if binarize:
        return (0 if label_name == "Good" else 1), True
    # 多分类
    if label_name not in MULTI_LABEL_MAP:
        return -1, False
    return MULTI_LABEL_MAP[label_name], True

def main():
    ap = argparse.ArgumentParser("IRWD 原始结构 → labels.csv")
    ap.add_argument("--root", required=True, help="Intel Robotic Welding 数据集根目录（原始结构）")
    ap.add_argument("--out_csv", default="data/labels.csv", help="输出 CSV 路径")
    ap.add_argument("--relative_to", default=None, help="将路径写成相对此目录（默认相对 root）")
    ap.add_argument("--binary", action="store_true", help="二分类（Good=0 / 其他=1）；默认多分类")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv)
    rel_base = Path(args.relative_to).resolve() if args.relative_to else root

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_audio = 0
    for v, a in collect_pairs(root):
        if a is None:
            missing_audio += 1
            continue
        y, is_lab = infer_label(v, binarize=args.binary)
        rows.append({
            "video_path": os.path.relpath(v, rel_base).replace("\\","/"),
            "audio_path": os.path.relpath(a, rel_base).replace("\\","/"),
            "label": y,
            "is_labeled": "true" if is_lab else "false"
        })

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_path","audio_path","label","is_labeled"])
        w.writeheader(); w.writerows(rows)

    print(f"[OK] 写出 {out_csv}  共 {len(rows)} 条；缺失音频配对 {missing_audio} 条被跳过。")
    # 简单统计
    labeled = sum(1 for r in rows if r["is_labeled"]=="true")
    unlabeled = len(rows) - labeled
    print(f"   有标签: {labeled}  无标签: {unlabeled}")
    if unlabeled>0:
        print("   提示：无标签样本将走半监督分支（FixMatch/MeanTeacher）。")

if __name__ == "__main__":
    sys.exit(main())
