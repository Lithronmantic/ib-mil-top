# tools/make_labels_from_irwd.py
# 作用：从 Intel Robotic Welding 的“原始目录结构”中递归扫描，
# 自动配对音视频，并生成 labels.csv（video_path,audio_path,label,is_labeled）
# 支持：
#   1) --alias_json 映射各种文件/目录别名到标准类别名（如 good_weld -> Good）
#   2) --binary 开关：二分类 Good=0 / 其他=1；默认多分类（12类）
#   3) --unknown_to_defect：猜不到类别时在二分类下当作缺陷(1)
#   4) --relative_to：把输出路径写成相对某个根（默认相对 --root）
#   5) --meta_csv：若类别不在路径里，可提供 stem->label_name 的外部表

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# 可识别的扩展名
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}

# 规范化：小写 + 移除非字母数字
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

# 标准 12 类 + 常见别名（可被 --alias_json 增补/覆盖）
CLASS_ALIASES: Dict[str, str] = {
    "good": "Good",
    "goodweld": "Good",
    "ok": "Good",
    "normal": "Good",
    "合格": "Good",
    "良品": "Good",

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

    # 常见统称
    "defect": "Defect",
    "defective": "Defect",
    "ng": "Defect",
    "bad": "Defect",
}

# 多分类标签映射（12类 + 兜底 Defect→1，便于二分类）
MULTI_LABEL_MAP: Dict[str, int] = {
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
    "Defect": 1,  # 方便二分类兜底
}

def guess_label_from_tokens(tokens) -> Optional[str]:
    """根据 tokens（文件名 & 若干级父目录）猜类别名"""
    # 优先 Good
    for t in tokens:
        if norm(t) == "good":
            return "Good"
    # 其他匹配
    for t in tokens:
        k = norm(t)
        if k in CLASS_ALIASES:
            return CLASS_ALIASES[k]
    return None

def find_audio_for_video(vid: Path, root: Path) -> Optional[Path]:
    """寻找和视频同 stem 的音频（先同目录、再常见子目录、最后全局）"""
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
    # 3) 父级下常见音频目录
    search_bases = [vid.parent]
    if vid.parent.parent:
        search_bases.append(vid.parent.parent)
    for base in search_bases:
        for sub in ("audio", "audios", "sound", "sounds", "audio_files"):
            d = base / sub
            if d.exists():
                for p in d.rglob("*"):
                    if p.is_file() and p.suffix.lower() in AUDIO_EXTS and p.stem == stem:
                        return p
    # 4) 全局同名 stem（最稳，但可能慢）
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and p.stem == stem:
            return p
    return None

def collect_pairs(root: Path):
    """收集所有视频及其配对音频"""
    video_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    for v in sorted(video_files):
        a = find_audio_for_video(v, root)
        yield v, a

def main():
    ap = argparse.ArgumentParser("IRWD 原始结构 → labels.csv 生成器")
    ap.add_argument("--root", required=True, help="IRWD 数据集根目录（原始结构）")
    ap.add_argument("--out_csv", default="data/labels.csv", help="输出 CSV 路径")
    ap.add_argument("--relative_to", default=None, help="将路径写成相对此目录（默认相对 --root）")
    ap.add_argument("--binary", action="store_true", help="二分类（Good=0 / 其他=1）；默认多分类（12类）")
    ap.add_argument("--alias_json", default=None,
                    help="类别别名映射 JSON，例如 {\"good_weld\":\"Good\",\"ng\":\"Defect\"}")
    ap.add_argument("--unknown_to_defect", action="store_true",
                    help="二分类模式下，猜不到类别时直接记为缺陷(1)")
    ap.add_argument("--meta_csv", default=None,
                    help="外部标注对照表：两列 stem,label_name（优先于路径猜测）")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv)
    rel_base = Path(args.relative_to).resolve() if args.relative_to else root
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 载入外部别名映射
    if args.alias_json:
        with open(args.alias_json, "r", encoding="utf-8") as f:
            user_alias = json.load(f)
        for k, v in user_alias.items():
            CLASS_ALIASES[norm(k)] = v

    # 载入外部标注（若有）
    meta_map: Dict[str, str] = {}
    if args.meta_csv:
        with open(args.meta_csv, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                # 期望两列：stem,label_name
                stem = str(row.get("stem", "")).strip()
                labn = str(row.get("label_name", "")).strip()
                if stem and labn:
                    meta_map[stem] = labn

    def infer_label(v: Path) -> Tuple[int, bool]:
        """返回(label, is_labeled)"""
        # 1) 优先外部标注
        if meta_map:
            name = meta_map.get(v.stem)
            if name:
                if args.binary:
                    return (0 if name == "Good" else 1), True
                return (MULTI_LABEL_MAP.get(name, -1), name in MULTI_LABEL_MAP)

        # 2) 路径猜测
        parts = [v.stem] + [p.name for p in v.parents][:3]  # 文件名 + 父/祖父/曾祖父
        label_name = guess_label_from_tokens(parts)
        if label_name is None:
            if args.binary and args.unknown_to_defect:
                return 1, True
            return -1, False

        if args.binary:
            return (0 if label_name == "Good" else 1), True

        # 多分类
        if label_name not in MULTI_LABEL_MAP:
            return -1, False
        return MULTI_LABEL_MAP[label_name], True

    rows = []
    missing_audio = 0
    total = 0
    for v, a in collect_pairs(root):
        total += 1
        if a is None:
            missing_audio += 1
            continue
        y, is_lab = infer_label(v)
        rows.append({
            "video_path": os.path.relpath(v, rel_base).replace("\\", "/"),
            "audio_path": os.path.relpath(a, rel_base).replace("\\", "/"),
            "label": y,
            "is_labeled": "true" if is_lab else "false",
        })

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_path", "audio_path", "label", "is_labeled"])
        w.writeheader()
        w.writerows(rows)

    labeled = sum(1 for r in rows if r["is_labeled"] == "true")
    unlabeled = len(rows) - labeled
    print(f"[OK] 写出 {out_csv}  共 {len(rows)} 条；缺失音频配对 {missing_audio} / 扫描视频 {total}")
    print(f"   有标签: {labeled}  无标签: {unlabeled}")
    if unlabeled > 0:
        print("   提示：无标签样本将走半监督分支（FixMatch/MeanTeacher）。")

if __name__ == "__main__":
    sys.exit(main())
