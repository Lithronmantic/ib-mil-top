# -*- coding: utf-8 -*-
"""
生成两个CSV：
  1) train_windows.csv : 供诊断/排错脚本使用（audio/video的时间窗与帧窗）
  2) annotations.csv   : 供 BinaryAVCSVDataset 直接训练使用（视频级/可选片段级标签）

默认规则：
  - 以文件名“主干”配对：xxx.mp4 与 xxx.wav 视为同一 clip
  - label 依据父目录名推断：{'pos','positive','abnormal','anomaly'} -> 1，其它常见 {neg,negative,normal} -> 0
  - 窗口时长 window_sec=0.96（VGGish 标准），步长 hop_sec=0.96（无重叠）
  - fps/时长优先从 ffprobe 获取；失败则 fps=30.0、时长取0（跳过）

用法:
  python tools/make_csvs.py \
      --root data/raw \
      --out-windows data/train_windows.csv \
      --out-ann     data/annotations.csv \
      --window-sec 0.96 --hop-sec 0.96
"""
import os, re, csv, json, argparse, subprocess, shlex
from pathlib import Path
from collections import defaultdict

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}

def run(cmd: str):
    try:
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def ffprobe_streams(path: str):
    # 取时长 + 帧率；r_frame_rate 是原始计数，需 eval 为 float
    # 参考: ffprobe 官方文档 show_streams / select_streams:contentReference[oaicite:8]{index=8}
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,duration,nb_frames -of default=nokw=1:noprint_wrappers=1 "{path}"'
    s = run(cmd)
    r_frame_rate = None; duration = None; nb_frames = None
    for line in s.strip().splitlines():
        line = line.strip()
        if "/" in line or line.isdigit():
            # 可能是 r_frame_rate（'30000/1001' 或 '30/1'）或 nb_frames
            if "/" in line:
                num, den = line.split("/", 1)
                try:
                    r_frame_rate = float(num) / float(den)
                except Exception:
                    pass
            elif line.isdigit():
                nb_frames = int(line)
        else:
            # 可能是 duration
            try:
                duration = float(line)
            except Exception:
                pass
    return r_frame_rate, duration, nb_frames

def ffprobe_duration_audio(path: str):
    # 只拿音频流时长
    cmd = f'ffprobe -v error -select_streams a:0 -show_entries stream=duration -of default=nw=1:nk=1 "{path}"'
    s = run(cmd).strip()
    try:
        return float(s)
    except Exception:
        # 退而求其次：容器层duration
        cmd2 = f'ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "{path}"'
        s2 = run(cmd2).strip()
        try:
            return float(s2)
        except Exception:
            return None

def is_video(p: Path): return p.suffix.lower() in VIDEO_EXTS
def is_audio(p: Path): return p.suffix.lower() in AUDIO_EXTS
def stem(p: Path):     return p.stem

def infer_label_from_dir(p: Path) -> int:
    name = p.parent.name.lower()
    if name in {"pos", "positive", "abnormal", "anomaly", "anomalous"}:
        return 1
    if name in {"neg", "negative", "normal"}:
        return 0
    # 兜底：不识别则 0
    return 0

def collect_pairs(root: Path):
    videos = {}
    audios = {}
    for fp in root.rglob("*"):
        if fp.is_file():
            if is_video(fp): videos.setdefault(stem(fp).lower(), []).append(fp)
            if is_audio(fp): audios.setdefault(stem(fp).lower(), []).append(fp)
    pairs = []
    for k in sorted(set(videos) & set(audios)):
        # 若有多个同名，择其一（可按最短路径优先）
        v = sorted(videos[k], key=lambda p: len(str(p)))[0]
        a = sorted(audios[k], key=lambda p: len(str(p)))[0]
        pairs.append((k, v, a))
    return pairs

def make_windows_csv(pairs, root: Path, out_csv: Path, window_sec: float, hop_sec: float):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 诊断脚本最低需求列（按你的代码）
        w.writerow(["clip_id", "video", "audio", "audio_t0", "audio_t1", "video_f0", "video_f1", "fps"])
        n_rows = 0
        for k, vpath, apath in pairs:
            fps, v_dur, nb = ffprobe_streams(str(vpath))
            if not fps or fps <= 0: fps = 30.0  # 容错
            a_dur = ffprobe_duration_audio(str(apath))
            if not a_dur or a_dur <= 0:
                # 两者都拿不到时长就跳过
                continue
            total = min(v_dur if v_dur else a_dur, a_dur)
            t = 0.0
            while t + window_sec <= total + 1e-6:
                t0 = max(0.0, t)
                t1 = min(total, t + window_sec)
                f0 = int(round(t0 * fps))
                f1 = int(round(t1 * fps))
                w.writerow([
                    k,
                    os.path.relpath(vpath, root),
                    os.path.relpath(apath, root),
                    f"{t0:.3f}", f"{t1:.3f}",
                    f0, f1, f"{fps:.6f}"
                ])
                n_rows += 1
                t += hop_sec
    return n_rows

def make_annotations_csv(pairs, root: Path, out_csv: Path):
    """
    符合 BinaryAVCSVDataset 读取需求的列：
      video_path, audio_path, label, clip_id, seg_starts, seg_ends
    见 __getitem__ 的取值。
    这里不给片段级标注，seg_* 置空字符串。
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "video_path", "audio_path", "label", "seg_starts", "seg_ends"])
        for k, vpath, apath in pairs:
            label = infer_label_from_dir(vpath)
            w.writerow([
                k,
                os.path.relpath(vpath, root),
                os.path.relpath(apath, root),
                int(label),
                "", ""  # 可后续填入 "起帧1,起帧2", "止帧1,止帧2"
            ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="数据根目录（包含视频/音频）")
    ap.add_argument("--out-windows", required=True, help="输出 train_windows.csv 路径")
    ap.add_argument("--out-ann", required=True, help="输出 annotations.csv 路径")
    ap.add_argument("--window-sec", type=float, default=0.96, help="窗口时长（秒），VGGish 标准 0.96s")
    ap.add_argument("--hop-sec", type=float, default=0.96, help="步长（秒），默认无重叠")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pairs = collect_pairs(root)
    print(f"[scan] 匹配到 {len(pairs)} 个 (video,audio) clip 对。")
    n = make_windows_csv(pairs, root, Path(args.out_windows), args.window_sec, args.hop_sec)
    print(f"[windows] 写入 {args.out_windows} 共 {n} 行。")
    make_annotations_csv(pairs, root, Path(args.out_ann))
    print(f"[ann] 写入 {args.out_ann}。")
    print("👍 完成。")

if __name__ == "__main__":
    main()
