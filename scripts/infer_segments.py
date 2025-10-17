#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
整段视频缺陷检测（滑窗 + 带标签视频渲染 + 音频可视化）
输出：
  1) segments.csv        : 每个时间窗口的缺陷概率与预测
  2) merged_segments.csv : 连续窗口合并后的缺陷/正常区段（秒）
  3) timeline.png        : 概率时间轴+阈值+合并区段
  4) summary.txt         : 视频级结论与关键统计
  5) annotated.mp4       : 叠加标签与音频可视化的演示视频   ← 新增

用法示例：
python scripts/infer_segments.py ^
  --video demo/xxx.mp4 ^
  --checkpoint outputs/sota_training/checkpoints/best_model.pth ^
  --config configs/real_binary_sota.yaml ^
  --out_dir outputs/infer_xxx ^
  --win_s 0.30 --hop_s 0.10 --v_frames 16 --v_stride 1 ^
  --thr 0.50 ^
  --render_video --show_audio
"""
import os
import sys
import math
import csv
import yaml
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

# 项目内导入
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))
from src.avtop.models.enhanced_detector import EnhancedAVDetector

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------- 基础 I/O ----------------

def load_audio(audio_path: str, target_sr: int) -> np.ndarray:
    """读音频为单通道 float32，采样率 target_sr。优先 librosa；否则 soundfile+线性重采样。"""
    try:
        import librosa
        y, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        y = y.astype(np.float32)
    except Exception:
        import soundfile as sf
        y, sr0 = sf.read(audio_path)
        if y.ndim == 2: y = y.mean(axis=1)
        if sr0 != target_sr:
            x = np.linspace(0, len(y), int(len(y) * target_sr / sr0), endpoint=False)
            y = np.interp(x, np.arange(len(y)), y)
        y = y.astype(np.float32)
    mx = float(np.max(np.abs(y))) if len(y) else 1.0
    return y / max(mx, 1e-6)

def extract_audio_from_video(video_path: str, target_sr: int) -> np.ndarray:
    """从视频提取音频，若失败提醒提供 --audio。"""
    try:
        import librosa
        y, _ = librosa.load(video_path, sr=target_sr, mono=True)
        y = y.astype(np.float32)
        mx = float(np.max(np.abs(y))) if len(y) else 1.0
        return y / max(mx, 1e-6)
    except Exception:
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(video_path)
            snd = clip.audio.to_soundarray(fps=target_sr)
            if snd.ndim == 2: snd = snd.mean(axis=1)
            y = snd.astype(np.float32)
            mx = float(np.max(np.abs(y))) if len(y) else 1.0
            return y / max(mx, 1e-6)
        except Exception as e:
            raise RuntimeError(f"无法从视频提取音频，请提供 --audio。原因：{e}")

def load_video_for_model_and_render(video_path: str, target_size: Tuple[int,int]):
    """
    用 OpenCV 读视频：
      - frames_model:   [N, H, W, 3] float32，已做 ImageNet 归一化（用于模型）
      - frames_render:  [N, H, W, 3] uint8，BGR（用于渲染叠加）
      - fps: float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"无法打开视频：{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_m, frames_r = [], []
    W, H = int(target_size[0]), int(target_size[1])
    while True:
        ok, bgr = cap.read()
        if not ok: break
        bgr_rs = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr_rs, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        frames_m.append(rgb)               # 模型输入（归一化）
        frames_r.append(bgr_rs.copy())     # 渲染底图（BGR uint8）
    cap.release()
    if not frames_m: raise RuntimeError("视频无帧")
    return np.stack(frames_m, 0), np.stack(frames_r, 0), float(fps)

# ---------------- 滑窗与切片 ----------------

def make_windows(duration_s: float, win_s: float, hop_s: float) -> List[Tuple[float,float]]:
    t, out = 0.0, []
    while t + win_s <= duration_s + 1e-6:
        out.append((t, t + win_s)); t += hop_s
    return out

def slice_audio(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    i0, i1 = int(round(start*sr)), int(round(end*sr))
    seg = y[i0:i1]
    if len(seg) == 0: seg = np.zeros(max(1, i1-i0), dtype=np.float32)
    mx = float(np.max(np.abs(seg))) if len(seg) else 1.0
    return (seg / max(mx, 1e-6)).astype(np.float32)

def slice_video(frames_norm: np.ndarray, fps: float, start: float, v_frames: int, v_stride: int) -> np.ndarray:
    n, h, w, c = frames_norm.shape
    start_idx = int(round(start * fps))
    idxs = [min(max(0, start_idx + i*v_stride), n-1) for i in range(v_frames)]
    clip = frames_norm[idxs]                 # [T,H,W,3]
    clip = np.transpose(clip, (0,3,1,2))     # [T,3,H,W]
    return clip.astype(np.float32)

# ---------------- 推理辅助 ----------------

def logits_to_prob_defect(logits: torch.Tensor) -> np.ndarray:
    return F.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()

def merge_binary_segments(times: List[Tuple[float,float]], preds: np.ndarray) -> List[Tuple[float,float,int]]:
    if len(times) == 0: return []
    out = []
    cur_s, cur_e, cur_y = times[0][0], times[0][1], int(preds[0])
    for (s,e), y in zip(times[1:], preds[1:]):
        y = int(y)
        if y == cur_y and abs(s - cur_e) <= 1e-6:
            cur_e = e
        else:
            out.append((cur_s, cur_e, cur_y)); cur_s, cur_e, cur_y = s, e, y
    out.append((cur_s, cur_e, cur_y))
    return out

# ---------------- 音频可视化（梅尔谱/波形） ----------------

def make_melspec_image(y: np.ndarray, sr: int, width: int, height: int) -> np.ndarray:
    """
    生成彩色梅尔谱 BGR 图像，大小 (height, width, 3) uint8；若 librosa 不可用，回退为波形图。
    """
    try:
        import librosa
        S = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=sr,
                                           n_fft=512, hop_length=160, win_length=400,
                                           n_mels=64, fmin=50, fmax=sr//2)
        S = librosa.power_to_db(S, ref=np.max)  # [n_mels, Tm]
        S = np.flipud(S)  # 低频在下
        S = (S - S.min()) / max(S.max() - S.min(), 1e-6)  # 0~1
        S = (S * 255.0).astype(np.uint8)
        S = cv2.resize(S, (width, height), interpolation=cv2.INTER_LINEAR)
        color = cv2.applyColorMap(S, cv2.COLORMAP_TURBO)  # BGR
        return color
    except Exception:
        # 回退：波形图
        img = np.zeros((height, width, 3), dtype=np.uint8)
        if len(y) > 1:
            # 取等距采样点绘折线
            xs = np.linspace(0, len(y)-1, width).astype(int)
            yy = y[xs]
            yy = (yy - yy.min()) / max(yy.max()-yy.min(), 1e-6)  # 0~1
            yy = (1.0 - yy) * (height-1)                         # 翻转 y 轴
            pts = np.stack([np.arange(width), yy], axis=1).astype(np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=(200, 200, 50), thickness=1)
        cv2.putText(img, "Waveform (librosa not found)", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)
        return img

# ---------------- 渲染带标签视频 ----------------

def render_video_with_overlays(frames_bgr: np.ndarray,
                               fps: float,
                               probs_def: np.ndarray,
                               preds_bin: np.ndarray,
                               windows: List[Tuple[float,float]],
                               audio: np.ndarray,
                               sr: int,
                               out_path: Path,
                               thr: float,
                               show_audio: bool = True):
    """
    在每帧上叠加：
      - 文本标签（GOOD/DEFECT）+ 背景块
      - 概率条（当前 p(defect)）
      - （可选）底部音频可视化（梅尔谱/波形）+ 当前时间竖线
    输出：MP4 视频（与输入帧同尺寸）
    """
    H, W = frames_bgr.shape[1], frames_bgr.shape[2]
    # 预计算：帧时间 -> 最近窗索引
    centers = np.array([(s+e)/2 for s,e in windows], dtype=float)
    num_frames = len(frames_bgr)
    times_f = np.arange(num_frames) / max(fps, 1e-6)
    idx = np.searchsorted(centers, times_f, side='left')
    idx = np.clip(idx, 0, len(centers)-1)
    # 比较相邻，选最近中心
    left = np.maximum(idx-1, 0)
    choose_left = (np.abs(centers[left] - times_f) < np.abs(centers[idx] - times_f))
    idx = np.where(choose_left, left, idx)
    prob_f = probs_def[idx]
    pred_f = preds_bin[idx]

    # 音频图（宽与帧一致，高度固定 96 像素）
    spec_h = 96
    spec_img = make_melspec_image(audio, sr, width=W, height=spec_h) if show_audio else None

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError("无法创建视频写入器，确认编解码器是否可用（建议使用 mp4v / H264）")

    # 叠加样式
    bar_h = 14            # 概率条高度
    pad = 6               # 内边距
    label_font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(num_frames):
        frame = frames_bgr[i].copy()

        # 1) 概率条（顶部）
        p = float(prob_f[i])
        cv2.rectangle(frame, (pad, pad), (W-pad, pad+bar_h), (60,60,60), thickness=-1)
        fill = int((W-2*pad) * np.clip(p, 0, 1))
        cv2.rectangle(frame, (pad, pad), (pad+fill, pad+bar_h), (0,0,255), thickness=-1)  # 红色表示缺陷概率

        # 2) 标签底板 + 文本
        label = "DEFECT" if pred_f[i]==1 else "GOOD"
        color = (40,40,220) if pred_f[i]==1 else (40,180,40)   # (B,G,R)
        text = f"{label}  P(def)={p:.2f}"
        (tw, th), bs = cv2.getTextSize(text, label_font, 0.7, 2)
        box_w, box_h = tw + 2*pad, th + 2*pad
        # 放在左上角、概率条下方
        cv2.rectangle(frame, (pad, pad*2+bar_h), (pad+box_w, pad*2+bar_h+box_h), color, thickness=-1)
        cv2.putText(frame, text, (pad+pad, pad*2+bar_h+th+1), label_font, 0.7, (255,255,255), 2, cv2.LINE_AA)

        # 3) 音频可视化（底部覆盖一条高 spec_h 的带状区域）+ 当前时间线
        if spec_img is not None:
            y0 = H - spec_h
            overlay = frame[y0:H, 0:W].copy()
            alpha = 0.55
            # 混合梅尔谱
            blended = cv2.addWeighted(spec_img, alpha, overlay, 1-alpha, 0)
            # 当前时间线（白色竖线）
            x_now = int((i / max(num_frames-1, 1)) * (W-1))
            cv2.line(blended, (x_now, 0), (x_now, spec_h-1), (255,255,255), 1, cv2.LINE_AA)
            frame[y0:H, 0:W] = blended

            # 阈值提示
            cv2.putText(frame, f"Audio Vis", (W-150, H-8), label_font, 0.5, (230,230,230), 1, cv2.LINE_AA)

        vw.write(frame)

    vw.release()
    print(f"🎬 已写出带标签视频：{out_path}")

# ---------------- 概率时间轴图 ----------------

def draw_timeline(times: List[Tuple[float,float]], p_def: np.ndarray, thr: float,
                  merged: List[Tuple[float,float,int]], out_png: Path):
    xs = np.array([(s+e)/2 for s,e in times], dtype=float)
    plt.figure(figsize=(12,4))
    plt.plot(xs, p_def, linewidth=1.5, label="P(defect)")
    plt.axhline(thr, ls='--', c='r', lw=1, label=f"thr={thr:.2f}")
    for s,e,y in merged:
        if y==1:
            plt.axvspan(s, e, color='tomato', alpha=0.2)
        else:
            plt.axvspan(s, e, color='steelblue', alpha=0.1)
    plt.ylim([-0.05, 1.05]); plt.xlabel("Time (s)"); plt.ylabel("Probability")
    plt.title("Defect Probability Timeline"); plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# ---------------- 模型加载 ----------------

def load_model(checkpoint: str, config: dict, device: torch.device) -> torch.nn.Module:
    import numpy as np
    print(f"[加载模型] {checkpoint}")
    model = EnhancedAVDetector(config)
    try:
        # 先尝试安全模式（PyTorch 2.4+）
        try:
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        except Exception:
            pass
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        safe_mode = True
    except Exception as e:
        print(f"⚠️ weights_only=True 失败（{e.__class__.__name__}），回退到传统加载。")
        ckpt = torch.load(checkpoint, map_location=device)  # 仅在你信任 ckpt 时使用
        safe_mode = False
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"ℹ️ state_dict 对齐：missing={len(missing)}, unexpected={len(unexpected)}")
    print(f"✅ 模型已加载（{'安全' if safe_mode else '传统'}模式）")
    return model.to(device).eval()

# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser("整段视频缺陷检测（滑窗 + 视频渲染）")
    ap.add_argument("--video", required=True, help="视频文件路径")
    ap.add_argument("--audio", default=None, help="（可选）音频文件路径；不提供则从视频提取")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="outputs/infer")
    # 滑窗/采样
    ap.add_argument("--win_s", type=float, default=None, help="音频窗口秒；默认从 config.data.max_audio_length")
    ap.add_argument("--hop_s", type=float, default=0.10)
    ap.add_argument("--v_frames", type=int, default=None, help="每窗视频帧数；默认从 config.data.max_video_frames")
    ap.add_argument("--v_stride", type=int, default=1)
    # 判定阈值
    ap.add_argument("--thr", type=float, default=None, help="缺陷概率阈值；默认 0.5 或 metrics.yaml 中 best_threshold")
    # 渲染开关
    ap.add_argument("--render_video", action="store_true", help="导出带标签的演示视频")
    ap.add_argument("--show_audio", action="store_true", help="在视频底部叠加音频可视化（梅尔谱/波形）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sr = int(cfg["data"].get("audio_sr", 16000))
    H, W = cfg["data"].get("video_size", [224,224])
    win_s    = float(args.win_s if args.win_s is not None else cfg["data"].get("max_audio_length", 0.3))
    v_frames = int(args.v_frames if args.v_frames is not None else cfg["data"].get("max_video_frames", 16))
    v_stride = int(args.v_stride)

    # 阈值：优先读取最近评估的 metrics.yaml；否则 0.5
    thr = args.thr
    if thr is None:
        metrics_yaml = Path(args.checkpoint).parent.parent / "evaluation" / "metrics.yaml"
        thr = 0.5
        if metrics_yaml.exists():
            try:
                m = yaml.safe_load(metrics_yaml.read_text("utf-8"))
                thr = float(m.get("best_threshold", 0.5))
            except Exception:
                pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, cfg, device)

    # 读音视频（模型帧 + 渲染帧）
    frames_model, frames_render, fps = load_video_for_model_and_render(args.video, (W, H))
    if args.audio and Path(args.audio).exists():
        audio = load_audio(args.audio, sr)
    else:
        audio = extract_audio_from_video(args.video, sr)

    dur_video = len(frames_model) / max(fps, 1e-6)
    dur_audio = len(audio) / float(sr)
    duration  = min(dur_video, dur_audio)
    windows   = make_windows(duration, win_s, args.hop_s)
    if not windows: raise RuntimeError("视频过短，未生成任何窗口。")

    # 推理
    probs_def, preds_bin = [], []
    with torch.no_grad():
        for (s, e) in windows:
            a_seg = slice_audio(audio, sr, s, e)                        # [L]  ← 波形 (B,T) 期望
            v_seg = slice_video(frames_model, fps, s, v_frames, v_stride)

            a_t = torch.from_numpy(a_seg).unsqueeze(0).to(device)       # [1, L] ✅
            v_t = torch.from_numpy(v_seg).unsqueeze(0).to(device)       # [1, T, 3, H, W]

            try:
                out = model(v_t, a_t, return_aux=False)                 # 优先走波形
            except ValueError as err:
                if "Cannot determine input type" not in str(err):
                    raise
                # 回退：log-mel（64 bins）
                import librosa
                S = librosa.feature.melspectrogram(
                    y=a_seg.astype(np.float32), sr=sr,
                    n_fft=512, hop_length=160, win_length=400,
                    n_mels=64, fmin=50, fmax=sr//2
                )
                S = librosa.power_to_db(S, ref=np.max).T                # [Tm, 64]
                a_t = torch.from_numpy(S.astype(np.float32)).unsqueeze(0).to(device)  # [1, Tm, 64]
                out = model(v_t, a_t, return_aux=False)

            p = logits_to_prob_defect(out["clip_logits"])               # [1] -> ndarray
            probs_def.append(float(p[0]))
            preds_bin.append(int(p[0] >= thr))

    probs_def = np.asarray(probs_def, dtype=float)
    preds_bin = np.asarray(preds_bin, dtype=int)

    # 合并连续区段并写报告/图表（保持你的原有产物）
    merged = merge_binary_segments(windows, preds_bin)

    seg_csv = out_dir / "segments.csv"
    with seg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["start_s","end_s","prob_defect","pred"])
        for (s,e), p, y in zip(windows, probs_def, preds_bin):
            w.writerow([f"{s:.3f}", f"{e:.3f}", f"{p:.6f}", int(y)])

    merged_csv = out_dir / "merged_segments.csv"
    with merged_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["start_s","end_s","label"])
        for s,e,y in merged:
            w.writerow([f"{s:.3f}", f"{e:.3f}", "defect" if y==1 else "good"])

    # 概率时间轴
    timeline_png = out_dir / "timeline.png"
    draw_timeline(windows, probs_def, thr, merged, timeline_png)

    # 视频级结论
    total = sum(e-s for s,e in windows)
    defect_time = sum(e-s for s,e,y in merged if y==1)
    ratio = defect_time / max(total, 1e-9)
    video_label = "defect" if ratio > 0.5 else "good"
    with (out_dir/"summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"video: {args.video}\n")
        f.write(f"duration_s: {duration:.3f}\n")
        f.write(f"windows: {len(windows)}, win_s={win_s}, hop_s={args.hop_s}, v_frames={v_frames}, v_stride={v_stride}\n")
        f.write(f"threshold: {thr:.3f}\n")
        f.write(f"defect_time_s: {defect_time:.3f} / {total:.3f} ({ratio*100:.2f}%)\n")
        f.write(f"video_label: {video_label}\n")

    print("✅ 推理完成（CSV/图表/报告 已写出）")
    print(f"  - {seg_csv}")
    print(f"  - {merged_csv}")
    print(f"  - {timeline_png}")
    print(f"  - {out_dir/'summary.txt'}")

    # 渲染带标签视频
    if args.render_video:
        out_mp4 = out_dir / "annotated.mp4"
        render_video_with_overlays(frames_render, fps, probs_def, preds_bin, windows,
                                   audio, sr, out_mp4, thr, show_audio=args.show_audio)

        print(f"🎯 视频级结论：{video_label}（缺陷时长占比 {ratio*100:.2f}%）")

if __name__ == "__main__":
    main()
