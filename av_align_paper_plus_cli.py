#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
av_align_paper_plus_cli.py
训练前 A/V 对齐与质检清单生成（最终整合版）

要点：
- 覆盖率配对：coverage_audio / bi_coverage / temporal_iou（默认 coverage_audio）
- 中心偏差：使用“交集中心”（在 pairing_utils.center_deviation_stats）
- 映射模式：
    * nominal_offset（默认）：斜率=名义FPS，仅估计平移；天然 drift≈0
    * affine：自由斜率 + 可选稳健重拟合
- 新质检门：resid_med_ms（音频窗中心→最近整数帧的时间残差中位数，单位 ms）
- CSV：稳健写出（字段并集），另导出 windows、windows_aligned
- 绘图：标题含 drift/dev/corr/（可选）resid

依赖：
- av_align_paper_plus.py（提供音频/视频预处理与 Mapping 结构）
- pairing_utils.py（配对与中心偏差）
- plot_utils.py（诊断图）
- video_activity_backends.py（可选；无则回退 basic）
"""
import argparse
import os
import csv
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

# ---- our libs ----
from av_align_paper_plus import (
    load_audio, audio_envelope_rms,
    detect_audio_bounds, detect_video_bounds_series,
    build_mapping, refine_mapping_with_windows,
    gen_audio_windows, gen_video_windows,
    Mapping, AudioBounds, VideoBounds,
    t_audio_to_frame
)

# 这些函数可能未在 av_align_paper_plus 中提供；下面有本地回退
try:
    from av_align_paper_plus import refine_mapping_offset_only, mapping_residual_ms
    HAVE_OFFSET_FUNCS = True
except Exception:
    HAVE_OFFSET_FUNCS = False

from pairing_utils import (
    pair_windows_audio_to_video, center_deviation_stats,
    t_audio_to_frame as map_t2f
)
from plot_utils import save_diagnostic_png

# ---- video activity backends ----
try:
    from video_activity_backends import BasicBackend, SlowFastBackend, VideoActivityBackend
except Exception:
    class VideoActivityBackend: ...


    class BasicBackend:
        class Result:
            def __init__(self, activity, fps): self.activity, self.fps = activity, fps

        def activity_series(self, video_path: str):
            import cv2, numpy as np
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): raise RuntimeError(f"cannot open video: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            ok, prev = cap.read()
            if not ok: raise RuntimeError(f"cannot read first frame: {video_path}")
            prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            vals = []
            while True:
                ok, frm = cap.read()
                if not ok: break
                g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(g, prev)
                vals.append(float(diff.mean()))
                prev = g
            cap.release()
            return BasicBackend.Result(np.array(vals, dtype=np.float32), float(fps))


    class SlowFastBackend(BasicBackend):
        def __init__(self): pass

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}
AUDIO_EXTS = {".wav", ".flac"}

# ---------------- utilities ----------------
def find_pairs(root: Path) -> List[Tuple[Path, Path]]:
    videos, audios = {}, {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            ext = p.suffix.lower(); stem = p.stem
            if ext in VIDEO_EXTS:
                videos.setdefault(stem, []).append(p)
            if ext in AUDIO_EXTS:
                audios.setdefault(stem, []).append(p)
    pairs = []
    for stem, vlist in videos.items():
        if stem not in audios:
            continue
        def score(v, a):
            vp, ap = v.parent.parts, a.parent.parts
            common = sum(1 for i in range(min(len(vp), len(ap))) if vp[i] == ap[i])
            return (-common, len(vp) + len(ap))
        best = min([(score(v, a), v, a) for v in vlist for a in audios[stem]],
                   key=lambda x: x[0])
        pairs.append((best[1], best[2]))
    return sorted(pairs)

def write_csv(path: Path, rows: list):
    """字段并集 + 保序，稳健写 CSV。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames, seen = [], set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def get_backend(name: str):
    if name == "basic":
        return BasicBackend()
    if name == "slowfast":
        try:
            b = SlowFastBackend()
            if hasattr(b, "available") and not getattr(b, "available"):
                return BasicBackend()
            return b
        except Exception:
            return BasicBackend()
    # auto
    try:
        b = SlowFastBackend()
        if hasattr(b, "available") and not getattr(b, "available"):
            return BasicBackend()
        return b
    except Exception:
        return BasicBackend()

# --- robust correlation (安全) ---
def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    a = a - a.mean(); b = b - b.mean()
    sa = float(a.std()); sb = float(b.std())
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (len(a) * sa * sb))

def short_time_corr(t_env: np.ndarray, env: np.ndarray,
                    v_series: np.ndarray, mapper: Mapping,
                    win_s: float = 2.0, hop_s: float = 0.5) -> float:
    frames = np.array([t_audio_to_frame(mapper, float(t)) for t in t_env], dtype=np.float32)
    x_old = np.arange(len(v_series), dtype=np.float32)
    v_on_audio = np.interp(frames, x_old, v_series).astype(np.float32)

    a = env.astype(np.float32); b = v_on_audio.astype(np.float32)
    if len(t_env) < 3:
        return 0.0
    dt = float(t_env[1] - t_env[0])
    W = max(3, int(round(win_s / max(dt, 1e-6))))
    H = max(1, int(round(hop_s / max(dt, 1e-6))))
    vals = []
    i = 0
    while i + W <= len(a):
        vals.append(_safe_corr(a[i:i+W], b[i:i+W]))
        i += H
    return float(np.median(np.array(vals, dtype=np.float32))) if vals else _safe_corr(a, b)

# --- mapping helpers（本地回退） ---
def _refine_mapping_offset_only_local(a_wins, fps_nominal: float, t0_audio: float, b_init: int) -> Mapping:
    ta = np.array([0.5*(s+e) for s,e in a_wins], dtype=np.float64)
    if ta.size == 0:
        return Mapping(fps_eff=fps_nominal, a=fps_nominal, t0_audio=t0_audio, f0_video=int(b_init))
    x = fps_nominal * (ta - t0_audio)
    b_resid = np.median(np.round(x) - x)
    b = float(b_init + b_resid)
    return Mapping(fps_eff=fps_nominal, a=fps_nominal, t0_audio=t0_audio, f0_video=int(round(b)))

def _mapping_residual_ms_local(a_wins, mapper: Mapping) -> float:
    if not a_wins:
        return 1e9
    ta = np.array([0.5*(s+e) for s,e in a_wins], dtype=np.float64)
    f_pred = mapper.a * (ta - mapper.t0_audio) + mapper.f0_video
    f_near = np.round(f_pred)
    dt = np.abs(f_near - f_pred) / max(mapper.a, 1e-9)
    return float(np.median(dt) * 1000.0)

def mapping_residual_ms_on_pairs(a_wins, pair_rows, mapper: Mapping, thr: float) -> float:
    """
    只在“得分>=thr 且有匹配视频窗”的音频窗中心上计算映射残差中位数（毫秒）。
    更稳健，避免边缘/噪声窗污染中位数。
    """
    idx = [int(r["a_idx"]) for r in pair_rows
           if r.get("v_idx", -1) >= 0 and float(r.get("score", 0.0)) >= float(thr)]
    if not idx:
        # 退回全局残差
        try:
            return mapping_residual_ms(a_wins, mapper)  # type: ignore
        except Exception:
            return _mapping_residual_ms_local(a_wins, mapper)
    ta = np.array([0.5*(a_wins[i][0] + a_wins[i][1]) for i in idx], dtype=np.float64)
    f_pred = mapper.a * (ta - mapper.t0_audio) + mapper.f0_video
    f_near = np.round(f_pred)
    dt = np.abs(f_near - f_pred) / max(mapper.a, 1e-9)
    return float(np.median(dt) * 1000.0)

# ---------------- core runner ----------------
def run_one(video: Path, audio: Path, args):
    # === Audio ===
    y, sr = load_audio(str(audio), target_sr=None)
    env, t_env = audio_envelope_rms(y, sr, frame_ms=args.a_frame_ms, hop_ms=args.a_hop_ms)
    ab = detect_audio_bounds(y, sr,
                             frame_ms=args.a_frame_ms, hop_ms=args.a_hop_ms,
                             warmup_sec=args.a_warmup, k=args.k_audio,
                             min_on_s=args.min_on_audio, close_max_gap_s=args.a_close_max_gap_s)

    # === Video activity ===
    backend = get_backend(args.video_backend)
    vres = backend.activity_series(str(video))
    vb = detect_video_bounds_series(vres.activity, vres.fps,
                                    baseline_sec=args.v_baseline,
                                    k_on=args.k_video_on, k_off=args.k_video_off,
                                    min_on_s=args.min_on_video,
                                    close_max_gap_s=args.v_close_max_gap_s)

    # === Windows ===
    a_wins = gen_audio_windows(ab, win_s=args.a_win_s, hop_s=args.a_hop_s)

    # === Mapping ===
    if args.mapping_mode == "nominal_offset":
        if HAVE_OFFSET_FUNCS:
            mapper = refine_mapping_offset_only(a_wins, vres.fps, ab.t0, vb.f0)  # type: ignore
        else:
            mapper = _refine_mapping_offset_only_local(a_wins, vres.fps, ab.t0, vb.f0)
    else:
        mapper = build_mapping(ab, vb)
        if args.refit_mapping:
            tmp_vwins = gen_video_windows(vb, frames_per_win=args.v_frames, stride_frames=args.v_stride)
            mapper = refine_mapping_with_windows(a_wins, tmp_vwins, mapper)

    v_wins = gen_video_windows(vb, frames_per_win=args.v_frames, stride_frames=args.v_stride)

    # === Pairing ===
    pair_rows, pair_pass_rate = pair_windows_audio_to_video(
        a_wins, v_wins, mapper, strategy=args.pair_strategy, thr=args.overlap_thr
    )

    # === Independent checks ===
    cd = center_deviation_stats(pair_rows, mapper)
    corr_med = short_time_corr(t_env, env, vres.activity, mapper,
                               win_s=args.corr_win_s, hop_s=args.corr_hop_s)

    # 漂移（名义模式下≈0）
    drift_pct = 100.0 * abs(mapper.fps_eff - vres.fps) / max(1e-6, vres.fps)

    # 只在“有效配对”上计算映射残差
    resid_ms = mapping_residual_ms_on_pairs(a_wins, pair_rows, mapper, args.overlap_thr)

    sample_keep = (resid_ms <= args.max_resid_ms) \
                  and (cd["center_dev_med"] <= args.center_dev_thr) \
                  and (drift_pct <= args.max_drift_pct) \
                  and (corr_med >= args.corr_thr)

    # === Plot ===
    plot_path = ""
    if args.save_plots:
        out_png = args.out_dir / f"{video.stem}_diag.png"
        save_diagnostic_png(str(out_png),
                            t_env, env, ab, a_wins,
                            vres.activity, vb, vres.fps, mapper.fps_eff, v_wins,
                            pair_rows, args.overlap_thr,
                            cd["center_dev_med"], corr_med,
                            resid_ms)
        plot_path = str(out_png)

    # === result row ===
    sample_row = {
        "sample": video.stem,
        "video": str(video), "audio": str(audio),
        "sr_audio": sr,
        "fps_nominal": round(float(vres.fps), 6),
        "fps_effective": round(float(mapper.fps_eff), 6),
        "fps_drift_pct": round(float(drift_pct), 3),
        "audio_t0": round(ab.t0, 6), "audio_t1": round(ab.t1, 6),
        "audio_duration": round(ab.t1 - ab.t0, 6),
        "video_f0": vb.f0, "video_f1": vb.f1,
        "video_n_frames": int(vb.f1 - vb.f0 + 1),
        "thr_audio": round(ab.thr, 6), "thr_video_on": round(float(vb.thr), 6),
        "method_video": backend.__class__.__name__,
        "pair_strategy": args.pair_strategy,
        "pair_pass_rate": round(float(pair_pass_rate), 4),
        "center_dev_med": round(float(cd["center_dev_med"]), 4),
        "center_dev_p95": round(float(cd["center_dev_p95"]), 4),
        "corr_med": round(float(corr_med), 4),
        "resid_med_ms": round(float(resid_ms), 2),
        "keep_sample": int(sample_keep),
        "n_audio_windows": len(a_wins),
        "n_video_windows": len(v_wins),
        "overlap_thr": args.overlap_thr,
        "mapping_mode": args.mapping_mode,
        "refit_mapping": int(args.refit_mapping),
        "plot": plot_path
    }

    # === windows rows（统一列）===
    win_rows = []
    for i, (ta, tb) in enumerate(a_wins):
        fc = map_t2f(mapper, (ta + tb) / 2.0)
        win_rows.append({
            "sample": video.stem, "modality": "audio", "win_idx": i,
            "audio_start_s": round(ta, 6), "audio_end_s": round(tb, 6),
            "map_center_frame": round(float(fc), 2),
            "video_start_frame": "", "video_end_frame": ""
        })
    for i, (f0, f1) in enumerate(v_wins):
        win_rows.append({
            "sample": video.stem, "modality": "video", "win_idx": i,
            "audio_start_s": "", "audio_end_s": "",
            "map_center_frame": "",
            "video_start_frame": int(f0), "video_end_frame": int(f1),
        })

    # === aligned rows ===
    aligned_rows = []
    for r in pair_rows:
        rr = {"sample": video.stem, **r}
        rr["keep"] = int(r["keep"] and sample_keep)
        aligned_rows.append(rr)

    return sample_row, win_rows, aligned_rows

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Paper-style A/V alignment (robust CLI)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_dir", default="./manifest_out_plus")
    ap.add_argument("--video_backend", choices=["auto", "basic", "slowfast"], default="auto")
    ap.add_argument("--save_plots", action="store_true")

    # Audio params
    ap.add_argument("--a_frame_ms", type=float, default=25.0)
    ap.add_argument("--a_hop_ms", type=float, default=10.0)
    ap.add_argument("--a_warmup", type=float, default=1.0)
    ap.add_argument("--k_audio", type=float, default=7.0)
    ap.add_argument("--min_on_audio", type=float, default=0.20)
    ap.add_argument("--a_win_s", type=float, default=0.20)
    ap.add_argument("--a_hop_s", type=float, default=0.10)
    ap.add_argument("--a_close_max_gap_s", type=float, default=0.15)

    # Video params
    ap.add_argument("--v_baseline", type=float, default=6.0)
    ap.add_argument("--k_video_on", type=float, default=9.0)
    ap.add_argument("--k_video_off", type=float, default=7.0)
    ap.add_argument("--min_on_video", type=float, default=0.80)
    ap.add_argument("--v_close_max_gap_s", type=float, default=0.25)
    ap.add_argument("--v_frames", type=int, default=16)
    ap.add_argument("--v_stride", type=int, default=1)

    # Mapping
    ap.add_argument("--mapping_mode", choices=["affine", "nominal_offset"], default="nominal_offset")
    ap.add_argument("--refit_mapping", action="store_true", default=True)
    ap.add_argument("--max_resid_ms", type=float, default=40.0)

    # Pairing
    ap.add_argument("--pair_strategy", choices=["coverage_audio", "bi_coverage", "temporal_iou"],
                    default="coverage_audio")
    ap.add_argument("--overlap_thr", type=float, default=0.70)

    # Independent gates
    ap.add_argument("--max_drift_pct", type=float, default=15.0)
    ap.add_argument("--center_dev_thr", type=float, default=0.30)
    ap.add_argument("--corr_thr", type=float, default=0.00)
    ap.add_argument("--corr_win_s", type=float, default=2.0)
    ap.add_argument("--corr_hop_s", type=float, default=0.5)

    args = ap.parse_args()
    args.root = Path(args.root)
    args.out_dir = Path(args.out_dir); args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.root)
    if not pairs:
        raise SystemExit(f"No A/V pairs found under: {args.root}")

    print(f"Found {len(pairs)} pairs\n")
    print(f"Strategy: {args.pair_strategy}, Threshold: {args.overlap_thr}\n")

    results, windows, aligned = [], [], []
    success = 0
    for i, (v, a) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {v.name}...", end=" ", flush=True)
        try:
            srow, wrows, arows = run_one(v, a, args)
            results.append(srow); windows.extend(wrows); aligned.extend(arows)
            success += 1
            status = "✓ pass={:.0f}%".format(srow["pair_pass_rate"]*100.0)
            if srow["fps_drift_pct"] > args.max_drift_pct:
                status += f" drift={srow['fps_drift_pct']:.1f}%⚠"
            if not srow["keep_sample"]:
                status += " REJECTED"
            print(status)
        except Exception as e:
            print("✗ FAILED")
            results.append({
                "sample": v.stem, "video": str(v), "audio": str(a),
                "pair_pass_rate": float("nan"),
                "keep_sample": 0,
                "error": str(e)
            })

    print("\nWriting results...")
    write_csv(args.out_dir / "results.csv", results)
    write_csv(args.out_dir / "windows.csv", windows)
    write_csv(args.out_dir / "windows_aligned.csv", aligned)

    # --- summary ---
    print("\n" + "="*70 + "\nSUMMARY\n" + "="*70)
    print(f"Total samples:      {len(results)}")
    print(f"Successfully processed: {success}")
    print(f"Failed:             {len(results) - success}")

    df = pd.DataFrame(results)
    if "pair_pass_rate" in df.columns and df["pair_pass_rate"].notna().any():
        valid = df[df["pair_pass_rate"].notna()]
        try:
            print(f"\nPair Pass Rate:     {valid['pair_pass_rate'].mean()*100:.1f}%")
        except Exception:
            pass
        if "fps_drift_pct" in valid.columns:
            try:
                print(f"FPS Drift (avg):    {pd.to_numeric(valid['fps_drift_pct'], errors='coerce').mean():.2f}%")
            except Exception:
                pass
        if "keep_sample" in valid.columns:
            try:
                print(f"Sample Keep Rate:   {valid['keep_sample'].mean()*100:.1f}%")
            except Exception:
                pass
    else:
        print("\nNo valid metrics (all samples failed before pairing).")

    print("="*70 + "\n")
    print(f"✓ Done! Results saved to {args.out_dir}")

if __name__ == "__main__":
    main()
