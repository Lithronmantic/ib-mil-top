# -*- coding: utf-8 -*-
"""
plot_utils.py
可视化 A/V 预处理与对齐诊断图

save_diagnostic_png(
    out_png,                # 输出路径
    t_env, env, ab, a_wins, # 音频包络与主段/窗口
    v_series, vb,           # 视频活动曲线与主段(帧)
    fps_nominal, fps_eff,   # 名义/有效 FPS
    v_wins,                 # 视频窗口(帧区间列表)
    pair_rows, overlap_thr, # 配对结果(含 score/keep)与阈值
    center_dev_med, corr_med,
    resid_ms=None           # 可选：映射残差中位数(毫秒)
)
"""
from __future__ import annotations
import os
from typing import Iterable, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 后端
import matplotlib.pyplot as plt


# ---------- helpers ----------
def _span_windows(ax, intervals: Iterable[Tuple[float, float]],
                  ylim: Tuple[float, float],
                  color: str = "#6BA5FF", alpha: float = 0.15, lw: float = 0.0):
    """在轴上绘制一组 [start, end] 的竖向窗（以秒为单位）"""
    y0, y1 = ylim
    for s, e in intervals:
        if e <= s:
            continue
        ax.add_patch(
            plt.Rectangle((float(s), y0),
                          float(e - s), float(y1 - y0),
                          facecolor=color, alpha=alpha, linewidth=lw, edgecolor=color)
        )


def _audio_win_seconds(a_wins: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [(float(s), float(e)) for s, e in a_wins]


def _video_win_seconds(v_wins: List[Tuple[int, int]], fps: float) -> List[Tuple[float, float]]:
    out = []
    for f0, f1 in v_wins:
        # f1 为包含端；右端取 (f1+1)/fps 更直观
        s = float(f0) / max(fps, 1e-9)
        e = float(f1 + 1) / max(fps, 1e-9)
        out.append((s, e))
    return out


def _safe_hist(ax, values: np.ndarray, thr: float, bins: int = 30):
    if values.size == 0:
        ax.text(0.5, 0.5, "No pairs", ha="center", va="center", transform=ax.transAxes)
        return
    ax.hist(values, bins=bins, color="#7DB1E8", edgecolor="white", linewidth=0.5)
    ax.axvline(thr, ls="--", lw=1.5, color="crimson")
    ax.legend([f"Threshold={thr:.2f}"], frameon=False, loc="upper right")


# ---------- main ----------
def save_diagnostic_png(out_png: str,
                        t_env: np.ndarray, env: np.ndarray,
                        ab, a_wins: List[Tuple[float, float]],
                        v_series: np.ndarray, vb,
                        fps_nominal: float, fps_eff: float,
                        v_wins: List[Tuple[int, int]],
                        pair_rows: List[dict], overlap_thr: float,
                        center_dev_med: float, corr_med: float,
                        resid_ms: Optional[float] = None):
    """
    绘制 3 行 4 轴：
      1) 音频 RMS 包络 + 主段 + 窗口
      2) 视频活动曲线 + 主段 + 窗口
      3) 左：配对分数直方图（Pass 率）；右：按音频窗索引的分数散点
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # ---------- 预处理时间轴 ----------
    # 音频时间轴已给出；视频时间轴按名义 FPS
    t_video = np.arange(len(v_series), dtype=np.float32) / max(fps_nominal, 1e-9)
    v_main_s = float(vb.f0) / max(fps_nominal, 1e-9)
    v_main_e = float(vb.f1 + 1) / max(fps_nominal, 1e-9)

    # 分数数组 & 通过率
    scores = []
    keeps = []
    for r in pair_rows:
        s = r.get("score", None)
        if s is not None:
            try:
                scores.append(float(s))
            except Exception:
                pass
        k = r.get("keep", None)
        if k is not None:
            try:
                keeps.append(int(k))
            except Exception:
                pass
    scores = np.array(scores, dtype=np.float32)
    keeps = np.array(keeps, dtype=np.int32) if keeps else np.zeros_like(scores, dtype=np.int32)
    pass_rate = (float(keeps.mean()) if keeps.size else 0.0) * 100.0

    # 漂移
    drift_pct = 100.0 * abs(float(fps_eff) - float(fps_nominal)) / max(float(fps_nominal), 1e-9)

    # ---------- Figure ----------
    fig = plt.figure(figsize=(14, 10), dpi=160)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 2.0, 1.2], hspace=0.35, wspace=0.25)

    # 1) Audio
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t_env, env, color="#2F6F9F", lw=1.0, label="RMS Envelope")
    ax0.axhline(getattr(ab, "thr", 0.0), ls="--", color="#E69F00", lw=1.2, label=f"Threshold={getattr(ab, 'thr', 0.0):.4f}")
    ax0.axvline(float(ab.t0), color="tab:green", lw=1.5, label="Start")
    ax0.axvline(float(ab.t1), color="crimson", lw=1.5, label="End")
    ax0.set_xlim(t_env[0] if len(t_env) else 0.0, t_env[-1] if len(t_env) else 1.0)
    ax0.set_ylabel("RMS")
    ax0.set_title(f"Audio | Duration={ (t_env[-1]-t_env[0]) if len(t_env) else 0:.3f}s | {len(a_wins)} windows")
    # 音频窗底色
    y0, y1 = ax0.get_ylim()
    _span_windows(ax0, _audio_win_seconds(a_wins), (y0, y1), color="#6BA5FF", alpha=0.18)
    ax0.legend(loc="upper right", frameon=False)

    # 2) Video
    ax1 = fig.add_subplot(gs[1, :])
    # 视频窗口底色（紫）
    y0, y1 = (np.min(v_series) if len(v_series) else 0.0, np.max(v_series) if len(v_series) else 1.0)
    if y0 == y1:
        y1 = y0 + 1.0
    _span_windows(ax1, _video_win_seconds(v_wins, fps_nominal), (y0, y1), color="#6F3FBF", alpha=0.20)
    # 主段上加一层更深的底色
    _span_windows(ax1, [(v_main_s, v_main_e)], (y0, y1), color="#6F3FBF", alpha=0.28)
    ax1.plot(t_video, v_series, color="#E67E22", lw=1.0, label="Video Activity")
    ax1.axvline(v_main_s, color="tab:green", lw=1.5, label="Start")
    ax1.axvline(v_main_e, color="crimson",  lw=1.5, label="End")

    # 标题包含漂移/中心偏差/相关/可选残差
    title = f"Video | FPS: {float(fps_nominal):.2f}→{float(fps_eff):.2f} (drift={drift_pct:.2f}%), dev={center_dev_med:.3f}s, corr={corr_med:.3f}"
    if resid_ms is not None:
        title += f", resid={float(resid_ms):.0f}ms"
    ax1.set_title(title)
    ax1.set_ylabel("Activity")
    ax1.set_xlim(t_video[0] if len(t_video) else 0.0, t_video[-1] if len(t_video) else 1.0)
    ax1.legend(loc="upper right", frameon=False)

    # 3a) Score histogram
    ax2 = fig.add_subplot(gs[2, 0])
    if scores.size:
        _safe_hist(ax2, scores, thr=float(overlap_thr), bins=30)
        ax2.set_xlabel("Score")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Pair Score Distribution | Pass={pass_rate:.1f}%")
        # x 轴更紧凑
        xmin = min(0.0, float(np.min(scores)) - 0.05)
        xmax = max(1.05, float(np.max(scores)) + 0.05)
        ax2.set_xlim(xmin, xmax)
    else:
        ax2.text(0.5, 0.5, "No pairs", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()

    # 3b) Score per audio window
    ax3 = fig.add_subplot(gs[2, 1])
    if scores.size:
        xs = np.arange(len(scores))
        ax3.scatter(xs, scores, s=8, alpha=0.8, color="#2C7A7B", edgecolors="none")
        ax3.axhline(float(overlap_thr), ls="--", color="crimson", lw=1.5, label=f"Threshold={float(overlap_thr):.2f}")
        # 通过的点上画一层半透明绿
        if keeps.size == len(xs):
            ok = keeps.astype(bool)
            ax3.scatter(xs[ok], scores[ok], s=10, alpha=0.9, color="#2ECC71", edgecolors="none")
        ax3.set_xlabel("Audio Window Index")
        ax3.set_ylabel("Score")
        ax3.set_ylim(0.0, 1.05)
        ax3.set_title("Score per Audio Window")
        ax3.legend(loc="lower right", frameon=False)
    else:
        ax3.text(0.5, 0.5, "No pairs", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_axis_off()

    for ax in [ax0, ax1, ax2, ax3]:
        ax.grid(True, ls=":", lw=0.6, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
