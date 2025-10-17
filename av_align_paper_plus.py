#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
av_align_paper_plus.py (merged update)
- 论文式训练前对齐：音频/视频主段起止（最大连通段 + 迟滞 + 以“秒”控制的闭运算）
- 映射：以音频时长为真值的线性映射；可选进行 Huber 稳健拟合 refine（基于窗中心）
- 窗口生成：audio(0.2s/0.1s)，video(64帧/步1)
"""
from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import math
import cv2

# -------------------- dataclasses --------------------
@dataclass
class AudioBounds:
    t0: float
    t1: float
    sr: int
    thr: float

@dataclass
class VideoBounds:
    f0: int
    f1: int
    fps_nominal: float
    thr: float

@dataclass
class Mapping:
    fps_eff: float
    a: float
    t0_audio: float
    f0_video: int

# -------------------- helpers --------------------
def load_audio(path: str, target_sr: Optional[int]=None):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        t_old = np.linspace(0, len(y)/sr, len(y), endpoint=False)
        t_new = np.linspace(0, len(y)/sr, int(round(len(y)*target_sr/sr)), endpoint=False)
        y = np.interp(t_new, t_old, y).astype(np.float32)
        sr = target_sr
    m = np.max(np.abs(y)) + 1e-8
    if m > 0: y = y / m
    return y.astype(np.float32), int(sr)

def audio_envelope_rms(y: np.ndarray, sr: int, frame_ms: float=25.0, hop_ms: float=10.0):
    L = max(1, int(round(frame_ms * 1e-3 * sr)))
    H = max(1, int(round(hop_ms * 1e-3 * sr)))
    n = 1 + max(0, (len(y)-L)//H)
    env = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = i*H; e = min(len(y), s+L)
        seg = y[s:e]
        env[i] = math.sqrt(float(np.mean(seg*seg) + 1e-12))
    times = (np.arange(n)*H + L/2) / float(sr)
    return env, times

def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x-med)) + 1e-8)

def _pick_main_segment(on: np.ndarray, score: np.ndarray=None):
    """返回连通段中“长度最大”或“得分最大”的那一段索引 [s,e]（包含端点）"""
    x = on.astype(np.int32)
    diff = np.diff(np.r_[0, x, 0])
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1
    if len(starts) == 0:
        return 0, len(on)-1
    if score is None:
        lengths = ends - starts + 1
        k = int(np.argmax(lengths))
    else:
        seg_scores = [float(score[s:e+1].sum()) for s,e in zip(starts,ends)]
        k = int(np.argmax(seg_scores))
    return int(starts[k]), int(ends[k])

# -------------------- bounds detection --------------------
def detect_audio_bounds(y: np.ndarray, sr: int,
                        frame_ms: float=25.0, hop_ms: float=10.0,
                        warmup_sec: float=1.0, k: float=6.0,
                        min_on_s: float=0.20, close_max_gap_s: float=0.15) -> AudioBounds:
    """主段起止：阈值 -> 闭运算（以秒限定核长）-> 只保留主段"""
    env, t = audio_envelope_rms(y, sr, frame_ms, hop_ms)
    hop_s = hop_ms * 1e-3
    warm_idx = int(max(1, round(warmup_sec / hop_s)))
    base = env[:min(len(env), warm_idx)] if len(env)>0 else env
    base_med = np.median(base) if len(base)>0 else np.median(env)
    base_mad = mad(base if len(base)>0 else env)
    thr = float(base_med + k*base_mad)

    on = (env >= thr).astype(np.uint8)
    klen = max(1, int(round(close_max_gap_s / max(hop_s,1e-6))))
    on = cv2.morphologyEx(on.reshape(-1,1), cv2.MORPH_CLOSE, np.ones((klen,1),np.uint8)).ravel()
    s, e = _pick_main_segment(on, score=env)
    t0 = float(t[max(0, s-1)])
    t1 = float(t[min(len(t)-1, e+1)])
    if (t1 - t0) < min_on_s:
        pad = 0.5*(min_on_s - (t1 - t0))
        t0 = max(float(t[0]), t0 - pad); t1 = min(float(t[-1]), t1 + pad)
    return AudioBounds(t0=t0, t1=t1, sr=sr, thr=thr)

def detect_video_bounds_series(v_series: np.ndarray, fps: float,
                               baseline_sec: float=5.0, k_on: float=7.5, k_off: float=6.0,
                               min_on_s: float=0.50, close_max_gap_s: float=0.20) -> VideoBounds:
    """视频主段：迟滞阈值 + 闭运算（秒）+ 主段选择"""
    n = len(v_series)
    base = v_series[:min(n, int(round(baseline_sec*fps)))]
    med = np.median(base) if len(base)>0 else np.median(v_series)
    madv = mad(base if len(base)>0 else v_series)
    on_thr  = float(med + k_on*madv)
    off_thr = float(med + k_off*madv)
    on = (v_series >= on_thr).astype(np.uint8)
    klen = max(1, int(round(close_max_gap_s * fps)))
    on = cv2.morphologyEx(on.reshape(-1,1), cv2.MORPH_CLOSE, np.ones((klen,1),np.uint8)).ravel()
    s, e = _pick_main_segment(on, score=v_series)
    # 用 off_thr 做边界外扩
    while s>0   and v_series[s-1] >= off_thr: s -= 1
    while e<n-1 and v_series[e+1] >= off_thr: e += 1
    if (e - s + 1)/max(1e-6,fps) < min_on_s:
        need = int(round(min_on_s*fps)); e = min(n-1, s+need-1)
    return VideoBounds(f0=int(s), f1=int(e), fps_nominal=float(fps), thr=on_thr)

# -------------------- mapping --------------------
def build_mapping(ab: AudioBounds, vb: VideoBounds) -> Mapping:
    Ta = max(ab.t1 - ab.t0, 1e-9)
    Nv = max(1, vb.f1 - vb.f0 + 1)
    fps_eff = float(Nv / Ta)
    a = fps_eff
    return Mapping(fps_eff=fps_eff, a=a, t0_audio=ab.t0, f0_video=vb.f0)

def t_audio_to_frame(mapper: Mapping, t_audio: float) -> float:
    return mapper.a * (t_audio - mapper.t0_audio) + mapper.f0_video

def frame_to_t_audio(mapper: Mapping, f: float) -> float:
    return (f - mapper.f0_video) / max(mapper.a,1e-9) + mapper.t0_audio

def refine_mapping_with_windows(a_wins: List[Tuple[float,float]],
                                v_wins: List[Tuple[int,int]],
                                mapper: Mapping, max_iter: int = 5) -> Mapping:
    """用窗中心做 Huber 稳健线性拟合，得到更稳的 a,b"""
    if not a_wins or not v_wins:
        return mapper
    ta = np.array([0.5*(s+e) for s,e in a_wins], dtype=np.float64)             # sec
    vc = np.array([0.5*(s+e) for s,e in v_wins], dtype=np.float64)             # frame
    # 初始：把音频中心映射到帧，再找最近视频窗中心
    f_pred = np.array([t_audio_to_frame(mapper, t) for t in ta], dtype=np.float64)
    # 最近中心索引
    idx = np.clip(np.searchsorted(vc, f_pred), 0, len(vc)-1)
    f_near = vc[idx].reshape(-1,1)
    # 拟合 f_near ≈ a*(t - t0) + b
    t0 = mapper.t0_audio
    x = (ta - t0).reshape(-1,1); y = f_near
    a, b = float(mapper.a), float(mapper.f0_video)
    for _ in range(max_iter):
        r = (a*x + b - y).ravel()
        s = np.median(np.abs(r)) + 1e-6
        w = 1.0 / np.maximum(1.0, np.abs(r)/(1.345*s))  # Huber
        X = np.c_[x, np.ones_like(x)]
        # 带权最小二乘
        WX = X * w[:,None]; Wy = y * w[:,None]
        beta, *_ = np.linalg.lstsq(WX, Wy, rcond=None)
        a, b = float(beta[0,0]), float(beta[1,0])
    return Mapping(fps_eff=a, a=a, t0_audio=t0, f0_video=int(round(b)))

def refine_mapping_offset_only(a_wins, fps_nominal: float, t0_audio: float, b_init: int) -> Mapping:
    """固定斜率=名义FPS，只估计截距b（稳健中位数），返回新的 Mapping"""
    ta = np.array([0.5*(s+e) for s,e in a_wins], dtype=np.float64)
    if ta.size == 0:
        return Mapping(fps_eff=fps_nominal, a=fps_nominal, t0_audio=t0_audio, f0_video=b_init)
    # 估计 b = median( round(fps*(t - t0)) - fps*(t - t0) ) + b_init
    x = fps_nominal * (ta - t0_audio)
    b_resid = np.median(np.round(x) - x)   # 稳健
    b = float(b_init + b_resid)
    return Mapping(fps_eff=fps_nominal, a=fps_nominal, t0_audio=t0_audio, f0_video=int(round(b)))

def mapping_residual_ms(a_wins, mapper: Mapping) -> float:
    """每个音频窗中心映射到最近整数帧的时间残差(毫秒)的中位数"""
    if not a_wins:
        return 1e9
    ta = np.array([0.5*(s+e) for s,e in a_wins], dtype=np.float64)
    f_pred = mapper.a * (ta - mapper.t0_audio) + mapper.f0_video
    f_near = np.round(f_pred)
    dt = np.abs(f_near - f_pred) / max(mapper.a, 1e-9)  # 秒
    return float(np.median(dt) * 1000.0)



# -------------------- windows --------------------
def gen_audio_windows(ab: AudioBounds, win_s: float, hop_s: float) -> list:
    out = []
    t = ab.t0
    if ab.t1 - ab.t0 <= 0: return out
    while t + win_s <= ab.t1 + 1e-9:
        out.append((t, t+win_s)); t += hop_s
    if not out:
        out.append((ab.t0, min(ab.t1, ab.t0+win_s)))
    return out

def gen_video_windows(vb: VideoBounds, frames_per_win: int=64, stride_frames: int=1) -> list:
    out = []
    f = vb.f0
    while f + frames_per_win - 1 <= vb.f1:
        out.append((f, f+frames_per_win-1)); f += stride_frames
    if not out:
        out.append((vb.f0, min(vb.f1, vb.f0+frames_per_win-1)))
    return out
