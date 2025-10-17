#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pairing_utils.py
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

@dataclass
class Mapping:
    fps_eff: float
    a: float
    t0_audio: float
    f0_video: int

def t_audio_to_frame(m, t_audio: float) -> float:
    return m.a * (t_audio - m.t0_audio) + m.f0_video

def frame_to_t_audio(m, f: float) -> float:
    return (f - m.f0_video) / max(m.a,1e-9) + m.t0_audio

def _overlap_components(m, a_win: Tuple[float,float], v_win_frames: Tuple[int,int]):
    ta, tb = a_win
    vf0, vf1 = v_win_frames
    tv0 = frame_to_t_audio(m, vf0)
    tv1 = frame_to_t_audio(m, vf1 + 1)   # exclusive
    if tv1 < tv0: tv0, tv1 = tv1, tv0
    inter = max(0.0, min(tb, tv1) - max(ta, tv0))
    a_len = max(1e-9, tb - ta)
    v_len = max(1e-9, tv1 - tv0)
    iou  = inter / max(1e-9, a_len + v_len - inter)
    cov_a = inter / a_len        # 音频窗被覆盖比例（上限=1）
    cov_v = inter / v_len        # 视频窗被覆盖比例
    return iou, cov_a, cov_v

def pair_windows_audio_to_video(a_wins, v_wins, mapper,
                                strategy: str="coverage_audio", thr: float=0.6):
    """
    strategy:
      - 'temporal_iou'      : IoU(时域) —— 原来的做法（不推荐用于 0.2s vs 2.1s）
      - 'coverage_audio'    : inter / len(audio_win)（推荐；满覆盖=1）
      - 'bi_coverage'       : min( inter/len(audio), inter/len(video) )
    """
    rows, pass_cnt = [], 0
    for ai,(ta,tb) in enumerate(a_wins):
        best = (-1.0, -1, None, 0.0, 0.0)  # (score, vi, (vf0,vf1), iou, cov_v)
        for vi,(vf0,vf1) in enumerate(v_wins):
            iou, cov_a, cov_v = _overlap_components(mapper, (ta,tb), (vf0,vf1))
            if strategy == "temporal_iou":
                score = iou
            elif strategy == "bi_coverage":
                score = min(cov_a, cov_v)
            else:  # coverage_audio
                score = cov_a
            if score > best[0]:
                best = (score, vi, (vf0,vf1), iou, cov_v)
        score, best_vi, (bf0,bf1), iou, cov_v = best
        keep = int(score >= thr)
        pass_cnt += keep
        rows.append({
            "a_idx": ai, "a_start_s": round(float(ta),6), "a_end_s": round(float(tb),6),
            "v_idx": int(best_vi), "v_start_frame": int(bf0), "v_end_frame": int(bf1),
            "score": round(float(score),6),
            "temporal_iou": round(float(iou),6),
            "coverage_audio": round(float(score),6) if strategy=="coverage_audio" else "",
            "coverage_video": round(float(cov_v),6),
            "keep": keep, "strategy": strategy
        })
    pass_rate = float(pass_cnt) / max(1, len(a_wins))
    return rows, pass_rate

def center_deviation_stats(rows: List[Dict], mapper: Mapping) -> Dict[str, float]:
    vals = []
    for r in rows:
        if r.get("v_idx", -1) < 0:
            continue
        ta = float(r["a_start_s"]); tb = float(r["a_end_s"])
        vf0 = int(r["v_start_frame"]); vf1 = int(r["v_end_frame"])
        # 把视频窗映射到“音频时间轴”
        tv0 = frame_to_t_audio(mapper, vf0)
        tv1 = frame_to_t_audio(mapper, vf1 + 1)  # 右开
        if tv1 < tv0: tv0, tv1 = tv1, tv0
        # 交集区间
        lo = max(ta, tv0); hi = min(tb, tv1)
        ca = 0.5 * (ta + tb)
        if hi > lo:  # 有交集：用“交集中心”
            cv = 0.5 * (lo + hi)
        else:       # 无交集：退化成两段中心
            cv = 0.5 * (tv0 + tv1)
        vals.append(abs(cv - ca))
    if not vals:
        return {"center_dev_med": 1e9, "center_dev_p95": 1e9}
    arr = np.array(vals, dtype=np.float32)
    return {
        "center_dev_med": float(np.median(arr)),
        "center_dev_p95": float(np.percentile(arr, 95)),
    }

