#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

@dataclass
class ActivityResult:
    activity: np.ndarray  # shape [T]
    fps: float

class VideoActivityBackend:
    def activity_series(self, video_path: str) -> ActivityResult:
        raise NotImplementedError

# ---------- Basic 后端：亮度+运动能量 ----------
class BasicBackend(VideoActivityBackend):
    def __init__(self, smooth: int = 5):
        self.smooth = int(max(1, smooth))

    def activity_series(self, video_path: str) -> ActivityResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        prev = None
        act = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if prev is None:
                mot = 0.0
            else:
                mot = float(np.mean(np.abs(gray - prev)))
            lum = float(np.mean(gray))
            act.append(0.9 * mot + 0.1 * lum)
            prev = gray
        cap.release()
        a = np.asarray(act, dtype=np.float32)
        if len(a) >= self.smooth:
            k = np.ones(self.smooth, dtype=np.float32) / self.smooth
            a = np.convolve(a, k, mode="same")
        return ActivityResult(activity=a, fps=float(fps))

# ---------- SlowFast（可选） ----------
def _try_load_slowfast_model():
    """
    多路径尝试：
    1) pytorchvideo.models.hub.slowfast_r50（推荐）
    2) torch.hub('facebookresearch/pytorchvideo', 'slowfast_r50')
    都失败则返回 (None, reason)
    """
    # 路径1：pytorchvideo
    try:
        from pytorchvideo.models.hub import slowfast_r50
        import torch
        model = slowfast_r50(pretrained=True).eval()
        return model, None
    except Exception as e1:
        reason1 = str(e1)
    # 路径2：torch.hub
    try:
        import torch
        model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
        model.eval()
        return model, None
    except Exception as e2:
        reason2 = str(e2)
    return None, f"pytorchvideo not available ({reason1}); torch.hub failed ({reason2})"

class SlowFastBackend(VideoActivityBackend):
    def __init__(self):
        self.model, self._reason = _try_load_slowfast_model()

    @property
    def available(self) -> bool:
        return self.model is not None

    def activity_series(self, video_path: str) -> ActivityResult:
        if not self.available:
            # 关键：不在 import 阶段抛异常，让上层能够回退
            raise RuntimeError(f"SlowFast unavailable: {self._reason}")
        # ——此处为了跨平台稳妥，给个简化实现：直接回退到 Basic 的活动序列——
        # 你如果安装好了 pytorchvideo，可以在这里把clip logits 的能量作为活动分数。
        return BasicBackend().activity_series(video_path)
