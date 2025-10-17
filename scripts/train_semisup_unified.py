#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/train_semisup_unified.py

- 从 YAML 读取配置（数据/模型/训练/SSL）
- CSV 数据集：支持有标注(train/val)与无标注(unlabeled)
- 模型：EnhancedAVDetector（融合 concat / coattention / early-fusion / 对齐 shift/none/softdtw）
- 半监督：Mean-Teacher + FixMatch + KL 一致性（Teacher EMA）
- AMP：torch.amp.autocast(device_type), GradScaler(device_type)
- 对齐调用的安全护栏：只有 align_mode=="shift" 才调用对齐接口
- 详细日志：维度/损失/伪标签命中率
"""

import os
import io
import csv
import glob
import math
import yaml
import time
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 终端输出 UTF-8（Windows 中文日志防乱码）
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from tqdm import tqdm

# 可选依赖（有就用）
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    import torchaudio
    HAS_TA = True
except Exception:
    HAS_TA = False

try:
    import librosa, soundfile as sf  # noqa: F401
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

# 项目内模块
ROOT = Path(__file__).resolve().parents[1]  # 项目根（.../avtop-top-tier）
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.avtop.models.enhanced_detector import EnhancedAVDetector

# 可选：你仓库里的完整损失。如果没有，脚本会降级到 BCE/CE
try:
    from src.avtop.losses.gram_contrastive import CompleteLossFunction as _CompleteLoss
    HAS_COMPLETE_LOSS = True
except Exception:
    HAS_COMPLETE_LOSS = False


# =========================
# 数据集：CSV (视频+音频)
# =========================
class AVFromCSV(Dataset):
    """
    CSV 字段支持：
      - video_path: 视频文件(.mp4/.avi) 或 帧目录（jpg/png）
      - audio_path: 可选；无则用静音
      - label: 单标签 0/1/... 或 类名
      - labels: 多标签 "Good|Defect" 或 "0|1"
      - audio_offset: 可选，秒
    输出：
      - 有标注： (video:(C,T,H,W), audio:(1,L) 或 (1,n_mels,T), y:(num_classes,))
      - 无标注： (video, audio)
    """
    def __init__(self,
                 csv_path: str,
                 data_root: Optional[str],
                 num_classes: int,
                 class_names: List[str],
                 video_cfg: Dict[str, Any],
                 audio_cfg: Dict[str, Any],
                 is_unlabeled: bool = False):
        super().__init__()
        self.rows = self._read_csv(csv_path)
        self.root = data_root
        self.num_classes = int(num_classes)
        self.class_names = list(class_names)
        # video
        self.img_size = int(video_cfg.get("img_size", 224))
        self.num_frames = int(video_cfg.get("num_frames", 8))
        # audio
        self.sample_rate = int(audio_cfg.get("sample_rate", 16000))
        self.duration_sec = float(audio_cfg.get("duration_sec", 2.56))
        self.n_mels = int(audio_cfg.get("n_mels", 128))
        self.use_waveform = bool(audio_cfg.get("use_waveform", True))
        self.normalize_audio = bool(audio_cfg.get("normalize", True))

        self.is_unlabeled = bool(is_unlabeled)
        print(f"[Data] load CSV: {csv_path} rows={len(self.rows)} unlabeled={self.is_unlabeled}")

    @staticmethod
    def _read_csv(path: str) -> List[Dict[str, str]]:
        rows = []
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
        if len(rows) == 0:
            raise RuntimeError(f"[Data] empty CSV: {path}")
        return rows

    def _abs(self, p: Optional[str]) -> Optional[str]:
        if not p or p == "": return None
        if os.path.isabs(p): return p
        return os.path.normpath(os.path.join(self.root or "", p))

    def __len__(self): return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        vpath = self._abs(r.get("video_path") or r.get("video") or "")
        apath = self._abs(r.get("audio_path") or r.get("audio") or "")
        if not vpath or not os.path.exists(vpath):
            raise FileNotFoundError(f"[Data] video_path not exists: {vpath}")

        video = self._read_video(vpath, self.num_frames, self.img_size)  # (C,T,H,W)

        # audio
        if apath and os.path.exists(apath):
            audio = self._read_audio(apath, self.sample_rate, self.duration_sec, float(r.get("audio_offset", 0) or 0.0))
            if not self.use_waveform:
                audio = self._to_mel(audio, self.sample_rate, self.n_mels).unsqueeze(0)  # (1,n_mels,T)
        else:
            L = int(round(self.sample_rate * self.duration_sec))
            audio = torch.zeros(1, L)

        if self.normalize_audio and self.use_waveform:
            eps = 1e-8
            std = audio.std(dim=1, keepdim=True).clamp_min(eps)
            audio = (audio - audio.mean(dim=1, keepdim=True)) / std

        if self.is_unlabeled:
            return video, audio

        y, _is_multi = self._parse_label(r, self.num_classes, self.class_names)
        return video, audio, y

    @staticmethod
    def _read_video(video_fp: str, num_frames: int, img_size: int) -> torch.Tensor:
        import numpy as np
        if os.path.isdir(video_fp):
            imgs = sorted(glob.glob(os.path.join(video_fp, "*.jpg")) + glob.glob(os.path.join(video_fp, "*.png")))
            if len(imgs) == 0:
                raise RuntimeError(f"[Data] no frames in dir: {video_fp}")
            idxs = torch.linspace(0, len(imgs)-1, steps=num_frames).round().long().tolist()
            frames = []
            from PIL import Image
            for i in idxs:
                im = Image.open(imgs[i]).convert("RGB").resize((img_size, img_size))
                arr = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
                frames.append(arr)
            return torch.stack(frames, dim=1)
        if not HAS_CV2:
            raise RuntimeError("[Data] need opencv-python for video files; or convert to frame folders.")
        cap = cv2.VideoCapture(video_fp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idxs = torch.linspace(0, max(total - 1, 0), steps=num_frames).round().long().tolist()
        frames = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros(3, img_size, img_size))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(t)
        cap.release()
        return torch.stack(frames, dim=1)

    @staticmethod
    def _read_audio(path: str, sr: int, duration_sec: float, offset: float) -> torch.Tensor:
        target_len = int(round(sr * duration_sec))
        if HAS_TA:
            wav, s = torchaudio.load(path)  # (C,L)
            if s != sr:
                wav = torchaudio.functional.resample(wav, s, sr)
            if wav.size(0) > 1: wav = wav.mean(dim=0, keepdim=True)
            off = int(round(offset * sr))
            if wav.size(1) - off >= target_len:
                wav = wav[:, off:off + target_len]
            else:
                seg = wav[:, off:] if off < wav.size(1) else torch.zeros(1, 0)
                pad = target_len - seg.size(1)
                wav = torch.cat([seg, torch.zeros(1, pad)], dim=1)
            return wav
        elif HAS_LIBROSA:
            y, s = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration_sec)
            y = torch.from_numpy(y).view(1, -1)
            L = y.size(1)
            if L < target_len:
                y = torch.cat([y, torch.zeros(1, target_len - L)], dim=1)
            elif L > target_len:
                y = y[:, :target_len]
            return y
        else:
            raise RuntimeError("[Data] need torchaudio or librosa to read audio")

    @staticmethod
    def _to_mel(wav: torch.Tensor, sr: int, n_mels: int) -> torch.Tensor:
        if HAS_TA:
            mel = torchaudio.transforms.MelSpectrogram(sr, n_mels=n_mels)(wav).squeeze(0)
            return mel
        elif HAS_LIBROSA:
            import numpy as np
            m = librosa.feature.melspectrogram(y=wav.squeeze(0).numpy(), sr=sr, n_mels=n_mels)
            return torch.from_numpy(m)
        else:
            raise RuntimeError("[Data] need torchaudio/librosa for mel")

    @staticmethod
    def _parse_label(row: Dict[str, str], num_classes: int, class_names: List[str]) -> Tuple[torch.Tensor, bool]:
        if "labels" in row and row["labels"]:
            raw = row["labels"].replace(",", "|")
            idxs: List[int] = []
            for p in [t.strip() for t in raw.split("|") if t.strip()]:
                if p.isdigit(): idxs.append(int(p))
                else:
                    try: idxs.append(class_names.index(p))
                    except ValueError: raise RuntimeError(f"[Data] label '{p}' not in {class_names}")
            y = torch.zeros(num_classes, dtype=torch.float32)
            for k in idxs:
                if 0 <= k < num_classes: y[k] = 1.0
            return y, True
        if "label" in row and row["label"] != "":
            v = row["label"]
            try: k = int(v)
            except Exception:
                try: k = class_names.index(v)
                except ValueError: raise RuntimeError(f"[Data] cannot parse label '{v}'")
            y = torch.zeros(num_classes, dtype=torch.float32)
            if 0 <= k < num_classes: y[k] = 1.0
            return y, False
        return torch.zeros(num_classes, dtype=torch.float32), False


def collate_av(batch):
    """pad 音频到 batch 内最大长度；视频固定 (C,T,H,W)"""
    has_label = (len(batch[0]) == 3)
    vids, auds, ys = [], [], []
    maxL = 0
    for item in batch:
        if has_label:
            v, a, y = item
            ys.append(y)
        else:
            v, a = item
        vids.append(v); auds.append(a); maxL = max(maxL, a.size(1))
    pad_a = []
    for a in auds:
        if a.size(1) == maxL: pad_a.append(a)
        elif a.size(1) < maxL:
            pad_a.append(torch.cat([a, torch.zeros(1, maxL - a.size(1), dtype=a.dtype)], dim=1))
        else:
            pad_a.append(a[:, :maxL])
    V = torch.stack(vids, dim=0)  # (B,C,T,H,W)
    A = torch.stack(pad_a, dim=0) # (B,1,L)
    if has_label:
        Y = torch.stack(ys, dim=0)  # (B,num_classes)
        return V, A, Y
    return V, A


def build_loaders(data_cfg: Dict[str, Any], batch_size: int, num_workers: int = 4):
    root = data_cfg.get("root", None)
    ncls = int(data_cfg["num_classes"])
    names = list(data_cfg["class_names"])
    video_cfg = data_cfg.get("video", {})
    audio_cfg = data_cfg.get("audio", {})

    ds_tr = AVFromCSV(data_cfg["train_csv"], root, ncls, names, video_cfg, audio_cfg, is_unlabeled=False)
    l_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(), drop_last=True, collate_fn=collate_av)

    u_loader = None
    if data_cfg.get("unlabeled_csv", None):
        ds_ul = AVFromCSV(data_cfg["unlabeled_csv"], root, ncls, names, video_cfg, audio_cfg, is_unlabeled=True)
        u_loader = DataLoader(ds_ul, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=torch.cuda.is_available(), drop_last=True, collate_fn=collate_av)

    v_loader = None
    if data_cfg.get("val_csv", None):
        ds_val = AVFromCSV(data_cfg["val_csv"], root, ncls, names, video_cfg, audio_cfg, is_unlabeled=False)
        v_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=torch.cuda.is_available(), drop_last=False, collate_fn=collate_av)

    print(f"[Data] train={len(ds_tr)} unlabeled={len(ds_ul) if u_loader else 0} val={len(ds_val) if v_loader else 0}")
    return l_loader, u_loader, v_loader


# =========================
# 工具：EMA / KL / 简单增广
# =========================
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: p.detach().clone() for k, p in model.state_dict().items() if p.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if k in self.shadow and p.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)


def kl_with_temperature(student_logits, teacher_logits, T=2.0):
    p = torch.log_softmax(student_logits / T, dim=-1)
    q = torch.softmax(teacher_logits / T, dim=-1)
    return nn.KLDivLoss(reduction="batchmean")(p, q) * (T * T)


def weak_aug(V, A):  # 预留：可加颜色抖动/时序裁剪等
    return V, A

def strong_aug(V, A):
    return V, A


# =========================
# 训练器
# =========================
class Trainer:
    def __init__(self, cfg: Dict[str, Any], output_dir: str):
        self.cfg = cfg
        self.out = Path(output_dir); self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "checkpoints").mkdir(exist_ok=True)

        # 压制 flash-attention 警告（可选）
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)
        except Exception:
            pass

        # 设备 & AMP
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.amp_device_type = "cuda"
            print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.amp_device_type = "mps"
            print("✅ 使用Apple MPS")
        else:
            self.device = torch.device("cpu")
            self.amp_device_type = "cpu"
            print("⚠️ 使用CPU")
        self.use_amp = bool(self.cfg.get("hardware", {}).get("mixed_precision", True) and self.amp_device_type != "cpu")
        self.scaler = GradScaler(self.amp_device_type, enabled=self.use_amp)

        # 随机种子
        seed = int(self.cfg.get("seed", 42))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"✅ 随机种子: {seed}")

        # 模型：Student / Teacher
        print("\n=== 模型构建 ===")
        self.model_s = EnhancedAVDetector(self.cfg.get("model", self.cfg)).to(self.device)
        self.model_t = EnhancedAVDetector(self.cfg.get("model", self.cfg)).to(self.device)
        self.model_t.load_state_dict(self.model_s.state_dict(), strict=False)
        for p in self.model_t.parameters(): p.requires_grad = False

        # 读取模型上的对齐模式作为“单一事实来源”
        self.align_mode = getattr(self.model_s, "align_mode", self.cfg.get("model", {}).get("align_mode", "none"))
        print(f"[DEBUG] Trainer.align_mode = {self.align_mode}")

        self.ema = EMA(self.model_s, decay=float(self.cfg.get("ssl", {}).get("ema_decay", 0.999)))

        # 数据
        print("\n=== 构建数据加载器 ===")
        bs = int(self.cfg.get("training", {}).get("batch_size", 8))
        nw = int(self.cfg.get("hardware", {}).get("num_workers", 4))
        self.l_loader, self.u_loader, self.v_loader = build_loaders(self.cfg["data"], batch_size=bs, num_workers=nw)

        # 监督损失
        print("\n=== 构建损失函数 ===")
        if HAS_COMPLETE_LOSS:
            loss_cfg = self.cfg.get("loss", {})
            self.sup_crit = _CompleteLoss(
                num_classes=self.cfg["data"]["num_classes"],
                lambda_contrastive=loss_cfg.get('lambda_contrastive', 0.0),
                lambda_kd=loss_cfg.get('lambda_kd', 0.0),
                lambda_consistency=loss_cfg.get('lambda_consistency', 0.0),
                temperature=loss_cfg.get('temperature', 0.07),
                kd_temperature=loss_cfg.get('kd_temperature', 4.0)
            ).to(self.device)
            print("✅ 使用仓库内 CompleteLossFunction（仅作监督主损）")
        else:
            self.sup_crit = None
            print("⚠️ 未检测到 CompleteLossFunction，改用 BCE/CE 监督损失")

        # 优化器/调度器
        print("\n=== 优化器/调度器 ===")
        tr = self.cfg.get("training", {})
        lr = float(tr.get("learning_rate", 1e-4))
        wd = float(tr.get("weight_decay", 1e-4))
        self.opt = optim.AdamW(self.model_s.parameters(), lr=lr, weight_decay=wd)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=int(tr.get("num_epochs", 10)), eta_min=lr*0.01)
        print(f"Optimizer=AdamW lr={lr} wd={wd}  Scheduler=Cosine T_max={tr.get('num_epochs',10)}")

        # 半监督
        ssl = self.cfg.get("ssl", {})
        self.fixmatch_thresh = float(ssl.get("fixmatch_thresh", 0.85))
        self.consistency_T = float(ssl.get("consistency_T", 2.0))
        self.unsup_weight_max = float(ssl.get("unsup_weight_max", 1.0))

        # 训练状态
        self.grad_accum = int(self.cfg.get("hardware", {}).get("gradient_accumulation_steps", 1))

    def _ramp_up(self, step: int, total_steps: int):
        if total_steps <= 0: return 1.0
        t = min(1.0, step / max(1, total_steps))
        return 0.5 - 0.5 * math.cos(math.pi * t)

    def train(self):
        epochs = int(self.cfg.get("training", {}).get("num_epochs", 10))
        log_every = int(self.cfg.get("training", {}).get("log_every", 10))

        u_iter = iter(self.u_loader) if self.u_loader else None
        total_steps = epochs * len(self.l_loader)

        print("\n=== 开始训练 ===")
        step = 0
        for ep in range(1, epochs + 1):
            self.model_s.train()
            ep_loss_sup = ep_loss_unsup = ep_loss_cons = ep_loss_fix = 0.0

            pbar = tqdm(self.l_loader, ncols=100, desc=f"Epoch {ep}/{epochs}")
            for batch_idx, batch in enumerate(pbar):
                if len(batch) != 3:
                    raise RuntimeError("[Train] labeled loader must return (video, audio, y)")
                v_l, a_l, y_l = batch
                v_l, a_l, y_l = v_l.to(self.device), a_l.to(self.device), y_l.to(self.device)

                # ===== 监督 =====
                with autocast(self.amp_device_type, enabled=self.use_amp):
                    out_l = self.model_s(v_l, a_l, return_aux=True)
                    s_logits_l = out_l["clip_logits"]
                    if y_l.ndim == 2 and y_l.dtype in (torch.float32, torch.float16):
                        loss_sup = nn.BCEWithLogitsLoss()(s_logits_l, y_l)
                    else:
                        y_idx = torch.argmax(y_l, dim=1)
                        loss_sup = nn.CrossEntropyLoss()(s_logits_l, y_idx)

                # ===== 无监督（若有） =====
                loss_unsup = loss_cons = loss_fix = torch.tensor(0.0, device=self.device)
                shift = None
                if self.u_loader is not None:
                    try:
                        u_batch = next(u_iter)
                    except StopIteration:
                        u_iter = iter(self.u_loader); u_batch = next(u_iter)
                    v_u, a_u = u_batch
                    v_u, a_u = v_u.to(self.device), a_u.to(self.device)

                    # 弱增强给 Teacher
                    v_w, a_w = weak_aug(v_u, a_u)
                    with torch.no_grad():
                        # 仅 shift 时才预测+缓存对齐
                        if getattr(self.model_t, "align_mode", "none") == "shift" and \
                           hasattr(self.model_t, "align_predict_and_cache"):
                            _, _, shift = self.model_t.align_predict_and_cache(v_w, a_w, return_shift=True)
                            if shift is not None and hasattr(self.model_s, "set_cached_shift"):
                                self.model_s.set_cached_shift(shift)

                        out_t = self.model_t(v_w, a_w)
                        t_logits = out_t["clip_logits"]
                        t_prob = torch.sigmoid(t_logits)
                        mask = (t_prob >= self.fixmatch_thresh).float()

                    # 强增强给 Student
                    v_s, a_s = strong_aug(v_u, a_u)
                    if getattr(self.model_s, "align_mode", "none") == "shift" and \
                       shift is not None and hasattr(self.model_s, "align_use_cached"):
                        v_s, a_s = self.model_s.align_use_cached(v_s, a_s)

                    out_s = self.model_s(v_s, a_s)
                    s_logits = out_s["clip_logits"]

                    with autocast(self.amp_device_type, enabled=self.use_amp):
                        # KL 一致性
                        loss_cons = kl_with_temperature(s_logits, t_logits, T=self.consistency_T)
                        # FixMatch（BCE + mask）
                        pseudo = (t_prob >= self.fixmatch_thresh).float()
                        loss_fix = (nn.BCEWithLogitsLoss(reduction="none")(s_logits, pseudo) * mask).mean()
                        wu = self._ramp_up(step, total_steps) * self.unsup_weight_max
                        loss_unsup = wu * (0.5 * loss_cons + 0.5 * loss_fix)

                # ===== 合成 / 反向 =====
                with autocast(self.amp_device_type, enabled=self.use_amp):
                    loss = loss_sup + loss_unsup

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model_s.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                # EMA
                with torch.no_grad():
                    self.ema.update(self.model_s)
                    self.ema.copy_to(self.model_t)

                # 日志
                ep_loss_sup += float(loss_sup.detach().item())
                ep_loss_unsup += float(loss_unsup.detach().item())
                ep_loss_cons += float(loss_cons.detach().item())
                ep_loss_fix += float(loss_fix.detach().item())

                if step % log_every == 0:
                    pos_ratio = float(mask.mean().detach().item()) if self.u_loader is not None else 0.0
                    print(f"[Step {step:06d}] sup={float(loss_sup):.4f} unsup={float(loss_unsup):.4f} "
                          f"cons={float(loss_cons):.4f} fix={float(loss_fix):.4f} "
                          f"wu={self._ramp_up(step, total_steps):.3f} mask_pos={pos_ratio:.3f}")
                step += 1
                pbar.set_postfix({"sup": f"{loss_sup.item():.3f}", "unsup": f"{loss_unsup.item():.3f}"})

            self.sched.step()

            # 验证
            val = self.validate()
            print(f"[Epoch {ep}/{epochs}] "
                  f"sup={ep_loss_sup/len(self.l_loader):.4f} "
                  f"unsup={ep_loss_unsup/max(1,len(self.l_loader)):.4f} "
                  f"val_loss={val['loss']:.4f} acc={val['acc']:.4f}")

            ckpt = {"epoch": ep, "model_s": self.model_s.state_dict(),
                    "opt": self.opt.state_dict(), "cfg": self.cfg}
            torch.save(ckpt, self.out / "checkpoints" / f"epoch{ep}.pth")

        print("✅ 训练完成！")

    @torch.no_grad()
    def validate(self):
        if self.v_loader is None:
            return {"loss": 0.0, "acc": 0.0}
        self.model_s.eval()
        tot = 0.0; n = 0
        all_pred = []; all_true = []
        for batch in tqdm(self.v_loader, ncols=100, desc="Validation"):
            v, a, y = batch
            v, a, y = v.to(self.device), a.to(self.device), y.to(self.device)
            out = self.model_s(v, a)
            logits = out["clip_logits"]
            if y.ndim == 2 and y.dtype in (torch.float32, torch.float16):
                loss = nn.BCEWithLogitsLoss()(logits, y)
                pred = torch.argmax(logits, dim=1)
                true = torch.argmax(y, dim=1)
            else:
                true = y.long().view(-1)
                loss = nn.CrossEntropyLoss()(logits, true)
                pred = torch.argmax(logits, dim=1)
            tot += float(loss.item()); n += 1
            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(true.detach().cpu().numpy())
        all_pred = np.concatenate(all_pred); all_true = np.concatenate(all_true)
        acc = (all_pred == all_true).mean()
        return {"loss": tot/max(1, n), "acc": float(acc)}


# =========================
# 主入口
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML 配置路径")
    ap.add_argument("--output_dir", default="outputs/csv_semisup", help="输出目录")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    # 打印一次 align_mode 供排查
    am = cfg.get("model", {}).get("align_mode", cfg.get("align_mode", "none"))
    print(f"[DEBUG] cfg.model.align_mode = {am}")

    trainer = Trainer(cfg, args.output_dir)
    trainer.train()


if __name__ == "__main__":
    main()
