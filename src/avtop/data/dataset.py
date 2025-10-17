
import os, csv, math
from typing import List, Dict, Optional, Tuple
import torch

class BinaryAVCSVDataset(torch.utils.data.Dataset):
    """
    CSV schema (header required):
      clip_id,video_path,audio_path,label[,seg_starts,seg_ends]
    - label: 0=good, 1=defect
    - seg_starts/seg_ends: optional, comma-separated frame indices for GT segments (same timeline as video frames)
    """
    def __init__(self, csv_path: str, root: str = "", T_v: int = 16, T_a: int = 32, mel_bins: int = 64,
                 sample_rate: int = 16000, size: Tuple[int,int]=(112,112)):
        super().__init__()
        self.rows = []
        self.root = root
        self.Tv, self.Ta, self.mel = T_v, T_a, mel_bins
        self.sample_rate = sample_rate
        self.size = size
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                self.rows.append(row)
        # 如果CSV中没有is_labeled列，默认为True
        if 'is_labeled' not in self.data.columns:
            self.data['is_labeled'] = True
    def __len__(self): return len(self.rows)

    def _load_video_uniform(self, path: str, T: int, size: Tuple[int,int]):
        try:
            import cv2
        except ImportError as e:
            raise ImportError("cv2 not found. Please `pip install opencv-python`.") from e
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or T
        idxs = [int(i) for i in torch.linspace(0, max(total-1,0), steps=T)]
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                f = (frames[-1] if frames else None)
                if f is None:
                    import numpy as np
                    f = (255*np.random.rand(size[1], size[0], 3)).astype("uint8")
            import cv2 as _cv2
            f = _cv2.resize(f, size, interpolation=_cv2.INTER_AREA)
            f = f[:, :, ::-1]  # BGR->RGB
            import torch as _T
            frames.append(_T.from_numpy(f).permute(2,0,1).float()/255.0)
        cap.release()
        import torch as _T
        v = _T.stack(frames, dim=0)  # [T, 3, H, W]
        return v

    def _load_audio_mel(self, path: str, Ta: int, mel_bins: int, sr: int):
        try:
            import torchaudio
        except ImportError as e:
            raise ImportError("torchaudio not found. Please `pip install torchaudio`.") from e
        wav, s = torchaudio.load(path)  # [C, N]
        if s != sr:
            wav = torchaudio.functional.resample(wav, s, sr)
        mono = wav.mean(dim=0, keepdim=True)
        mel_extractor = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=mel_bins)
        mel = mel_extractor(mono)  # [1, mel, T_spec]
        import torch as _T
        mel = _T.log(mel + 1e-6)
        mel = mel.squeeze(0).transpose(0,1)  # [T_spec, mel]
        if mel.size(0) >= Ta:
            idx = _T.linspace(0, mel.size(0)-1, steps=Ta).long()
            mel = mel.index_select(0, idx)
        else:
            pad = Ta - mel.size(0)
            mel = _T.cat([mel, mel.new_zeros(pad, mel.size(1))], dim=0)
        return mel  # [Ta, mel]

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        vid = os.path.join(self.root, r["video_path"]) if self.root else r["video_path"]
        aud = os.path.join(self.root, r["audio_path"]) if self.root else r["audio_path"]
        label = int(r["label"])

        video = self._load_video_uniform(vid, self.Tv, self.size)
        audio = self._load_audio_mel(aud, self.Ta, self.mel, self.sample_rate)

        segs = []
        if "seg_starts" in r and "seg_ends" in r and r["seg_starts"] and r["seg_ends"]:
            ss = [s for s in r["seg_starts"].split(",") if s.strip()!=""]
            ee = [e for e in r["seg_ends"].split(",") if e.strip()!=""]
            for s,e in zip(ss,ee):
                try:
                    s_i, e_i = int(s), int(e)
                    segs.append((s_i, e_i, label))
                except:
                    pass

        return {
            "video": video,
            "audio": audio,
            "label_idx": label,
            "clip_id": r["clip_id"],
            "gt_segments": segs
        }

def collate(batch: List[Dict]):
    import torch
    video = torch.stack([b["video"] for b in batch], dim=0)
    audio = torch.stack([b["audio"] for b in batch], dim=0)
    labels = torch.tensor([b["label_idx"] for b in batch], dtype=torch.long)
    clip_id = [b["clip_id"] for b in batch]
    gt_segments = [b["gt_segments"] for b in batch]
    return {"video": video, "audio": audio, "label_idx": labels, "clip_id": clip_id, "gt_segments": gt_segments}
