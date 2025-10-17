# -*- coding: utf-8 -*-
import os
os.environ.setdefault("AVTOP_DEBUG", "1")     # 可关
import torch
import torch.nn.functional as F

from src.avtop.models.enhanced_detector import EnhancedAVTopDetector

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ——最小配置（与你工程一致的键，必要时自行调整）——
    cfg = {
        "model": {
            "video": {"backbone": "resnet18_2d"},
            "audio": {"backbone": "mel_spectrogram_cnn"},  # 确保不是 vggish
            "fusion": {"d_model": 512},                    # 常见融合每路 512 → concat 1024
            "temporal": {"d_model": 256, "n_layers": 2},
            "num_classes": 2,
        },
        "data": {"audio": {"sample_rate": 16000}},
    }

    B, Tv, H, W = 4, 16, 224, 224
    sr, seconds = 16000, 3.0
    wav_len = int(sr * seconds)

    # ——合成一批数据（不依赖数据集）——
    torch.manual_seed(0)
    video = torch.randn(B, Tv, 3, H, W, device=device)
    audio = torch.randn(B, wav_len, device=device)

    # ——建模 & 前后检查——
    model = EnhancedAVTopDetector(cfg).to(device).eval()

    # 1) 直接看两路 backbone 输出与对齐
    with torch.no_grad():  # 推理时建议关闭梯度，省显存并更快
        v_feat = model.video_backbone(video)                 # (B, Tv, Dv)
        a_feat = model.audio_backbone(audio)                 # (B, Ta, Da)

    print(f"[check] v_feat={tuple(v_feat.shape)}  a_feat={tuple(a_feat.shape)}")
    Tv = v_feat.size(1)
    a_aligned = F.interpolate(a_feat.permute(0, 2, 1), size=Tv,
                              mode="linear", align_corners=False).permute(0, 2, 1)
    print(f"[check] a_aligned={tuple(a_aligned.shape)} → Tv={Tv}")

    # 基础数值健检：无 NaN/Inf
    assert torch.isfinite(v_feat).all(), "v_feat 出现 NaN/Inf"
    assert torch.isfinite(a_feat).all(), "a_feat 出现 NaN/Inf"
    assert a_aligned.size(1) == Tv, "音频时间步未对齐到视频 Tv"

    # 2) 端到端跑 forward（带内部调试打印）
    with torch.no_grad():
        out = model(video, audio)

    # out 可能是张量/元组/字典，这里宽松打印一下
    def _shape(x):
        if torch.is_tensor(x): return tuple(x.shape)
        if isinstance(x, (list, tuple)): return tuple(_shape(t) for t in x)
        if isinstance(x, dict): return {k: _shape(v) for k, v in x.items()}
        return type(x).__name__

    print(f"[check] model(video,audio) => { _shape(out) }")
    print("✅ AV 形状/对齐 Sanity Check 通过。")

if __name__ == "__main__":
    main()
