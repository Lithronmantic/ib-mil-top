import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import torch, yaml
from torch.utils.data import DataLoader
from avtop.models.detector import AVTopDetector
from avtop.data.csv_dataset import BinaryAVCSVDataset, collate as csv_collate
from avtop.experiments.sota import simple_metrics_from_logits
from avtop.eval.temporal_loc import TemporalLocalizationEvaluator

CFG = "configs/real_binary.yaml"
def make_loader(csv_path, cfg, shuffle=False):
    ds = BinaryAVCSVDataset(csv_path, root=cfg["data"].get("root",""),
                            T_v=cfg["data"]["T_v"], T_a=cfg["data"]["T_a"],
                            mel_bins=cfg["data"]["mel"], sample_rate=cfg["data"]["sr"])
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                      shuffle=shuffle, collate_fn=csv_collate,
                      num_workers=cfg["train"].get("workers",4), pin_memory=True)

if __name__ == "__main__":
    cfg = yaml.safe_load(open(CFG, "r", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader  = make_loader(cfg["data"]["test_csv"],  cfg, shuffle=False)
    model = AVTopDetector(ib_beta=cfg["train"]["ib_beta"]).to(device)
    ckpt = (Path(cfg["experiment"]["workdir"]) / cfg["experiment"]["name"] / "ckpts" / "best_model.pth")
    if ckpt.exists():
        print(f"[load] {ckpt}"); state = torch.load(ckpt, map_location=device); model.load_state_dict(state["model"], strict=False)
    model.eval()
    logits = []; labels = []
    with torch.no_grad():
        for batch in test_loader:
            v = batch["video"].to(device); a = batch["audio"].to(device)
            out = model(a, v); logits.append(out["clip_logits"].cpu()); labels.append(batch["label_idx"])
    import torch as _T
    logits = _T.cat(logits, 0); labels = _T.cat(labels, 0); clip_res = simple_metrics_from_logits(logits, labels)
    loc = TemporalLocalizationEvaluator(); map_res = loc.evaluate(model, test_loader, device=device.type)
    rep = Path("reports") / "comprehensive_report.md"; rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(f"# Comprehensive Evaluation\n\nclip: {clip_res}\n\nloc: {map_res}\n", encoding="utf-8")
    print("[report]", rep.as_posix())
