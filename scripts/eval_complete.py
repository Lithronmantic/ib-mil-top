#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版评估脚本：
- 全指标：Acc/Balanced-Acc/Per-class P/R/F1/MCC/Kappa/AUC/AP/Brier/ECE
- 可视化：ROC/PR/Calibration/Score-Hist/CM(Counts+Norm)/Threshold-Sweep
- 单模态对比：Audio-only / Video-only
- 按样本聚合：sample_* 指标与 CSV
- 无标签分析：伪标签分布、覆盖率、直方图、明细导出
- 温度缩放（可选）：校准后指标与图
- YAML 安全序列化
"""

import os
import sys
import csv
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    precision_recall_curve, roc_curve, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, brier_score_loss
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 项目路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.avtop.models.enhanced_detector import EnhancedAVDetector
from src.avtop.data.window_dataset import WindowDataset, collate_fn

# ---------------- 工具函数 ----------------

def _to_py(obj):
    import numpy as np
    if isinstance(obj, (np.bool_, np.bool8)): return bool(obj)
    if isinstance(obj, np.integer):           return int(obj)
    if isinstance(obj, np.floating):          return float(obj)
    if isinstance(obj, np.ndarray):           return obj.tolist()
    if isinstance(obj, dict):                 return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):        return [_to_py(x) for x in obj]
    return obj

def _extract_samples(meta_obj, batch_size: int) -> List[str]:
    names = [""] * batch_size
    if meta_obj is None: return names
    if isinstance(meta_obj, list):
        for i, m in enumerate(meta_obj[:batch_size]):
            if isinstance(m, dict):
                names[i] = str(m.get("sample", ""))
        return names
    if isinstance(meta_obj, dict):
        arr = meta_obj.get("sample", None)
        if arr is None: return names
        for i in range(min(len(arr), batch_size)):
            names[i] = str(arr[i])
    return names

def _per_class_metrics(labels: np.ndarray, preds: np.ndarray, num_classes: int, class_names: List[str]) -> Dict[str, float]:
    out = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0: continue
        out[f'class_{class_names[c]}_acc']       = float((preds[mask] == labels[mask]).mean())
        out[f'class_{class_names[c]}_precision'] = float(precision_score(labels, preds, labels=[c], average='macro', zero_division=0))
        out[f'class_{class_names[c]}_recall']    = float(recall_score(labels, preds, labels=[c],  average='macro', zero_division=0))
        out[f'class_{class_names[c]}_f1']        = float(f1_score(labels, preds,   labels=[c],  average='macro', zero_division=0))
    return out

def _specificity_npvs(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    if len(np.unique(labels)) != 2: return {}
    cm = confusion_matrix(labels, preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {"specificity": float(spec), "npv": float(npv)}

def _threshold_sweep(labels: np.ndarray, probs: np.ndarray):
    best_thr, best = 0.5, -1.0
    best_stats, rows = {}, []
    for thr in np.linspace(0.0, 1.0, 101):
        pred_t = (probs[:,1] >= thr).astype(np.int64)
        prec = precision_score(labels, pred_t, zero_division=0)
        rec  = recall_score(labels, pred_t, zero_division=0)
        f1   = f1_score(labels, pred_t, zero_division=0)
        acc  = accuracy_score(labels, pred_t)
        cm = confusion_matrix(labels, pred_t, labels=[0,1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            J = tpr + tnr - 1.0
        else:
            J = -1.0
        rows.append((thr, prec, rec, f1, acc))
        if (f1, J) > (best, -np.inf):
            best = f1; best_thr = thr
            best_stats = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc), "youdenJ": float(J)}
    return best_thr, best_stats, rows

def _calibration_bins(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10):
    assert probs.ndim == 2 and probs.shape[1] >= 2
    p = probs[:,1].astype(float)
    y = labels.astype(int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, edges, right=False) - 1
    idx = np.clip(idx, 0, n_bins-1)
    confs   = np.zeros(n_bins, dtype=float)
    posrate = np.zeros(n_bins, dtype=float)
    counts  = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = (idx == i)
        counts[i] = int(m.sum())
        if counts[i] > 0:
            confs[i]   = float(p[m].mean())
            posrate[i] = float(y[m].mean())
        else:
            confs[i]   = float((edges[i]+edges[i+1])/2.0)
            posrate[i] = np.nan
    valid = counts > 0
    freq  = counts[valid] / counts[valid].sum() if valid.any() else np.array([1.0])
    ece   = float(np.sum(np.abs(posrate[valid] - confs[valid]) * freq)) if valid.any() else 0.0
    return edges, confs, posrate, counts, ece

def _save_csv(path: Path, header: List[str], rows: List[List]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

# ---------------- 评估器 ----------------

class ModelEvaluator:
    def __init__(self, model: nn.Module, val_loader: DataLoader, device: torch.device,
                 num_classes: int, class_names: List[str], output_dir: Path):
        self.model = model.eval()
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        self.out_dir = output_dir

    def evaluate(self, return_predictions: bool = False):
        all_preds, all_labels, all_probs = [], [], []
        all_preds_v, all_probs_v = [], []
        all_preds_a, all_probs_a = [], []
        all_samples, all_logits = [], []

        print("\n" + "="*70)
        print("🔍 开始评估")
        print("="*70)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                v = batch['video'].to(self.device)
                a = batch['audio'].to(self.device)
                y = batch['label'].to(self.device)

                out = self.model(v, a, return_aux=True)
                logits = out['clip_logits']
                probs  = torch.softmax(logits, dim=-1)
                preds  = torch.argmax(logits, dim=-1)

                all_logits.append(logits.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())

                # 单模态
                v_logits = out.get('video_logits'); a_logits = out.get('audio_logits')
                if v_logits is not None:
                    pv = torch.softmax(v_logits, dim=-1)
                    all_preds_v.append(torch.argmax(v_logits, dim=-1).cpu().numpy())
                    all_probs_v.append(pv.cpu().numpy())
                if a_logits is not None:
                    pa = torch.softmax(a_logits, dim=-1)
                    all_preds_a.append(torch.argmax(a_logits, dim=-1).cpu().numpy())
                    all_probs_a.append(pa.cpu().numpy())

                all_samples += _extract_samples(batch.get('metadata'), v.shape[0])

        all_logits = np.concatenate(all_logits, 0)
        all_probs  = np.concatenate(all_probs,  0)
        all_preds  = np.concatenate(all_preds,  0)
        all_labels = np.concatenate(all_labels, 0)

        metrics = self._compute_metrics(all_labels, all_preds, all_probs)

        # 单模态
        if len(all_preds_v):
            pv = np.concatenate(all_probs_v, 0); yv = np.concatenate(all_preds_v, 0)
            metrics['video_only_acc'] = accuracy_score(all_labels, yv)
            metrics['video_only_f1']  = f1_score(all_labels, yv, average='macro', zero_division=0)
            if self.num_classes == 2:
                metrics['video_only_auc'] = roc_auc_score(all_labels, pv[:,1])
                metrics['video_only_ap']  = average_precision_score(all_labels, pv[:,1])
        if len(all_preds_a):
            pa = np.concatenate(all_probs_a, 0); ya = np.concatenate(all_preds_a, 0)
            metrics['audio_only_acc'] = accuracy_score(all_labels, ya)
            metrics['audio_only_f1']  = f1_score(all_labels, ya, average='macro', zero_division=0)
            if self.num_classes == 2:
                metrics['audio_only_auc'] = roc_auc_score(all_labels, pa[:,1])
                metrics['audio_only_ap']  = average_precision_score(all_labels, pa[:,1])

        # 保存预测（窗口级，含二值化 @0.5 与 @best）
        thr_best = metrics.get('best_threshold', 0.5)
        pred_rows = []
        for i in range(len(all_labels)):
            row = [all_samples[i], int(all_labels[i]), int(all_preds[i])]
            row += list(np.round(all_probs[i], 6))
            row += [int(all_probs[i,1] >= 0.5), int(all_probs[i,1] >= thr_best)]
            pred_rows.append(row)
        header = ["sample","label","pred"] + [f"prob_{k}" for k in range(self.num_classes)] + ["pred@0.5","pred@best"]
        _save_csv(self.out_dir / "predictions.csv", header, pred_rows)

        # 错误样本 Top-50
        err_mask = (all_preds != all_labels)
        if err_mask.any():
            conf = all_probs.max(axis=1); idx = np.where(err_mask)[0]
            top = idx[np.argsort(-conf[idx])][:50]
            rows = [[all_samples[i], int(all_labels[i]), int(all_preds[i]), float(conf[i])] for i in top]
            _save_csv(self.out_dir / "errors_topk.csv", ["sample","label","pred","confidence"], rows)

        # 按样本聚合
        if any(s != "" for s in all_samples):
            agg = {}
            for s, y, p in zip(all_samples, all_labels, all_probs):
                if s not in agg: agg[s] = {"y": y, "scores": []}
                agg[s]["scores"].append(p)
            s_labels, s_probs = [], []
            for s in agg:
                s_labels.append(agg[s]["y"])
                s_probs.append(np.mean(np.stack(agg[s]["scores"],0), axis=0))
            s_labels = np.asarray(s_labels); s_probs = np.asarray(s_probs)
            s_preds  = np.argmax(s_probs, axis=1)
            metrics.update(self._compute_metrics(s_labels, s_preds, s_probs, prefix="sample_"))
            _save_csv(self.out_dir / "predictions_sample.csv",
                      ["sample","label"]+[f"prob_{i}" for i in range(self.num_classes)],
                      [[k, int(agg[k]["y"])] + list(np.round(np.mean(np.stack(agg[k]["scores"],0),0), 6)) for k in agg])

        if return_predictions:
            return metrics, all_preds, all_labels, all_probs, all_samples, all_logits
        return metrics

    def _compute_metrics(self, labels, preds, probs, prefix=""):
        m = {}
        m[prefix+'accuracy']  = accuracy_score(labels, preds)
        m[prefix+'balanced_accuracy'] = balanced_accuracy_score(labels, preds)
        m[prefix+'precision'] = precision_score(labels, preds, average='macro', zero_division=0)
        m[prefix+'recall']    = recall_score(labels, preds, average='macro', zero_division=0)
        m[prefix+'f1']        = f1_score(labels, preds, average='macro', zero_division=0)
        m[prefix+'mcc']       = matthews_corrcoef(labels, preds)
        m[prefix+'kappa']     = cohen_kappa_score(labels, preds)
        m.update(_per_class_metrics(labels, preds, self.num_classes, self.class_names))

        if self.num_classes == 2:
            try: m[prefix+'auc'] = roc_auc_score(labels, probs[:,1])
            except ValueError: m[prefix+'auc'] = 0.0
            try: m[prefix+'ap']  = average_precision_score(labels, probs[:,1])
            except ValueError: m[prefix+'ap']  = 0.0
            m.update(_specificity_npvs(labels, preds))
            try: m[prefix+'brier'] = brier_score_loss(labels, probs[:,1])
            except ValueError: m[prefix+'brier'] = 0.0

            # 校准（正确版）
            edges, confs, posrate, counts, ece = _calibration_bins(labels, probs, n_bins=10)
            m[prefix+'ece_10bins'] = ece
            self._plot_calibration(confs, posrate, counts, self.out_dir / f"{prefix}calibration.png")

            # 阈值扫描 & 最优阈值
            best_thr, best_stats, rows = _threshold_sweep(labels, probs)
            m[prefix+'best_threshold'] = float(best_thr)
            for k, v in best_stats.items():
                m[prefix+f'best_{k}'] = v
            _save_csv(self.out_dir / f"{prefix}thresholds_metrics.csv",
                      ["threshold","precision","recall","f1","accuracy"],
                      [[float(t),float(p),float(r),float(f1),float(a)] for t,p,r,f1,a in rows])
            self._plot_threshold_sweep(rows, self.out_dir / f"{prefix}threshold_sweep.png")

            # 最优阈值下的混淆矩阵
            pred_best = (probs[:,1] >= m[prefix+'best_threshold']).astype(np.int64)
            cm_best   = confusion_matrix(labels, pred_best, labels=[0,1])
            self._plot_confusion_matrix(cm_best, self.class_names, self.out_dir / f"{prefix}confusion_matrix@best.png", normalize=False)
            self._plot_confusion_matrix(cm_best, self.class_names, self.out_dir / f"{prefix}confusion_matrix@best_norm.png", normalize=True)

            # 曲线
            self._plot_roc(labels, probs, self.out_dir / f"{prefix}roc_curve.png")
            self._plot_pr(labels, probs, self.out_dir / f"{prefix}pr_curve.png")
            self._plot_score_hist(labels, probs, self.out_dir / f"{prefix}score_hist.png")

        # 基础混淆矩阵
        cm = confusion_matrix(labels, preds)
        m[prefix+'confusion_matrix'] = cm
        self._plot_confusion_matrix(cm, self.class_names, self.out_dir / f"{prefix}confusion_matrix_counts.png", normalize=False)
        self._plot_confusion_matrix(cm, self.class_names, self.out_dir / f"{prefix}confusion_matrix_norm.png", normalize=True)
        return m

    # --------- 画图 ---------
    def _plot_confusion_matrix(self, cm, class_names, save_path, normalize=False):
        plt.figure(figsize=(8,6))
        if normalize:
            cmn = cm.astype('float') / (cm.sum(axis=1, keepdims=True)+1e-9)
            sns.heatmap(cmn, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title('Normalized Confusion Matrix')
        else:
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix (Counts)')
        plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150); plt.close()

    def _plot_roc(self, labels, probs, save_path):
        fpr, tpr, _ = roc_curve(labels, probs[:,1])
        auc = roc_auc_score(labels, probs[:,1])
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(save_path, dpi=150); plt.close()

    def _plot_pr(self, labels, probs, save_path):
        prec, rec, _ = precision_recall_curve(labels, probs[:,1])
        ap = average_precision_score(labels, probs[:,1])
        plt.figure(figsize=(6,5))
        plt.plot(rec, prec, label=f"AP={ap:.4f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(save_path, dpi=150); plt.close()

    def _plot_calibration(self, confs, posrate, counts, save_path):
        valid = (~np.isnan(posrate)) & (counts > 0)
        if valid.sum() == 0:
            print("⚠️ 校准图：所有 bin 为空，跳过绘制。"); return
        sizes = (counts[valid] / counts[valid].max()) * 80 + 20
        ece_approx = float(np.mean(np.abs(posrate[valid] - confs[valid])))
        plt.figure(figsize=(6,5))
        plt.plot([0,1],[0,1],'--', color='gray', linewidth=1)
        plt.scatter(confs[valid], posrate[valid], s=sizes)
        plt.xlabel("Predicted probability (bin mean)")
        plt.ylabel("Empirical positive rate")
        plt.title(f"Reliability Diagram (ECE≈{ece_approx:.3f})")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150); plt.close()

    def _plot_score_hist(self, labels, probs, save_path):
        plt.figure(figsize=(6,5))
        p1 = probs[:,1]
        plt.hist(p1[labels==0], bins=30, alpha=0.6, label=f"{self.class_names[0]}")
        plt.hist(p1[labels==1], bins=30, alpha=0.6, label=f"{self.class_names[1]}")
        plt.xlabel("Predicted probability for class 1"); plt.ylabel("Count"); plt.legend()
        plt.title("Score Histogram by Class"); plt.tight_layout()
        plt.savefig(save_path, dpi=150); plt.close()

    def _plot_threshold_sweep(self, rows, save_path):
        arr = np.asarray(rows)
        thr, prec, rec, f1, acc = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
        plt.figure(figsize=(7,5))
        plt.plot(thr, f1, label='F1'); plt.plot(thr, prec, label='Precision'); plt.plot(thr, rec, label='Recall'); plt.plot(thr, acc, label='Accuracy')
        plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("Threshold Sweep")
        plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

    # 保留旧接口
    def visualize_confusion_matrix(self, metrics: Dict, save_path: str):
        self._plot_confusion_matrix(metrics['confusion_matrix'], self.class_names, Path(save_path), normalize=False)

    def visualize_attention(self, num_samples: int = 5, save_dir: str = 'visualizations'):
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📸 可视化注意力权重（前{num_samples}个样本）")
        with torch.no_grad():
            count = 0
            for batch in self.val_loader:
                if count >= num_samples: break
                v = batch['video'].to(self.device)
                a = batch['audio'].to(self.device)
                y = batch['label']
                out = self.model(v, a, return_aux=True, return_attn=True)
                if 'attn_weights' not in out or 'mil_weights' not in out:
                    print("⚠️ 模型不支持注意力可视化"); return
                attn_w = out['attn_weights']; mil_w = out['mil_weights']
                for i in range(min(v.shape[0], num_samples - count)):
                    self._plot_attention(attn_w, mil_w[i], y[i].item(), save_dir / f"attention_sample{count}.png", i)
                    count += 1
                    if count >= num_samples: break
        print(f"✅ 注意力可视化已保存到: {save_dir}")

    def _plot_attention(self, attn_weights: Dict, mil_weights: torch.Tensor, label: int, save_path: Path, batch_idx: int = 0):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        mil = mil_weights.detach().cpu().numpy()
        axes[0].bar(range(len(mil)), mil, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Frame Index'); axes[0].set_ylabel('Attention Weight')
        axes[0].set_title(f'MIL Attention (Label: {self.class_names[label]})'); axes[0].grid(True, alpha=0.3)
        modal = attn_weights.get('modal_weights', None)
        if modal is not None:
            mw = modal[batch_idx].detach().cpu().numpy()
            axes[1].bar(['Video','Audio'], mw, color=['coral','skyblue'], alpha=0.7)
            axes[1].set_ylim([0,1]); axes[1].set_ylabel('Modality Weight'); axes[1].set_title('Cross-modal Attention')
            axes[1].grid(True, alpha=0.3)
        plt.tight_layout(); save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

# ------------- 加载 / 无标签分析 / 主程序 -------------

def load_model(checkpoint_path: str, config: Dict, device: torch.device) -> nn.Module:
    import numpy as np
    print(f"\n📥 加载模型: {checkpoint_path}")
    model = EnhancedAVDetector(config)

    # 先尝试“安全加载”；失败则回退到传统（可信文件才使用）
    try:
        # 某些旧 ckpt 需要放行 numpy 的 scalar；能缓解一部分错误
        try:
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        except Exception:
            pass
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        safe_mode = True
    except Exception as e:
        print(f"⚠️ weights_only=True 加载失败（{e.__class__.__name__}）。回退到传统加载方式。")
        ckpt = torch.load(checkpoint_path, map_location=device)  # 仅在你信任该文件时使用
        safe_mode = False

    # 兼容 {'model_state_dict': ...} 或直接 state_dict
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"ℹ️ state_dict 对齐：missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device).eval()
    ep = ckpt.get('epoch', '?')
    how = "安全模式" if safe_mode else "传统模式"
    print(f"✅ 模型已加载 ({how}, Epoch {ep})")
    return model
def evaluate_unlabeled(model, csv_path, config, device, out_dir: Path, p_cutoff: float = 0.95):
    print("\n📂 加载无标签集:", csv_path)
    ds = WindowDataset(
        csv_path=csv_path,
        target_sr=config['data'].get('audio_sr',16000),
        target_video_size=tuple(config['data'].get('video_size',[224,224])),
        max_audio_length=config['data'].get('max_audio_length',0.3),
        max_video_frames=config['data'].get('max_video_frames',16)
    )
    dl = DataLoader(ds, batch_size=config['training'].get('batch_size',8),
                    shuffle=False, num_workers=4, collate_fn=collate_fn)
    probs_all, preds_all, samples_all = [], [], []
    model.eval()
    with torch.no_grad():
        for b in tqdm(dl, desc="Unlabeled Infer"):
            v = b['video'].to(device); a = b['audio'].to(device)
            out = model(v, a, return_aux=False)
            p = torch.softmax(out['clip_logits'], dim=-1).cpu().numpy()
            yhat = p.argmax(axis=1)
            probs_all.append(p); preds_all.append(yhat)
            samples_all += _extract_samples(b.get('metadata'), v.shape[0])
    probs = np.concatenate(probs_all, 0); preds = np.concatenate(preds_all, 0)
    conf  = probs.max(axis=1); high = conf >= p_cutoff
    _save_csv(out_dir / "unlabeled_predictions.csv",
              ["sample","pseudo_label","confidence"]+[f"prob_{k}" for k in range(probs.shape[1])],
              [[samples_all[i], int(preds[i]), float(conf[i])] + list(map(float, probs[i])) for i in range(len(preds))])
    _save_csv(out_dir / f"unlabeled_highconf@{p_cutoff:.2f}.csv",
              ["sample","pseudo_label","confidence"],
              [[samples_all[i], int(preds[i]), float(conf[i])] for i in np.where(high)[0]])
    # 图
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,5)); plt.hist(conf, bins=30); plt.title(f"Unlabeled Confidence (p_cutoff={p_cutoff})")
    plt.xlabel("max prob"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(out_dir / "figs/unlabeled_conf_hist.png", dpi=150); plt.close()
    plt.figure(figsize=(6,4))
    uniq, cnts = np.unique(preds, return_counts=True); plt.bar([str(int(u)) for u in uniq], cnts)
    plt.title("Unlabeled Pseudo-label Distribution"); plt.xlabel("class"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(out_dir / "figs/unlabeled_pseudolabel_dist.png", dpi=150); plt.close()
    with open(out_dir / "unlabeled_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Total: {len(preds)}\n")
        f.write(f"High-conf @ {p_cutoff:.2f}: {high.sum()} ({high.mean()*100:.2f}%)\n")
        if probs.shape[1] == 2:
            f.write(f"Mean prob class1: {probs[:,1].mean():.4f}\n")
    print(f"✅ 无标签分析完成（覆盖率={high.mean()*100:.2f}%） -> {out_dir}")

class TemperatureScaler(torch.nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.tensor([init_T], dtype=torch.float32))
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.clamp(self.temperature, min=1e-3)
        return logits / T
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200):
        opt = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
        nll = torch.nn.CrossEntropyLoss()
        logits = logits.detach(); labels = labels.detach()
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = nll(self.forward(logits), labels)
            loss.backward(); return loss
        opt.step(closure); return float(self.temperature.item())

def main():
    ap = argparse.ArgumentParser("模型评估（增强版）")
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output_dir', default='evaluation_results')
    ap.add_argument('--visualize', action='store_true')
    ap.add_argument('--eval_unlabeled', action='store_true', help='同时评估 unlabeled（非监督）')
    ap.add_argument('--p_cutoff', type=float, default=0.95, help='无标签高置信阈值')
    ap.add_argument('--temp_scale', action='store_true', help='拟合温度缩放并输出校准后指标')
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")

    model = load_model(args.checkpoint, config, device)

    print("\n📂 加载验证集")
    val_dataset = WindowDataset(
        csv_path=config['data']['val_csv'],
        target_sr=config['data'].get('audio_sr', 16000),
        target_video_size=tuple(config['data'].get('video_size', [224, 224])),
        max_audio_length=config['data'].get('max_audio_length', 0.3),
        max_video_frames=config['data'].get('max_video_frames', 16)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].get('batch_size', 8),
        shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    print(f"验证集: {len(val_dataset)} 样本")
    class_names = config['data'].get('class_names', [f'class_{i}' for i in range(config['model']['num_classes'])])

    evaluator = ModelEvaluator(model, val_loader, device, config['model']['num_classes'], class_names, out_dir)
    metrics, preds, labels, probs, samples, logits = evaluator.evaluate(return_predictions=True)

    # 保存指标
    with open(out_dir / "metrics.yaml", 'w', encoding='utf-8') as f:
        yaml.safe_dump(_to_py(metrics), f, allow_unicode=True, sort_keys=False)
    print("✅ 指标已保存: metrics.yaml")

    if args.visualize:
        print("\n🎨 生成可视化")
        attn_dir = out_dir / "attention_visualizations"
        evaluator.visualize_attention(num_samples=10, save_dir=str(attn_dir))
        err_mask = (np.asarray(preds) != np.asarray(labels))
        if err_mask.any():
            conf = np.asarray(probs).max(axis=1); idx = np.where(err_mask)[0]
            top = idx[np.argsort(-conf[idx])][:10]
            print("\n🔎 Top-10 高置信度错误样本：")
            for i in top:
                print(f"  sample={samples[i]}  true={class_names[labels[i]]}  pred={class_names[preds[i]]}  conf={conf[i]:.4f}")

    # 无标签集分析
    unl_csv = config['data'].get('unlabeled_csv')
    if args.eval_unlabeled and unl_csv and Path(unl_csv).exists():
        evaluate_unlabeled(model, unl_csv, config, device, out_dir / "unlabeled", args.p_cutoff)
    elif args.eval_unlabeled:
        print("⚠️ 未发现 data.unlabeled_csv 或路径不存在，跳过无标签分析。")

    # 温度缩放（后验校准）
    if args.temp_scale:
        print("\n🧪 温度缩放（后验校准）")
        logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        scaler = TemperatureScaler().to(device)
        T = scaler.fit(logits_t, labels_t)
        print(f"  最优温度 T = {T:.3f}")

        with torch.no_grad():
            logits_cal = scaler(logits_t).cpu().numpy()
        probs_cal = torch.softmax(torch.from_numpy(logits_cal), dim=-1).numpy()
        preds_cal = probs_cal.argmax(axis=1)

        metrics_cal = evaluator._compute_metrics(labels, preds_cal, probs_cal, prefix="cal_")

        to_save_cal = dict(metrics_cal)  # ← 新增
        to_save_cal["temperature"] = float(T)  # ← 新增
        with open(out_dir / "metrics_calibrated.yaml", 'w', encoding='utf-8') as f:
            yaml.safe_dump(_to_py(to_save_cal), f, allow_unicode=True, sort_keys=False)
        print("✅ 校准后指标已保存: metrics_calibrated.yaml")

    print("\n" + "="*70)
    print("🎉 评估完成！结果保存在：", out_dir)
    print("="*70)

if __name__ == "__main__":
    main()
