# metrics.py
# -*- coding: utf-8 -*-
"""
Métricas e visualizações alinhadas ao seu treino v3 (best_model.keras)
- Usa libras_actions.npy e results_v3/norm_stats.json (mu, sd, T, F, feature_mode)
- Procura dados em libras_data/<classe>/*.npy por padrão (pode mudar com --data_dir)
- Checa compatibilidade: nº de saídas do modelo == nº de classes
- Salva em --outdir (padrão: report_figs_v3)
"""

import os, sys, json, csv, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ========================= Carregar modelo (.keras/.h5) =========================
def load_model_any(path):
    """Carrega modelo salvo (Keras 3/tf.keras). Faz fallback para 'unsafe' se tiver Lambda."""
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e1:
        try:
            import keras
            keras.config.enable_unsafe_deserialization()
            from keras.saving import load_model
            return load_model(path, compile=False)
        except Exception as e2:
            raise RuntimeError(
                f"Falha ao carregar modelo '{path}'.\n"
                f"tf.keras error: {repr(e1)}\nkeras.saving error: {repr(e2)}"
            )

# ========================= Dataset & transforms =========================
def load_actions(path):
    a = np.load(path)
    if a.ndim != 1:
        raise ValueError("libras_actions.npy deve ser 1D.")
    return a

def load_norm_stats(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"norm_stats.json não encontrado em {path}.")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    mu = np.array(d["mu"], dtype=np.float32).reshape(1,1,-1)
    sd = np.array(d["sd"], dtype=np.float32).reshape(1,1,-1)
    T  = int(d["T"]); F = int(d["F"])
    feature_mode = d.get("feature_mode", "absolute")
    return mu, sd, T, F, feature_mode

def pad_or_crop_to_T(x, T):
    t = x.shape[0]
    if t == T: return x
    if t > T:
        start = (t - T) // 2
        return x[start:start+T]
    pad = np.zeros((T - t, x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)

def to_wrist_centered(x):
    """ x: (T, F). Espera F ∈ {63,126}. """
    F = x.shape[1]
    if F not in (63, 126):
        return x
    if F == 63:
        pts = x.reshape(x.shape[0], 21, 3)
        wrist = pts[:, 0:1, :]
        pts = pts - wrist
        return pts.reshape(x.shape[0], -1)
    else:
        pts = x.reshape(x.shape[0], 42, 3)
        wrist_r = pts[:, 0:1, :]
        wrist_l = pts[:, 21:22, :]
        pts[:, 0:21, :]  = pts[:, 0:21, :]  - wrist_r
        pts[:, 21:42, :] = pts[:, 21:42, :] - wrist_l
        return pts.reshape(x.shape[0], -1)

def apply_feature_mode(x, mode):
    return to_wrist_centered(x) if mode == "wrist_centered" else x

def scan_labeled_dir(root_dir, actions):
    """Retorna lista [(path, class_idx), ...] seguindo os nomes em actions."""
    items = []
    for idx, cls in enumerate(actions):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir): continue
        for f in os.listdir(cls_dir):
            if f.endswith(".npy"):
                items.append((os.path.join(cls_dir, f), idx))
    return items

def build_dataset(root_dir, actions, T, F, mu, sd, feature_mode):
    files = scan_labeled_dir(root_dir, actions)
    if not files:
        raise RuntimeError(f"Nenhum .npy encontrado em {root_dir}/<classe>/*.npy")
    X_list, y_list = [], []
    for path, y in files:
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2:
            continue
        arr = apply_feature_mode(arr, feature_mode)  # MESMO modo do treino
        if arr.shape[1] != F:
            raise ValueError(f"{path}: F={arr.shape[1]}, esperado F={F}. Verifique feature_mode/treino.")
        arr = pad_or_crop_to_T(arr, T)
        X_list.append(arr); y_list.append(y)
    X = np.stack(X_list, axis=0)          # (N, T, F)
    y = np.array(y_list, dtype=int)
    X = (X - mu) / (sd + 1e-8)            # z-score com stats do treino
    return X, y, files

# ========================= Plots =========================
def ensure_outdir(d): os.makedirs(d, exist_ok=True)

def plot_class_distribution(y, actions, outpath):
    counts = np.bincount(y, minlength=len(actions))
    plt.figure(figsize=(10,4))
    plt.bar(range(len(actions)), counts)
    plt.xticks(range(len(actions)), actions, rotation=45, ha='right')
    plt.ylabel("Amostras"); plt.title("Distribuição de classes")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_confusions(y_true, y_pred, actions, out_raw, out_norm):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(actions)))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    # abs
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest'); plt.title('Matriz de confusão (abs)')
    plt.colorbar(); plt.xticks(range(len(actions)), actions, rotation=45, ha='right')
    plt.yticks(range(len(actions)), actions); plt.xlabel('Predito'); plt.ylabel('Verdadeiro')
    plt.tight_layout(); plt.savefig(out_raw); plt.close()
    # norm
    plt.figure(figsize=(6,5))
    plt.imshow(cmn, interpolation='nearest', vmin=0, vmax=1); plt.title('Matriz de confusão (norm)')
    plt.colorbar(); plt.xticks(range(len(actions)), actions, rotation=45, ha='right')
    plt.yticks(range(len(actions)), actions); plt.xlabel('Predito'); plt.ylabel('Verdadeiro')
    plt.tight_layout(); plt.savefig(out_norm); plt.close()

def plot_f1_bars(report_dict, actions, outpath):
    f1 = [report_dict.get(cls, {}).get("f1-score", np.nan) for cls in actions]
    plt.figure(figsize=(10,4))
    plt.bar(range(len(actions)), f1)
    plt.xticks(range(len(actions)), actions, rotation=45, ha='right')
    plt.ylim(0,1); plt.ylabel("F1-score"); plt.title("F1 por classe")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_confidence_hist(probs, y_pred, outpath):
    conf = probs[np.arange(len(y_pred)), y_pred]
    plt.figure(figsize=(6,4))
    plt.hist(conf, bins=20)
    plt.xlabel("Confiança da classe predita"); plt.ylabel("Amostras")
    plt.title("Histograma das confianças")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_calibration(probs, y_true, outpath, bins=10):
    y_pred = probs.argmax(1)
    conf = probs[np.arange(len(y_pred)), y_pred]
    acc  = (y_pred == y_true).astype(float)
    edges = np.linspace(0,1,bins+1)
    mids, accs = [], []
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i+1])
        if m.sum()==0: continue
        mids.append((edges[i]+edges[i+1])/2)
        accs.append(acc[m].mean())
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1],'--', label='Ideal')
    plt.plot(mids, accs, marker='o', label='Observado')
    plt.xlabel("Confiança predita"); plt.ylabel("Acurácia empírica")
    plt.title("Reliability / Calibration"); plt.legend()
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_roc_curves(y_true, probs, actions, outpath):
    if len(actions) < 2 or probs.shape[1] != len(actions): return
    Y = label_binarize(y_true, classes=range(len(actions)))
    plt.figure(figsize=(8,6))
    fpr, tpr, _ = roc_curve(Y.ravel(), probs.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"micro-average AUC={roc_auc:.3f}")
    for i in range(len(actions)):
        fpr_i, tpr_i, _ = roc_curve(Y[:, i], probs[:, i])
        plt.plot(fpr_i, tpr_i, alpha=0.3, label=f"{actions[i]}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC multi-classe")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_pr_curves(y_true, probs, actions, outpath):
    if len(actions) < 2 or probs.shape[1] != len(actions): return
    Y = label_binarize(y_true, classes=range(len(actions)))
    plt.figure(figsize=(8,6))
    precision, recall, _ = precision_recall_curve(Y.ravel(), probs.ravel())
    ap_micro = average_precision_score(Y, probs, average="micro")
    plt.plot(recall, precision, label=f"micro AP={ap_micro:.3f}")
    for i in range(len(actions)):
        p, r, _ = precision_recall_curve(Y[:, i], probs[:, i])
        plt.plot(r, p, alpha=0.3, label=f"{actions[i]}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.savefig(outpath); plt.close()

# ========================= Main =========================
def main():
    parser = argparse.ArgumentParser(description="Métricas/visualizações (v3).")
    parser.add_argument("--data_dir", default="newMINDS", help="Pasta com subpastas por classe (*.npy).")
    parser.add_argument("--model",    default="best_model_MINDS.keras")
    parser.add_argument("--actions",  default="libras_actions_MINDS.npy")
    parser.add_argument("--norm",     default=os.path.join("results_MINDS","norm_stats.json"))
    parser.add_argument("--outdir",   default="report_figs_MINDS")
    args = parser.parse_args()

    # Artefatos
    actions = load_actions(args.actions)
    mu, sd, T, F, feature_mode = load_norm_stats(args.norm)
    model = load_model_any(args.model)

    # Checagem de classes
    n_out = int(model.output_shape[-1])
    if n_out != len(actions):
        print(f"[ERRO] Saídas do modelo ({n_out}) ≠ nº de classes em actions.npy ({len(actions)}).")
        print("Use o MESMO libras_actions.npy salvo no treino que gerou best_model.keras.")
        sys.exit(1)

    # Dataset
    X, y, files = build_dataset(args.data_dir, actions, T, F, mu, sd, feature_mode)
    print(f"[INFO] Avaliação: X={X.shape} | classes={len(actions)} | feature_mode={feature_mode}")

    # Predição
    probs = model.predict(X, verbose=0)  # (N, n_out)
    if probs.shape[1] != len(actions):
        print("[ERRO] Probabilidades não batem com nº de classes.")
        sys.exit(1)
    y_pred = probs.argmax(1)

    # Métricas texto
    print("\n=== Classification report ===")
    print(classification_report(y, y_pred, target_names=actions.tolist(), digits=4, zero_division=0))

    # Figuras/CSVs
    ensure_outdir(args.outdir)
    plot_class_distribution(y, actions, os.path.join(args.outdir, "class_distribution.png"))
    plot_confusions(y, y_pred, actions,
                    os.path.join(args.outdir, "confusion_raw.png"),
                    os.path.join(args.outdir, "confusion_norm.png"))

    rep = classification_report(y, y_pred, target_names=actions.tolist(), digits=4, zero_division=0, output_dict=True)
    with open(os.path.join(args.outdir, "per_class_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["class","precision","recall","f1","support"])
        for cls in actions:
            row = rep.get(cls, {})
            w.writerow([cls, row.get("precision",""), row.get("recall",""), row.get("f1-score",""), row.get("support","")])

    plot_f1_bars(rep, actions, os.path.join(args.outdir, "f1_per_class.png"))
    plot_confidence_hist(probs, y_pred, os.path.join(args.outdir, "confidence_hist.png"))
    plot_calibration(probs, y, os.path.join(args.outdir, "calibration_reliability.png"))
    if len(actions) >= 2:
        plot_roc_curves(y, probs, actions, os.path.join(args.outdir, "roc_curves.png"))
        plot_pr_curves(y, probs, actions, os.path.join(args.outdir, "pr_curves.png"))

    # Misclassifications (corrigido: 4 iteráveis => 4 alvos)
    conf = probs[np.arange(len(y_pred)), y_pred]
    with open(os.path.join(args.outdir, "misclassifications.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file","true","pred","confidence"])
        for (path, file_true), y_true, y_hat, c in zip(files, y, y_pred, conf):
            # y_true já é o rótulo inteiro; file_true é o índice da classe derivado da pasta
            if y_true != y_hat:
                w.writerow([path, str(actions[y_true]), str(actions[y_hat]), f"{c:.4f}"])

    print(f"\n✅ Relatórios salvos em: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
