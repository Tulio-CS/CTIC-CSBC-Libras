# metrics_and_landmarks_viz.py
# -*- coding: utf-8 -*-
"""
Métricas + visualizações a partir dos landmarks salvos (.npy)
Compatível com seu treino v3:
  - best_model.keras
  - libras_actions.npy
  - results_v3/norm_stats.json (mu, sd, T, F, feature_mode)

Saídas em --outdir:
  - métricas: classification_report (CSV), Cohen's kappa, MCC, top-k, ECE
  - figuras: confusão abs/norm, PR/ROC, hist de confiança, calibration
  - landmarks: 2D e 3D dos gestos salvos (por classe), usando conexões do MediaPipe

NÃO usa cv2; tudo é feito com NumPy + Matplotlib.
"""

import os, sys, json, csv, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    cohen_kappa_score, matthews_corrcoef, top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (necessário para projeção 3D)

# ========================= Utilidades básicas =========================
def load_model_any(path):
    """Carrega modelo salvo (tf.keras -> Keras 3 unsafe como fallback)."""
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

def load_actions(path):
    a = np.load(path)
    if a.ndim != 1:
        raise ValueError("libras_actions.npy deve ser 1D.")
    return a

def load_norm_stats(path):
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
    """ x: (T, F). F ∈ {63,126}. Subtrai o pulso (landmark 0) por mão. """
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
    """Retorna [(path, class_idx)] conforme nomes em actions."""
    items = []
    for idx, cls in enumerate(actions):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir): continue
        for f in os.listdir(cls_dir):
            if f.endswith(".npy"):
                items.append((os.path.join(cls_dir, f), idx))
    return items

def build_dataset(root_dir, actions, T, F, mu, sd, feature_mode):
    """Retorna X_norm, y, files, X_raw (sem normalizar/sem feature_mode para visualização)."""
    files = scan_labeled_dir(root_dir, actions)
    if not files:
        raise RuntimeError(f"Nenhum .npy encontrado em {root_dir}/<classe>/*.npy")

    X_list, X_raw_list, y_list = [], [], []
    for path, y in files:
        arr = np.load(path).astype(np.float32)      # (T0, F0)
        if arr.ndim != 2:
            continue
        X_raw_list.append(pad_or_crop_to_T(arr, T)) # guardamos bruto p/ viz
        arr = apply_feature_mode(arr, feature_mode) # aplica mesmo modo do treino
        if arr.shape[1] != F:
            raise ValueError(f"{path}: F={arr.shape[1]}, esperado F={F}. Verifique feature_mode/treino.")
        arr = pad_or_crop_to_T(arr, T)
        X_list.append(arr); y_list.append(y)

    X = np.stack(X_list, axis=0)                # (N, T, F)
    X_raw = np.stack(X_raw_list, axis=0)        # (N, T, F0_ou_F)
    y = np.array(y_list, dtype=int)
    X = (X - mu) / (sd + 1e-8)                  # normaliza para o modelo
    return X, y, files, X_raw

# ========================= Conexões da mão (MediaPipe) =========================
# Índices (0..20) – mão única
MP_EDGES_ONE = [
    (0,1),(1,2),(2,3),(3,4),          # polegar
    (0,5),(5,6),(6,7),(7,8),          # indicador
    (0,9),(9,10),(10,11),(11,12),     # médio
    (0,13),(13,14),(14,15),(15,16),   # anelar
    (0,17),(17,18),(18,19),(19,20)    # mínimo
]

def edges_for_F(F):
    """Retorna lista de arestas para 21 pts (F=63) ou 42 pts (F=126)."""
    if F == 63:
        return MP_EDGES_ONE
    elif F == 126:
        # duas mãos: (0..20) e (21..41)
        return MP_EDGES_ONE + [(a+21, b+21) for (a,b) in MP_EDGES_ONE]
    else:
        # F desconhecido: sem arestas (só scatter)
        return []

# ========================= Plots (métricas) =========================
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

# ========================= Visualização dos landmarks salvos =========================
def frame_selector_middle(seq):
    """Escolhe o frame central da sequência (T,F)."""
    return seq.shape[0] // 2

def plot_landmarks_2d(seq_raw, outpath, title="Landmarks 2D"):
    """
    seq_raw: (T, F_raw) em ABSOLUTE (sem centragem), se possível.
    Plota o frame central. Assume F_raw ∈ {63,126} (21/42 pts * 3).
    """
    T, F = seq_raw.shape
    t = frame_selector_middle(seq_raw)
    pts = seq_raw[t].reshape(-1, 3)
    xs, ys = pts[:,0], pts[:,1]

    edges = edges_for_F(F)
    plt.figure(figsize=(5,5))
    # conexões
    if edges:
        for a,b in edges:
            plt.plot([xs[a], xs[b]], [ys[a], ys[b]], linewidth=2, alpha=0.7)
    # pontos
    plt.scatter(xs, ys, s=20)
    plt.gca().invert_yaxis()  # como em imagens (origem no canto sup.)
    plt.title(title); plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_landmarks_3d(seq_raw, outpath, title="Landmarks 3D"):
    T, F = seq_raw.shape
    t = frame_selector_middle(seq_raw)
    pts = seq_raw[t].reshape(-1, 3)
    xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
    edges = edges_for_F(F)

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    # conexões
    if edges:
        for a,b in edges:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]], linewidth=1.5, alpha=0.8)
    ax.scatter(xs, ys, zs, s=20)
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def export_landmark_gallery(X_raw, y, actions, outdir, per_class=3):
    """
    Cria uma pequena galeria 2D/3D por classe, usando exemplos do frame central.
    """
    os.makedirs(outdir, exist_ok=True)
    for cls_idx, cls in enumerate(actions):
        idxs = np.where(y == cls_idx)[0][:per_class]
        for k, i in enumerate(idxs):
            seq = X_raw[i]   # (T, F_raw)
            base = f"class_{cls}_ex{k}"
            plot_landmarks_2d(seq, os.path.join(outdir, base + "_2d.png"),
                              title=f"{cls} (2D)")
            plot_landmarks_3d(seq, os.path.join(outdir, base + "_3d.png"),
                              title=f"{cls} (3D)")

# ========================= Métricas adicionais =========================
def expected_calibration_error(probs, y_true, n_bins=10):
    """ECE multi-classe usando confiança da classe predita."""
    y_pred = probs.argmax(1)
    conf = probs[np.arange(len(y_pred)), y_pred]
    acc  = (y_pred == y_true).astype(float)
    edges = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i+1])
        if m.sum() == 0: continue
        bin_acc = acc[m].mean()
        bin_conf = conf[m].mean()
        ece += (m.mean()) * abs(bin_acc - bin_conf)
    return float(ece)

def compute_extra_metrics(y_true, y_pred, probs, actions, outdir):
    """Salva CSV com métricas + imprime κ, MCC, top-k, ECE."""
    rep = classification_report(
        y_true, y_pred, target_names=actions.tolist(),
        digits=4, zero_division=0, output_dict=True
    )
    with open(os.path.join(outdir, "classification_report_detailed.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["class","precision","recall","f1","support"])
        for cls in actions:
            row = rep.get(cls, {})
            w.writerow([cls, row.get("precision",""), row.get("recall",""),
                        row.get("f1-score",""), row.get("support","")])

    kappa = cohen_kappa_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)
    top1  = top_k_accuracy_score(y_true, probs, k=1, labels=range(len(actions)))
    top3  = top_k_accuracy_score(y_true, probs, k=min(3, len(actions)), labels=range(len(actions)))
    ece   = expected_calibration_error(probs, y_true, n_bins=10)

    with open(os.path.join(outdir, "extra_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Cohen's kappa: {kappa:.6f}\n")
        f.write(f"Matthews corrcoef (MCC): {mcc:.6f}\n")
        f.write(f"Top-1 accuracy: {top1:.6f}\n")
        f.write(f"Top-3 accuracy: {top3:.6f}\n")
        f.write(f"Expected Calibration Error (ECE, 10 bins): {ece:.6f}\n")

    print(f"\n[Extras] κ={kappa:.4f} | MCC={mcc:.4f} | Top-1={top1:.4f} | Top-3={top3:.4f} | ECE={ece:.4f}")

# ========================= Main =========================
def main():
    ap = argparse.ArgumentParser(description="Métricas + visualizações usando landmarks salvos (.npy)")
    ap.add_argument("--data_dir", default="libras_data", help="Pasta Classe/*.npy")
    ap.add_argument("--model",    default="best_model.keras")
    ap.add_argument("--actions",  default="libras_actions.npy")
    ap.add_argument("--norm",     default=os.path.join("results_v3","norm_stats.json"))
    ap.add_argument("--outdir",   default="report_figs_v3")
    ap.add_argument("--per_class_examples", type=int, default=2, help="Exemplos de landmarks por classe (2D/3D)")
    args = ap.parse_args()

    actions = load_actions(args.actions)
    mu, sd, T, F, feature_mode = load_norm_stats(args.norm)
    model = load_model_any(args.model)

    n_out = int(model.output_shape[-1])
    if n_out != len(actions):
        print(f"[ERRO] Saídas do modelo ({n_out}) ≠ nº de classes em actions.npy ({len(actions)}).")
        sys.exit(1)

    # Dataset (X normalizado para predição + X_raw para visualização)
    X, y, files, X_raw = build_dataset(args.data_dir, actions, T, F, mu, sd, feature_mode)
    print(f"[INFO] Avaliação: X={X.shape} | X_raw={X_raw.shape} | classes={len(actions)} | feature_mode={feature_mode}")

    probs = model.predict(X, verbose=0)                # (N, C)
    if probs.shape[1] != len(actions):
        print("[ERRO] Probabilidades não batem com nº de classes."); sys.exit(1)
    y_pred = probs.argmax(1)

    ensure_outdir(args.outdir)

    # ---- Relatórios principais
    print("\n=== Classification report ===")
    print(classification_report(y, y_pred, target_names=actions.tolist(), digits=4, zero_division=0))
    plot_class_distribution(y, actions, os.path.join(args.outdir, "class_distribution.png"))
    plot_confusions(y, y_pred, actions,
                    os.path.join(args.outdir, "confusion_raw.png"),
                    os.path.join(args.outdir, "confusion_norm.png"))
    plot_confidence_hist(probs, y_pred, os.path.join(args.outdir, "confidence_hist.png"))
    plot_calibration(probs, y, os.path.join(args.outdir, "calibration_reliability.png"))
    if len(actions) >= 2:
        plot_roc_curves(y, probs, actions, os.path.join(args.outdir, "roc_curves.png"))
        plot_pr_curves(y, probs, actions, os.path.join(args.outdir, "pr_curves.png"))

    # ---- Extras (κ, MCC, Top-k, ECE)
    compute_extra_metrics(y, y_pred, probs, actions, args.outdir)

    # ---- Visualizações dos landmarks salvos (2D/3D)
    viz_dir = os.path.join(args.outdir, "landmarks_viz")
    export_landmark_gallery(X_raw, y, actions, viz_dir, per_class=args.per_class_examples)
    print(f"[OK] Visualizações de landmarks salvas em: {viz_dir}")

    # ---- Misclassifications
    conf = probs[np.arange(len(y_pred)), y_pred]
    with open(os.path.join(args.outdir, "misclassifications.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file","true","pred","confidence"])
        for (path, file_true), y_true, y_hat, c in zip(files, y, y_pred, conf):
            if y_true != y_hat:
                w.writerow([path, str(actions[y_true]), str(actions[y_hat]), f"{c:.4f}"])

    print(f"\n✅ Relatórios e figuras em: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
