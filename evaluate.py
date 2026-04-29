# evaluate.py
# -*- coding: utf-8 -*-
"""
Avaliação completa de um modelo LIBRAS treinado.

Gera:
  - Classification report (CSV + txt)
  - Métricas globais com bootstrap ± std
  - Cohen's κ, MCC, Top-3 accuracy, ECE
  - Matriz de confusão (abs + norm, seaborn heatmap)
  - Curvas ROC e Precision-Recall (por classe + micro-average)
  - F1 por classe (barras)
  - Histograma de confiança (acertos vs erros)
  - Diagrama de calibração
  - t-SNE das embeddings

Uso:
    python evaluate.py                         # usa modelo padrão (bilstm_attn)
    python evaluate.py --model bilstm
    python evaluate.py --model bilstm_attn --data dataset
"""

import os, sys, json, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # sem janela gráfica
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    cohen_kappa_score, matthews_corrcoef, top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

import config as cfg
from data_utils import (
    load_sequences, make_split, compute_norm_stats,
    load_norm_stats, make_dataset, materialize_dataset,
    bootstrap_metrics, expected_calibration_error, scan_labeled_dir,
    apply_feature_mode, pad_or_crop_to_T,
)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de modelo
# ─────────────────────────────────────────────────────────────────────────────

def load_model_safe(path):
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
                f"  tf.keras: {e1}\n  keras.saving: {e2}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Plots individuais
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(path, dpi=150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_confusion(y_true, y_pred, actions, outdir):
    cm  = confusion_matrix(y_true, y_pred, labels=range(len(actions)))
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    for matrix, fmt, title, fname, cmap in [
        (cm,  "d",    "Matriz de Confusão (Absoluta)",    "confusion_abs.png",  "Reds"),
        (cmn, ".2f",  "Matriz de Confusão (Normalizada)", "confusion_norm.png", "Blues"),
    ]:
        fig, ax = plt.subplots(figsize=(max(8, len(actions) * 0.45),
                                        max(6, len(actions) * 0.40)))
        if HAS_SEABORN:
            sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap,
                        xticklabels=actions, yticklabels=actions,
                        ax=ax, linewidths=0.3)
        else:
            im = ax.imshow(matrix, cmap=cmap, vmin=0,
                           vmax=(1.0 if fmt == ".2f" else None))
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(actions))); ax.set_xticklabels(actions)
            ax.set_yticks(range(len(actions))); ax.set_yticklabels(actions)
        ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
        _savefig(os.path.join(outdir, fname))

    return cm, cmn


def plot_roc_curves(y_true, probs, actions, outdir):
    Y = label_binarize(y_true, classes=range(len(actions)))
    plt.figure(figsize=(9, 7))
    # micro-average
    fpr, tpr, _ = roc_curve(Y.ravel(), probs.ravel())
    plt.plot(fpr, tpr, lw=2.5, label=f"micro-avg AUC={auc(fpr, tpr):.3f}")
    # por classe
    for i, cls in enumerate(actions):
        fi, ti, _ = roc_curve(Y[:, i], probs[:, i])
        plt.plot(fi, ti, alpha=0.25, lw=1, label=cls)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curvas ROC")
    plt.legend(ncol=3, fontsize=7, loc="lower right")
    _savefig(os.path.join(outdir, "roc_curves.png"))


def plot_pr_curves(y_true, probs, actions, outdir):
    Y  = label_binarize(y_true, classes=range(len(actions)))
    ap = average_precision_score(Y, probs, average="micro")
    p, r, _ = precision_recall_curve(Y.ravel(), probs.ravel())
    plt.figure(figsize=(9, 7))
    plt.plot(r, p, lw=2.5, label=f"micro-avg AP={ap:.3f}")
    for i, cls in enumerate(actions):
        pi, ri, _ = precision_recall_curve(Y[:, i], probs[:, i])
        plt.plot(ri, pi, alpha=0.25, lw=1, label=cls)
    plt.xlabel("Recall"); plt.ylabel("Precisão"); plt.title("Curvas Precision-Recall")
    plt.legend(ncol=3, fontsize=7, loc="lower left")
    _savefig(os.path.join(outdir, "pr_curves.png"))


def plot_f1_bars(report_dict, actions, outdir):
    f1 = [report_dict.get(cls, {}).get("f1-score", 0.0) for cls in actions]
    colors = ["#e74c3c" if v < 0.80 else "#f39c12" if v < 0.90 else "#2ecc71"
              for v in f1]
    plt.figure(figsize=(max(10, len(actions) * 0.35), 5))
    bars = plt.bar(range(len(actions)), f1, color=colors)
    plt.axhline(np.mean(f1), color="gray", linestyle="--", label=f"média={np.mean(f1):.3f}")
    plt.xticks(range(len(actions)), actions, rotation=45, ha="right", fontsize=8)
    plt.ylim(0, 1.05); plt.ylabel("F1-score"); plt.title("F1-score por Classe")
    plt.legend()
    _savefig(os.path.join(outdir, "f1_per_class.png"))


def plot_confidence_split(probs, y_pred, y_true, outdir):
    """Histograma de confiança separado em acertos e erros."""
    conf    = probs[np.arange(len(y_pred)), y_pred]
    correct = conf[y_pred == y_true]
    wrong   = conf[y_pred != y_true]
    plt.figure(figsize=(8, 4))
    plt.hist(correct, bins=25, alpha=0.65, color="#2ecc71",
             label=f"Acertos (n={len(correct)})")
    plt.hist(wrong,   bins=25, alpha=0.65, color="#e74c3c",
             label=f"Erros (n={len(wrong)})")
    plt.xlabel("Confiança da predição"); plt.ylabel("Amostras")
    plt.title("Distribuição de Confiança: Acertos vs Erros")
    plt.legend()
    _savefig(os.path.join(outdir, "confidence_split.png"))


def plot_calibration(probs, y_true, outdir, n_bins=10):
    y_pred = probs.argmax(1)
    conf   = probs[np.arange(len(y_pred)), y_pred]
    acc    = (y_pred == y_true).astype(float)
    edges  = np.linspace(0, 1, n_bins + 1)
    mids, accs, sizes = [], [], []
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() == 0:
            continue
        mids.append((edges[i] + edges[i + 1]) / 2)
        accs.append(acc[m].mean())
        sizes.append(m.sum())
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Calibração ideal")
    sc = plt.scatter(mids, accs, c=sizes, cmap="Blues", s=80, zorder=5,
                     edgecolors="black", linewidths=0.5)
    plt.colorbar(sc, label="Amostras no bin")
    plt.plot(mids, accs, "o-", color="#3498db", alpha=0.7)
    plt.xlabel("Confiança predita"); plt.ylabel("Acurácia empírica")
    plt.title("Diagrama de Calibração (Reliability Diagram)")
    plt.legend()
    _savefig(os.path.join(outdir, "calibration.png"))


def plot_class_distribution(y, actions, outdir):
    counts = np.bincount(y, minlength=len(actions))
    plt.figure(figsize=(max(10, len(actions) * 0.35), 4))
    plt.bar(range(len(actions)), counts, color="#3498db")
    plt.axhline(counts.mean(), color="gray", linestyle="--",
                label=f"média={counts.mean():.1f}")
    plt.xticks(range(len(actions)), actions, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Amostras"); plt.title("Distribuição de Classes")
    plt.legend()
    _savefig(os.path.join(outdir, "class_distribution.png"))


def plot_tsne(model, X_test, y_true, actions, outdir):
    """t-SNE das embeddings da penúltima camada."""
    if not HAS_TSNE:
        print("[WARN] sklearn não tem TSNE — pulando t-SNE.")
        return
    import tensorflow as tf
    try:
        # extrai penúltima camada (antes da Dense softmax)
        feat_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output,
        )
        features = feat_model.predict(X_test, verbose=0)
    except Exception as e:
        print(f"[WARN] Falha ao extrair features para t-SNE: {e}")
        return

    perp = min(30, len(X_test) // 5)
    tsne = TSNE(n_components=2, perplexity=max(5, perp), random_state=42, n_jobs=-1)
    emb  = tsne.fit_transform(features)

    n_cls  = len(actions)
    cmap   = plt.cm.get_cmap("tab20", n_cls)
    plt.figure(figsize=(14, 10))
    for i, cls in enumerate(actions):
        mask = y_true == i
        plt.scatter(emb[mask, 0], emb[mask, 1],
                    color=cmap(i), label=cls, s=25, alpha=0.75)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=7, ncol=2, markerscale=1.5)
    plt.title("t-SNE das Embeddings (camada penúltima)")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    _savefig(os.path.join(outdir, "tsne_embeddings.png"), dpi=120)
    print("[OK] t-SNE salvo.")


# ─────────────────────────────────────────────────────────────────────────────
# Função principal de avaliação (usada por train.py e pelo CLI)
# ─────────────────────────────────────────────────────────────────────────────

def eval_and_save(model, X_test, y_test_onehot, actions, outdir,
                  n_bootstrap=1000, seed=42, compute_tsne=True):
    """
    Avaliação completa.  X_test e y_test_onehot são numpy arrays.
    y_test_onehot pode ser one-hot ou inteiro — detecta automaticamente.
    """
    os.makedirs(outdir, exist_ok=True)

    probs  = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    if y_test_onehot.ndim == 2:
        y_true = np.argmax(y_test_onehot, axis=1)
    else:
        y_true = y_test_onehot.astype(int)

    # ── Classification report ──────────────────────────────────────────────
    rep_str  = classification_report(y_true, y_pred,
                                     target_names=actions.tolist(),
                                     digits=4, zero_division=0)
    rep_dict = classification_report(y_true, y_pred,
                                     target_names=actions.tolist(),
                                     digits=4, zero_division=0,
                                     output_dict=True)
    print("\n" + "="*60)
    print(rep_str)

    with open(os.path.join(outdir, "classification_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(rep_str)

    with open(os.path.join(outdir, "per_class_metrics.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for cls in actions:
            r = rep_dict.get(cls, {})
            w.writerow([cls, r.get("precision",""), r.get("recall",""),
                        r.get("f1-score",""), r.get("support","")])

    # ── Métricas extras ────────────────────────────────────────────────────
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)
    top1  = top_k_accuracy_score(y_true, probs, k=1, labels=range(len(actions)))
    top3  = top_k_accuracy_score(y_true, probs, k=min(3, len(actions)),
                                  labels=range(len(actions)))
    ece   = expected_calibration_error(probs, y_true)

    extra = (
        f"Cohen's kappa          : {kappa:.6f}\n"
        f"Matthews corrcoef (MCC): {mcc:.6f}\n"
        f"Top-1 accuracy         : {top1:.6f}\n"
        f"Top-3 accuracy         : {top3:.6f}\n"
        f"ECE (10 bins)          : {ece:.6f}\n"
    )
    print(extra)
    with open(os.path.join(outdir, "extra_metrics.txt"), "w",
              encoding="utf-8") as f:
        f.write(extra)

    # ── Bootstrap ─────────────────────────────────────────────────────────
    boot = bootstrap_metrics(y_true, y_pred, n_boot=n_bootstrap, seed=seed)
    pm   = lambda mean, std: f"{mean:.4f} ± {std:.4f}"
    print("─" * 60)
    print(f"Acurácia  (bootstrap): {pm(boot['accuracy_mean'],         boot['accuracy_std'])}")
    print(f"Precisão  (bootstrap): {pm(boot['precision_macro_mean'],  boot['precision_macro_std'])}")
    print(f"Recall    (bootstrap): {pm(boot['recall_macro_mean'],     boot['recall_macro_std'])}")
    print(f"F1-macro  (bootstrap): {pm(boot['f1_macro_mean'],         boot['f1_macro_std'])}")

    with open(os.path.join(outdir, "metrics_bootstrap.json"), "w",
              encoding="utf-8") as f:
        json.dump(boot, f, indent=2)

    with open(os.path.join(outdir, "metrics_bootstrap.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std"])
        for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            w.writerow([k, f"{boot[k+'_mean']:.6f}", f"{boot[k+'_std']:.6f}"])

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_confusion(y_true, y_pred, actions, outdir)
    plot_f1_bars(rep_dict, actions, outdir)
    plot_confidence_split(probs, y_pred, y_true, outdir)
    plot_calibration(probs, y_true, outdir)
    plot_class_distribution(y_true, actions, outdir)

    if len(actions) >= 2:
        plot_roc_curves(y_true, probs, actions, outdir)
        plot_pr_curves(y_true, probs, actions, outdir)

    if compute_tsne:
        plot_tsne(model, X_test, y_true, actions, outdir)

    # ── Misclassifications ─────────────────────────────────────────────────
    conf = probs[np.arange(len(y_pred)), y_pred]
    with open(os.path.join(outdir, "misclassifications.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true", "pred", "confidence"])
        for yt, yp, c in zip(y_true, y_pred, conf):
            if yt != yp:
                w.writerow([actions[yt], actions[yp], f"{c:.4f}"])

    print(f"\n[OK] Resultados salvos em: {os.path.abspath(outdir)}")
    return boot


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Avalia modelo LIBRAS treinado.")
    p.add_argument("--model",  default=cfg.DEFAULT_MODEL_NAME,
                   help="Nome do modelo (pasta em models/)")
    p.add_argument("--data",   default=cfg.DATA_DIR)
    p.add_argument("--no_tsne", action="store_true")
    return p.parse_args()


def main():
    args       = parse_args()
    model_dir  = os.path.join(cfg.MODELS_DIR, args.model)
    model_path = os.path.join(model_dir, "model.keras")
    norm_path  = os.path.join(model_dir, "norm_stats.json")
    act_path   = os.path.join(model_dir, "actions.npy")
    out_dir    = os.path.join(cfg.RESULTS_DIR, args.model)

    for p in [model_path, norm_path, act_path]:
        if not os.path.exists(p):
            print(f"[ERRO] Arquivo não encontrado: {p}")
            print("       Treine o modelo primeiro: python train.py --model " + args.model)
            sys.exit(1)

    model   = load_model_safe(model_path)
    actions = np.load(act_path)
    mu, sd, T, F, feature_mode = load_norm_stats(norm_path)

    # Valida compatibilidade
    n_out = int(model.output_shape[-1])
    if n_out != len(actions):
        print(f"[ERRO] Modelo tem {n_out} saídas mas actions.npy tem {len(actions)} classes.")
        sys.exit(1)

    # Carrega dataset e normaliza
    files = scan_labeled_dir(args.data, actions)
    if not files:
        print(f"[ERRO] Nenhum .npy encontrado em {args.data}/<classe>/")
        sys.exit(1)

    X_list, y_list = [], []
    for path, yi in files:
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2:
            continue
        arr = apply_feature_mode(arr, feature_mode)
        arr = pad_or_crop_to_T(arr, T)
        X_list.append(arr); y_list.append(yi)

    X = (np.stack(X_list) - mu) / (sd + 1e-8)
    y = np.array(y_list, int)

    print(f"[INFO] {len(X)} sequências | {len(actions)} classes | feature_mode={feature_mode}")

    eval_and_save(model, X, y, actions, out_dir,
                  n_bootstrap=cfg.N_BOOTSTRAP, seed=cfg.SEED,
                  compute_tsne=not args.no_tsne)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
