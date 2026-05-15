# cross_dataset_eval.py
# -*- coding: utf-8 -*-
import sys, io
# força UTF-8 no stdout para evitar UnicodeEncodeError no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
Avaliação de generalização cross-dataset para o modelo LIBRAS.

Responde à crítica dos revisores sobre generalização entre conjuntos de dados.

Experimentos realizados:
  A) Modelo treinado no dataset CUSTOM avaliado no MINDS
  B) Modelo treinado no dataset MINDS  avaliado no CUSTOM
  C) Comparação com avaliação in-domain (treino/teste no mesmo dataset)

Gera:
  - Tabela comparativa de métricas (in-domain vs cross-domain)
  - Gráfico de barras: queda de performance cross-domain
  - Matriz de confusão por cenário
  - Relatório de classes compartilhadas vs exclusivas

Uso:
    python cross_dataset_eval.py
    python cross_dataset_eval.py --model bilstm_attn
    python cross_dataset_eval.py --custom dataset --minds dataset_minds
    python cross_dataset_eval.py --out results_cross
"""

import os, sys, json, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

import config as cfg
from evaluate import load_model_safe
from data_utils import (
    load_norm_stats, scan_labeled_dir,
    apply_feature_mode, pad_or_crop_to_T,
    list_classes,
)
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de dataset externo (sem treino)
# ─────────────────────────────────────────────────────────────────────────────

def load_external_dataset(data_dir, actions, mu, sd, T, feature_mode):
    """
    Carrega sequências de data_dir usando as actions e normalização de outro modelo.
    Retorna (X, y_true, classes_presentes).
    """
    files = scan_labeled_dir(data_dir, actions)
    if not files:
        return None, None, []

    X_list, y_list = [], []
    for path, yi in files:
        try:
            arr = np.load(path).astype(np.float32)
            if arr.ndim != 2:
                continue
            arr = apply_feature_mode(arr, feature_mode)
            arr = pad_or_crop_to_T(arr, T)
            X_list.append(arr)
            y_list.append(yi)
        except Exception as e:
            print(f"[WARN] Falha ao carregar {path}: {e}")

    if not X_list:
        return None, None, []

    X = (np.stack(X_list) - mu) / (sd + 1e-8)
    y = np.array(y_list, int)

    present = sorted(set(y_list))
    return X, y, present


# ─────────────────────────────────────────────────────────────────────────────
# Avaliação de um modelo num dataset
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model_on_dataset(model_name, eval_data_dir, label):
    """
    Carrega modelo model_name, avalia em eval_data_dir.
    Retorna dict com métricas.
    """
    model_dir  = os.path.join(cfg.MODELS_DIR, model_name)
    model_path = os.path.join(model_dir, "model.keras")
    norm_path  = os.path.join(model_dir, "norm_stats.json")
    act_path   = os.path.join(model_dir, "actions.npy")

    for p in [model_path, norm_path, act_path]:
        if not os.path.exists(p):
            print(f"[WARN] Arquivo não encontrado: {p}  — experimento '{label}' ignorado.")
            return None

    model   = load_model_safe(model_path)
    actions = np.load(act_path)
    mu, sd, T, F, feature_mode = load_norm_stats(norm_path)

    X, y_true, present = load_external_dataset(
        eval_data_dir, actions, mu, sd, T, feature_mode
    )

    if X is None:
        print(f"[WARN] Nenhuma amostra encontrada em '{eval_data_dir}' "
              f"para o modelo '{model_name}'.")
        return None

    probs = model.predict(X, verbose=0)

    # Restringe o argmax apenas às classes presentes no dataset avaliado.
    # Sem isso, amostras de palavras (ex: "Banheiro") podem ser classificadas
    # como letras (ex: "b") que não existem neste split — degradando as
    # métricas artificialmente.
    present_arr = np.array(present)
    probs_restricted = probs[:, present_arr]
    y_pred_local = np.argmax(probs_restricted, axis=1)   # índice dentro de present
    y_true_local = np.array([present.index(yt) for yt in y_true])

    present_actions = [actions[i] for i in present]
    n_present = len(present)
    rep = classification_report(
        y_true_local, y_pred_local,
        labels=list(range(n_present)),
        target_names=present_actions,
        digits=4, zero_division=0, output_dict=True,
    )
    # Reconverte para índices globais para compatibilidade com confusion matrix
    y_pred = present_arr[y_pred_local]

    # Métricas usando índices locais (restritos às classes presentes)
    return {
        "label":       label,
        "model":       model_name,
        "eval_data":   eval_data_dir,
        "n_samples":   len(y_true_local),
        "n_classes":   len(present),
        "actions":     actions,
        "present":     present,
        "y_true":      y_true,        # índices globais (para confusion matrix)
        "y_pred":      y_pred,        # índices globais (para confusion matrix)
        "y_true_local": y_true_local, # índices locais (0..N-1)
        "y_pred_local": y_pred_local, # índices locais (0..N-1)
        "probs":       probs,
        "accuracy":    accuracy_score(y_true_local, y_pred_local),
        "f1_macro":    f1_score(y_true_local, y_pred_local, average="macro", zero_division=0),
        "precision":   precision_score(y_true_local, y_pred_local, average="macro", zero_division=0),
        "recall":      recall_score(y_true_local, y_pred_local, average="macro", zero_division=0),
        "report_dict": rep,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(path, dpi=150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"[OK] Salvo: {path}")


def plot_comparison_bars(results, outdir):
    """
    Gráfico de barras agrupadas comparando métricas de cada experimento.
    """
    labels       = [r["label"] for r in results]
    metrics      = ["accuracy", "precision", "recall", "f1_macro"]
    metric_names = ["Acurácia", "Precisão", "Recall", "F1-macro"]
    colors       = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    n_exp     = len(results)
    n_metrics = len(metrics)
    x         = np.arange(n_exp)
    bar_w     = 0.18

    fig, ax = plt.subplots(figsize=(max(8, n_exp * 2), 6))

    for mi, (mk, ml, color) in enumerate(zip(metrics, metric_names, colors)):
        vals   = [r[mk] for r in results]
        offset = (mi - n_metrics / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, vals, bar_w,
                        label=ml, color=color, alpha=0.85)
        for rect, val in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Valor da métrica")
    ax.set_title("Comparação In-Domain vs Cross-Domain")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    _savefig(os.path.join(outdir, "cross_dataset_comparison.png"))


def plot_drop_heatmap(results, outdir):
    """
    Heatmap de degradação: (cross - in_domain) por métrica.
    Útil para visualizar quanto cada métrica piora no cenário cross-domain.
    """
    if len(results) < 2:
        return

    metrics      = ["accuracy", "f1_macro", "precision", "recall"]
    metric_names = ["Acurácia", "F1-macro", "Precisão", "Recall"]

    n_rows = len(results)
    data   = np.array([[r[m] for m in metrics] for r in results])

    fig, ax = plt.subplots(figsize=(8, max(3, n_rows * 0.7 + 1)))
    labels = [r["label"] for r in results]

    if HAS_SEABORN:
        sns.heatmap(data, annot=True, fmt=".3f", cmap="YlOrRd_r",
                    xticklabels=metric_names, yticklabels=labels,
                    vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5)
    else:
        im = ax.imshow(data, cmap="YlOrRd_r", vmin=0.5, vmax=1.0)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metric_names)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(labels)
        for i in range(n_rows):
            for j in range(len(metrics)):
                ax.text(j, i, f"{data[i, j]:.3f}",
                        ha="center", va="center", fontsize=9)

    ax.set_title("Métricas por Cenário de Avaliação")
    _savefig(os.path.join(outdir, "cross_dataset_heatmap.png"))


def plot_confusion_cross(result, outdir):
    """Matriz de confusão para um resultado."""
    if result is None:
        return
    actions = result["actions"]
    present = result["present"]
    y_true  = result["y_true"]
    y_pred  = result["y_pred"]
    label   = result["label"].replace(" ", "_").replace("/", "-")

    present_names = [actions[i] for i in present]
    cm  = confusion_matrix(y_true, y_pred, labels=present)
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(
        max(6, len(present) * 0.45),
        max(5, len(present) * 0.40),
    ))

    if HAS_SEABORN:
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=present_names, yticklabels=present_names,
                    ax=ax, linewidths=0.3)
    else:
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(present_names)))
        ax.set_xticklabels(present_names)
        ax.set_yticks(range(len(present_names)))
        ax.set_yticklabels(present_names)

    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(f"Matriz de Confusão — {result['label']}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)

    fname = f"confusion_{label}.png"
    _savefig(os.path.join(outdir, fname))


# ─────────────────────────────────────────────────────────────────────────────
# Relatório textual
# ─────────────────────────────────────────────────────────────────────────────

def save_report(results, outdir):
    """Salva tabela comparativa em TXT e CSV."""
    header = (f"{'Cenário':<40} {'N':>6} {'Acc':>7} {'F1':>7} "
              f"{'Prec':>7} {'Recall':>7}")
    sep    = "-" * len(header)

    lines = [
        "=== Avaliacao Cross-Dataset ===",
        sep, header, sep,
    ]
    for r in results:
        lines.append(
            f"{r['label']:<40} {r['n_samples']:>6} "
            f"{r['accuracy']:>7.4f} {r['f1_macro']:>7.4f} "
            f"{r['precision']:>7.4f} {r['recall']:>7.4f}"
        )
    lines.append(sep)
    txt = "\n".join(lines)
    safe = txt.encode("ascii", errors="replace").decode("ascii")
    print("\n" + safe)

    txt_path = os.path.join(outdir, "cross_dataset_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"[OK] Relatório: {txt_path}")

    csv_path = os.path.join(outdir, "cross_dataset_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "model", "eval_data", "n_samples", "n_classes",
                    "accuracy", "f1_macro", "precision", "recall"])
        for r in results:
            w.writerow([
                r["label"], r["model"], r["eval_data"],
                r["n_samples"], r["n_classes"],
                f"{r['accuracy']:.4f}", f"{r['f1_macro']:.4f}",
                f"{r['precision']:.4f}", f"{r['recall']:.4f}",
            ])
    print(f"[OK] CSV: {csv_path}")


def report_shared_classes(custom_dir, minds_dir, outdir):
    """Mostra quantas classes são compartilhadas entre os dois datasets."""
    if not os.path.isdir(custom_dir) or not os.path.isdir(minds_dir):
        return

    custom_classes = set(list_classes(custom_dir).tolist())
    minds_classes  = set(list_classes(minds_dir).tolist())
    shared         = custom_classes & minds_classes
    only_custom    = custom_classes - minds_classes
    only_minds     = minds_classes  - custom_classes

    lines = [
        f"Classes no dataset CUSTOM : {len(custom_classes)}",
        f"Classes no dataset MINDS  : {len(minds_classes)}",
        f"Classes compartilhadas     : {len(shared)}",
        f"Apenas no CUSTOM           : {len(only_custom)} — {sorted(only_custom)}",
        f"Apenas no MINDS            : {len(only_minds)} — {sorted(only_minds)}",
    ]
    txt = "\n".join(lines)
    print("\n" + txt)

    path = os.path.join(outdir, "class_overlap.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"[OK] Overlap de classes: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Avaliação cross-dataset LIBRAS.")
    p.add_argument("--model",  default=cfg.DEFAULT_MODEL_NAME,
                   help="Nome do modelo a avaliar (pasta em models/)")
    p.add_argument("--custom", default=cfg.DATA_DIR,
                   help="Pasta do dataset custom (padrão: dataset)")
    p.add_argument("--minds",  default=cfg.MINDS_DATA_DIR,
                   help="Pasta do dataset MINDS (padrão: dataset_minds)")
    p.add_argument("--out",    default="results_cross",
                   help="Diretório de saída (padrão: results_cross)")
    p.add_argument("--model_minds", default=None,
                   help="Modelo treinado no MINDS (para experimento B); "
                        "se não fornecido, usa o mesmo --model para tudo")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    report_shared_classes(args.custom, args.minds, args.out)

    experiments = []

    # A) Modelo treinado no CUSTOM, avaliado no CUSTOM (in-domain)
    r = evaluate_model_on_dataset(
        args.model, args.custom,
        label=f"CUSTOM→CUSTOM (in-domain)",
    )
    if r:
        experiments.append(r)

    # B) Modelo treinado no CUSTOM, avaliado no MINDS (cross-domain)
    if os.path.isdir(args.minds):
        r = evaluate_model_on_dataset(
            args.model, args.minds,
            label=f"CUSTOM→MINDS (cross-domain)",
        )
        if r:
            experiments.append(r)
    else:
        print(f"[WARN] Dataset MINDS não encontrado em '{args.minds}' — "
              f"experimento cross-domain ignorado.")

    # C) Modelo treinado no MINDS, avaliado no CUSTOM (se disponível)
    if args.model_minds:
        model_minds_dir = os.path.join(cfg.MODELS_DIR, args.model_minds)
        if os.path.isdir(model_minds_dir):
            r = evaluate_model_on_dataset(
                args.model_minds, args.custom,
                label=f"MINDS→CUSTOM (cross-domain)",
            )
            if r:
                experiments.append(r)
        else:
            print(f"[WARN] Modelo MINDS '{args.model_minds}' não encontrado.")

    if not experiments:
        print("[ERRO] Nenhum experimento foi executado com sucesso.")
        sys.exit(1)

    save_report(experiments, args.out)

    if len(experiments) >= 1:
        plot_comparison_bars(experiments, args.out)
        plot_drop_heatmap(experiments, args.out)
        for r in experiments:
            plot_confusion_cross(r, args.out)

    print(f"\n[OK] Avaliação cross-dataset concluída. Resultados em: "
          f"{os.path.abspath(args.out)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
