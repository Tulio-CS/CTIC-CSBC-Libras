# minds_study.py
# -*- coding: utf-8 -*-
"""
Estudo de desempenho do BiLSTM+Atenção no dataset MINDS e no MINDS+Custom combinado.

Executa 6 cenários de avaliação e 2 treinamentos novos:

  Modelos treinados
  ─────────────────
  • bilstm_attn          — treinado no custom (55 classes)  [já deve existir]
  • bilstm_attn_minds    — treinado SOMENTE no MINDS (20 classes) [treinado aqui]
  • bilstm_attn_combined — treinado no custom + MINDS (55 classes, mais dados
                           para as 20 classes compartilhadas) [treinado aqui]

  Cenários de avaliação
  ──────────────────────
  A) Custom → Custom          bilstm_attn       avaliado em dataset/
  B) Custom → MINDS           bilstm_attn       avaliado em dataset_minds/
  C) MINDS → MINDS            bilstm_attn_minds avaliado em dataset_minds/
  D) MINDS → Custom[shared]   bilstm_attn_minds avaliado em dataset/ (20 classes MINDS)
  E) Combined → Custom        bilstm_attn_combined avaliado em dataset/
  F) Combined → MINDS         bilstm_attn_combined avaliado em dataset_minds/

  Gráficos e relatórios gerados
  ──────────────────────────────
  • Barras comparativas de todas as métricas (todos os cenários)
  • Heatmap de métricas (cenários × métricas)
  • F1 por classe das 20 classes MINDS (comparando cenários B, C, F)
  • Matrizes de confusão (todos os cenários)
  • Análise de volume de dados por classe (custom vs MINDS vs combined)
  • Tabela resumo em TXT e CSV

Uso:
    python minds_study.py                         # executa tudo
    python minds_study.py --skip_training         # pula treino, só avalia
    python minds_study.py --epochs 80
    python minds_study.py --out results_minds
"""

import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, json, csv, argparse, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

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
    list_classes, load_sequences,
)
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CUSTOM   = "bilstm_attn"
MODEL_MINDS    = "bilstm_attn_minds"
MODEL_COMBINED = "bilstm_attn_combined"

SCENARIO_COLORS = {
    "A": "#3498db",
    "B": "#e74c3c",
    "C": "#2ecc71",
    "D": "#f39c12",
    "E": "#9b59b6",
    "F": "#1abc9c",
}


# ─────────────────────────────────────────────────────────────────────────────
# Treinamento de novos modelos
# ─────────────────────────────────────────────────────────────────────────────

def train_model(run_name, data_dir, extra_data, epochs, skip_existing):
    """Treina um modelo via subprocess do train.py."""
    model_path = os.path.join(cfg.MODELS_DIR, run_name, "model.keras")
    if skip_existing and os.path.exists(model_path):
        print(f"[SKIP] Modelo '{run_name}' já existe.")
        return True

    cmd = [
        sys.executable, "train.py",
        "--model",   "bilstm_attn",
        "--data",    data_dir,
        "--epochs",  str(epochs),
        "--run_name", run_name,
    ]
    if extra_data:
        cmd += ["--extra_data", extra_data]

    print(f"\n{'='*60}")
    print(f"[TREINO] {run_name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[WARN] Treino '{run_name}' terminou com código {ret.returncode}.")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Avaliação de um modelo num dataset
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_scenario(model_name, eval_data_dir, label, scenario_id,
                      restrict_to_classes=None):
    """
    Carrega o modelo model_name e avalia em eval_data_dir.

    restrict_to_classes: set de nomes de classes para restringir o argmax
      (útil quando um modelo de 55 classes é avaliado em dados de 20 classes,
       ou vice-versa).

    Retorna dict com métricas e dados para plots.
    """
    model_dir  = os.path.join(cfg.MODELS_DIR, model_name)
    model_path = os.path.join(model_dir, "model.keras")
    norm_path  = os.path.join(model_dir, "norm_stats.json")
    act_path   = os.path.join(model_dir, "actions.npy")

    for p in [model_path, norm_path, act_path]:
        if not os.path.exists(p):
            print(f"[WARN] {p} não encontrado — cenário '{label}' ignorado.")
            return None

    model   = load_model_safe(model_path)
    actions = np.load(act_path)
    mu, sd, T, F, feature_mode = load_norm_stats(norm_path)

    # Carrega dados de avaliação
    files = scan_labeled_dir(eval_data_dir, actions)

    # Se restrict_to_classes, filtra os arquivos
    if restrict_to_classes:
        files = [(p, yi) for p, yi in files
                 if actions[yi] in restrict_to_classes]

    if not files:
        print(f"[WARN] Nenhum arquivo encontrado em '{eval_data_dir}' "
              f"para o modelo '{model_name}'.")
        return None

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
            print(f"[WARN] {path}: {e}")

    if not X_list:
        return None

    X      = (np.stack(X_list) - mu) / (sd + 1e-8)
    y_true = np.array(y_list, int)

    probs   = model.predict(X, verbose=0)
    present = sorted(set(y_list))

    # Restringe argmax às classes presentes no split de avaliação
    present_arr      = np.array(present)
    probs_restricted = probs[:, present_arr]
    y_pred_local     = np.argmax(probs_restricted, axis=1)
    y_true_local     = np.array([present.index(yt) for yt in y_true])

    present_names = [actions[i] for i in present]
    rep = classification_report(
        y_true_local, y_pred_local,
        labels=list(range(len(present))),
        target_names=present_names,
        digits=4, zero_division=0, output_dict=True,
    )
    y_pred = present_arr[y_pred_local]

    acc  = accuracy_score(y_true_local, y_pred_local)
    f1   = f1_score(y_true_local, y_pred_local, average="macro", zero_division=0)
    prec = precision_score(y_true_local, y_pred_local, average="macro", zero_division=0)
    rec  = recall_score(y_true_local, y_pred_local, average="macro", zero_division=0)

    print(f"  [{scenario_id}] {label:<46}  "
          f"Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
          f"(n={len(y_true_local)}, cls={len(present)})")

    return {
        "id":           scenario_id,
        "label":        label,
        "model":        model_name,
        "eval_data":    eval_data_dir,
        "n_samples":    len(y_true_local),
        "n_classes":    len(present),
        "actions":      actions,
        "present":      present,
        "present_names": present_names,
        "y_true":       y_true,
        "y_pred":       y_pred,
        "y_true_local": y_true_local,
        "y_pred_local": y_pred_local,
        "probs":        probs,
        "accuracy":     acc,
        "f1_macro":     f1,
        "precision":    prec,
        "recall":       rec,
        "report_dict":  rep,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(path, dpi=150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"[OK] {path}")


def plot_comparison_bars(results, outdir):
    """Barras agrupadas: métricas por cenário."""
    labels       = [f"({r['id']}) {r['label']}" for r in results]
    metric_keys  = ["accuracy", "precision", "recall", "f1_macro"]
    metric_names = ["Acurácia", "Precisão (macro)", "Recall (macro)", "F1-macro"]
    colors       = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    n_exp     = len(results)
    n_metrics = len(metric_keys)
    x         = np.arange(n_exp)
    bar_w     = 0.18

    fig, ax = plt.subplots(figsize=(max(10, n_exp * 2.2), 6))

    for mi, (mk, ml, color) in enumerate(zip(metric_keys, metric_names, colors)):
        vals   = [r[mk] for r in results]
        offset = (mi - n_metrics / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, vals, bar_w,
                        label=ml, color=color, alpha=0.85)
        for rect, val in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Valor da métrica")
    ax.set_title("Desempenho por Cenário — BiLSTM+Atenção no MINDS e no MINDS+Custom")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _savefig(os.path.join(outdir, "minds_comparison_bars.png"))


def plot_metrics_heatmap(results, outdir):
    """Heatmap de métricas: cenários × métricas."""
    metric_keys  = ["accuracy", "f1_macro", "precision", "recall"]
    metric_names = ["Acurácia", "F1-macro", "Precisão", "Recall"]
    labels       = [f"({r['id']}) {r['label']}" for r in results]
    data         = np.array([[r[m] for m in metric_keys] for r in results])

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(results) * 0.75 + 1)))
    if HAS_SEABORN:
        sns.heatmap(data, annot=True, fmt=".4f", cmap="YlOrRd_r",
                    xticklabels=metric_names, yticklabels=labels,
                    vmin=max(0, data.min() - 0.05), vmax=1.0,
                    ax=ax, linewidths=0.5,
                    annot_kws={"size": 9})
    else:
        vmin = max(0, data.min() - 0.05)
        im = ax.imshow(data, cmap="YlOrRd_r", vmin=vmin, vmax=1.0)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(metric_keys)))
        ax.set_xticklabels(metric_names)
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels(labels)
        for i in range(len(results)):
            for j in range(len(metric_keys)):
                ax.text(j, i, f"{data[i, j]:.4f}",
                        ha="center", va="center", fontsize=8.5)

    ax.set_title("Métricas por Cenário de Avaliação")
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    _savefig(os.path.join(outdir, "minds_metrics_heatmap.png"))


def plot_f1_per_minds_class(results_subset, minds_classes, outdir):
    """
    F1 por classe das 20 classes MINDS, comparando múltiplos cenários.
    results_subset: lista de (label, report_dict) dos cenários que avaliaram no MINDS.
    minds_classes : lista ordenada dos nomes das 20 classes MINDS.
    """
    if not results_subset:
        return

    n_cls     = len(minds_classes)
    n_results = len(results_subset)
    x         = np.arange(n_cls)
    bar_w     = 0.8 / n_results
    colors    = ["#e74c3c", "#2ecc71", "#1abc9c", "#f39c12"]

    fig, ax = plt.subplots(figsize=(max(14, n_cls * 0.65), 6))

    for ri, (label, rep) in enumerate(results_subset):
        f1s = []
        for cls in minds_classes:
            f1s.append(rep.get(cls, {}).get("f1-score", 0.0))
        offset = (ri - n_results / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, f1s, bar_w,
                      label=label, color=colors[ri % len(colors)], alpha=0.85)
        for rect, val in zip(bars, f1s):
            if val > 0:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(minds_classes, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1-score")
    ax.set_title("F1 por Classe — 20 Sinais do MINDS\n"
                 "(comparação entre cenários que avaliam no MINDS)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(0.9, color="gray", linestyle=":", alpha=0.5, label="0.90")
    _savefig(os.path.join(outdir, "minds_f1_per_class.png"))


def plot_delta_from_indomain(results, outdir):
    """
    Barras mostrando a queda (Δ) de cada cenário em relação ao melhor in-domain.
    Negativo = degradação; positivo = melhoria.
    """
    # Encontra baseline in-domain para cada "domínio de treino"
    baselines = {}
    for r in results:
        if "→" in r["label"]:
            train_domain = r["label"].split("→")[0].strip().split()[0]
            eval_domain  = r["label"].split("→")[1].strip().split()[0]
            if train_domain == eval_domain:
                baselines[train_domain] = r["accuracy"]

    deltas, labels, colors_list = [], [], []
    for r in results:
        if "→" in r["label"]:
            train_domain = r["label"].split("→")[0].strip().split()[0]
            eval_domain  = r["label"].split("→")[1].strip().split()[0]
            base = baselines.get(eval_domain)
            if base is None:
                base = baselines.get(train_domain)
            if base is not None and train_domain != eval_domain:
                delta = r["accuracy"] - base
                deltas.append(delta)
                labels.append(f"({r['id']}) {r['label']}")
                colors_list.append("#e74c3c" if delta < 0 else "#2ecc71")

    if not deltas:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(deltas) * 1.8), 5))
    bars = ax.bar(range(len(deltas)), deltas, color=colors_list, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    for rect, val in zip(bars, deltas):
        ax.text(rect.get_x() + rect.get_width() / 2,
                val + (0.005 if val >= 0 else -0.015),
                f"{val:+.4f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Δ Acurácia vs. in-domain")
    ax.set_title("Degradação Cross-Domain (negativo = pior que in-domain)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _savefig(os.path.join(outdir, "minds_delta_accuracy.png"))


def plot_data_volume(custom_dir, minds_dir, outdir):
    """Barras de volume de dados: amostras por classe nos 3 datasets."""
    minds_classes = sorted(list_classes(minds_dir).tolist())
    custom_counts = {}
    minds_counts  = {}

    for cls in minds_classes:
        import glob
        cp = os.path.join(custom_dir, cls, "*.npy")
        mp = os.path.join(minds_dir,  cls, "*.npy")
        custom_counts[cls] = len(glob.glob(cp))
        minds_counts[cls]  = len(glob.glob(mp))

    n   = len(minds_classes)
    x   = np.arange(n)
    w   = 0.3

    fig, ax = plt.subplots(figsize=(max(12, n * 0.65), 5))
    b1 = ax.bar(x - w/2, [custom_counts[c] for c in minds_classes],
                w, label="Custom", color="#3498db", alpha=0.85)
    b2 = ax.bar(x + w/2, [minds_counts[c] for c in minds_classes],
                w, label="MINDS", color="#e74c3c", alpha=0.85)

    # Combined total (linha)
    combined = [custom_counts[c] + minds_counts[c] for c in minds_classes]
    ax.plot(x, combined, "D--", color="#2c3e50", markersize=5,
            label="Combined (total)", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(minds_classes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Número de sequências")
    ax.set_title("Volume de dados por classe — Custom vs MINDS vs Combined")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _savefig(os.path.join(outdir, "minds_data_volume.png"))


def plot_confusion(result, outdir):
    """Matriz de confusão normalizada para um cenário."""
    if result is None:
        return
    actions = result["actions"]
    present = result["present"]
    y_true  = result["y_true"]
    y_pred  = result["y_pred"]
    label   = result["label"].replace(" ", "_").replace("/", "-").replace("→", "to")
    sid     = result["id"]

    present_names = [actions[i] for i in present]
    cm  = confusion_matrix(y_true, y_pred, labels=present)
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    n = len(present)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.48), max(5, n * 0.42)))
    if HAS_SEABORN:
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=present_names, yticklabels=present_names,
                    ax=ax, linewidths=0.3, vmin=0, vmax=1)
    else:
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n)); ax.set_xticklabels(present_names)
        ax.set_yticks(range(n)); ax.set_yticklabels(present_names)
    ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")
    ax.set_title(f"Confusão ({sid}) {result['label']}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    _savefig(os.path.join(outdir, f"confusion_{sid}_{label}.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Relatório textual / CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_report(results, outdir):
    """Tabela comparativa TXT + CSV."""
    header = (f"{'ID':<4} {'Cenário':<48} {'N':>5} {'Cls':>4} "
              f"{'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}")
    sep    = "─" * len(header)

    lines  = ["", "═"*len(header),
              "  ESTUDO DE DESEMPENHO: BiLSTM+Atenção no MINDS e MINDS+Custom",
              "═"*len(header),
              "", header, sep]

    for r in results:
        lines.append(
            f"({r['id']})  {r['label']:<48} {r['n_samples']:>5} {r['n_classes']:>4} "
            f"{r['accuracy']:>7.4f} {r['f1_macro']:>7.4f} "
            f"{r['precision']:>7.4f} {r['recall']:>7.4f}"
        )
    lines.append(sep)

    # Legenda dos modelos
    lines += [
        "",
        "  Modelos:",
        f"  • {MODEL_CUSTOM:<26} treinado em dataset/ (55 classes, dados coletados)",
        f"  • {MODEL_MINDS:<26} treinado em dataset_minds/ (20 classes MINDS)",
        f"  • {MODEL_COMBINED:<26} treinado em dataset/ + dataset_minds/ (55 cl.)",
        "",
        "  Cenários:",
        "  A) Custom→Custom        : baseline in-domain (55 classes)",
        "  B) Custom→MINDS         : zero-shot transfer (sem ver MINDS no treino)",
        "  C) MINDS→MINDS          : in-domain MINDS (treino e teste no MINDS)",
        "  D) MINDS→Custom[shared] : modelo MINDS avaliado no custom (20 classes)",
        "  E) Combined→Custom      : modelo combinado avaliado no custom",
        "  F) Combined→MINDS       : modelo combinado avaliado no MINDS",
    ]

    txt = "\n".join(lines)
    safe = txt.encode("ascii", errors="replace").decode("ascii")
    print("\n" + safe)

    txt_path = os.path.join(outdir, "minds_study_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"[OK] {txt_path}")

    csv_path = os.path.join(outdir, "minds_study_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "model", "eval_data",
                    "n_samples", "n_classes",
                    "accuracy", "f1_macro", "precision", "recall"])
        for r in results:
            w.writerow([
                r["id"], r["label"], r["model"], r["eval_data"],
                r["n_samples"], r["n_classes"],
                f"{r['accuracy']:.4f}", f"{r['f1_macro']:.4f}",
                f"{r['precision']:.4f}", f"{r['recall']:.4f}",
            ])
    print(f"[OK] {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Estudo de desempenho do BiLSTM+Atenção no MINDS e MINDS+Custom."
    )
    p.add_argument("--custom",  default=cfg.DATA_DIR,
                   help=f"Dataset custom (padrão: {cfg.DATA_DIR})")
    p.add_argument("--minds",   default=cfg.MINDS_DATA_DIR,
                   help=f"Dataset MINDS (padrão: {cfg.MINDS_DATA_DIR})")
    p.add_argument("--epochs",  type=int, default=80,
                   help="Épocas para os novos modelos (padrão: 80)")
    p.add_argument("--out",     default="results_minds",
                   help="Diretório de saída (padrão: results_minds)")
    p.add_argument("--skip_training", action="store_true",
                   help="Pula os treinamentos, só executa avaliação")
    p.add_argument("--skip_existing", action="store_true",
                   help="Pula treinamento se o modelo já existe")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.path.isdir(args.minds):
        print(f"[ERRO] Dataset MINDS não encontrado: '{args.minds}'")
        print(f"       Rode process_minds.py primeiro para gerar as sequências.")
        sys.exit(1)

    # ── Classes compartilhadas ────────────────────────────────────────────────
    custom_classes = set(list_classes(args.custom).tolist())
    minds_classes  = sorted(list_classes(args.minds).tolist())
    shared         = sorted(custom_classes & set(minds_classes))

    print(f"\n[INFO] Classes em custom : {len(custom_classes)}")
    print(f"[INFO] Classes em MINDS  : {len(minds_classes)}")
    print(f"[INFO] Classes shared    : {len(shared)} — {shared}")

    # ── Etapa 1: Treinamentos ─────────────────────────────────────────────────
    if not args.skip_training:
        print(f"\n{'#'*60}")
        print("# ETAPA 1: TREINAMENTO")
        print(f"{'#'*60}")

        # Modelo treinado só no MINDS (20 classes)
        train_model(
            run_name=MODEL_MINDS,
            data_dir=args.minds,
            extra_data=None,
            epochs=args.epochs,
            skip_existing=args.skip_existing,
        )

        # Modelo treinado no custom + MINDS (55 classes, mais dados para as 20 shared)
        train_model(
            run_name=MODEL_COMBINED,
            data_dir=args.custom,
            extra_data=args.minds,
            epochs=args.epochs,
            skip_existing=args.skip_existing,
        )
    else:
        print("[INFO] --skip_training ativo: pulando treinamento.")

    # ── Etapa 2: Avaliação ────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print("# ETAPA 2: AVALIAÇÃO DOS 6 CENÁRIOS")
    print(f"{'#'*60}\n")

    results = []
    shared_set = set(shared)

    # A) Custom → Custom (in-domain, 55 classes)
    r = evaluate_scenario(MODEL_CUSTOM,   args.custom, "Custom → Custom (in-domain, 55 cls)", "A")
    if r: results.append(r)

    # B) Custom → MINDS (zero-shot, 20 classes)
    r = evaluate_scenario(MODEL_CUSTOM,   args.minds,  "Custom → MINDS (zero-shot, 20 cls)",  "B")
    if r: results.append(r)

    # C) MINDS → MINDS (in-domain MINDS, 20 classes)
    r = evaluate_scenario(MODEL_MINDS,    args.minds,  "MINDS → MINDS (in-domain, 20 cls)",   "C")
    if r: results.append(r)

    # D) MINDS → Custom[shared] (reverso, 20 classes)
    r = evaluate_scenario(MODEL_MINDS,    args.custom, "MINDS → Custom[shared] (20 cls)",     "D",
                          restrict_to_classes=shared_set)
    if r: results.append(r)

    # E) Combined → Custom (55 classes)
    r = evaluate_scenario(MODEL_COMBINED, args.custom, "Combined → Custom (55 cls)",           "E")
    if r: results.append(r)

    # F) Combined → MINDS (20 classes)
    r = evaluate_scenario(MODEL_COMBINED, args.minds,  "Combined → MINDS (20 cls)",            "F")
    if r: results.append(r)

    if not results:
        print("[ERRO] Nenhum cenário foi executado com sucesso.")
        sys.exit(1)

    # ── Etapa 3: Relatório e visualizações ────────────────────────────────────
    print(f"\n{'#'*60}")
    print("# ETAPA 3: RELATÓRIOS E GRÁFICOS")
    print(f"{'#'*60}\n")

    save_report(results, args.out)
    plot_comparison_bars(results, args.out)
    plot_metrics_heatmap(results, args.out)
    plot_delta_from_indomain(results, args.out)

    # Volume de dados por classe
    if os.path.isdir(args.custom) and os.path.isdir(args.minds):
        plot_data_volume(args.custom, args.minds, args.out)

    # F1 por classe MINDS: cenários B, C e F (todos avaliam no MINDS)
    minds_eval_scenarios = []
    for r in results:
        if r["id"] in ("B", "C", "F"):
            minds_eval_scenarios.append((
                f"({r['id']}) {r['label']}",
                r["report_dict"],
            ))
    if minds_eval_scenarios:
        plot_f1_per_minds_class(minds_eval_scenarios, minds_classes, args.out)

    # Matrizes de confusão
    for r in results:
        plot_confusion(r, args.out)

    # ── Resumo final ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESUMO FINAL")
    print(f"{'='*60}")
    for r in results:
        print(f"  ({r['id']}) {r['label']:<46}  "
              f"Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}")

    # Análise específica da questão de pesquisa
    b = next((r for r in results if r["id"] == "B"), None)
    c = next((r for r in results if r["id"] == "C"), None)
    f = next((r for r in results if r["id"] == "F"), None)
    e = next((r for r in results if r["id"] == "E"), None)
    a = next((r for r in results if r["id"] == "A"), None)

    print(f"\n  Questões de pesquisa:")
    if b and c:
        gap = b["accuracy"] - c["accuracy"]
        print(f"  → Gap zero-shot vs MINDS in-domain: {gap:+.4f} "
              f"({'pior' if gap < 0 else 'melhor'} sem ver MINDS no treino)")
    if b and f:
        gain = f["accuracy"] - b["accuracy"]
        print(f"  → Ganho de adicionar MINDS ao treino (B→F): {gain:+.4f}")
    if a and e:
        delta = e["accuracy"] - a["accuracy"]
        print(f"  → Impacto no custom ao adicionar MINDS (A→E): {delta:+.4f} "
              f"({'melhora' if delta > 0 else 'não piora' if delta == 0 else 'piora'})")
    print(f"{'='*60}")
    print(f"\n[OK] Estudo concluído. Resultados em: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
