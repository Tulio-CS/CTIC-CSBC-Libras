# error_analysis.py
# -*- coding: utf-8 -*-
"""
Análise de erros por categoria de sinal LIBRAS.

Responde diretamente às perguntas dos revisores:
  - Sinais com 2 mãos têm desempenho inferior?
  - Sinais com movimento são mais difíceis que estáticos?
  - Quais são os padrões de confusão mais frequentes?

Gera:
  - Boxplot de F1 por categoria (1 mão vs 2 mãos, estático vs dinâmico)
  - Tabela de métricas por grupo (média ± std)
  - Top-20 pares de confusão com contagem e confiança média
  - Curvas PR separadas por grupo
  - Gráfico de barras de F1 colorido por categoria

Uso:
    python error_analysis.py                          # usa resultados do modelo padrão
    python error_analysis.py --model bilstm_attn
    python error_analysis.py --results results_v4/bilstm_attn
    python error_analysis.py --reeval               # reavalia o modelo antes de analisar
"""

import os, sys, json, argparse, csv
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

METADATA_PATH = "sign_metadata.json"


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de metadados e resultados
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"'{METADATA_PATH}' não encontrado. "
            "Gere-o antes de rodar este script."
        )
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_per_class_metrics(results_dir):
    """Carrega per_class_metrics.csv do diretório de resultados."""
    path = os.path.join(results_dir, "per_class_metrics.csv")
    if not os.path.exists(path):
        return None
    metrics = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metrics[row["class"]] = {
                "precision": float(row["precision"]) if row["precision"] else 0.0,
                "recall":    float(row["recall"])    if row["recall"]    else 0.0,
                "f1":        float(row["f1"])        if row["f1"]        else 0.0,
                "support":   int(float(row["support"])) if row["support"] else 0,
            }
    return metrics


def load_misclassifications(results_dir):
    """Carrega misclassifications.csv (true, pred, confidence)."""
    path = os.path.join(results_dir, "misclassifications.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "true": row["true"],
                "pred": row["pred"],
                "conf": float(row["confidence"]),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Reavaliação (opcional)
# ─────────────────────────────────────────────────────────────────────────────

def reeval_model(model_name, data_dir, results_dir):
    """Recarrega o modelo treinado e re-executa eval_and_save."""
    from evaluate import load_model_safe, eval_and_save
    from data_utils import (
        load_norm_stats, scan_labeled_dir,
        apply_feature_mode, pad_or_crop_to_T,
    )

    model_dir  = os.path.join(cfg.MODELS_DIR, model_name)
    model_path = os.path.join(model_dir, "model.keras")
    norm_path  = os.path.join(model_dir, "norm_stats.json")
    act_path   = os.path.join(model_dir, "actions.npy")

    for p in [model_path, norm_path, act_path]:
        if not os.path.exists(p):
            print(f"[ERRO] Arquivo não encontrado: {p}")
            sys.exit(1)

    model   = load_model_safe(model_path)
    actions = np.load(act_path)
    mu, sd, T, F, feature_mode = load_norm_stats(norm_path)

    files = scan_labeled_dir(data_dir, actions)
    X_list, y_list = [], []
    for path, yi in files:
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2:
            continue
        arr = apply_feature_mode(arr, feature_mode)
        arr = pad_or_crop_to_T(arr, T)
        X_list.append(arr)
        y_list.append(yi)

    X = (np.stack(X_list) - mu) / (sd + 1e-8)
    y = np.array(y_list, int)

    eval_and_save(model, X, y, actions, results_dir,
                  n_bootstrap=cfg.N_BOOTSTRAP, seed=cfg.SEED,
                  compute_tsne=False)


# ─────────────────────────────────────────────────────────────────────────────
# Agrupamento por categoria
# ─────────────────────────────────────────────────────────────────────────────

def build_groups(per_class, metadata):
    """
    Retorna dict com 4 grupos:
      '1 mão'     / '2 mãos'
      'estático'  / 'dinâmico'
    Cada grupo é {sign_name: metrics_dict}.
    """
    groups = {
        "1 mão":    {},
        "2 mãos":   {},
        "estático": {},
        "dinâmico": {},
    }
    missing = []
    for sign, m in per_class.items():
        key = sign.lower() if sign not in metadata else sign
        # busca case-insensitive
        meta = metadata.get(sign) or metadata.get(sign.lower()) or None
        if meta is None:
            missing.append(sign)
            continue
        groups["1 mão" if meta["hands"] == 1 else "2 mãos"][sign] = m
        groups["estático" if not meta["movement"] else "dinâmico"][sign] = m

    if missing:
        print(f"[WARN] {len(missing)} sinais sem metadado: {missing}")
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(path, dpi=150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"[OK] Salvo: {path}")


def plot_f1_by_category(groups, outdir):
    """
    Boxplot duplo: (1 mão vs 2 mãos) e (estático vs dinâmico).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (g1, g2), title in zip(
        axes,
        [("1 mão", "2 mãos"), ("estático", "dinâmico")],
        ["Número de mãos", "Tipo de sinal"],
    ):
        f1_g1 = [m["f1"] for m in groups[g1].values()]
        f1_g2 = [m["f1"] for m in groups[g2].values()]
        n1, n2 = len(f1_g1), len(f1_g2)

        data   = [f1_g1, f1_g2]
        labels = [f"{g1}\n(n={n1})", f"{g2}\n(n={n2})"]
        colors = ["#3498db", "#e74c3c"]

        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Adiciona pontos individuais (jitter)
        for i, (vals, color) in enumerate(zip(data, colors), 1):
            jitter = np.random.normal(i, 0.06, size=len(vals))
            ax.scatter(jitter, vals, alpha=0.5, s=20, color=color, zorder=3)

        # Médias
        for i, vals in enumerate(data, 1):
            if vals:
                ax.scatter(i, np.mean(vals), marker="D", s=60,
                           color="white", edgecolors="black", zorder=5,
                           label=f"média={np.mean(vals):.3f}")

        ax.set_xticklabels(labels)
        ax.set_ylabel("F1-score")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Distribuição de F1-score por Categoria de Sinal", fontsize=13)
    _savefig(os.path.join(outdir, "f1_by_category_boxplot.png"))


def plot_f1_bars_colored(per_class, metadata, outdir):
    """Barras de F1 coloridas por categoria (azul=1mão, vermelho=2mãos)."""
    signs  = sorted(per_class.keys())
    f1vals = [per_class[s]["f1"] for s in signs]
    colors = []
    for s in signs:
        meta = metadata.get(s) or metadata.get(s.lower())
        if meta is None:
            colors.append("#aaaaaa")
        elif meta["hands"] == 2 and meta["movement"]:
            colors.append("#c0392b")   # 2 mãos + dinâmico
        elif meta["hands"] == 2:
            colors.append("#e67e22")   # 2 mãos + estático
        elif meta["movement"]:
            colors.append("#2980b9")   # 1 mão + dinâmico
        else:
            colors.append("#27ae60")   # 1 mão + estático

    fig, ax = plt.subplots(figsize=(max(12, len(signs) * 0.38), 5))
    ax.bar(range(len(signs)), f1vals, color=colors)
    ax.axhline(np.mean(f1vals), color="gray", linestyle="--",
               label=f"média={np.mean(f1vals):.3f}")
    ax.set_xticks(range(len(signs)))
    ax.set_xticklabels(signs, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score por Classe (cor = categoria)")

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#27ae60", label="1 mão, estático"),
        Patch(facecolor="#2980b9", label="1 mão, dinâmico"),
        Patch(facecolor="#e67e22", label="2 mãos, estático"),
        Patch(facecolor="#c0392b", label="2 mãos, dinâmico"),
        Patch(facecolor="#aaaaaa", label="sem metadado"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right")
    ax.legend(handles=legend_elems + [
        plt.Line2D([0], [0], color="gray", linestyle="--",
                   label=f"média={np.mean(f1vals):.3f}")
    ], fontsize=8, loc="lower right")
    _savefig(os.path.join(outdir, "f1_bars_colored.png"))


def plot_top_confusions(misclassifications, outdir, top_n=20):
    """Gráfico de barras dos top-N pares confundidos."""
    from collections import Counter
    pairs   = [(r["true"], r["pred"]) for r in misclassifications]
    counter = Counter(pairs)

    if not counter:
        print("[INFO] Nenhuma misclassificação encontrada.")
        return

    top_pairs  = counter.most_common(top_n)
    labels     = [f"{t} → {p}" for (t, p), _ in top_pairs]
    counts     = [c for _, c in top_pairs]

    # Confiança média por par
    conf_map = {}
    for r in misclassifications:
        k = (r["true"], r["pred"])
        conf_map.setdefault(k, []).append(r["conf"])
    avg_conf = [np.mean(conf_map[(t, p)]) for (t, p), _ in top_pairs]

    fig, ax1 = plt.subplots(figsize=(max(10, top_n * 0.6), 6))
    x = np.arange(len(labels))
    bars = ax1.bar(x, counts, color="#e74c3c", alpha=0.8, label="Contagem de erros")
    ax1.set_ylabel("Contagem de erros")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.set_title(f"Top-{top_n} Pares de Confusão (verdadeiro → predito)")

    ax2 = ax1.twinx()
    ax2.plot(x, avg_conf, "o-", color="#2c3e50", linewidth=1.5,
             markersize=6, label="Confiança média")
    ax2.set_ylabel("Confiança média da predição errada")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    _savefig(os.path.join(outdir, "top_confusions.png"))


def plot_pr_by_group(model_name, data_dir, results_dir, groups, outdir):
    """
    Curvas PR separadas por grupo, usando o modelo carregado.
    Requer TensorFlow e o modelo treinado.
    """
    try:
        from evaluate import load_model_safe
        from data_utils import (
            load_norm_stats, scan_labeled_dir,
            apply_feature_mode, pad_or_crop_to_T,
        )
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize
    except ImportError as e:
        print(f"[WARN] Não foi possível gerar curvas PR: {e}")
        return

    model_dir  = os.path.join(cfg.MODELS_DIR, model_name)
    model_path = os.path.join(model_dir, "model.keras")
    norm_path  = os.path.join(model_dir, "norm_stats.json")
    act_path   = os.path.join(model_dir, "actions.npy")

    for p in [model_path, norm_path, act_path]:
        if not os.path.exists(p):
            print(f"[WARN] PR por grupo ignorada — arquivo não encontrado: {p}")
            return

    model   = load_model_safe(model_path)
    actions = np.load(act_path)
    mu, sd, T, F, feature_mode = load_norm_stats(norm_path)

    files = scan_labeled_dir(data_dir, actions)
    X_list, y_list = [], []
    for path, yi in files:
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2:
            continue
        arr = apply_feature_mode(arr, feature_mode)
        arr = pad_or_crop_to_T(arr, T)
        X_list.append(arr)
        y_list.append(yi)

    X       = (np.stack(X_list) - mu) / (sd + 1e-8)
    y_true  = np.array(y_list, int)
    probs   = model.predict(X, verbose=0)

    act_list = actions.tolist()
    Y_bin    = label_binarize(y_true, classes=range(len(act_list)))

    group_pairs = [("1 mão", "2 mãos"), ("estático", "dinâmico")]
    colors_map  = {"1 mão": "#3498db", "2 mãos": "#e74c3c",
                   "estático": "#27ae60", "dinâmico": "#e67e22"}

    for g1, g2 in group_pairs:
        fig, ax = plt.subplots(figsize=(8, 6))
        for gname in (g1, g2):
            class_indices = [
                act_list.index(s) for s in groups[gname]
                if s in act_list
            ]
            if not class_indices:
                continue
            probs_g = probs[:, class_indices]
            Y_g     = Y_bin[:, class_indices]
            mask    = Y_g.sum(axis=1) > 0      # só amostras que pertencem ao grupo
            if mask.sum() == 0:
                continue
            ap = average_precision_score(Y_g[mask].ravel(), probs_g[mask].ravel())
            p_vals, r_vals, _ = precision_recall_curve(
                Y_g[mask].ravel(), probs_g[mask].ravel()
            )
            ax.plot(r_vals, p_vals, lw=2,
                    color=colors_map[gname],
                    label=f"{gname} (AP={ap:.3f}, n={len(class_indices)} sinais)")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precisão")
        ax.set_title(f"Curvas Precision-Recall por Grupo: {g1} vs {g2}")
        ax.legend(fontsize=10)
        ax.grid(linestyle="--", alpha=0.4)
        fname = f"pr_group_{g1.replace(' ', '_')}_vs_{g2.replace(' ', '_')}.png"
        _savefig(os.path.join(outdir, fname))


# ─────────────────────────────────────────────────────────────────────────────
# Relatório textual por grupo
# ─────────────────────────────────────────────────────────────────────────────

def print_group_report(groups, outdir):
    """Tabela de métricas médias por grupo."""
    lines = []
    header = f"{'Grupo':<15} {'N sinais':>9} {'Acc (F1 macro)':>16} {'F1 min':>8} {'F1 max':>8} {'F1 std':>8}"
    sep    = "-" * len(header)
    lines += [sep, header, sep]

    for gname, sign_metrics in sorted(groups.items()):
        if not sign_metrics:
            continue
        f1s = [m["f1"] for m in sign_metrics.values()]
        lines.append(
            f"{gname:<15} {len(f1s):>9} {np.mean(f1s):>16.4f} "
            f"{np.min(f1s):>8.4f} {np.max(f1s):>8.4f} {np.std(f1s):>8.4f}"
        )

    lines.append(sep)
    txt = "\n".join(lines)
    print("\n=== Métricas por Grupo ===")
    print(txt)

    path = os.path.join(outdir, "group_metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"[OK] Relatório de grupos salvo em: {path}")

    # CSV por grupo
    csv_path = os.path.join(outdir, "group_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "sign", "precision", "recall", "f1", "support"])
        for gname, sign_metrics in sorted(groups.items()):
            for sign, m in sorted(sign_metrics.items()):
                w.writerow([gname, sign,
                             f"{m['precision']:.4f}",
                             f"{m['recall']:.4f}",
                             f"{m['f1']:.4f}",
                             m["support"]])
    print(f"[OK] CSV por grupo salvo em: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Análise de erros por categoria.")
    p.add_argument("--model",   default=cfg.DEFAULT_MODEL_NAME)
    p.add_argument("--data",    default=cfg.DATA_DIR)
    p.add_argument("--results", default=None,
                   help="Diretório com per_class_metrics.csv e misclassifications.csv")
    p.add_argument("--out",     default=None,
                   help="Diretório de saída (padrão: <results>/error_analysis)")
    p.add_argument("--reeval",  action="store_true",
                   help="Reavalia o modelo antes de analisar")
    p.add_argument("--no_pr",   action="store_true",
                   help="Não gera curvas PR por grupo (mais rápido)")
    return p.parse_args()


def main():
    args = parse_args()

    results_dir = args.results or os.path.join(cfg.RESULTS_DIR, args.model)
    outdir      = args.out     or os.path.join(results_dir, "error_analysis")
    os.makedirs(outdir, exist_ok=True)

    if args.reeval or not os.path.exists(
        os.path.join(results_dir, "per_class_metrics.csv")
    ):
        print("[INFO] Executando reavaliação do modelo...")
        reeval_model(args.model, args.data, results_dir)

    metadata   = load_metadata()
    per_class  = load_per_class_metrics(results_dir)
    misclasses = load_misclassifications(results_dir)

    if per_class is None:
        print(f"[ERRO] per_class_metrics.csv não encontrado em {results_dir}")
        print("       Rode primeiro: python evaluate.py")
        sys.exit(1)

    # Remove entradas agregadas do classification_report
    for key in ("accuracy", "macro avg", "weighted avg"):
        per_class.pop(key, None)

    groups = build_groups(per_class, metadata)
    print_group_report(groups, outdir)

    plot_f1_by_category(groups, outdir)
    plot_f1_bars_colored(per_class, metadata, outdir)

    if misclasses:
        plot_top_confusions(misclasses, outdir)
    else:
        print("[WARN] misclassifications.csv vazio — pulando gráfico de confusões.")

    if not args.no_pr:
        plot_pr_by_group(args.model, args.data, results_dir, groups, outdir)

    print(f"\n[OK] Análise de erros salva em: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
