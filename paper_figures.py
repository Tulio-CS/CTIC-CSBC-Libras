# paper_figures.py
# -*- coding: utf-8 -*-
"""
Gera figuras de publicação (artigo científico) comparando
LSTM padrão vs BiLSTM+Atenção Combinado (dataset custom + MINDS).

Uso:
    py -3.11 paper_figures.py

Saída em results_paper/:
  fig1_metricas.png/pdf       — barras Acc/Prec/Rec/F1 com eixo cortado
  fig2_confiancas.png/pdf     — histogramas de confiança acertos vs erros
  fig3_pr_curves.png/pdf      — curvas Precision-Recall macro
  fig4_f1_categorias.png/pdf  — F1 por categoria de sinal
  fig5_ij_analise.png/pdf     — análise específica I vs J
  tabela_resultados.csv       — tabela comparativa
  tabela_resultados.tex       — tabela LaTeX pronta para incluir
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, f1_score,
)
from sklearn.preprocessing import label_binarize

import config as cfg
from data_utils import (
    load_sequences, make_split, load_norm_stats,
    apply_feature_mode, pad_or_crop_to_T, bootstrap_metrics,
)

try:
    import tensorflow as tf
    TF_OK = True
except ImportError:
    TF_OK = False

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Estilo para publicacao
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "axes.titlepad":     6,
    "axes.linewidth":    0.8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize":   8,
    "legend.frameon":    True,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "0.75",
    "legend.handlelength": 1.4,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.35,
    "lines.linewidth":   1.6,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "figure.dpi":        150,
})

OUT_DIR = "results_paper"
os.makedirs(OUT_DIR, exist_ok=True)

# Paleta de cores — acessível e diferenciável em B&W
C_LSTM   = "#D55E00"   # vermelho-laranja
C_COMB   = "#0072B2"   # azul

LABELS = {
    "lstm":     "LSTM",
    "combined": "BiLSTM+Atenção\nCombinado",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

def savefig(name: str):
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    -> {os.path.join(OUT_DIR, name)}.png / .pdf")


def load_model_safe(path: str):
    try:
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
                f"  tf.keras: {e1}\n  keras: {e2}"
            )


def _add_axis_break(ax_top, ax_bot):
    """Desenha marcas diagonais indicando quebra no eixo Y."""
    d = 0.018
    kw = dict(color="k", clip_on=False, linewidth=1.0, transform=ax_top.transAxes)
    ax_top.plot((-d, +d), (-d, +d), **kw)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw["transform"] = ax_bot.transAxes
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de dados e inferência
# ─────────────────────────────────────────────────────────────────────────────

def get_test_data():
    """Carrega dataset/ e retorna o split de teste determinístico."""
    print("[INFO] Carregando dataset/ ...")
    X_raw, y, actions, meta = load_sequences(cfg.DATA_DIR)
    tr_idx, te_idx = make_split(
        X_raw, y, meta,
        test_size=cfg.TEST_SIZE,
        seed=cfg.SEED,
        use_group_split=cfg.USE_GROUP_SPLIT,
    )
    X_test = [X_raw[i] for i in te_idx]
    y_test  = y[te_idx]
    print(f"       {len(X_test)} amostras | {len(actions)} classes")
    return X_test, y_test, actions


def predict_with_model(model_name: str, X_test_raw: list, ref_actions: np.ndarray):
    """
    Carrega modelo, aplica normalização própria e retorna (probs, preds).
    Alinha ações do modelo com ref_actions para comparações justas.
    """
    model_dir = os.path.join(cfg.MODELS_DIR, model_name)
    mu, sd, T, F, feature_mode = load_norm_stats(
        os.path.join(model_dir, "norm_stats.json")
    )

    # Pré-processamento  (mu/sd têm shape (1,1,F) -> achata para (F,) para broadcast limpo)
    mu2 = mu.reshape(-1)   # (F,)
    sd2 = sd.reshape(-1)   # (F,)
    X_prep = np.stack([
        (apply_feature_mode(pad_or_crop_to_T(x, T), feature_mode) - mu2) / (sd2 + 1e-8)
        for x in X_test_raw
    ], axis=0).astype(np.float32)

    print(f"       Carregando {model_name} ...")
    model = load_model_safe(os.path.join(model_dir, "model.keras"))
    probs_raw = model.predict(X_prep, batch_size=64, verbose=0)

    # Alinhamento de classes
    model_actions = np.load(
        os.path.join(model_dir, "actions.npy"), allow_pickle=True
    )
    if list(model_actions) == list(ref_actions):
        return probs_raw, np.argmax(probs_raw, axis=1)

    # Remapeia probabilidades para a ordem de ref_actions
    model_idx = {a: i for i, a in enumerate(model_actions)}
    probs_aligned = np.zeros((len(probs_raw), len(ref_actions)), dtype=np.float32)
    for j, a in enumerate(ref_actions):
        if a in model_idx:
            probs_aligned[:, j] = probs_raw[:, model_idx[a]]
    return probs_aligned, np.argmax(probs_aligned, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Figura 1 — Métricas comparativas com eixo cortado
# ─────────────────────────────────────────────────────────────────────────────

def fig1_metricas(stats: dict):
    """
    stats[model_key][metric] = (mean, std)
    """
    metrics  = ["Acurácia", "Precisão", "Revocação", "F1-Score"]
    keys     = ["acc",      "prec",     "rec",       "f1"]
    models   = ["lstm", "combined"]
    colors   = [C_LSTM, C_COMB]
    n        = len(metrics)
    x        = np.arange(n)
    w        = 0.32

    # Limites dinâmicos
    all_vals = [stats[m][k][0] for m in models for k in keys]
    y_lo_top = max(0.60, min(all_vals) - 0.04)
    y_hi_top = min(1.002, max(all_vals) + 0.035)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(5.6, 4.0),
        gridspec_kw={"height_ratios": [4, 0.75]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.055)

    for ax, (y_lo, y_hi) in zip((ax_top, ax_bot),
                                 ((y_lo_top, y_hi_top), (0.0, 0.055))):
        for model_key, color, off in zip(models, colors, (-w / 2, w / 2)):
            vals = [stats[model_key][k][0] for k in keys]
            errs = [stats[model_key][k][1] for k in keys]
            ax.bar(
                x + off, vals, w,
                color=color, alpha=0.88, zorder=3,
                linewidth=0.5, edgecolor="white",
                label=LABELS[model_key] if ax is ax_top else "_nolegend_",
            )
            ax.errorbar(
                x + off, vals, yerr=errs,
                fmt="none", color="0.25", linewidth=0.9,
                capsize=3.5, capthick=0.8, zorder=4,
            )
            # Rótulos de valor — apenas painel superior
            if ax is ax_top:
                for j, (v, e) in enumerate(zip(vals, errs)):
                    y_text = min(v + e + 0.004, y_hi - 0.005)
                    ax.text(j + off, y_text, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=6.8, color="0.15")

        ax.set_ylim(y_lo, y_hi)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ax.spines["top"].set_visible(ax is ax_bot)
        ax.spines["bottom"].set_visible(ax is ax_bot)
        ax.tick_params(bottom=(ax is ax_bot))

    # Ticks do eixo Y — painel superior com passo fixo
    step = 0.05
    y_ticks = np.arange(
        np.ceil(y_lo_top / step) * step,
        y_hi_top + 1e-9,
        step,
    )
    ax_top.set_yticks(y_ticks)
    ax_bot.set_yticks([0.0])

    _add_axis_break(ax_top, ax_bot)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(metrics, fontsize=9)
    ax_top.set_ylabel("Valor da Métrica", fontsize=9)
    ax_top.legend(loc="lower right", ncol=2, fontsize=8.5,
                  handlelength=1.2, handleheight=0.9)
    ax_top.set_title(
        "Comparação de Desempenho:\nLSTM vs. BiLSTM+Atenção Combinado",
        fontsize=10, pad=6,
    )

    savefig("fig1_metricas")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 2 — Histogramas de confiança: acertos vs erros
# ─────────────────────────────────────────────────────────────────────────────

def fig2_confiancas(conf_data: dict):
    """
    conf_data[model_key] = {'correct_conf': arr, 'wrong_conf': arr}
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=False)
    bins = np.linspace(0, 1, 26)
    C_CORRECT = "#009E73"  # verde
    C_WRONG   = "#CC79A7"  # magenta

    for ax, model_key, color in zip(axes, ["lstm", "combined"], [C_LSTM, C_COMB]):
        c_conf = conf_data[model_key]["correct_conf"]
        w_conf = conf_data[model_key]["wrong_conf"]

        n_c, n_w = len(c_conf), len(w_conf)
        acc_plot = n_c / (n_c + n_w) * 100

        ax.hist(c_conf, bins=bins, density=True, alpha=0.60,
                color=C_CORRECT, label=f"Acertos (n={n_c})", zorder=2)
        ax.hist(w_conf, bins=bins, density=True, alpha=0.60,
                color=C_WRONG,   label=f"Erros (n={n_w})",   zorder=2)

        # Curvas KDE
        for conf, col in ((c_conf, C_CORRECT), (w_conf, C_WRONG)):
            if len(conf) >= 4 and HAS_SCIPY:
                try:
                    kde = gaussian_kde(conf, bw_method=0.15)
                    xg  = np.linspace(0, 1, 300)
                    ax.plot(xg, kde(xg), color=col, linewidth=1.8, zorder=3)
                except Exception:
                    pass

        # Linhas de média
        for conf, col, ls in ((c_conf, C_CORRECT, "--"), (w_conf, C_WRONG, ":")):
            if len(conf):
                ax.axvline(np.mean(conf), color=col, linewidth=1.2,
                           linestyle=ls, alpha=0.85, zorder=4)

        ax.set_xlabel("Confiança máxima (softmax)", fontsize=9)
        ax.set_ylabel("Densidade de probabilidade", fontsize=9)
        ax.set_title(
            f"{LABELS[model_key].replace(chr(10), ' ')}\n(acurácia = {acc_plot:.1f}%)",
            fontsize=9.5,
        )
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Anotação de média
        for conf, col, va in ((c_conf, C_CORRECT, "bottom"), (w_conf, C_WRONG, "top")):
            if len(conf):
                ax.annotate(
                    f"μ={np.mean(conf):.2f}",
                    xy=(np.mean(conf), ax.get_ylim()[1] * 0.95),
                    fontsize=6.5, color=col, ha="center",
                    va="top" if va == "top" else "bottom",
                )

    fig.suptitle(
        "Distribuição de Confiança: Acertos vs. Erros de Classificação",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    savefig("fig2_confiancas")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 3 — Curvas Precisão-Revocação
# ─────────────────────────────────────────────────────────────────────────────

def fig3_pr_curves(probs_all: dict, y_test: np.ndarray, actions: np.ndarray):
    n_cls  = len(actions)
    Y_bin  = label_binarize(y_test, classes=np.arange(n_cls))

    fig, ax = plt.subplots(figsize=(4.6, 3.8))

    for model_key, color, ls in zip(
        ["lstm", "combined"], [C_LSTM, C_COMB], ["-", "--"]
    ):
        probs = probs_all[model_key]
        rec_lin = np.linspace(0, 1, 500)
        prec_interp = np.zeros(500)
        ap_list = []

        for c in range(n_cls):
            p, r, _ = precision_recall_curve(Y_bin[:, c], probs[:, c])
            ap = average_precision_score(Y_bin[:, c], probs[:, c])
            ap_list.append(ap)
            # Curvas por classe (fundo)
            ax.plot(r, p, color=color, alpha=0.06, linewidth=0.7, zorder=1)
            prec_interp += np.interp(rec_lin, r[::-1], p[::-1])

        prec_interp /= n_cls
        map_val = np.mean(ap_list)
        label = f"{LABELS[model_key].replace(chr(10), ' ')}  (mAP = {map_val:.3f})"
        ax.plot(rec_lin, prec_interp, color=color, linewidth=2.0,
                linestyle=ls, label=label, zorder=4)
        ax.fill_between(rec_lin, prec_interp, alpha=0.08, color=color, zorder=2)

    ax.set_xlabel("Revocação (Recall)", fontsize=9)
    ax.set_ylabel("Precisão (Precision)", fontsize=9)
    ax.set_title("Curvas Precisão–Revocação\n(média macro — curvas por classe em cinza)", fontsize=9.5)
    ax.set_xlim(-0.01, 1.02)
    ax.set_ylim(0.0, 1.08)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig("fig3_pr_curves")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 4 — Distribuição de F1 por categoria de sinal
# ─────────────────────────────────────────────────────────────────────────────

def fig4_f1_categorias(f1_data: dict, actions: np.ndarray):
    """
    f1_data[model_key]['f1_per_class'] = array shape (n_classes,)
    """
    # Carregar metadados
    meta_path = "sign_metadata.json"
    if not os.path.isfile(meta_path):
        print("  [AVISO] sign_metadata.json não encontrado — fig4 ignorada.")
        return

    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    def categorize(sign):
        info = meta.get(sign, {})
        h, mv = info.get("hands", 1), info.get("movement", False)
        if h == 1 and not mv: return "1 mão\nEstático"
        if h == 1 and mv:     return "1 mão\nDinâmico"
        if h == 2 and not mv: return "2 mãos\nEstático"
        return "2 mãos\nDinâmico"

    cat_order  = ["1 mão\nEstático", "1 mão\nDinâmico",
                  "2 mãos\nEstático", "2 mãos\nDinâmico"]
    cats       = [categorize(a) for a in actions]

    n_cats = len(cat_order)
    x      = np.arange(n_cats)
    w      = 0.33

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    rng = np.random.RandomState(42)

    for model_key, color, off in zip(["lstm", "combined"], [C_LSTM, C_COMB], (-w / 2, w / 2)):
        f1_arr = f1_data[model_key]["f1_per_class"]
        cat_f1 = {c: [] for c in cat_order}
        for val, cat in zip(f1_arr, cats):
            if cat in cat_f1:
                cat_f1[cat].append(val)

        means  = [np.mean(cat_f1[c]) if cat_f1[c] else 0.0 for c in cat_order]
        stds   = [np.std(cat_f1[c])  if len(cat_f1[c]) > 1 else 0.0 for c in cat_order]
        counts = [len(cat_f1[c]) for c in cat_order]

        ax.bar(x + off, means, w,
               color=color, alpha=0.85, zorder=3,
               linewidth=0.5, edgecolor="white",
               label=LABELS[model_key].replace("\n", " "))
        ax.errorbar(x + off, means, yerr=stds,
                    fmt="none", color="0.25", linewidth=0.9,
                    capsize=4, capthick=0.8, zorder=4)

        # Pontos individuais (jitter)
        for j, c in enumerate(cat_order):
            vals = cat_f1[c]
            if vals:
                jitter = rng.uniform(-w * 0.38, w * 0.38, len(vals))
                ax.scatter(j + off + jitter, vals,
                           s=10, color=color, alpha=0.45, zorder=5, linewidths=0)

        # Contagem de sinais abaixo
        for j, (m, n) in enumerate(zip(means, counts)):
            ax.text(j + off, ax.get_ylim()[0] + 0.002 if ax.get_ylim()[0] > 0.5 else 0.002,
                    f"n={n}", ha="center", va="bottom", fontsize=6.5, color="0.45")

    # Eixo Y com corte se valores altos
    all_means = [
        np.mean(f1_data[m]["f1_per_class"][[i for i, c in enumerate(cats) if c == cat]])
        for m in ["lstm", "combined"]
        for cat in cat_order
        if any(c == cat for c in cats)
    ]
    y_lo = max(0.0, min(all_means) - 0.07) if all_means else 0.0
    y_hi = 1.02
    ax.set_ylim(y_lo, y_hi)

    if y_lo > 0.05:
        ax.annotate(
            "//", xy=(-0.045, y_lo), xycoords=("axes fraction", "data"),
            fontsize=10, color="0.4", ha="right", va="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cat_order, fontsize=9)
    ax.set_ylabel("F1-Score (média por categoria ± desvio)", fontsize=9)
    ax.set_title("Distribuição do F1-Score por Categoria de Sinal", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig("fig4_f1_categorias")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 5 — Análise I vs J
# ─────────────────────────────────────────────────────────────────────────────

def fig5_ij_analise(probs_all: dict, y_test: np.ndarray, actions: np.ndarray):
    actions_list = list(actions)
    if "i" not in actions_list or "j" not in actions_list:
        print("  [AVISO] 'i' ou 'j' não encontrado em actions — fig5 ignorada.")
        return

    idx_i = actions_list.index("i")
    idx_j = actions_list.index("j")

    fig = plt.figure(figsize=(9.0, 3.6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax_cm   = fig.add_subplot(gs[0])
    ax_trI  = fig.add_subplot(gs[1])
    ax_trJ  = fig.add_subplot(gs[2])

    mask_i = (y_test == idx_i)
    mask_j = (y_test == idx_j)

    # ── (a) Contagens de confusão I↔J ───────────────────────────────────────
    ax = ax_cm
    ax.set_title("(a) Confusões I ↔ J", fontsize=9.5, pad=5)

    cm_labels = ["I->I", "I->J", "J->I", "J->J"]
    x_cm = np.arange(4)
    w_cm = 0.35

    for model_key, color, off in zip(["lstm", "combined"], [C_LSTM, C_COMB], (-w_cm/2, w_cm/2)):
        probs = probs_all[model_key]
        preds = np.argmax(probs, axis=1)
        ii = int(np.sum((y_test == idx_i) & (preds == idx_i)))
        ij = int(np.sum((y_test == idx_i) & (preds == idx_j)))
        ji = int(np.sum((y_test == idx_j) & (preds == idx_i)))
        jj = int(np.sum((y_test == idx_j) & (preds == idx_j)))
        vals = [ii, ij, ji, jj]
        bars = ax.bar(x_cm + off, vals, w_cm,
                      color=color, alpha=0.85, zorder=3,
                      linewidth=0.5, edgecolor="white",
                      label=LABELS[model_key].replace("\n", " "))
        for xi, v in zip(x_cm + off, vals):
            ax.text(xi, v + 0.15, str(v),
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x_cm)
    ax.set_xticklabels(cm_labels, fontsize=8.5)
    ax.set_ylabel("Número de amostras", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Destaca confusões (2ª e 3ª barras) com fundo
    for xi in (1, 2):
        ax.axvspan(xi - 0.5, xi + 0.5, color="0.93", zorder=0)

    # ── (b) Probabilidades para amostras verdadeiramente I ───────────────────
    ax = ax_trI
    ax.set_title("(b) Amostras com rótulo I\n(estático — sem movimento)", fontsize=9, pad=5)

    positions_lstm = [0.6, 1.4]
    positions_comb = [2.6, 3.4]
    bp_data  = []
    bp_pos   = []
    bp_color = []
    bp_label = []

    for model_key, color, pos in zip(
        ["lstm", "combined"], [C_LSTM, C_COMB],
        [positions_lstm, positions_comb]
    ):
        probs = probs_all[model_key]
        pi = probs[mask_i, idx_i]
        pj = probs[mask_i, idx_j]
        for data, p, lbl in zip([pi, pj], pos, ["P(I)", "P(J)"]):
            bp_data.append(data)
            bp_pos.append(p)
            bp_color.append(color)

    bplots = ax.boxplot(
        bp_data, positions=bp_pos, widths=0.5,
        patch_artist=True, notch=False,
        medianprops=dict(color="white", linewidth=2.0),
        flierprops=dict(marker="o", markersize=3, alpha=0.4, linewidth=0),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bplots["boxes"], [C_LSTM, C_LSTM, C_COMB, C_COMB]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_xticks([1.0, 3.0])
    ax.set_xticklabels(["LSTM", "Combinado"], fontsize=8.5)
    ax.set_ylabel("Probabilidade softmax", fontsize=8.5)
    ax.set_ylim(-0.04, 1.12)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)

    # Legenda manual para P(I) e P(J)
    for xpos, lbl in zip([0.6, 1.4], ["P(I)", "P(J)"]):
        ax.text(xpos, -0.09, lbl, ha="center", va="top", fontsize=7.5,
                transform=ax.get_xaxis_transform())
    for xpos, lbl in zip([2.6, 3.4], ["P(I)", "P(J)"]):
        ax.text(xpos, -0.09, lbl, ha="center", va="top", fontsize=7.5,
                transform=ax.get_xaxis_transform())

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── (c) Probabilidades para amostras verdadeiramente J ───────────────────
    ax = ax_trJ
    ax.set_title("(c) Amostras com rótulo J\n(dinâmico — com movimento)", fontsize=9, pad=5)

    bp_data2 = []
    bp_pos2  = []

    for model_key, color, pos in zip(
        ["lstm", "combined"], [C_LSTM, C_COMB],
        [positions_lstm, positions_comb]
    ):
        probs = probs_all[model_key]
        pj = probs[mask_j, idx_j]
        pi = probs[mask_j, idx_i]
        for data, p in zip([pj, pi], pos):
            bp_data2.append(data)
            bp_pos2.append(p)

    bplots2 = ax.boxplot(
        bp_data2, positions=bp_pos2, widths=0.5,
        patch_artist=True, notch=False,
        medianprops=dict(color="white", linewidth=2.0),
        flierprops=dict(marker="o", markersize=3, alpha=0.4, linewidth=0),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bplots2["boxes"], [C_LSTM, C_LSTM, C_COMB, C_COMB]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_xticks([1.0, 3.0])
    ax.set_xticklabels(["LSTM", "Combinado"], fontsize=8.5)
    ax.set_ylabel("Probabilidade softmax", fontsize=8.5)
    ax.set_ylim(-0.04, 1.12)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)

    for xpos, lbl in zip([0.6, 1.4], ["P(J)", "P(I)"]):
        ax.text(xpos, -0.09, lbl, ha="center", va="top", fontsize=7.5,
                transform=ax.get_xaxis_transform())
    for xpos, lbl in zip([2.6, 3.4], ["P(J)", "P(I)"]):
        ax.text(xpos, -0.09, lbl, ha="center", va="top", fontsize=7.5,
                transform=ax.get_xaxis_transform())

    ax.text(
        0.97, 0.97,
        "I e J compartilham\na mesma configuração\nmanual (handshape).\n"
        "J inclui movimento\nadicional da mão.",
        transform=ax.transAxes, fontsize=7.0, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFACD",
                  edgecolor="0.65", alpha=0.90),
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Análise de Par Confundível: I (Estático) vs. J (Dinâmico)",
        fontsize=11, y=1.02,
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.16, top=0.88, wspace=0.38)
    savefig("fig5_ij_analise")


# ─────────────────────────────────────────────────────────────────────────────
# Figura — Curvas de treinamento (Loss + Acurácia)
# ─────────────────────────────────────────────────────────────────────────────

def fig_curvas_treino():
    """
    Compara curvas de Loss e Acurácia (treino + validação) para
    LSTM vs BiLSTM+Atenção (bilstm_attn_best).
    Lê training_history.csv de cada modelo em results/<nome>/.
    """
    import csv as csvmod

    hist_paths = {
        "lstm":     os.path.join("results", "lstm",            "training_history.csv"),
        "combined": os.path.join("results", "bilstm_attn_best","training_history.csv"),
    }
    display = {
        "lstm":     "LSTM",
        "combined": "BiLSTM+Atenção (melhor)",
    }

    # Verifica existência
    for key, path in hist_paths.items():
        if not os.path.isfile(path):
            print(f"  [AVISO] {path} não encontrado — fig_curvas_treino ignorada.")
            return

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    ax_loss, ax_acc = axes

    # Estilo: LSTM=sólido, BiLSTM+Attn=tracejado; trem=opaco, val=transparente
    line_styles = {
        "lstm":     {"ls": "-",  "lw": 1.6},
        "combined": {"ls": "--", "lw": 1.8},
    }
    colors = {"lstm": C_LSTM, "combined": C_COMB}

    for key in ["lstm", "combined"]:
        epochs, loss_tr, loss_val, acc_tr, acc_val = [], [], [], [], []
        with open(hist_paths[key], newline="", encoding="utf-8") as fh:
            reader = csvmod.DictReader(fh)
            for row in reader:
                epochs.append(int(row["epoch"]) + 1)
                loss_tr.append(float(row["loss"]))
                loss_val.append(float(row["val_loss"]))
                acc_tr.append(float(row["categorical_accuracy"]))
                acc_val.append(float(row["val_categorical_accuracy"]))

        ep   = np.array(epochs)
        col  = colors[key]
        ls   = line_styles[key]
        lbl  = display[key]

        # Loss
        ax_loss.plot(ep, loss_tr,  color=col, alpha=0.90, label=f"{lbl} — treino",     **ls)
        ax_loss.plot(ep, loss_val, color=col, alpha=0.50, label=f"{lbl} — validação",  **ls)

        # Accuracy
        ax_acc.plot(ep, acc_tr,  color=col, alpha=0.90, label=f"{lbl} — treino",  **ls)
        ax_acc.plot(ep, acc_val, color=col, alpha=0.50, label=f"{lbl} — validação",**ls)

        # Marca o ponto de melhor validação
        best_ep_loss = ep[np.argmin(loss_val)]
        best_ep_acc  = ep[np.argmax(acc_val)]
        ax_loss.axvline(best_ep_loss, color=col, linewidth=0.7, linestyle=":", alpha=0.6)
        ax_acc.axvline(best_ep_acc,   color=col, linewidth=0.7, linestyle=":", alpha=0.6)

    # Formatação Loss
    ax_loss.set_xlabel("Época", fontsize=9)
    ax_loss.set_ylabel("Loss (entropia cruzada)", fontsize=9)
    ax_loss.set_title("(a) Evolução da Loss\n(treino vs. validação)", fontsize=9.5)
    ax_loss.set_xlim(left=1)
    ax_loss.set_ylim(bottom=0)
    ax_loss.legend(fontsize=7, loc="upper right", ncol=1)
    ax_loss.grid(linestyle="--", alpha=0.3)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)

    # Formatação Acurácia — eixo cortado para destacar diferenças
    all_acc = []
    for key in ["lstm", "combined"]:
        with open(hist_paths[key], newline="", encoding="utf-8") as fh:
            reader = csvmod.DictReader(fh)
            for row in reader:
                all_acc.extend([float(row["categorical_accuracy"]),
                                float(row["val_categorical_accuracy"])])
    y_lo = max(0.0, min(all_acc) - 0.04)

    ax_acc.set_xlabel("Época", fontsize=9)
    ax_acc.set_ylabel("Acurácia categórica", fontsize=9)
    ax_acc.set_title("(b) Evolução da Acurácia\n(treino vs. validação)", fontsize=9.5)
    ax_acc.set_xlim(left=1)
    ax_acc.set_ylim(y_lo, 1.02)
    if y_lo > 0.05:
        ax_acc.text(0.01, y_lo + 0.004, "//", fontsize=9, color="0.5",
                    transform=ax_acc.get_yaxis_transform(), ha="right")
    ax_acc.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_acc.legend(fontsize=7, loc="lower right", ncol=1)
    ax_acc.grid(linestyle="--", alpha=0.3)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)

    fig.suptitle(
        "Curvas de Aprendizado: LSTM vs. BiLSTM+Atenção (melhor configuração)",
        fontsize=10.5, y=1.01,
    )
    plt.tight_layout()
    savefig("fig_curvas_treino")


# ─────────────────────────────────────────────────────────────────────────────
# Figura — Comparação MINDS: treinado só no MINDS vs treinado no combinado
# ─────────────────────────────────────────────────────────────────────────────

def fig_minds_comparacao():
    """
    Compara BiLSTM+Atenção treinado SOMENTE no MINDS (bilstm_attn_minds)
    vs treinado no MINDS+Custom (bilstm_attn_combined), ambos avaliados
    no dataset MINDS (dataset_minds/).
    Figura com dois painéis:
      (a) Métricas globais (Acc/Prec/Rec/F1) com eixo cortado
      (b) F1 por classe MINDS (barras horizontais)
    """
    from data_utils import scan_labeled_dir, load_norm_stats, apply_feature_mode, pad_or_crop_to_T

    minds_dir = cfg.MINDS_DATA_DIR
    if not os.path.isdir(minds_dir):
        print(f"  [AVISO] {minds_dir} não encontrado — fig_minds_comparacao ignorada.")
        return

    model_map = {
        "minds":    "bilstm_attn_minds",
        "combined": "bilstm_attn_combined",
    }
    display_labels = {
        "minds":    "Treinado\nSó MINDS",
        "combined": "Treinado\nMINDS + Custom",
    }
    colors_minds = {"minds": "#009E73", "combined": C_COMB}  # verde vs azul

    # ── Inferência no MINDS data ─────────────────────────────────────────────
    results = {}
    for key, model_name in model_map.items():
        model_dir  = os.path.join(cfg.MODELS_DIR, model_name)
        model_path = os.path.join(model_dir, "model.keras")
        norm_path  = os.path.join(model_dir, "norm_stats.json")
        act_path   = os.path.join(model_dir, "actions.npy")

        for p in (model_path, norm_path, act_path):
            if not os.path.exists(p):
                print(f"  [AVISO] {p} não encontrado.")
                return

        model   = load_model_safe(model_path)
        actions = np.load(act_path, allow_pickle=True)
        mu, sd, T, F, feature_mode = load_norm_stats(norm_path)
        mu2, sd2 = mu.reshape(-1), sd.reshape(-1)

        files = scan_labeled_dir(minds_dir, actions)
        if not files:
            print(f"  [AVISO] Nenhum arquivo MINDS encontrado para {model_name}.")
            return

        X_list, y_list = [], []
        for path, yi in files:
            arr = np.load(path).astype(np.float32)
            if arr.ndim != 2:
                continue
            arr = apply_feature_mode(pad_or_crop_to_T(arr, T), feature_mode)
            X_list.append(arr)
            y_list.append(yi)

        X_np   = (np.stack(X_list, axis=0) - mu2) / (sd2 + 1e-8)
        y_true = np.array(y_list, int)
        probs  = model.predict(X_np, batch_size=64, verbose=0)
        preds  = np.argmax(probs, axis=1)

        # Métricas globais
        bs = bootstrap_metrics(y_true, preds, n_boot=1000, seed=cfg.SEED)
        # F1 por classe presente
        present = sorted(set(y_list))
        f1_pc   = f1_score(y_true, preds, average=None,
                           labels=np.arange(len(actions)), zero_division=0)
        results[key] = {
            "actions": actions,
            "present": present,
            "f1_per_class": f1_pc,
            "bs": bs,
        }
        print(f"       {model_name}: Acc={bs['accuracy_mean']:.4f}  F1={bs['f1_macro_mean']:.4f}")

    # Ações MINDS presentes (intersecção das duas listas de classes)
    minds_actions = [results["minds"]["actions"][i] for i in results["minds"]["present"]]

    # ── Layout ───────────────────────────────────────────────────────────────
    fig, (ax_bar, ax_cls) = plt.subplots(
        1, 2, figsize=(8.5, 4.0),
        gridspec_kw={"width_ratios": [1.1, 2.2]},
    )

    # ── Painel (a): Métricas globais com eixo cortado ────────────────────────
    ax  = ax_bar
    metrics = [("Acurácia", "accuracy"), ("Precisão", "precision_macro"),
               ("Revocação","recall_macro"),  ("F1-Score",  "f1_macro")]
    n   = len(metrics)
    x   = np.arange(n)
    w   = 0.32
    models_order = ["minds", "combined"]

    vals_all = [results[mk]["bs"][f"{mkey}_mean"] for mk in models_order for _, mkey in metrics]
    y_lo_top = max(0.65, min(vals_all) - 0.04)
    y_hi_top = min(1.002, max(vals_all) + 0.03)

    # Painel duplo (eixo cortado)
    ax_top = ax.inset_axes([0, 0.22, 1, 0.78])
    ax_bot = ax.inset_axes([0, 0,    1, 0.15])
    ax.axis("off")

    for sub_ax, (y_lo, y_hi) in zip((ax_top, ax_bot),
                                     ((y_lo_top, y_hi_top), (0.0, 0.04))):
        for mk, col, off in zip(models_order, [colors_minds["minds"], colors_minds["combined"]],
                                (-w/2, w/2)):
            vals = [results[mk]["bs"][f"{mkey}_mean"] for _, mkey in metrics]
            errs = [results[mk]["bs"][f"{mkey}_std"]  for _, mkey in metrics]
            sub_ax.bar(x + off, vals, w, color=col, alpha=0.88, zorder=3,
                       linewidth=0.5, edgecolor="white",
                       label=display_labels[mk] if sub_ax is ax_top else "_nl_")
            sub_ax.errorbar(x + off, vals, yerr=errs, fmt="none",
                            color="0.25", linewidth=0.9, capsize=3.5,
                            capthick=0.8, zorder=4)
            if sub_ax is ax_top:
                for j, (v, e) in enumerate(zip(vals, errs)):
                    y_text = min(v + e + 0.004, y_hi_top - 0.004)
                    sub_ax.text(j + off, y_text, f"{v:.3f}", ha="center",
                                va="bottom", fontsize=6.5, color="0.15")

        sub_ax.set_ylim(y_lo, y_hi)
        sub_ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        sub_ax.spines["top"].set_visible(sub_ax is ax_bot)
        sub_ax.spines["bottom"].set_visible(sub_ax is ax_bot)
        sub_ax.tick_params(bottom=(sub_ax is ax_bot))

    step = 0.05
    yticks = np.arange(np.ceil(y_lo_top / step) * step, y_hi_top + 1e-9, step)
    ax_top.set_yticks(yticks)
    ax_bot.set_yticks([0.0])
    _add_axis_break(ax_top, ax_bot)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([m[0] for m in metrics], fontsize=8.5)
    ax_top.set_ylabel("Métrica", fontsize=8.5)
    ax_top.legend(fontsize=7.5, loc="lower right", ncol=1)
    ax_top.set_title("(a) Métricas no MINDS\n(eixo cortado)", fontsize=9)

    # ── Painel (b): F1 por classe ─────────────────────────────────────────────
    ax = ax_cls
    ax.set_title("(b) F1-Score por classe MINDS", fontsize=9)

    n_cls  = len(minds_actions)
    y_pos  = np.arange(n_cls)
    h      = 0.32

    # Cria mapeamento de nome de classe para F1
    for mk, col, off in zip(models_order, [colors_minds["minds"], colors_minds["combined"]],
                             (-h/2, h/2)):
        f1_arr = results[mk]["f1_per_class"]
        act    = results[mk]["actions"]
        # Mapeia nome→F1 para o painel
        name_to_f1 = {act[i]: f1_arr[i] for i in results[mk]["present"]}
        f1_vals = [name_to_f1.get(a, 0.0) for a in minds_actions]

        ax.barh(y_pos + off, f1_vals, h, color=col, alpha=0.85, zorder=3,
                linewidth=0.5, edgecolor="white",
                label=display_labels[mk].replace("\n", " "))

        for yi, v in zip(y_pos + off, f1_vals):
            if v > 0.01:
                ax.text(v + 0.005, yi, f"{v:.2f}", va="center",
                        ha="left", fontsize=6.5, color="0.2")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(minds_actions, fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1-Score", fontsize=8.5)
    ax.axvline(1.0, color="0.6", linewidth=0.7, linestyle=":")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Comparação no MINDS: modelo treinado só no MINDS vs treinado no combinado",
        fontsize=10.5, y=1.01,
    )
    plt.tight_layout()
    savefig("fig_minds_comparacao")


# ─────────────────────────────────────────────────────────────────────────────
# Tabela de resultados
# ─────────────────────────────────────────────────────────────────────────────

def save_table(stats: dict):
    model_rows = [
        ("LSTM",                    "lstm"),
        ("BiLSTM+Atenção Combinado","combined"),
    ]
    metric_cols = [
        ("Acurácia",  "acc"),
        ("Precisão",  "prec"),
        ("Revocação", "rec"),
        ("F1-Score",  "f1"),
    ]

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "tabela_resultados.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        header = "Modelo," + ",".join(
            f"{m} (média),{m} (±dp)" for m, _ in metric_cols
        )
        fh.write(header + "\n")
        for label, key in model_rows:
            row = ",".join(
                f"{stats[key][k][0]:.4f},{stats[key][k][1]:.4f}"
                for _, k in metric_cols
            )
            fh.write(f"{label},{row}\n")
    print(f"    -> {csv_path}")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    tex_path = os.path.join(OUT_DIR, "tabela_resultados.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("% Tabela gerada por paper_figures.py\n")
        fh.write("\\begin{table}[ht]\n\\centering\n")
        fh.write(
            "\\caption{Desempenho comparativo (média $\\pm$ desvio-padrão, "
            "bootstrap com 1\\,000 reamostras). Melhores resultados em negrito.}\n"
        )
        fh.write("\\label{tab:resultados}\n")
        fh.write(f"\\begin{{tabular}}{{l{'c' * len(metric_cols)}}}\n\\hline\n")
        fh.write(
            "\\textbf{Modelo} & "
            + " & ".join(f"\\textbf{{{m}}}" for m, _ in metric_cols)
            + " \\\\\n\\hline\n"
        )
        # Determina o melhor valor por métrica
        best = {
            k: max(stats[mk][k][0] for mk, _ in [("lstm", "lstm"), ("combined", "combined")]
                   if mk in stats)
            for _, k in metric_cols
        }
        for label, key in model_rows:
            cells = []
            for _, k in metric_cols:
                m, s = stats[key][k]
                val_str = f"${m:.4f} \\pm {s:.4f}$"
                if abs(m - best[k]) < 1e-6:
                    val_str = f"\\textbf{{{val_str}}}"
                cells.append(val_str)
            fh.write(f"{label} & " + " & ".join(cells) + " \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"    -> {tex_path}")

    # ── Impressão no terminal ─────────────────────────────────────────────────
    print("\n  ┌" + "─" * 68 + "┐")
    header_parts = [f"{'Modelo':<28}"] + [f"{m[0]:>9}" for m in metric_cols]
    print("  │ " + "  ".join(header_parts) + " │")
    print("  ├" + "─" * 68 + "┤")
    for label, key in model_rows:
        cols = [f"{label:<28}"] + [
            f"{stats[key][k][0]:>8.4f}" for _, k in metric_cols
        ]
        print("  │ " + "  ".join(cols) + " │")
    print("  └" + "─" * 68 + "┘")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not TF_OK:
        sys.exit("[ERRO] TensorFlow não encontrado.")

    print("=" * 60)
    print("  GERAÇÃO DE FIGURAS PARA PUBLICAÇÃO")
    print("=" * 60)

    # 1. Dados de teste
    X_test, y_test, actions = get_test_data()

    # 2. Inferência
    model_map = {"lstm": "lstm", "combined": "bilstm_attn_combined"}
    probs_all = {}
    preds_all = {}
    for key, name in model_map.items():
        print(f"\n[INFERÊNCIA] {name}")
        probs_all[key], preds_all[key] = predict_with_model(name, X_test, actions)
        acc = np.mean(preds_all[key] == y_test)
        print(f"       Acurácia no teste: {acc:.4f}")

    # 3. Bootstrap de métricas
    print("\n[MÉTRICAS] Bootstrap (1000 amostras) ...")
    stats = {}
    for key in ["lstm", "combined"]:
        bs = bootstrap_metrics(y_test, preds_all[key], n_boot=1000, seed=cfg.SEED)
        stats[key] = {
            "acc":  (bs["accuracy_mean"],        bs["accuracy_std"]),
            "prec": (bs["precision_macro_mean"],  bs["precision_macro_std"]),
            "rec":  (bs["recall_macro_mean"],     bs["recall_macro_std"]),
            "f1":   (bs["f1_macro_mean"],         bs["f1_macro_std"]),
        }

    # 4. F1 por classe
    f1_data = {}
    for key in ["lstm", "combined"]:
        f1_pc = f1_score(
            y_test, preds_all[key],
            average=None, labels=np.arange(len(actions)), zero_division=0,
        )
        f1_data[key] = {"f1_per_class": f1_pc}

    # 5. Dados de confiança
    conf_data = {}
    for key in ["lstm", "combined"]:
        max_conf = np.max(probs_all[key], axis=1)
        correct  = (preds_all[key] == y_test)
        conf_data[key] = {
            "correct_conf": max_conf[correct],
            "wrong_conf":   max_conf[~correct],
        }

    # 6. Gerar figuras
    print("\n[FIG 1] Métricas comparativas (eixo cortado) ...")
    fig1_metricas(stats)

    print("[FIG 2] Histogramas de confiança ...")
    fig2_confiancas(conf_data)

    print("[FIG 3] Curvas Precisão–Revocação ...")
    fig3_pr_curves(probs_all, y_test, actions)

    print("[FIG 4] F1 por categoria de sinal ...")
    fig4_f1_categorias(f1_data, actions)

    print("[FIG 5] Análise I vs J ...")
    fig5_ij_analise(probs_all, y_test, actions)

    print("[FIG] Curvas de treinamento ...")
    fig_curvas_treino()

    print("[FIG] Comparação MINDS: minds-only vs combinado ...")
    fig_minds_comparacao()

    print("\n[TABELA] Resultados comparativos ...")
    save_table(stats)

    print(f"\n{'=' * 60}")
    print(f"  Figuras salvas em '{OUT_DIR}/'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
