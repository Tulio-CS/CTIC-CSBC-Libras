# ablation.py
# -*- coding: utf-8 -*-
"""
Estudo de ablação sistemático e abrangente para o modelo LIBRAS.

Treina múltiplas configurações e gera tabela comparativa + gráficos.

═══════════════════════════════════════════════════════════════════════════════
Suítes disponíveis (22 no total)
═══════════════════════════════════════════════════════════════════════════════

 # Arquitetura e estrutura do modelo
  arch        — LSTM vs BiLSTM vs BiLSTM+Atenção
  depth       — 1, 2, 3 camadas BiLSTM
  attn_heads  — 1, 2, 4, 8 cabeças de atenção
  attn_key_dim— key_dim = 16, 32, 64
  units       — tamanho pequeno / médio / grande das camadas LSTM
  dense_head  — cabeça densa pequena / média / grande
  pooling     — GlobalAvgPool vs GlobalMaxPool vs último timestep
  layer_norm  — com vs sem LayerNormalization
  residual    — com vs sem conexão residual na atenção

 # Regularização
  dropout     — 0.0, 0.2, 0.4, 0.6
  rec_dropout — 0.0, 0.10, 0.15, 0.30
  label_smooth— 0.0, 0.05, 0.10, 0.20
  class_weight— com vs sem ponderação de classes

 # Treinamento
  lr_sched    — ReduceLROnPlateau vs CosineDecay
  optimizer   — AdamW vs Adam vs SGD

 # Dados e features
  feat        — wrist-centered vs coordenadas absolutas
  aug         — com vs sem augmentação
  aug_ind     — cada augmentação individualmente (5 tipos)
  seq         — T = 8, 16, 32 frames
  seq_ext     — T = 4, 8, 12, 16, 24, 32 frames (grade fina)
  data_frac   — 25%, 50%, 75%, 100% dos dados de treino

 # Análise pós-treinamento (sem retreino)
  confusable  — pares de sinais mais confundidos (requer modelo já treinado)

═══════════════════════════════════════════════════════════════════════════════
Uso
═══════════════════════════════════════════════════════════════════════════════
  python ablation.py                            # todas as suítes de treinamento
  python ablation.py --suite arch               # só arquiteturas
  python ablation.py --suite depth              # profundidade de camadas
  python ablation.py --suite attn_heads         # cabeças de atenção
  python ablation.py --suite confusable         # análise de confundíveis (sem treino)
  python ablation.py --suite confusable --baseline_model bilstm_attn
  python ablation.py --epochs 60                # épocas por run (padrão 80)
  python ablation.py --skip_existing            # pula runs já executados
  python ablation.py --data dataset             # pasta de dados
  python ablation.py --no_tsne                  # pula t-SNE (mais rápido)
"""

import os, sys, json, argparse, csv, subprocess
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Definição das suítes de ablação
# ─────────────────────────────────────────────────────────────────────────────

SUITES = {

    # ── Arquitetura base ──────────────────────────────────────────────────────
    "arch": [
        {"name": "lstm",        "args": ["--model", "lstm"]},
        {"name": "bilstm",      "args": ["--model", "bilstm"]},
        {"name": "bilstm_attn", "args": ["--model", "bilstm_attn"]},
    ],

    # ── Profundidade: número de camadas BiLSTM ────────────────────────────────
    "depth": [
        {"name": "bilstm_attn_L1", "args": ["--model", "bilstm_attn", "--n_layers", "1"]},
        {"name": "bilstm_attn_L2", "args": ["--model", "bilstm_attn", "--n_layers", "2"]},
        {"name": "bilstm_attn_L3", "args": ["--model", "bilstm_attn", "--n_layers", "3"]},
    ],

    # ── Atenção: número de cabeças ────────────────────────────────────────────
    "attn_heads": [
        {"name": "attn_h1", "args": ["--model", "bilstm_attn", "--attn_heads", "1"]},
        {"name": "attn_h2", "args": ["--model", "bilstm_attn", "--attn_heads", "2"]},
        {"name": "attn_h4", "args": ["--model", "bilstm_attn", "--attn_heads", "4"]},
        {"name": "attn_h8", "args": ["--model", "bilstm_attn", "--attn_heads", "8"]},
    ],

    # ── Atenção: dimensão key_dim ─────────────────────────────────────────────
    "attn_key_dim": [
        {"name": "attn_kd16", "args": ["--model", "bilstm_attn", "--attn_key_dim", "16"]},
        {"name": "attn_kd32", "args": ["--model", "bilstm_attn", "--attn_key_dim", "32"]},
        {"name": "attn_kd64", "args": ["--model", "bilstm_attn", "--attn_key_dim", "64"]},
    ],

    # ── Tamanho das camadas LSTM ──────────────────────────────────────────────
    "units": [
        {"name": "units_small",
         "args": ["--model", "bilstm_attn",
                  "--lstm_units", "64", "32",
                  "--dense_units", "64"]},
        {"name": "units_medium",   # padrão: [160,128] + [192,128]
         "args": ["--model", "bilstm_attn"]},
        {"name": "units_large",
         "args": ["--model", "bilstm_attn",
                  "--lstm_units", "256", "192",
                  "--dense_units", "256", "192"]},
    ],

    # ── Cabeça densa ──────────────────────────────────────────────────────────
    "dense_head": [
        {"name": "dense_small",
         "args": ["--model", "bilstm_attn", "--dense_units", "64"]},
        {"name": "dense_medium",   # padrão: [192,128]
         "args": ["--model", "bilstm_attn"]},
        {"name": "dense_large",
         "args": ["--model", "bilstm_attn", "--dense_units", "256", "192", "128"]},
    ],

    # ── Estratégia de pooling ─────────────────────────────────────────────────
    "pooling": [
        {"name": "pool_avg",  "args": ["--model", "bilstm_attn", "--pooling", "avg"]},
        {"name": "pool_max",  "args": ["--model", "bilstm_attn", "--pooling", "max"]},
        {"name": "pool_last", "args": ["--model", "bilstm_attn", "--pooling", "last"]},
    ],

    # ── LayerNorm ─────────────────────────────────────────────────────────────
    "layer_norm": [
        {"name": "with_layer_norm",
         "args": ["--model", "bilstm_attn"]},
        {"name": "without_layer_norm",
         "args": ["--model", "bilstm_attn", "--no_layer_norm"]},
    ],

    # ── Conexão residual ──────────────────────────────────────────────────────
    "residual": [
        {"name": "with_residual",
         "args": ["--model", "bilstm_attn"]},
        {"name": "without_residual",
         "args": ["--model", "bilstm_attn", "--no_residual"]},
    ],

    # ── Dropout ───────────────────────────────────────────────────────────────
    "dropout": [
        {"name": "drop_0.0", "args": ["--model", "bilstm_attn", "--dropout", "0.0"]},
        {"name": "drop_0.2", "args": ["--model", "bilstm_attn", "--dropout", "0.2"]},
        {"name": "drop_0.4", "args": ["--model", "bilstm_attn", "--dropout", "0.4"]},
        {"name": "drop_0.6", "args": ["--model", "bilstm_attn", "--dropout", "0.6"]},
    ],

    # ── Dropout recorrente ────────────────────────────────────────────────────
    "rec_dropout": [
        {"name": "recdrop_0.00",
         "args": ["--model", "bilstm_attn", "--rec_dropout", "0.0"]},
        {"name": "recdrop_0.10",
         "args": ["--model", "bilstm_attn", "--rec_dropout", "0.1"]},
        {"name": "recdrop_0.15",
         "args": ["--model", "bilstm_attn", "--rec_dropout", "0.15"]},
        {"name": "recdrop_0.30",
         "args": ["--model", "bilstm_attn", "--rec_dropout", "0.3"]},
    ],

    # ── Label smoothing ───────────────────────────────────────────────────────
    "label_smooth": [
        {"name": "ls_0.00",
         "args": ["--model", "bilstm_attn", "--label_smooth", "0.0"]},
        {"name": "ls_0.05",
         "args": ["--model", "bilstm_attn", "--label_smooth", "0.05"]},
        {"name": "ls_0.10",
         "args": ["--model", "bilstm_attn", "--label_smooth", "0.1"]},
        {"name": "ls_0.20",
         "args": ["--model", "bilstm_attn", "--label_smooth", "0.2"]},
    ],

    # ── Ponderação de classes ─────────────────────────────────────────────────
    "class_weight": [
        {"name": "with_cw",    "args": ["--model", "bilstm_attn"]},
        {"name": "without_cw", "args": ["--model", "bilstm_attn", "--no_weights"]},
    ],

    # ── Agendador de taxa de aprendizado ──────────────────────────────────────
    "lr_sched": [
        {"name": "lrsched_reduce",
         "args": ["--model", "bilstm_attn"]},
        {"name": "lrsched_cosine",
         "args": ["--model", "bilstm_attn", "--cosine"]},
    ],

    # ── Otimizador ────────────────────────────────────────────────────────────
    "optimizer": [
        {"name": "opt_adamw", "args": ["--model", "bilstm_attn", "--optimizer", "adamw"]},
        {"name": "opt_adam",  "args": ["--model", "bilstm_attn", "--optimizer", "adam"]},
        {"name": "opt_sgd",   "args": ["--model", "bilstm_attn", "--optimizer", "sgd"]},
    ],

    # ── Modo de feature ───────────────────────────────────────────────────────
    "feat": [
        {"name": "bilstm_attn_wrist",
         "args": ["--model", "bilstm_attn"]},
        {"name": "bilstm_attn_absolute",
         "args": ["--model", "bilstm_attn", "--absolute"]},
    ],

    # ── Augmentação: com vs sem ───────────────────────────────────────────────
    "aug": [
        {"name": "bilstm_attn_aug",
         "args": ["--model", "bilstm_attn"]},
        {"name": "bilstm_attn_no_aug",
         "args": ["--model", "bilstm_attn", "--no_aug"]},
    ],

    # ── Augmentações individuais ──────────────────────────────────────────────
    "aug_ind": [
        {"name": "aug_none",
         "args": ["--model", "bilstm_attn", "--no_aug"]},
        {"name": "aug_all",
         "args": ["--model", "bilstm_attn"]},
        {"name": "aug_jitter",
         "args": ["--model", "bilstm_attn", "--aug_types", "jitter"]},
        {"name": "aug_rotation",
         "args": ["--model", "bilstm_attn", "--aug_types", "rotation"]},
        {"name": "aug_scale",
         "args": ["--model", "bilstm_attn", "--aug_types", "scale"]},
        {"name": "aug_temp_dropout",
         "args": ["--model", "bilstm_attn", "--aug_types", "temp_dropout"]},
        {"name": "aug_time_mask",
         "args": ["--model", "bilstm_attn", "--aug_types", "time_mask"]},
    ],

    # ── Comprimento de sequência (grade original) ─────────────────────────────
    "seq": [
        {"name": "bilstm_attn_T8",
         "args": ["--model", "bilstm_attn", "--T", "8"]},
        {"name": "bilstm_attn_T16",
         "args": ["--model", "bilstm_attn", "--T", "16"]},
        {"name": "bilstm_attn_T32",
         "args": ["--model", "bilstm_attn", "--T", "32"]},
    ],

    # ── Comprimento de sequência (grade fina) ─────────────────────────────────
    "seq_ext": [
        {"name": "T4",  "args": ["--model", "bilstm_attn", "--T", "4"]},
        {"name": "T8",  "args": ["--model", "bilstm_attn", "--T", "8"]},
        {"name": "T12", "args": ["--model", "bilstm_attn", "--T", "12"]},
        {"name": "T16", "args": ["--model", "bilstm_attn", "--T", "16"]},
        {"name": "T24", "args": ["--model", "bilstm_attn", "--T", "24"]},
        {"name": "T32", "args": ["--model", "bilstm_attn", "--T", "32"]},
    ],

    # ── Fração dos dados de treino ────────────────────────────────────────────
    "data_frac": [
        {"name": "data_25pct",
         "args": ["--model", "bilstm_attn", "--data_fraction", "0.25"]},
        {"name": "data_50pct",
         "args": ["--model", "bilstm_attn", "--data_fraction", "0.50"]},
        {"name": "data_75pct",
         "args": ["--model", "bilstm_attn", "--data_fraction", "0.75"]},
        {"name": "data_100pct",
         "args": ["--model", "bilstm_attn"]},
    ],
}

# Suítes que são apenas análise (sem treinamento)
ANALYSIS_SUITES = {"confusable"}

SUITE_LABELS = {
    "arch":         "Arquitetura (LSTM vs BiLSTM vs BiLSTM+Atenção)",
    "depth":        "Profundidade — nº de camadas BiLSTM",
    "attn_heads":   "Nº de cabeças de atenção",
    "attn_key_dim": "Dimensão key_dim da atenção",
    "units":        "Tamanho das unidades LSTM",
    "dense_head":   "Tamanho da cabeça densa",
    "pooling":      "Estratégia de pooling temporal",
    "layer_norm":   "LayerNormalization",
    "residual":     "Conexão residual na atenção",
    "dropout":      "Taxa de dropout",
    "rec_dropout":  "Taxa de dropout recorrente",
    "label_smooth": "Label smoothing",
    "class_weight": "Ponderação de classes",
    "lr_sched":     "Agendador de taxa de aprendizado",
    "optimizer":    "Otimizador",
    "feat":         "Modo de feature (wrist-centered vs absoluto)",
    "aug":          "Augmentação de dados (com vs sem)",
    "aug_ind":      "Augmentações individuais",
    "seq":          "Comprimento da sequência T (grade original)",
    "seq_ext":      "Comprimento da sequência T (grade fina)",
    "data_frac":    "Curva de aprendizado — fração dos dados",
    "confusable":   "Análise de pares de sinais confundíveis",
}

DISPLAY_NAMES = {
    # arch
    "lstm":                    "LSTM",
    "bilstm":                  "BiLSTM",
    "bilstm_attn":             "BiLSTM + Atenção",
    # depth
    "bilstm_attn_L1":          "1 camada BiLSTM",
    "bilstm_attn_L2":          "2 camadas BiLSTM (padrão)",
    "bilstm_attn_L3":          "3 camadas BiLSTM",
    # attn_heads
    "attn_h1":                 "1 head",
    "attn_h2":                 "2 heads",
    "attn_h4":                 "4 heads (padrão)",
    "attn_h8":                 "8 heads",
    # attn_key_dim
    "attn_kd16":               "key_dim=16",
    "attn_kd32":               "key_dim=32 (padrão)",
    "attn_kd64":               "key_dim=64",
    # units
    "units_small":             "Pequeno [64, 32]",
    "units_medium":            "Médio [160, 128] (padrão)",
    "units_large":             "Grande [256, 192]",
    # dense_head
    "dense_small":             "Dense pequeno [64]",
    "dense_medium":            "Dense médio [192, 128] (padrão)",
    "dense_large":             "Dense grande [256, 192, 128]",
    # pooling
    "pool_avg":                "GlobalAvgPool (padrão)",
    "pool_max":                "GlobalMaxPool",
    "pool_last":               "Último timestep",
    # layer_norm
    "with_layer_norm":         "Com LayerNorm (padrão)",
    "without_layer_norm":      "Sem LayerNorm",
    # residual
    "with_residual":           "Com resíduo (padrão)",
    "without_residual":        "Sem resíduo",
    # dropout
    "drop_0.0":                "Dropout=0.0",
    "drop_0.2":                "Dropout=0.2",
    "drop_0.4":                "Dropout=0.4 (padrão)",
    "drop_0.6":                "Dropout=0.6",
    # rec_dropout
    "recdrop_0.00":            "RecDrop=0.00",
    "recdrop_0.10":            "RecDrop=0.10",
    "recdrop_0.15":            "RecDrop=0.15 (padrão)",
    "recdrop_0.30":            "RecDrop=0.30",
    # label_smooth
    "ls_0.00":                 "LabelSmooth=0.00",
    "ls_0.05":                 "LabelSmooth=0.05 (padrão)",
    "ls_0.10":                 "LabelSmooth=0.10",
    "ls_0.20":                 "LabelSmooth=0.20",
    # class_weight
    "with_cw":                 "Com class weights (padrão)",
    "without_cw":              "Sem class weights",
    # lr_sched
    "lrsched_reduce":          "ReduceLROnPlateau (padrão)",
    "lrsched_cosine":          "CosineDecay",
    # optimizer
    "opt_adamw":               "AdamW (padrão)",
    "opt_adam":                "Adam",
    "opt_sgd":                 "SGD",
    # feat
    "bilstm_attn_wrist":       "Wrist-centered (padrão)",
    "bilstm_attn_absolute":    "Coordenadas absolutas",
    # aug
    "bilstm_attn_aug":         "Com augmentação (padrão)",
    "bilstm_attn_no_aug":      "Sem augmentação",
    # aug_ind
    "aug_none":                "Nenhuma augmentação",
    "aug_all":                 "Todas (padrão)",
    "aug_jitter":              "Só ruído gaussiano (jitter)",
    "aug_rotation":            "Só rotação 2D",
    "aug_scale":               "Só escala espacial",
    "aug_temp_dropout":        "Só dropout temporal",
    "aug_time_mask":           "Só time-mask",
    # seq
    "bilstm_attn_T8":          "T=8 frames",
    "bilstm_attn_T16":         "T=16 frames (padrão)",
    "bilstm_attn_T32":         "T=32 frames",
    # seq_ext
    "T4":                      "T=4",
    "T8":                      "T=8",
    "T12":                     "T=12",
    "T16":                     "T=16 (padrão)",
    "T24":                     "T=24",
    "T32":                     "T=32",
    # data_frac
    "data_25pct":              "25% dos dados",
    "data_50pct":              "50% dos dados",
    "data_75pct":              "75% dos dados",
    "data_100pct":             "100% dos dados (padrão)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Leitura de resultados
# ─────────────────────────────────────────────────────────────────────────────

def read_metrics(run_name, results_base="results"):
    """Lê métricas de bootstrap de um run já executado."""
    path = os.path.join(results_base, run_name, "metrics_bootstrap.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {
        "accuracy":  (d["accuracy_mean"],        d["accuracy_std"]),
        "precision": (d["precision_macro_mean"],  d["precision_macro_std"]),
        "recall":    (d["recall_macro_mean"],     d["recall_macro_std"]),
        "f1":        (d["f1_macro_mean"],         d["f1_macro_std"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Execução de um run de treino
# ─────────────────────────────────────────────────────────────────────────────

def run_config(config, data_dir, epochs, results_base, skip_existing, extra_train_args):
    """Executa train.py para uma configuração de ablação."""
    run_name    = config["name"]
    result_path = os.path.join(results_base, run_name, "metrics_bootstrap.json")

    if skip_existing and os.path.exists(result_path):
        print(f"\n[SKIP] {run_name} — resultado já existe.")
        return read_metrics(run_name, results_base)

    cmd = [
        sys.executable, "train.py",
        "--data",        data_dir,
        "--epochs",      str(epochs),
        "--run_name",    run_name,
        "--results_dir", os.path.join(results_base, run_name),
    ] + config["args"] + extra_train_args

    print(f"\n{'='*60}")
    print(f"[RUN] {run_name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[WARN] Run '{run_name}' terminou com código {ret.returncode}.")

    return read_metrics(run_name, results_base)


# ─────────────────────────────────────────────────────────────────────────────
# Análise de pares confundíveis (pós-treinamento, sem retreino)
# ─────────────────────────────────────────────────────────────────────────────

def run_confusable_analysis(baseline_model, results_base, outdir, data_dir):
    """
    Analisa quais pares de sinais são mais frequentemente confundidos.

    Carrega misclassifications.csv do modelo baseline e gera:
      1. Barras horizontais — top-20 pares confundidos
      2. Heatmap focado nos sinais mais confundidos
      3. Análise por tipo de sinal (estático vs dinâmico, 1 vs 2 mãos)
      4. Destaque de pares fonologicamente similares (ex: i/j — mesmo início)
    """
    misc_path = os.path.join(results_base, baseline_model, "misclassifications.csv")
    pcm_path  = os.path.join(results_base, baseline_model, "per_class_metrics.csv")

    if not os.path.exists(misc_path):
        # Tenta também em results/<baseline_model>
        misc_path_alt = os.path.join("results", baseline_model, "misclassifications.csv")
        pcm_path_alt  = os.path.join("results", baseline_model, "per_class_metrics.csv")
        if os.path.exists(misc_path_alt):
            misc_path = misc_path_alt
            pcm_path  = pcm_path_alt
        else:
            print(f"\n[WARN] misclassifications.csv não encontrado em:")
            print(f"       {misc_path}")
            print(f"       {misc_path_alt}")
            print(f"       Treine o modelo '{baseline_model}' primeiro:")
            print(f"       python train.py --model {baseline_model}")
            return

    os.makedirs(outdir, exist_ok=True)

    # ── Carrega confusões ──────────────────────────────────────────────────
    pairs   = defaultdict(int)
    per_cls = defaultdict(int)
    total   = 0

    with open(misc_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true = row["true"].strip()
            pred = row["pred"].strip()
            if true != pred:
                pairs[(true, pred)] += 1
                per_cls[true] += 1
                total += 1

    if total == 0:
        print("[INFO] Nenhuma confusão encontrada — modelo perfeito no teste!")
        return

    # ── Carrega metadados de sinais ────────────────────────────────────────
    meta_path = "sign_metadata.json"
    sign_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            sign_meta = json.load(f)
        sign_meta.pop("_comment", None)

    # ── 1. Barras — top-20 pares confundidos ──────────────────────────────
    top_pairs = sorted(pairs.items(), key=lambda kv: -kv[1])[:20]

    fig, ax = plt.subplots(figsize=(10, 7))
    pair_labels = [f'"{a}" → "{b}"' for (a, b), _ in top_pairs]
    counts      = [c for _, c in top_pairs]
    colors      = ["#e74c3c" if counts[0] > 0 else "#95a5a6"] * len(counts)
    ax.barh(pair_labels[::-1], counts[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Número de confusões no conjunto de teste")
    ax.set_title(f"Top-{len(top_pairs)} pares de sinais mais confundidos\n"
                 f"(modelo: {baseline_model} | total de erros: {total})")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(outdir, "confusable_top_pairs.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[OK] {path}")

    # ── 2. Heatmap dos sinais mais confundidos ────────────────────────────
    # Identifica os N sinais mais envolvidos em confusões
    all_confused_cls = set()
    for (a, b), _ in top_pairs:
        all_confused_cls.add(a); all_confused_cls.add(b)
    confused_list = sorted(all_confused_cls)
    n_cls = len(confused_list)
    idx_map = {c: i for i, c in enumerate(confused_list)}

    matrix = np.zeros((n_cls, n_cls), dtype=int)
    for (true, pred), cnt in pairs.items():
        if true in idx_map and pred in idx_map:
            matrix[idx_map[true], idx_map[pred]] = cnt

    fig, ax = plt.subplots(figsize=(max(8, n_cls * 0.55), max(6, n_cls * 0.50)))
    try:
        import seaborn as sns
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Reds",
                    xticklabels=confused_list, yticklabels=confused_list,
                    ax=ax, linewidths=0.3, cbar_kws={"label": "Confusões"})
    except ImportError:
        im = ax.imshow(matrix, cmap="Reds")
        plt.colorbar(im, ax=ax, label="Confusões")
        ax.set_xticks(range(n_cls)); ax.set_xticklabels(confused_list)
        ax.set_yticks(range(n_cls)); ax.set_yticklabels(confused_list)
    ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")
    ax.set_title("Matriz de Confusão (sinais mais confundidos)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    path = os.path.join(outdir, "confusable_heatmap.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[OK] {path}")

    # ── 3. Análise por tipo de sinal ──────────────────────────────────────
    if sign_meta:
        _plot_confusion_by_type(pairs, sign_meta, outdir)

    # ── 4. Destaque de pares fonologicamente similares ────────────────────
    _analyze_phonological_pairs(pairs, sign_meta, total, outdir)

    # ── 5. Relatório CSV de todas as confusões ────────────────────────────
    csv_out = os.path.join(outdir, "confusable_all_pairs.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true", "pred", "count", "true_hands", "true_movement",
                    "pred_hands", "pred_movement", "same_hands", "same_movement"])
        for (true, pred), cnt in sorted(pairs.items(), key=lambda kv: -kv[1]):
            tm = sign_meta.get(true, {})
            pm = sign_meta.get(pred, {})
            w.writerow([
                true, pred, cnt,
                tm.get("hands", "?"), tm.get("movement", "?"),
                pm.get("hands", "?"), pm.get("movement", "?"),
                tm.get("hands") == pm.get("hands"),
                tm.get("movement") == pm.get("movement"),
            ])
    print(f"[OK] {csv_out}")

    # ── 6. Resumo textual ─────────────────────────────────────────────────
    _print_confusable_summary(pairs, sign_meta, total, top_pairs)


def _plot_confusion_by_type(pairs, sign_meta, outdir):
    """Barras de erros agrupadas por tipo de sinal (mãos e movimento)."""
    categories = {
        "1 mão estático":  {"hands": 1, "movement": False},
        "1 mão dinâmico":  {"hands": 1, "movement": True},
        "2 mãos estático": {"hands": 2, "movement": False},
        "2 mãos dinâmico": {"hands": 2, "movement": True},
    }
    cat_counts = defaultdict(int)
    inter_counts = defaultdict(int)  # entre categorias diferentes

    for (true, pred), cnt in pairs.items():
        tm = sign_meta.get(true, {})
        pm = sign_meta.get(pred, {})
        if not tm or not pm:
            continue
        t_cat = _get_category(tm, categories)
        p_cat = _get_category(pm, categories)
        cat_counts[t_cat] += cnt
        if t_cat != p_cat:
            inter_counts[f"{t_cat}\n→ {p_cat}"] += cnt

    if not cat_counts:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Erros por categoria da classe verdadeira
    cats  = list(cat_counts.keys())
    vals  = [cat_counts[c] for c in cats]
    axes[0].bar(cats, vals, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"][:len(cats)],
                alpha=0.85)
    axes[0].set_title("Erros por tipo de sinal verdadeiro")
    axes[0].set_ylabel("Número de confusões")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    plt.setp(axes[0].get_xticklabels(), rotation=15)

    # Transferências entre categorias diferentes
    if inter_counts:
        inter_sorted = sorted(inter_counts.items(), key=lambda kv: -kv[1])[:10]
        il, iv = zip(*inter_sorted)
        axes[1].barh(list(il)[::-1], list(iv)[::-1], color="#9b59b6", alpha=0.85)
        axes[1].set_title("Transferências entre categorias distintas")
        axes[1].set_xlabel("Número de confusões")
        axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    plt.suptitle("Análise de Confusões por Tipo de Sinal", fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, "confusable_by_type.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {path}")


def _get_category(meta, categories):
    """Retorna a categoria de um sinal dado seu metadado."""
    for name, crit in categories.items():
        if (meta.get("hands") == crit["hands"] and
                meta.get("movement") == crit["movement"]):
            return name
    return "outro"


def _analyze_phonological_pairs(pairs, sign_meta, total_errors, outdir):
    """
    Destaca pares fonologicamente similares: mesma configuração de mão,
    diferindo apenas no movimento (ex: i=estático, j=dinâmico).
    """
    # Pares que começam igual mas diferem no movimento
    static_dynamic_pairs = []
    if sign_meta:
        for (true, pred), cnt in pairs.items():
            tm = sign_meta.get(true, {})
            pm = sign_meta.get(pred, {})
            if (tm and pm and
                    tm.get("hands") == pm.get("hands") and
                    tm.get("movement") != pm.get("movement")):
                static_dynamic_pairs.append(((true, pred), cnt))

    static_dynamic_pairs.sort(key=lambda x: -x[1])

    # Pares de letras do alfabeto (mesma família)
    alphabet_pairs = []
    alphabet = set("abcdefghijklmnopqrstuvwxyz")
    for (true, pred), cnt in pairs.items():
        if true.lower() in alphabet and pred.lower() in alphabet:
            alphabet_pairs.append(((true, pred), cnt))
    alphabet_pairs.sort(key=lambda x: -x[1])

    # Gera figura com dois painéis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Painel 1: estático vs dinâmico confundidos
    if static_dynamic_pairs:
        labels = [f'"{a}"({"din" if sign_meta.get(a,{}).get("movement") else "est"})'
                  f' → "{b}"({"din" if sign_meta.get(b,{}).get("movement") else "est"})'
                  for (a, b), _ in static_dynamic_pairs[:12]]
        vals   = [c for _, c in static_dynamic_pairs[:12]]
        axes[0].barh(labels[::-1], vals[::-1], color="#e67e22", alpha=0.85)
        axes[0].set_title("Confusão estático ↔ dinâmico\n(mesmo nº de mãos)")
        axes[0].set_xlabel("Confusões")
        axes[0].grid(axis="x", linestyle="--", alpha=0.4)
    else:
        axes[0].text(0.5, 0.5, "Nenhum par estático/dinâmico\nconfundido",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("Confusão estático ↔ dinâmico")

    # Painel 2: letras do alfabeto confundidas
    if alphabet_pairs:
        top_alpha = alphabet_pairs[:12]
        labels    = [f'"{a}" → "{b}"' for (a, b), _ in top_alpha]
        vals      = [c for _, c in top_alpha]
        axes[1].barh(labels[::-1], vals[::-1], color="#8e44ad", alpha=0.85)
        axes[1].set_title("Confusão entre letras do alfabeto")
        axes[1].set_xlabel("Confusões")
        axes[1].grid(axis="x", linestyle="--", alpha=0.4)
    else:
        axes[1].text(0.5, 0.5, "Nenhuma confusão entre letras",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Confusão entre letras do alfabeto")

    plt.suptitle("Análise de Similaridade Fonológica", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "confusable_phonological.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {path}")


def _print_confusable_summary(pairs, sign_meta, total, top_pairs):
    """Imprime resumo textual dos resultados de confundíveis."""
    print("\n" + "="*65)
    print("ANÁLISE DE SINAIS CONFUNDÍVEIS")
    print("="*65)
    print(f"  Total de erros de classificação: {total}")
    print(f"\n  Top-10 pares confundidos:")
    for i, ((true, pred), cnt) in enumerate(top_pairs[:10], 1):
        pct = 100 * cnt / total
        tm  = sign_meta.get(true, {})
        pm  = sign_meta.get(pred, {})
        t_desc = f"{tm.get('hands','?')}mão {'din' if tm.get('movement') else 'est'}" if tm else ""
        p_desc = f"{pm.get('hands','?')}mão {'din' if pm.get('movement') else 'est'}" if pm else ""
        print(f"  {i:2d}. \"{true}\"({t_desc}) → \"{pred}\"({p_desc})"
              f"  : {cnt:3d} confusões ({pct:.1f}%)")

    # Destaque especial: i vs j (estático vs dinâmico, mesmo início)
    ij_fwd = pairs.get(("i", "j"), 0)
    ij_bck = pairs.get(("j", "i"), 0)
    if ij_fwd + ij_bck > 0:
        print(f"\n  ► Par especial i/j (mesmo início, j tem movimento):")
        print(f"    i → j : {ij_fwd} confusões")
        print(f"    j → i : {ij_bck} confusões")
        print(f"    Total : {ij_fwd + ij_bck} ({100*(ij_fwd+ij_bck)/total:.1f}% dos erros)")
    else:
        print(f"\n  ► Par i/j: nenhuma confusão detectada no conjunto de teste.")

    print("="*65)


# ─────────────────────────────────────────────────────────────────────────────
# Geração de gráficos e relatório
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(path, dpi=150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"[OK] Salvo: {path}")


def plot_suite(suite_name, run_names, metrics_map, outdir):
    """
    Gráfico de barras com 4 métricas para cada run da suíte,
    com barras de erro (±std bootstrap).
    """
    labels    = [DISPLAY_NAMES.get(n, n) for n in run_names]
    m_keys    = ["accuracy", "precision", "recall", "f1"]
    m_labels  = ["Acurácia", "Precisão (macro)", "Recall (macro)", "F1-macro"]
    colors    = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    n_runs    = len(run_names)
    n_metrics = len(m_keys)
    x         = np.arange(n_runs)
    bar_w     = 0.18

    fig, ax = plt.subplots(figsize=(max(8, n_runs * 2.0), 6))

    for mi, (mk, ml, color) in enumerate(zip(m_keys, m_labels, colors)):
        means, stds = [], []
        for rn in run_names:
            m = metrics_map.get(rn)
            if m and mk in m:
                means.append(m[mk][0])
                stds.append(m[mk][1])
            else:
                means.append(0.0)
                stds.append(0.0)
        offset = (mi - n_metrics / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, means, bar_w,
                        yerr=stds, capsize=4,
                        label=ml, color=color, alpha=0.85)
        for rect, mean in zip(bars, means):
            if mean > 0:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.005,
                        f"{mean:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Valor da métrica")
    ax.set_title(f"Ablação — {SUITE_LABELS.get(suite_name, suite_name)}")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fname = os.path.join(outdir, f"ablation_{suite_name}.png")
    _savefig(fname)


def save_table(all_results, outdir):
    """Salva tabela CSV e TXT com todos os resultados e delta em relação ao baseline."""

    # Baseline = bilstm_attn ou bilstm_attn_aug (melhor resultado padrão)
    baseline_acc = None
    for bname in ("bilstm_attn", "bilstm_attn_aug", "bilstm_attn_T16", "data_100pct"):
        for suite_res in all_results.values():
            m = suite_res.get(bname)
            if m and m.get("accuracy"):
                baseline_acc = m["accuracy"][0]
                break
        if baseline_acc is not None:
            break

    rows = []
    for suite_name, suite_runs in all_results.items():
        for run_name, metrics in suite_runs.items():
            if metrics is None:
                continue
            acc_mean = metrics["accuracy"][0]
            delta    = (f"{acc_mean - baseline_acc:+.4f}"
                        if baseline_acc is not None else "N/A")
            rows.append({
                "suite":     suite_name,
                "run":       run_name,
                "label":     DISPLAY_NAMES.get(run_name, run_name),
                "accuracy":  f"{metrics['accuracy'][0]:.4f}",
                "acc_std":   f"{metrics['accuracy'][1]:.4f}",
                "precision": f"{metrics['precision'][0]:.4f}",
                "prec_std":  f"{metrics['precision'][1]:.4f}",
                "recall":    f"{metrics['recall'][0]:.4f}",
                "rec_std":   f"{metrics['recall'][1]:.4f}",
                "f1":        f"{metrics['f1'][0]:.4f}",
                "f1_std":    f"{metrics['f1'][1]:.4f}",
                "delta_acc": delta,
            })

    if not rows:
        print("[WARN] Nenhum resultado para tabular.")
        return

    # CSV
    csv_path = os.path.join(outdir, "ablation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] CSV: {csv_path}")

    # TXT legível
    txt_path = os.path.join(outdir, "ablation_results.txt")
    col_w = [
        ("Suíte",        10), ("Configuração", 34),
        ("Acc ± std",    16), ("Prec ± std",   16),
        ("Rec ± std",    16), ("F1 ± std",     16),
        ("ΔAcc",         8),
    ]
    header = "  ".join(f"{h:<{w}}" for h, w in col_w)
    sep    = "─" * len(header)

    lines = [sep, header, sep]
    prev_suite = None

    for r in rows:
        if r["suite"] != prev_suite:
            if prev_suite is not None:
                lines.append("")
            lines.append(f"\n  [{SUITE_LABELS.get(r['suite'], r['suite'])}]")
            prev_suite = r["suite"]

        row_str = (
            f"  {r['suite']:<10}  {r['label']:<34}  "
            f"{r['accuracy']} ± {r['acc_std']:<10}  "
            f"{r['precision']} ± {r['prec_std']:<10}  "
            f"{r['recall']} ± {r['rec_std']:<10}  "
            f"{r['f1']} ± {r['f1_std']:<10}  "
            f"{r['delta_acc']:<8}"
        )
        lines.append(row_str)

    lines.append(sep)
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"[OK] TXT: {txt_path}")


def plot_learning_curve(suite_results, outdir):
    """Curva de aprendizado (data_frac suite) — accuracy vs fração de dados."""
    frac_map = {
        "data_25pct": 0.25, "data_50pct": 0.50,
        "data_75pct": 0.75, "data_100pct": 1.00,
    }
    fracs, accs, errs = [], [], []
    for name, frac in frac_map.items():
        m = suite_results.get(name)
        if m:
            fracs.append(frac)
            accs.append(m["accuracy"][0])
            errs.append(m["accuracy"][1])

    if len(fracs) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(fracs, accs, yerr=errs, fmt="o-", capsize=5,
                color="#3498db", linewidth=2, markersize=7)
    ax.fill_between(fracs,
                    [a - e for a, e in zip(accs, errs)],
                    [a + e for a, e in zip(accs, errs)],
                    alpha=0.2, color="#3498db")
    ax.set_xlabel("Fração dos dados de treino")
    ax.set_ylabel("Acurácia (bootstrap)")
    ax.set_title("Curva de Aprendizado — BiLSTM+Atenção")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_xticks([0.25, 0.50, 0.75, 1.00])
    ax.set_xticklabels(["25%", "50%", "75%", "100%"])
    _savefig(os.path.join(outdir, "learning_curve.png"))


def plot_seq_sensitivity(suite_results, outdir):
    """Sensibilidade ao comprimento de sequência T (seq_ext suite)."""
    seq_map = {"T4": 4, "T8": 8, "T12": 12, "T16": 16, "T24": 24, "T32": 32}
    ts, accs, errs = [], [], []
    for name, t in seq_map.items():
        m = suite_results.get(name)
        if m:
            ts.append(t)
            accs.append(m["accuracy"][0])
            errs.append(m["accuracy"][1])

    if len(ts) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(ts, accs, yerr=errs, fmt="s-", capsize=5,
                color="#e67e22", linewidth=2, markersize=7)
    ax.fill_between(ts,
                    [a - e for a, e in zip(accs, errs)],
                    [a + e for a, e in zip(accs, errs)],
                    alpha=0.2, color="#e67e22")
    ax.set_xlabel("Comprimento de sequência T (frames)")
    ax.set_ylabel("Acurácia (bootstrap)")
    ax.set_title("Sensibilidade ao Comprimento de Sequência")
    ax.set_xticks(ts)
    ax.grid(linestyle="--", alpha=0.4)
    _savefig(os.path.join(outdir, "seq_sensitivity.png"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    all_suites = list(SUITES) + list(ANALYSIS_SUITES)
    p = argparse.ArgumentParser(description="Estudo de ablação abrangente — LIBRAS.")
    p.add_argument("--suite",
                   choices=all_suites + ["all"],
                   default="all",
                   help="Suíte a executar (padrão: all). 'confusable' não treina.")
    p.add_argument("--epochs",    type=int, default=80,
                   help="Épocas por run de treino (padrão: 80)")
    p.add_argument("--data",      default="dataset",
                   help="Pasta de dados (padrão: dataset)")
    p.add_argument("--out",       default="results_ablation",
                   help="Diretório de saída (padrão: results_ablation)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Pula runs cujo resultado já existe")
    p.add_argument("--no_tsne",   action="store_true",
                   help="Passa --no_tsne para evaluate (mais rápido)")
    p.add_argument("--baseline_model", default="bilstm_attn",
                   help="Modelo base para análise de confundíveis (padrão: bilstm_attn)")
    p.add_argument("--baseline_results", default=None,
                   help="Diretório de resultados do modelo base (padrão: --out)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Extra args passados ao train.py (ex: --no_tsne)
    extra_train_args = []
    if args.no_tsne:
        extra_train_args.append("--no_tsne")   # repassa ao evaluate via train.py
        # (train.py não aceita --no_tsne diretamente; o evaluate.py sim)
        # Se quiser, pode remover e só usar no evaluate separado.
        extra_train_args = []  # train.py não tem --no_tsne

    # Suítes a executar
    if args.suite == "all":
        train_suites    = list(SUITES.keys())
        analysis_suites = list(ANALYSIS_SUITES)
    elif args.suite in ANALYSIS_SUITES:
        train_suites    = []
        analysis_suites = [args.suite]
    else:
        train_suites    = [args.suite]
        analysis_suites = []

    all_results = {}

    # ── Suítes de treinamento ─────────────────────────────────────────────────
    for suite_name in train_suites:
        print(f"\n{'#'*65}")
        print(f"# SUÍTE: {SUITE_LABELS.get(suite_name, suite_name)}")
        print(f"{'#'*65}")

        suite_results = {}
        run_names     = []

        for config in SUITES[suite_name]:
            run_name = config["name"]
            run_names.append(run_name)
            metrics = run_config(
                config, args.data, args.epochs,
                args.out, args.skip_existing, extra_train_args,
            )
            suite_results[run_name] = metrics
            if metrics:
                acc = metrics["accuracy"]
                f1  = metrics["f1"]
                print(f"  → {DISPLAY_NAMES.get(run_name, run_name):<38} "
                      f"acc={acc[0]:.4f}±{acc[1]:.4f}  f1={f1[0]:.4f}±{f1[1]:.4f}")

        all_results[suite_name] = suite_results
        plot_suite(suite_name, run_names, suite_results, args.out)

        # Gráficos especiais por suíte
        if suite_name == "data_frac":
            plot_learning_curve(suite_results, args.out)
        if suite_name == "seq_ext":
            plot_seq_sensitivity(suite_results, args.out)

    # ── Suítes de análise (sem treino) ────────────────────────────────────────
    if "confusable" in analysis_suites:
        print(f"\n{'#'*65}")
        print(f"# ANÁLISE: {SUITE_LABELS['confusable']}")
        print(f"{'#'*65}")
        baseline_res = args.baseline_results or args.out
        run_confusable_analysis(
            baseline_model=args.baseline_model,
            results_base=baseline_res,
            outdir=os.path.join(args.out, "confusable"),
            data_dir=args.data,
        )

    # ── Tabela final ──────────────────────────────────────────────────────────
    if all_results:
        save_table(all_results, args.out)

    print(f"\n[OK] Ablação concluída. Resultados em: {os.path.abspath(args.out)}")

    # Imprime resumo de suítes executadas
    print("\n  Suítes executadas:")
    for s in train_suites:
        n_ok = sum(1 for m in all_results.get(s, {}).values() if m)
        n_tot = len(SUITES[s])
        print(f"    {s:<15} {n_ok}/{n_tot} runs com resultado")
    for s in analysis_suites:
        print(f"    {s:<15} análise pós-treino")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
