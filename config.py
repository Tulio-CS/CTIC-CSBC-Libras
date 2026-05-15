# config.py
# -*- coding: utf-8 -*-
"""
Configuração centralizada do projeto LIBRAS Recognition.
Altere aqui — todos os outros scripts importam deste arquivo.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR       = "dataset"       # dataset coletado manualmente — Classe/*.npy (55 classes)
MINDS_DATA_DIR = "dataset_minds" # dataset MINDS processado — Classe/*.npy (20 classes)
MODELS_DIR      = "models"            # models/<nome>/ → model.keras + norm_stats.json + actions.npy
RESULTS_DIR     = "results"           # results/<nome>/ → plots e métricas

# ─────────────────────────────────────────────────────────────────────────────
# Treino
# ─────────────────────────────────────────────────────────────────────────────
SEED              = 42
TEST_SIZE         = 0.20
BATCH_SIZE        = 32
EPOCHS            = 500
INIT_LR           = 3e-4
WEIGHT_DECAY      = 1e-4
LABEL_SMOOTH      = 0.05
FEATURE_MODE      = "wrist_centered"  # 'absolute' | 'wrist_centered'
USE_GROUP_SPLIT   = True              # separa grupos (sessão/pessoa) entre treino e teste
USE_CLASS_WEIGHTS = True              # balanceia classes desiguais
N_BOOTSTRAP       = 1000             # reamostragens para intervalo de confiança

# ─────────────────────────────────────────────────────────────────────────────
# Augmentação de dados
# ─────────────────────────────────────────────────────────────────────────────
ROT_DEG         = 8.0   # rotação máxima em graus (x,y)
SCALE_MIN       = 0.9   # faixa de escala espacial
SCALE_MAX       = 1.1
JITTER_STD      = 0.01  # desvio do ruído gaussiano
TIME_MASK_RATIO = 0.10  # fração de timesteps mascarados (bloco contíguo)
TEMP_DROPOUT_P  = 0.05  # probabilidade de zerar um timestep

# ─────────────────────────────────────────────────────────────────────────────
# Arquiteturas disponíveis
#   Use:  python train.py --model lstm
#         python train.py --model bilstm
#         python train.py --model bilstm_attn        (padrão)
#         python train.py --model bilstm_attn_best   (melhor configuração dos estudos de ablação)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "lstm": {
        "bidirectional": False,
        "attention":     False,
        "lstm_units":    [128, 64],
        "dense_units":   [128],
        "dropout":       0.4,
        "rec_dropout":   0.0,
    },
    "bilstm": {
        "bidirectional": True,
        "attention":     False,
        "lstm_units":    [128, 64],
        "dense_units":   [128],
        "dropout":       0.4,
        "rec_dropout":   0.15,
    },
    "bilstm_attn": {
        "bidirectional": True,
        "attention":     True,
        "lstm_units":    [160, 128],
        "attn_heads":    4,
        "attn_key_dim":  32,
        "dense_units":   [192, 128],
        "dropout":       0.4,
        "rec_dropout":   0.15,
    },

    # ── Melhor configuração derivada dos estudos de ablação ───────────────────
    # Cada hiperparâmetro foi escolhido com base no melhor resultado observado:
    #
    #  lstm_units   [256, 192]  → suite "units":       +0.0083 vs médio
    #  attn_heads   8           → suite "attn_heads":  +0.0081 vs 4 heads
    #  attn_key_dim 64          → suite "attn_key_dim":+0.0054 vs key_dim=32
    #  dense_units  [64]        → suite "dense_head":  +0.0081 vs médio [192,128]
    #  dropout      0.0         → suite "dropout":     +0.0220 vs dropout=0.4 (maior ganho!)
    #  rec_dropout  0.15        → suite "rec_dropout": melhor valor individual
    #  pooling      "max"       → suite "pooling":     GlobalMaxPool marginal melhor
    #  no_residual  True        → suite "residual":    sem resíduo +0.0081
    #  no_layer_norm False      → suite "layer_norm":  com LN +0.0082 (mantém LN)
    #
    # Parâmetros de treino (via CLI em train.py):
    #   --label_smooth 0.10   → suite "label_smooth": 0.10 melhor (+0.0055)
    #   --no_weights          → suite "class_weight": sem CW marginal melhor
    #   --T 32                → suite "seq_ext":      T=32 melhor (+0.0081)
    #   (augmentação padrão — todas ativas, conforme suite "aug": +0.0109)
    "bilstm_attn_best": {
        "bidirectional": True,
        "attention":     True,
        "lstm_units":    [256, 192],
        "attn_heads":    8,
        "attn_key_dim":  64,
        "dense_units":   [64],
        "dropout":       0.0,
        "rec_dropout":   0.15,
        "pooling":       "max",
        "no_residual":   True,
        "no_layer_norm": False,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Inferência ao vivo
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "bilstm_attn"   # pasta dentro de MODELS_DIR
CONF_THRESH        = 0.30            # confiança mínima para aceitar predição
EMA_ALPHA          = 0.60            # suavização exponencial das probabilidades
MAJORITY_K         = 8               # janelas para majority vote
INFER_EVERY_N      = 2               # roda inferência a cada N frames (web)
                                     # 1 = todo frame, 2 = metade das vezes
CAM_INDEX          = 0
LOG_LIVE_CSV       = True
LIVE_CSV_PATH      = "logs/live_predictions.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Coleta de dados
# ─────────────────────────────────────────────────────────────────────────────
COLLECT_NUM_SEQUENCES   = 30
COLLECT_SEQUENCE_LENGTH = 15          # deve bater com T usado no treino
COLLECT_OUTPUT_DIR      = DATA_DIR    # salva direto em dataset/<SIGN_NAME>/
