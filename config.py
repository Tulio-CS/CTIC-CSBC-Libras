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
DATA_DIR        = "dataset"           # pasta Classe/*.npy (seu próprio dataset)
LIBRAS_DATA_DIR = "libras_data"       # alias legado (coleta manual)
MINDS_DATA_DIR  = "libras_data_minds" # dataset MINDS processado
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
#         python train.py --model bilstm_attn   (padrão — melhor resultado)
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
