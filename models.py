# models.py
# -*- coding: utf-8 -*-
"""
Definições de arquiteturas disponíveis:
  - lstm         : LSTM empilhado
  - bilstm       : BiLSTM empilhado
  - bilstm_attn  : BiLSTM + MultiHeadAttention (melhor resultado)

Uso:
    from models import build_model
    model = build_model("bilstm_attn", input_shape=(15, 126), n_classes=55)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import config as cfg


def build_model(model_name: str, input_shape: tuple, n_classes: int,
                lr_schedule=None) -> tf.keras.Model:
    """
    Constrói e compila o modelo especificado.

    Args:
        model_name   : 'lstm' | 'bilstm' | 'bilstm_attn'
        input_shape  : (T, F)
        n_classes    : número de classes
        lr_schedule  : schedule de LR (usa INIT_LR se None)

    Returns:
        modelo compilado
    """
    if model_name not in cfg.MODEL_CONFIGS:
        raise ValueError(
            f"Modelo '{model_name}' desconhecido. Escolha: {list(cfg.MODEL_CONFIGS)}"
        )

    mcfg = cfg.MODEL_CONFIGS[model_name]

    if model_name == "lstm":
        model = _build_lstm(input_shape, n_classes, mcfg)
    elif model_name == "bilstm":
        model = _build_bilstm(input_shape, n_classes, mcfg)
    else:  # bilstm_attn
        model = _build_bilstm_attn(input_shape, n_classes, mcfg)

    lr = lr_schedule if lr_schedule is not None else cfg.INIT_LR
    optimizer = _make_optimizer(lr)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=cfg.LABEL_SMOOTH)
    model.compile(optimizer=optimizer, loss=loss, metrics=["categorical_accuracy"])
    model.summary()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Arquiteturas
# ─────────────────────────────────────────────────────────────────────────────

def _build_lstm(input_shape, n_classes, mcfg):
    units   = mcfg["lstm_units"]
    dense_u = mcfg["dense_units"]
    drop    = mcfg["dropout"]

    inp = layers.Input(shape=input_shape)
    x   = layers.LSTM(units[0], return_sequences=True,  dropout=drop)(inp)
    x   = layers.LSTM(units[1], return_sequences=False, dropout=drop)(x)
    for d in dense_u:
        x = layers.Dense(d, activation="relu")(x)
        x = layers.Dropout(drop)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="lstm")


def _build_bilstm(input_shape, n_classes, mcfg):
    units   = mcfg["lstm_units"]
    dense_u = mcfg["dense_units"]
    drop    = mcfg["dropout"]
    rec_d   = mcfg.get("rec_dropout", 0.0)

    inp = layers.Input(shape=input_shape)
    x   = layers.Bidirectional(
              layers.LSTM(units[0], return_sequences=True, dropout=drop,
                          recurrent_dropout=rec_d)
          )(inp)
    x   = layers.LayerNormalization()(x)
    x   = layers.Bidirectional(
              layers.LSTM(units[1], return_sequences=False, dropout=drop,
                          recurrent_dropout=rec_d)
          )(x)
    for d in dense_u:
        x = layers.Dense(d, activation="relu")(x)
        x = layers.Dropout(drop)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="bilstm")


def _build_bilstm_attn(input_shape, n_classes, mcfg):
    units     = mcfg["lstm_units"]
    dense_u   = mcfg["dense_units"]
    drop      = mcfg["dropout"]
    rec_d     = mcfg.get("rec_dropout", 0.15)
    n_heads   = mcfg.get("attn_heads", 4)
    key_dim   = mcfg.get("attn_key_dim", 32)

    inp = layers.Input(shape=input_shape)

    # BiLSTM 1
    x = layers.Bidirectional(
            layers.LSTM(units[0], return_sequences=True, dropout=drop,
                        recurrent_dropout=rec_d)
        )(inp)
    x = layers.LayerNormalization()(x)

    # BiLSTM 2
    x2 = layers.Bidirectional(
             layers.LSTM(units[1], return_sequences=True, dropout=drop,
                         recurrent_dropout=rec_d)
         )(x)
    x2 = layers.LayerNormalization()(x2)

    # Multi-Head Self-Attention
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=key_dim)(x2, x2)
    x    = layers.LayerNormalization()(x2 + attn)       # residual
    x    = layers.GlobalAveragePooling1D()(x)

    # Dense head
    drops = [drop, drop * 0.75]                          # dropout decrescente
    for i, d in enumerate(dense_u):
        x = layers.Dense(d, activation="relu")(x)
        x = layers.Dropout(drops[min(i, len(drops) - 1)])(x)

    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="bilstm_attn")


# ─────────────────────────────────────────────────────────────────────────────
# Otimizador
# ─────────────────────────────────────────────────────────────────────────────

def _make_optimizer(lr):
    try:
        return tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=cfg.WEIGHT_DECAY,
            clipnorm=1.0,
        )
    except Exception:
        print("[WARN] AdamW indisponível — usando Adam com clipnorm.")
        return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)


def make_cosine_schedule(init_lr, steps_per_epoch, epochs):
    """CosineDecay com mínimo de 15% do LR inicial."""
    decay_steps = steps_per_epoch * max(50, epochs // 2)
    return tf.keras.optimizers.schedules.CosineDecay(
        init_lr, decay_steps, alpha=0.15
    )


def make_callbacks(model_path: str, use_cosine: bool = False):
    """
    Retorna lista de callbacks padrão.
    Se use_cosine=True, omite ReduceLROnPlateau (incompatível com CosineDecay).
    """
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_categorical_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    if not use_cosine:
        cbs.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            )
        )
    return cbs
