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

Parâmetros de ablação via `overrides` dict:
  n_layers      : número de camadas LSTM (padrão: 2)
  no_layer_norm : desativa LayerNormalization (padrão: False)
  pooling       : 'avg' | 'max' | 'last' (padrão: 'avg') — só bilstm_attn
  no_residual   : desativa conexão residual na atenção (padrão: False)
  optimizer_type: 'adamw' | 'adam' | 'sgd' (padrão: 'adamw')
  dropout       : sobrescreve taxa de dropout
  rec_dropout   : sobrescreve taxa de dropout recorrente
  attn_heads    : sobrescreve número de cabeças de atenção
  attn_key_dim  : sobrescreve dimensão key_dim
  lstm_units    : sobrescreve lista de unidades LSTM
  dense_units   : sobrescreve lista de unidades densas
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import config as cfg


def build_model(model_name: str, input_shape: tuple, n_classes: int,
                lr_schedule=None, overrides: dict = None,
                label_smooth: float = None) -> tf.keras.Model:
    """
    Constrói e compila o modelo especificado.

    Args:
        model_name   : 'lstm' | 'bilstm' | 'bilstm_attn'
        input_shape  : (T, F)
        n_classes    : número de classes
        lr_schedule  : schedule de LR (usa INIT_LR se None)
        overrides    : dict com parâmetros a sobrescrever no MODEL_CONFIG
        label_smooth : label smoothing (usa cfg.LABEL_SMOOTH se None)

    Returns:
        modelo compilado
    """
    if model_name not in cfg.MODEL_CONFIGS:
        raise ValueError(
            f"Modelo '{model_name}' desconhecido. Escolha: {list(cfg.MODEL_CONFIGS)}"
        )

    # Mescla config base com overrides
    base_cfg = cfg.MODEL_CONFIGS[model_name].copy()
    if overrides:
        base_cfg.update(overrides)
    mcfg = base_cfg

    if model_name == "lstm":
        model = _build_lstm(input_shape, n_classes, mcfg)
    elif model_name == "bilstm":
        model = _build_bilstm(input_shape, n_classes, mcfg)
    else:  # bilstm_attn
        model = _build_bilstm_attn(input_shape, n_classes, mcfg)

    lr     = lr_schedule if lr_schedule is not None else cfg.INIT_LR
    ls     = label_smooth if label_smooth is not None else cfg.LABEL_SMOOTH
    opt_tp = mcfg.get("optimizer_type", "adamw")

    optimizer = _make_optimizer(lr, opt_tp)
    loss      = tf.keras.losses.CategoricalCrossentropy(label_smoothing=ls)
    model.compile(optimizer=optimizer, loss=loss, metrics=["categorical_accuracy"])
    model.summary()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _expand_units(units, n_layers):
    """Garante que units tenha exatamente n_layers entradas (replica o último se necessário)."""
    units = list(units)
    while len(units) < n_layers:
        units.append(units[-1])
    return units[:n_layers]


# ─────────────────────────────────────────────────────────────────────────────
# Arquiteturas
# ─────────────────────────────────────────────────────────────────────────────

def _build_lstm(input_shape, n_classes, mcfg):
    units    = mcfg["lstm_units"]
    dense_u  = mcfg["dense_units"]
    drop     = mcfg["dropout"]
    n_layers = mcfg.get("n_layers", 2)
    units    = _expand_units(units, n_layers)

    inp = layers.Input(shape=input_shape)
    x   = inp
    for i, u in enumerate(units):
        ret_seq = (i < n_layers - 1)
        x = layers.LSTM(u, return_sequences=ret_seq, dropout=drop)(x)

    for d in dense_u:
        x = layers.Dense(d, activation="relu")(x)
        x = layers.Dropout(drop)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="lstm")


def _build_bilstm(input_shape, n_classes, mcfg):
    units    = mcfg["lstm_units"]
    dense_u  = mcfg["dense_units"]
    drop     = mcfg["dropout"]
    rec_d    = mcfg.get("rec_dropout", 0.0)
    no_ln    = mcfg.get("no_layer_norm", False)
    n_layers = mcfg.get("n_layers", 2)
    units    = _expand_units(units, n_layers)

    inp = layers.Input(shape=input_shape)
    x   = inp
    for i, u in enumerate(units):
        ret_seq = (i < n_layers - 1)
        x = layers.Bidirectional(
            layers.LSTM(u, return_sequences=ret_seq, dropout=drop,
                        recurrent_dropout=rec_d)
        )(x)
        if not no_ln and ret_seq:
            x = layers.LayerNormalization()(x)

    for d in dense_u:
        x = layers.Dense(d, activation="relu")(x)
        x = layers.Dropout(drop)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="bilstm")


def _build_bilstm_attn(input_shape, n_classes, mcfg):
    units    = mcfg["lstm_units"]
    dense_u  = mcfg["dense_units"]
    drop     = mcfg["dropout"]
    rec_d    = mcfg.get("rec_dropout", 0.15)
    n_heads  = mcfg.get("attn_heads", 4)
    key_dim  = mcfg.get("attn_key_dim", 32)
    no_ln    = mcfg.get("no_layer_norm", False)
    no_res   = mcfg.get("no_residual", False)
    pooling  = mcfg.get("pooling", "avg")
    n_layers = mcfg.get("n_layers", 2)
    units    = _expand_units(units, n_layers)

    inp = layers.Input(shape=input_shape)
    x   = inp

    # Todas as camadas BiLSTM retornam sequências (necessário para atenção)
    for u in units:
        x = layers.Bidirectional(
            layers.LSTM(u, return_sequences=True, dropout=drop,
                        recurrent_dropout=rec_d)
        )(x)
        if not no_ln:
            x = layers.LayerNormalization()(x)

    # Multi-Head Self-Attention
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=key_dim)(x, x)
    if no_res:
        x = layers.LayerNormalization()(attn) if not no_ln else attn
    else:
        combined = x + attn
        x = layers.LayerNormalization()(combined) if not no_ln else combined

    # Pooling temporal
    if pooling == "avg":
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling1D()(x)
    else:  # "last"
        x = x[:, -1, :]

    # Cabeça densa com dropout decrescente
    drops = [drop, drop * 0.75]
    for i, d in enumerate(dense_u):
        x = layers.Dense(d, activation="relu")(x)
        x = layers.Dropout(drops[min(i, len(drops) - 1)])(x)

    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="bilstm_attn")


# ─────────────────────────────────────────────────────────────────────────────
# Otimizador
# ─────────────────────────────────────────────────────────────────────────────

def _make_optimizer(lr, optimizer_type="adamw"):
    if optimizer_type == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=lr, momentum=0.9, clipnorm=1.0
        )
    if optimizer_type == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    # adamw (padrão)
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
