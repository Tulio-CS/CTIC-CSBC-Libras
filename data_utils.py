# data_utils.py
# -*- coding: utf-8 -*-
"""
Utilitários compartilhados de dados:
  - Carregamento e listagem de classes
  - Transformações de features (wrist-centered)
  - Split estratificado por grupo
  - Pipeline tf.data com augmentações
  - Bootstrap de métricas
"""

import os, json
import numpy as np
import tensorflow as tf
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento
# ─────────────────────────────────────────────────────────────────────────────

def list_classes(data_dir: str) -> np.ndarray:
    """Retorna array de nomes de classes (subpastas ordenadas)."""
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    return np.array(classes)


def load_sequences(data_dir: str):
    """
    Carrega todas as sequências .npy de data_dir/<classe>/*.npy.
    Retorna: X (object array), y (int array), actions (str array), meta (list de dict)
    """
    actions = list_classes(data_dir)
    X, y, meta = [], [], []

    for ci, act in enumerate(actions):
        for fp in sorted(glob(os.path.join(data_dir, act, "*.npy"))):
            try:
                arr = np.load(fp)           # (T, F)
                if arr.ndim != 2:
                    print(f"[WARN] Shape inesperado em {fp}: {arr.shape} — ignorado.")
                    continue
                X.append(arr)
                y.append(ci)
                base  = os.path.splitext(os.path.basename(fp))[0]
                group = base.split("_")[0] if "_" in base else base
                meta.append({"path": fp, "class": act, "group": group})
            except Exception as e:
                print(f"[WARN] Falha ao carregar {fp}: {e}")

    if not X:
        raise RuntimeError(f"Nenhuma sequência .npy encontrada em '{data_dir}'.")

    return np.array(X, dtype=object), np.array(y, dtype=int), actions, meta


def scan_labeled_dir(root_dir: str, actions: np.ndarray):
    """Retorna lista de (path, class_idx) para avaliação externa."""
    items = []
    for idx, cls in enumerate(actions):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in sorted(os.listdir(cls_dir)):
            if f.endswith(".npy"):
                items.append((os.path.join(cls_dir, f), idx))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Transformações de features
# ─────────────────────────────────────────────────────────────────────────────

def to_wrist_centered(x: np.ndarray) -> np.ndarray:
    """
    Centraliza os landmarks no pulso de cada mão.
    x: (T, F) onde F ∈ {63, 126}.  Retorna x sem modificação se F for outro.
    Ordem esperada: [mão_direita(63), mão_esquerda(63)] — padrão do projeto.
    """
    F = x.shape[1]
    if F == 63:
        pts   = x.reshape(x.shape[0], 21, 3)
        pts  -= pts[:, 0:1, :]          # subtrai pulso (landmark 0)
        return pts.reshape(x.shape[0], -1)
    elif F == 126:
        pts = x.reshape(x.shape[0], 42, 3)
        # Mão direita: pontos 0..20, pulso em 0
        pts[:, 0:21, :]  -= pts[:, 0:1,  :]
        # Mão esquerda: pontos 21..41, pulso em 21
        pts[:, 21:42, :] -= pts[:, 21:22, :]
        return pts.reshape(x.shape[0], -1)
    else:
        return x  # F desconhecido — retorna sem alterar


def apply_feature_mode(x: np.ndarray, mode: str) -> np.ndarray:
    """Aplica a transformação de features configurada."""
    if mode == "wrist_centered":
        return to_wrist_centered(x)
    return x  # 'absolute'


# ─────────────────────────────────────────────────────────────────────────────
# Padding / crop temporal
# ─────────────────────────────────────────────────────────────────────────────

def pad_or_crop_to_T(x: np.ndarray, T: int) -> np.ndarray:
    """Garante que x tenha exatamente T timesteps (crop centralizado ou zero-pad no final)."""
    t = x.shape[0]
    if t == T:
        return x
    if t > T:
        start = (t - T) // 2
        return x[start:start + T]
    pad = np.zeros((T - t, x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Split por grupo (separa sessões/pessoas entre treino e teste)
# ─────────────────────────────────────────────────────────────────────────────

def group_stratified_split(X, y, meta, test_size=0.20, seed=42):
    """
    Divide o dataset mantendo grupos (prefixo do nome do arquivo antes de '_')
    separados entre treino e teste para evitar vazamento de dados.
    """
    by_group        = defaultdict(list)
    groups_by_class = defaultdict(list)

    for i, m in enumerate(meta):
        by_group[(y[i], m["group"])].append(i)
    for (cls, grp), idxs in by_group.items():
        groups_by_class[cls].append((grp, idxs))

    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []

    for cls, grp_list in groups_by_class.items():
        rng.shuffle(grp_list)
        all_idxs = sum((idxs for _, idxs in grp_list), [])
        target   = int(np.ceil(len(all_idxs) * test_size))

        picked, count = [], 0
        for _, idxs in grp_list:
            if count >= target:
                break
            picked.extend(idxs)
            count += len(idxs)

        picked_set = set(picked)
        for _, idxs in grp_list:
            for i in idxs:
                (test_idx if i in picked_set else train_idx).append(i)

    return np.array(train_idx, int), np.array(test_idx, int)


def make_split(X, y, meta, test_size, seed, use_group_split):
    """Wrapper — usa split por grupo ou split aleatório estratificado."""
    if use_group_split:
        return group_stratified_split(X, y, meta, test_size=test_size, seed=seed)
    return train_test_split(
        np.arange(len(X)), test_size=test_size, random_state=seed, stratify=y
    )


# ─────────────────────────────────────────────────────────────────────────────
# Estatísticas de normalização
# ─────────────────────────────────────────────────────────────────────────────

def compute_norm_stats(X_train_list, T_fixed, feature_mode):
    """Calcula média e desvio-padrão do conjunto de treino (para z-score)."""
    X_tmp = np.stack([
        apply_feature_mode(pad_or_crop_to_T(xi, T_fixed), feature_mode)
        for xi in X_train_list
    ], axis=0).astype(np.float32)

    mu = X_tmp.mean(axis=(0, 1), keepdims=True)   # (1, 1, F)
    sd = X_tmp.std(axis=(0, 1),  keepdims=True) + 1e-8
    return mu, sd, X_tmp.shape[2]   # retorna também F


def save_norm_stats(mu, sd, T, F, feature_mode, out_path):
    import json
    data = {
        "mu":           mu.squeeze().tolist(),
        "sd":           sd.squeeze().tolist(),
        "T":            int(T),
        "F":            int(F),
        "feature_mode": feature_mode,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] norm_stats salvo em: {out_path}")


def load_norm_stats(path):
    """
    Carrega mu, sd, T, F e feature_mode de um norm_stats.json.
    Retorna: mu (1,1,F), sd (1,1,F), T, F, feature_mode
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    mu = np.array(d["mu"], dtype=np.float32).reshape(1, 1, -1)
    sd = np.array(d["sd"], dtype=np.float32).reshape(1, 1, -1)
    return mu, sd, int(d["T"]), int(d["F"]), d.get("feature_mode", "absolute")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline tf.data
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(X_list, y, T_fixed, mu, sd, batch_size, training, feature_mode,
                 aug_types=None):
    """
    Constrói um tf.data.Dataset com normalização e (opcionalmente) augmentações.
    X_list   : lista de arrays (T_i, F)
    aug_types: set de strings com augmentações a aplicar, ou None = todas.
               Ignorado quando training=False.
               Valores válidos: 'jitter', 'rotation', 'scale', 'temp_dropout', 'time_mask'
    """
    X2 = np.stack([
        apply_feature_mode(pad_or_crop_to_T(xi, T_fixed), feature_mode)
        for xi in X_list
    ], axis=0).astype(np.float32)

    X2 = (X2 - mu) / (sd + 1e-8)
    y2 = tf.keras.utils.to_categorical(y).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((X2, y2))

    if training:
        ds = ds.shuffle(len(X2), seed=cfg.SEED, reshuffle_each_iteration=True)
        aug_fn = make_augment_fn(aug_types)
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_augment_fn(aug_types=None):
    """
    Fábrica de funções de augmentação para uso no tf.data pipeline.

    aug_types=None  → aplica TODAS as augmentações (comportamento padrão).
    aug_types=set() → aplica nenhuma (equivalente a training=False, mas via factory).
    aug_types={'jitter', 'rotation'} → aplica apenas as listadas.

    Tipos disponíveis: 'jitter', 'rotation', 'scale', 'temp_dropout', 'time_mask'

    Os condicionais Python são avaliados em tempo de rastreamento pelo TF,
    portanto branches inativos não entram no grafo computacional.
    """
    ALL = {"jitter", "rotation", "scale", "temp_dropout", "time_mask"}
    active = ALL if aug_types is None else (set(aug_types) & ALL)

    def _aug(x, y):
        T  = tf.shape(x)[0]
        F  = tf.shape(x)[1]
        P  = F // 3
        x3 = tf.reshape(x, (T, P, 3))

        if "jitter" in active:
            x3 += tf.random.normal(tf.shape(x3), stddev=cfg.JITTER_STD)

        if "rotation" in active:
            theta = tf.random.uniform([], -_deg2rad(cfg.ROT_DEG), _deg2rad(cfg.ROT_DEG))
            c, s  = tf.cos(theta), tf.sin(theta)
            rot   = tf.stack([[c, -s], [s, c]])
            xy    = tf.reshape(x3[..., :2], (-1, 2)) @ rot
            xy    = tf.reshape(xy, (T, P, 2))
            x3    = tf.concat([xy, x3[..., 2:3]], axis=-1)

        if "scale" in active:
            x3 *= tf.random.uniform([], cfg.SCALE_MIN, cfg.SCALE_MAX)

        if "temp_dropout" in active:
            keep = tf.cast(tf.random.uniform((T,)) > cfg.TEMP_DROPOUT_P, tf.float32)
            x3  *= tf.reshape(keep, (T, 1, 1))

        if "time_mask" in active:
            L = tf.cast(tf.round(cfg.TIME_MASK_RATIO * tf.cast(T, tf.float32)), tf.int32)
            L = tf.maximum(L, 0)

            def apply_time_mask(x3_):
                start = tf.random.uniform([], 0, tf.maximum(T - L, 1), dtype=tf.int32)
                mask  = tf.concat([
                    tf.zeros((start, 1, 1)),
                    tf.ones((L, 1, 1)),
                    tf.zeros((T - start - L, 1, 1))
                ], axis=0)
                return x3_ * (1.0 - mask)

            x3 = tf.cond(L > 0, lambda: apply_time_mask(x3), lambda: x3)

        return tf.reshape(x3, (T, F)), y

    return _aug


# Mantido para compatibilidade retroativa com código legado que importa _augment_fn
def _augment_fn(x, y):
    return make_augment_fn(None)(x, y)


def _deg2rad(deg):
    import math
    return math.pi * deg / 180.0


def materialize_dataset(ds, max_samples=50_000):
    """
    Materializa um tf.data.Dataset em arrays numpy.
    Substitui o frágil next(iter(ds.unbatch().batch(N))).
    """
    X_parts, y_parts = [], []
    for xb, yb in ds:
        X_parts.append(xb.numpy())
        y_parts.append(yb.numpy())
    return np.concatenate(X_parts), np.concatenate(y_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap de métricas
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_metrics(y_true, y_pred, n_boot=1000, seed=42):
    """
    Calcula média e desvio-padrão (bootstrap) para accuracy, precision, recall e F1 (macro).
    Retorna dict com keys: *_mean e *_std.
    """
    rng = np.random.default_rng(seed)
    N   = len(y_true)
    accs, precs, recs, f1s = [], [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        yt  = y_true[idx]
        yp  = y_pred[idx]
        accs.append(accuracy_score(yt, yp))
        precs.append(precision_score(yt, yp, average="macro", zero_division=0))
        recs.append(recall_score(yt, yp, average="macro", zero_division=0))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))

    def _s(lst): return np.array(lst, dtype=np.float64)
    accs, precs, recs, f1s = map(_s, (accs, precs, recs, f1s))

    return {
        "accuracy_mean":        float(accs.mean()),
        "accuracy_std":         float(accs.std(ddof=1)),
        "precision_macro_mean": float(precs.mean()),
        "precision_macro_std":  float(precs.std(ddof=1)),
        "recall_macro_mean":    float(recs.mean()),
        "recall_macro_std":     float(recs.std(ddof=1)),
        "f1_macro_mean":        float(f1s.mean()),
        "f1_macro_std":         float(f1s.std(ddof=1)),
        "n_bootstrap":          int(n_boot),
    }


def expected_calibration_error(probs, y_true, n_bins=10):
    """ECE multi-classe usando confiança da classe predita."""
    y_pred = probs.argmax(1)
    conf   = probs[np.arange(len(y_pred)), y_pred]
    acc    = (y_pred == y_true).astype(float)
    edges  = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)
