# train_libras_bilstm_v3_plus.py
# -*- coding: utf-8 -*-
"""
LIBRAS landmarks -> BiLSTM + Temporal Attention
- wrist-centered features (default)
- tf.data com jitter/rotate/scale/time-mask/temporal-dropout
- AdamW + CosineDecay + grad clipping + label smoothing
- Class weights opcionais (balanceamento)
- Salva métricas no formato solicitado (Classification Report)
"""

import os, json, random, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_class_weight

# --------------------- CONFIG ---------------------
DATA_DIR         = "dataset"
RESULTS_DIR      = "results_v4"
MODEL_OUT        = "best_model.keras"
ACTIONS_OUT      = "libras_actions.npy"
SEED             = 13
TEST_SIZE        = 0.2
BATCH_SIZE       = 32
EPOCHS           = 500
INIT_LR          = 3e-4
WEIGHT_DECAY     = 1e-4
LABEL_SMOOTH     = 0.05
FEATURE_MODE     = "wrist_centered"
USE_GROUP_SPLIT  = True
ROT_DEG          = 8.0
SCALE_MIN, SCALE_MAX = 0.9, 1.1
JITTER_STD       = 0.01
TIME_MASK_RATIO  = 0.1
TEMP_DROPOUT_P   = 0.05
SAVE_HISTORY_PNG = True
USE_CLASS_WEIGHTS= True
ATTN_HEADS       = 4
ATTN_KEY_DIM     = 32

os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ------------------ LOAD DATA --------------------
def list_classes(data_dir):
    acts = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    acts.sort()
    return np.array(acts)

def load_sequences(data_dir):
    actions = list_classes(data_dir)
    X, y, meta = [], [], []
    for ci, act in enumerate(actions):
        for fp in glob(os.path.join(data_dir, act, "*.npy")):
            try:
                arr = np.load(fp)
                X.append(arr)
                y.append(ci)
                base = os.path.splitext(os.path.basename(fp))[0]
                group = base.split("_")[0] if "_" in base else base
                meta.append({"path": fp, "class": act, "group": group})
            except Exception as e:
                print(f"[WARN] {fp}: {e}")
    return np.array(X, dtype=object), np.array(y, dtype=int), actions, meta

def pad_or_crop_to_T(x, T):
    t = x.shape[0]
    if t == T: return x
    if t > T:
        start = (t - T) // 2
        return x[start:start+T]
    pad = np.zeros((T - t, x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)

def infer_TF_from_first(X):
    x0 = X[0]
    return x0.shape[0], x0.shape[1]

# ---------- FEATURE TRANSFORMS ----------
def to_wrist_centered(x):
    F = x.shape[1]
    if F not in (63, 126): return x
    if F == 63:
        pts = x.reshape(x.shape[0], 21, 3)
        return (pts - pts[:, 0:1, :]).reshape(x.shape[0], -1)
    pts = x.reshape(x.shape[0], 42, 3)
    pts[:, 0:21, :]  -= pts[:, 0:1, :]
    pts[:, 21:42, :] -= pts[:, 21:22, :]
    return pts.reshape(x.shape[0], -1)

def apply_feature_mode(x, mode):
    return to_wrist_centered(x) if mode == "wrist_centered" else x

# ----------------- SPLIT ------------------------
def group_stratified_split(X, y, meta, test_size=0.2, seed=42):
    by_group = defaultdict(list)
    for i, m in enumerate(meta):
        by_group[(y[i], m["group"])].append(i)
    groups_by_class = defaultdict(list)
    for (cls, grp), idxs in by_group.items():
        groups_by_class[cls].append((grp, idxs))
    rng = np.random.RandomState(seed)
    test_idx, train_idx = [], []
    for cls, grp_list in groups_by_class.items():
        rng.shuffle(grp_list)
        cls_all = sum((idxs for (_, idxs) in grp_list), [])
        target = int(np.ceil(len(cls_all) * test_size))
        picked, count = [], 0
        for grp, idxs in grp_list:
            if count >= target: break
            picked.extend(idxs); count += len(idxs)
        cls_test = set(picked)
        for _, idxs in grp_list:
            for i in idxs:
                (test_idx if i in cls_test else train_idx).append(i)
    return np.array(train_idx, int), np.array(test_idx, int)

# ------------- tf.data + AUGMENT ---------------
def make_dataset(X, y, T_fixed, mu, sd, batch, training, feature_mode):
    X2 = [(apply_feature_mode(pad_or_crop_to_T(xi, T_fixed), feature_mode)).astype(np.float32) for xi in X]
    X2 = np.stack(X2, axis=0)
    y2 = tf.keras.utils.to_categorical(y).astype(np.float32)
    X2 = (X2 - mu) / (sd + 1e-8)
    ds = tf.data.Dataset.from_tensor_slices((X2, y2))
    if training:
        ds = ds.shuffle(len(X2), seed=SEED, reshuffle_each_iteration=True)
        def aug_fn(x, y):
            T = tf.shape(x)[0]; F = tf.shape(x)[1]; P = F // 3
            x3 = tf.reshape(x, (T, P, 3))
            x3 += tf.random.normal(tf.shape(x3), stddev=JITTER_STD)
            theta = tf.random.uniform([], -np.deg2rad(ROT_DEG), np.deg2rad(ROT_DEG))
            c, s = tf.cos(theta), tf.sin(theta)
            rot = tf.stack([[c, -s],[s, c]])
            xy = tf.reshape(x3[..., :2], (-1, 2)) @ rot
            xy = tf.reshape(xy, (T, P, 2))
            x3 = tf.concat([xy, x3[..., 2:3]], axis=-1)
            scale = tf.random.uniform([], SCALE_MIN, SCALE_MAX)
            x3 *= scale
            mask_keep = tf.cast(tf.random.uniform((T,)) > TEMP_DROPOUT_P, tf.float32)
            x3 *= tf.reshape(mask_keep, (T,1,1))
            L = tf.cast(tf.round(TIME_MASK_RATIO * tf.cast(T, tf.float32)), tf.int32)
            def time_mask_block(x3_):
                start = tf.random.uniform([], 0, tf.maximum(T - L, 1), dtype=tf.int32)
                mask = tf.concat([tf.zeros((start,1,1)), tf.ones((L,1,1)), tf.zeros((T-start-L,1,1))], axis=0)
                return x3_ * (1.0 - mask)
            x3 = tf.cond(L > 0, lambda: time_mask_block(x3), lambda: x3)
            return tf.reshape(x3, (T, F)), y
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

# ----------------- MODEL ------------------------
def build_model(input_shape, n_classes, lr_schedule):
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, clipnorm=1.0)
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR, clipnorm=1.0)
    inp = Input(shape=input_shape)
    x  = Bidirectional(LSTM(160, return_sequences=True, dropout=0.3, recurrent_dropout=0.15))(inp)
    x  = LayerNormalization()(x)
    x2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.15))(x)
    x2 = LayerNormalization()(x2)
    attn = MultiHeadAttention(num_heads=ATTN_HEADS, key_dim=ATTN_KEY_DIM)(x2, x2)
    x   = LayerNormalization()(x2 + attn)
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(192, activation="relu")(x)
    x   = Dropout(0.4)(x)
    x   = Dense(128, activation="relu")(x)
    x   = Dropout(0.3)(x)
    out = Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)
    model.compile(optimizer=opt, loss=loss, metrics=["categorical_accuracy"])
    return model

# ----------------- PLOTS + METRICS ------------------------
def plot_training(history, outdir):
    plt.figure(figsize=(12,6))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss.png")); plt.close()
    plt.figure(figsize=(12,6))
    plt.plot(history.history["categorical_accuracy"], label="train")
    plt.plot(history.history["val_categorical_accuracy"], label="val")
    plt.title("Accuracy"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy.png")); plt.close()

def eval_and_plots(model, X_test, y_test, actions, outdir):
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    report_str = classification_report(y_true, y_pred, target_names=actions, digits=3)
    acc = accuracy_score(y_true, y_pred)
    print("\n=== Classification Report ===")
    print(report_str)
    print(f"\naccuracy                          {acc:0.3f}      {len(y_true)}")
    with open(os.path.join(outdir, "metrics_report.txt"), "w", encoding="utf-8") as f:
        f.write("=== Classification Report ===\n")
        f.write(report_str)
        f.write(f"\naccuracy                          {acc:0.3f}      {len(y_true)}\n")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(actions)))
    ConfusionMatrixDisplay(cm, display_labels=actions).plot(xticks_rotation=45, colorbar=False)
    plt.title("Matriz de confusão (abs)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "confusion_abs.png")); plt.close()
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cmn, display_labels=actions).plot(xticks_rotation=45, colorbar=False, values_format=".2f")
    plt.title("Matriz de confusão (norm)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "confusion_norm.png")); plt.close()
    Yb = label_binarize(y_true, classes=range(len(actions)))
    ap_macro = average_precision_score(Yb, y_prob, average="macro")
    p, r, _ = precision_recall_curve(Yb.ravel(), y_prob.ravel())
    plt.figure(figsize=(7,5))
    plt.plot(r, p); plt.title(f"Precision-Recall (AP macro={ap_macro:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precisão"); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_macro.png")); plt.close()
    with open(os.path.join(outdir, "metrics_report.txt"), "a", encoding="utf-8") as f:
        f.write(f"\nAverage Precision (macro): {ap_macro:.3f}\n")

# ----------------- MAIN ------------------------
def main():
    X_raw, y, actions, meta = load_sequences(DATA_DIR)
    if len(X_raw) == 0: raise RuntimeError("Sem dados.")
    T0, F = infer_TF_from_first(X_raw)
    if USE_GROUP_SPLIT:
        tr_idx, te_idx = group_stratified_split(X_raw, y, meta, test_size=TEST_SIZE, seed=SEED)
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(X_raw)), test_size=TEST_SIZE, random_state=SEED, stratify=y)
    X_train_list = [X_raw[i] for i in tr_idx]
    X_test_list  = [X_raw[i] for i in te_idx]
    y_train = y[tr_idx]; y_test = y[te_idx]
    T_fixed = max(int(np.median([xi.shape[0] for xi in X_raw])), 16)
    X_tmp = np.stack([apply_feature_mode(pad_or_crop_to_T(xi, T_fixed), FEATURE_MODE) for xi in X_train_list], axis=0).astype(np.float32)
    mu = X_tmp.mean(axis=(0,1), keepdims=True); sd = X_tmp.std(axis=(0,1), keepdims=True) + 1e-8
    with open(os.path.join(RESULTS_DIR, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"mu": mu.squeeze().tolist(), "sd": sd.squeeze().tolist(), "feature_mode": FEATURE_MODE, "T": T_fixed, "F": int(X_tmp.shape[2])}, f, indent=2)
    ds_train = make_dataset(X_train_list, y_train, T_fixed, mu, sd, BATCH_SIZE, training=True,  feature_mode=FEATURE_MODE)
    ds_test  = make_dataset(X_test_list,  y_test,  T_fixed, mu, sd, BATCH_SIZE, training=False, feature_mode=FEATURE_MODE)
    steps_per_epoch = max(1, int(np.ceil(len(X_train_list) / BATCH_SIZE)))
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(INIT_LR, steps_per_epoch * max(50, EPOCHS // 2), alpha=0.15)
    model = build_model((T_fixed, X_tmp.shape[2]), len(actions), lr_schedule)
    # Callbacks recomendados com CosineDecay (sem ReduceLROnPlateau)
    cbs = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_OUT,
        monitor="val_categorical_accuracy",
        mode="max",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    ),
    tf.keras.callbacks.TerminateOnNaN()
    ]

    class_weight = None
    if USE_CLASS_WEIGHTS:
        cw = compute_class_weight(class_weight="balanced", classes=np.arange(len(actions)), y=y_train)
        class_weight = {i: float(w) for i, w in enumerate(cw)}
        print("[INFO] class_weight:", class_weight)
    hist = model.fit(ds_train, validation_data=ds_test, epochs=EPOCHS, callbacks=cbs, class_weight=class_weight)
    if SAVE_HISTORY_PNG: plot_training(hist, RESULTS_DIR)
    X_test_np, y_test_np = next(iter(ds_test.unbatch().batch(10_000)))
    eval_and_plots(model, X_test_np.numpy(), y_test_np.numpy(), actions, RESULTS_DIR)
    np.save(ACTIONS_OUT, actions)

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(f"[FATAL] {e}"); sys.exit(1)
