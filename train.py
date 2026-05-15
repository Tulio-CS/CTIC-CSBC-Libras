# train.py
# -*- coding: utf-8 -*-
"""
Script unificado de treinamento LIBRAS.

Exemplos de uso:
    py -3.11 train.py                                          # modelo padrão (bilstm_attn)
    py -3.11 train.py --model bilstm
    py -3.11 train.py --model lstm --data dataset
    py -3.11 train.py --model bilstm_attn --cosine            # usa CosineDecay
    py -3.11 train.py --model bilstm_attn --no_aug            # sem augmentação
    py -3.11 train.py --model bilstm_attn --absolute          # coordenadas absolutas
    py -3.11 train.py --model bilstm_attn --aug_types jitter rotation  # augmentações seletivas

Parâmetros de ablação (sobrescrevem config.py):
    --dropout FLOAT          taxa de dropout
    --rec_dropout FLOAT      taxa de dropout recorrente
    --label_smooth FLOAT     label smoothing
    --attn_heads INT         número de cabeças de atenção (bilstm_attn)
    --attn_key_dim INT       dimensão key_dim da atenção (bilstm_attn)
    --lstm_units INT [INT]   unidades por camada LSTM
    --dense_units INT [INT]  unidades por camada densa
    --n_layers INT           número de camadas BiLSTM (1, 2, 3)
    --no_layer_norm          desativa LayerNormalization
    --pooling {avg,max,last} estratégia de pooling (bilstm_attn)
    --no_residual            desativa conexão residual na atenção
    --data_fraction FLOAT    fração dos dados de treino a usar (0.0–1.0)
    --optimizer {adamw,adam,sgd}  otimizador
    --aug_types STR [STR]    tipos de augmentação (None = todas)
"""

import os, sys, json, argparse, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

import config as cfg
from data_utils import (
    load_sequences, make_split, compute_norm_stats,
    save_norm_stats, make_dataset, materialize_dataset, bootstrap_metrics
)
from models import build_model, make_cosine_schedule, make_callbacks
from evaluate import eval_and_save


# ─────────────────────────────────────────────────────────────────────────────
# Argumentos
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Treina modelo LIBRAS.")

    # ── Modelo e dados ────────────────────────────────────────────────────────
    p.add_argument("--model",    default=cfg.DEFAULT_MODEL_NAME,
                   choices=list(cfg.MODEL_CONFIGS),
                   help="Arquitetura: lstm | bilstm | bilstm_attn")
    p.add_argument("--data",     default=cfg.DATA_DIR,
                   help="Pasta com subpastas por classe (*.npy)")
    p.add_argument("--epochs",   type=int, default=cfg.EPOCHS)
    p.add_argument("--batch",    type=int, default=cfg.BATCH_SIZE)

    # ── Opções de treino ──────────────────────────────────────────────────────
    p.add_argument("--cosine",     action="store_true",
                   help="Usa CosineDecay em vez de ReduceLROnPlateau")
    p.add_argument("--no_aug",     action="store_true",
                   help="Desativa augmentações")
    p.add_argument("--absolute",   action="store_true",
                   help="Usa coordenadas absolutas em vez de wrist-centered")
    p.add_argument("--no_weights", action="store_true",
                   help="Não usa class weights")
    p.add_argument("--T",          type=int, default=None,
                   help="Comprimento fixo da sequência; None = auto-detectado")

    # ── Ablação: arquitetura ──────────────────────────────────────────────────
    p.add_argument("--dropout",     type=float, default=None,
                   help="Sobrescreve taxa de dropout (ex: 0.2, 0.4)")
    p.add_argument("--rec_dropout", type=float, default=None,
                   help="Sobrescreve taxa de dropout recorrente")
    p.add_argument("--attn_heads",  type=int,   default=None,
                   help="Número de cabeças de atenção (bilstm_attn)")
    p.add_argument("--attn_key_dim",type=int,   default=None,
                   help="Dimensão key_dim da atenção (bilstm_attn)")
    p.add_argument("--lstm_units",  type=int,   nargs="+", default=None,
                   help="Unidades por camada LSTM (ex: 128 64)")
    p.add_argument("--dense_units", type=int,   nargs="+", default=None,
                   help="Unidades por camada densa (ex: 192 128)")
    p.add_argument("--n_layers",    type=int,   default=None,
                   help="Número de camadas BiLSTM (1, 2 ou 3)")
    p.add_argument("--no_layer_norm", action="store_true",
                   help="Desativa LayerNormalization")
    p.add_argument("--pooling",     choices=["avg", "max", "last"], default=None,
                   help="Estratégia de pooling: avg | max | last")
    p.add_argument("--no_residual", action="store_true",
                   help="Desativa conexão residual na atenção")
    p.add_argument("--optimizer",   choices=["adamw", "adam", "sgd"], default=None,
                   help="Otimizador: adamw | adam | sgd")

    # ── Ablação: dados e treinamento ──────────────────────────────────────────
    p.add_argument("--label_smooth",   type=float, default=None,
                   help="Label smoothing (ex: 0.0, 0.05, 0.1)")
    p.add_argument("--data_fraction",  type=float, default=None,
                   help="Fração dos dados de treino a usar (0.0–1.0)")
    p.add_argument("--aug_types",
                   nargs="+",
                   choices=["jitter", "rotation", "scale", "temp_dropout", "time_mask"],
                   default=None,
                   help="Tipos de augmentação (padrão: todas). Ignorado com --no_aug")

    # ── Dados extras (para combinar datasets) ────────────────────────────────
    p.add_argument("--extra_data",  default=None,
                   help="Diretório adicional de dados a mesclar com --data. "
                        "Classes do extra_data que não existam em --data são ignoradas.")

    # ── Saída ─────────────────────────────────────────────────────────────────
    p.add_argument("--run_name",    default=None,
                   help="Nome do run para ablação (sobrescreve --model como nome da pasta)")
    p.add_argument("--results_dir", default=None,
                   help="Diretório de resultados (sobrescreve padrão)")
    return p.parse_args()


def _build_overrides(args):
    """Constrói dict de overrides para build_model() a partir dos args de ablação."""
    overrides = {}
    if args.dropout      is not None: overrides["dropout"]       = args.dropout
    if args.rec_dropout  is not None: overrides["rec_dropout"]   = args.rec_dropout
    if args.attn_heads   is not None: overrides["attn_heads"]    = args.attn_heads
    if args.attn_key_dim is not None: overrides["attn_key_dim"]  = args.attn_key_dim
    if args.lstm_units:               overrides["lstm_units"]    = args.lstm_units
    if args.dense_units:              overrides["dense_units"]   = args.dense_units
    if args.n_layers     is not None: overrides["n_layers"]      = args.n_layers
    if args.no_layer_norm:            overrides["no_layer_norm"] = True
    if args.pooling      is not None: overrides["pooling"]       = args.pooling
    if args.no_residual:              overrides["no_residual"]   = True
    if args.optimizer    is not None: overrides["optimizer_type"]= args.optimizer
    return overrides


# ─────────────────────────────────────────────────────────────────────────────
# Plots de treino
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(history, outdir):
    """Salva curvas de loss e acurácia (treino vs validação)."""
    for metric, title, fname in [
        ("loss",                 "Loss (Treino vs Validação)",     "loss.png"),
        ("categorical_accuracy", "Acurácia (Treino vs Validação)", "accuracy.png"),
    ]:
        val_key = f"val_{metric}"
        plt.figure(figsize=(10, 5))
        plt.plot(history.history[metric],  label="Treino")
        plt.plot(history.history[val_key], label="Validação")
        plt.title(title)
        plt.xlabel("Época")
        plt.ylabel(metric.replace("categorical_", "").capitalize())
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Ajustes de config via CLI ─────────────────────────────────────────────
    feature_mode = "absolute" if args.absolute else cfg.FEATURE_MODE
    use_weights  = cfg.USE_CLASS_WEIGHTS and not args.no_weights
    run_name     = args.run_name if args.run_name else args.model

    # Augmentação: --no_aug desativa tudo; --aug_types limita quais aplicar
    if args.no_aug:
        use_aug   = False
        aug_types = None
    elif args.aug_types:
        use_aug   = True
        aug_types = set(args.aug_types)
    else:
        use_aug   = True
        aug_types = None  # todas

    # Seeds
    np.random.seed(cfg.SEED); tf.random.set_seed(cfg.SEED); random.seed(cfg.SEED)

    # ── Diretórios de saída ───────────────────────────────────────────────────
    model_dir   = os.path.join(cfg.MODELS_DIR, run_name)
    results_dir = args.results_dir if args.results_dir else os.path.join(cfg.RESULTS_DIR, run_name)
    os.makedirs(model_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_path   = os.path.join(model_dir, "model.keras")
    norm_path    = os.path.join(model_dir, "norm_stats.json")
    actions_path = os.path.join(model_dir, "actions.npy")

    print(f"[INFO] Modelo       : {args.model}")
    print(f"[INFO] Run name     : {run_name}")
    print(f"[INFO] Dados        : {args.data}")
    print(f"[INFO] Saída        : {model_dir}")

    # ── Carregamento ──────────────────────────────────────────────────────────
    X_raw, y, actions, meta = load_sequences(args.data)
    print(f"[INFO] {len(X_raw)} sequências | {len(actions)} classes")

    # ── Mescla dataset extra (ex: MINDS + custom) ─────────────────────────────
    if args.extra_data and os.path.isdir(args.extra_data):
        X_extra, y_extra, actions_extra, meta_extra = load_sequences(args.extra_data)
        prim_map = {a: i for i, a in enumerate(actions)}
        ex_X, ex_y, ex_meta = [], [], []
        skipped = set()
        for xi, yi, mi in zip(X_extra, y_extra, meta_extra):
            cls = actions_extra[yi]
            if cls in prim_map:
                ex_X.append(xi)
                ex_y.append(prim_map[cls])
                ex_meta.append(mi)
            else:
                skipped.add(cls)
        if skipped:
            print(f"[WARN] extra_data: {len(skipped)} classes ignoradas "
                  f"(não estão em --data): {sorted(skipped)}")
        if ex_X:
            X_raw = np.concatenate([X_raw, np.array(ex_X, dtype=object)])
            y     = np.concatenate([y, np.array(ex_y, int)])
            meta  = meta + ex_meta
            print(f"[INFO] extra_data: +{len(ex_X)} sequências de '{args.extra_data}' "
                  f"(total: {len(X_raw)})")
    elif args.extra_data:
        print(f"[WARN] --extra_data '{args.extra_data}' não encontrado — ignorado.")

    # ── Split ─────────────────────────────────────────────────────────────────
    tr_idx, te_idx = make_split(
        X_raw, y, meta,
        test_size=cfg.TEST_SIZE,
        seed=cfg.SEED,
        use_group_split=cfg.USE_GROUP_SPLIT,
    )
    X_train = [X_raw[i] for i in tr_idx]
    X_test  = [X_raw[i] for i in te_idx]
    y_train = y[tr_idx]
    y_test  = y[te_idx]
    print(f"[INFO] Treino: {len(X_train)} | Teste: {len(X_test)}")

    # ── Subsampling de dados (ablação de fração) ──────────────────────────────
    if args.data_fraction is not None and 0.0 < args.data_fraction < 1.0:
        rng    = np.random.RandomState(cfg.SEED)
        n_keep = max(1, int(round(len(X_train) * args.data_fraction)))
        idx    = rng.choice(len(X_train), size=n_keep, replace=False)
        X_train = [X_train[i] for i in idx]
        y_train = y_train[idx]
        print(f"[INFO] data_fraction={args.data_fraction:.2f} → {len(X_train)} amostras de treino")

    # ── T e estatísticas de normalização (calculadas SÓ no treino) ───────────
    T_fixed = args.T if args.T else max(int(np.median([xi.shape[0] for xi in X_raw])), 16)
    mu, sd, F = compute_norm_stats(X_train, T_fixed, feature_mode)
    save_norm_stats(mu, sd, T_fixed, F, feature_mode, norm_path)
    np.save(actions_path, actions)
    print(f"[INFO] T={T_fixed} | F={F} | feature_mode={feature_mode}")
    if aug_types is not None:
        print(f"[INFO] aug_types={sorted(aug_types)}")

    # ── tf.data ───────────────────────────────────────────────────────────────
    ds_train = make_dataset(X_train, y_train, T_fixed, mu, sd,
                            args.batch, training=use_aug,
                            feature_mode=feature_mode, aug_types=aug_types)
    ds_test  = make_dataset(X_test,  y_test,  T_fixed, mu, sd,
                            args.batch, training=False,
                            feature_mode=feature_mode)

    # ── Modelo ────────────────────────────────────────────────────────────────
    steps_per_epoch = max(1, int(np.ceil(len(X_train) / args.batch)))
    use_cosine      = args.cosine
    lr_schedule     = None

    if use_cosine:
        lr_schedule = make_cosine_schedule(cfg.INIT_LR, steps_per_epoch, args.epochs)
        print("[INFO] Usando CosineDecay.")
    else:
        print("[INFO] Usando ReduceLROnPlateau.")

    overrides    = _build_overrides(args)
    label_smooth = args.label_smooth

    if overrides:
        print(f"[INFO] Overrides de modelo: {overrides}")
    if label_smooth is not None:
        print(f"[INFO] label_smooth={label_smooth}")

    model = build_model(
        args.model, (T_fixed, F), len(actions),
        lr_schedule=lr_schedule,
        overrides=overrides,
        label_smooth=label_smooth,
    )

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weight = None
    if use_weights:
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(actions)),
            y=y_train,
        )
        class_weight = {i: float(w) for i, w in enumerate(cw)}
        print(f"[INFO] Usando class_weights (min={min(cw):.3f}, max={max(cw):.3f})")

    # ── Treino ────────────────────────────────────────────────────────────────
    history_csv = os.path.join(results_dir, "training_history.csv")
    callbacks = make_callbacks(model_path, use_cosine=use_cosine,
                               history_csv=history_csv)
    history   = model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # ── Plots de treino ───────────────────────────────────────────────────────
    plot_training(history, results_dir)

    # ── Avaliação ─────────────────────────────────────────────────────────────
    X_test_np, y_test_np = materialize_dataset(ds_test)
    eval_and_save(model, X_test_np, y_test_np, actions, results_dir,
                  n_bootstrap=cfg.N_BOOTSTRAP, seed=cfg.SEED)

    print(f"\n[OK] Modelo salvo em     : {model_path}")
    print(f"[OK] norm_stats salvo em : {norm_path}")
    print(f"[OK] Resultados em       : {results_dir}")
    if run_name == args.model:
        print(f"\nPara inferência ao vivo, rode:")
        print(f"  python infer_live.py --model {args.model}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
