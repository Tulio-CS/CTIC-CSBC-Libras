# train.py
# -*- coding: utf-8 -*-
"""
Script unificado de treinamento LIBRAS.

Exemplos de uso:
    py -3.11 train.py                                # modelo padrão (bilstm_attn)
    py -3.11 train.py --model bilstm
    py -3.11 train.py --model lstm --data dataset
    py -3.11 train.py --model bilstm_attn --cosine   # usa CosineDecay
    py -3.11 train.py --model bilstm_attn --no_aug   # sem augmentação (ablation)
    py -3.11 train.py --model bilstm_attn --absolute # wrist_centered desligado
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
    p.add_argument("--model",    default=cfg.DEFAULT_MODEL_NAME,
                   choices=list(cfg.MODEL_CONFIGS),
                   help="Arquitetura: lstm | bilstm | bilstm_attn")
    p.add_argument("--data",     default=cfg.DATA_DIR,
                   help="Pasta com subpastas por classe (*.npy)")
    p.add_argument("--epochs",   type=int, default=cfg.EPOCHS)
    p.add_argument("--batch",    type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--cosine",   action="store_true",
                   help="Usa CosineDecay em vez de ReduceLROnPlateau")
    p.add_argument("--no_aug",   action="store_true",
                   help="Desativa augmentações (ablation study)")
    p.add_argument("--absolute", action="store_true",
                   help="Usa coordenadas absolutas em vez de wrist-centered")
    p.add_argument("--no_weights", action="store_true",
                   help="Não usa class weights")
    p.add_argument("--T", type=int, default=None,
                   help="Comprimento fixo da sequência (ablation); None = auto-detectado")
    p.add_argument("--run_name", default=None,
                   help="Nome do run para ablação (sobrescreve --model como nome da pasta)")
    p.add_argument("--results_dir", default=None,
                   help="Diretório de resultados (sobrescreve padrão)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Plots de treino
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(history, outdir):
    """Salva curvas de loss e acurácia (treino vs validação)."""
    for metric, title, fname in [
        ("loss",                  "Loss (Treino vs Validação)",    "loss.png"),
        ("categorical_accuracy",  "Acurácia (Treino vs Validação)", "accuracy.png"),
    ]:
        val_key = f"val_{metric}"
        plt.figure(figsize=(10, 5))
        plt.plot(history.history[metric],     label="Treino")
        plt.plot(history.history[val_key],    label="Validação")
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

    # Ajustes de config via CLI
    feature_mode = "absolute" if args.absolute else cfg.FEATURE_MODE
    use_aug      = not args.no_aug
    use_weights  = cfg.USE_CLASS_WEIGHTS and not args.no_weights

    # run_name permite ablação com subpastas separadas por configuração
    run_name = args.run_name if args.run_name else args.model

    # Seeds
    np.random.seed(cfg.SEED); tf.random.set_seed(cfg.SEED); random.seed(cfg.SEED)

    # Diretórios de saída
    model_dir   = os.path.join(cfg.MODELS_DIR, run_name)
    results_dir = args.results_dir if args.results_dir else os.path.join(cfg.RESULTS_DIR, run_name)
    os.makedirs(model_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_path   = os.path.join(model_dir, "model.keras")
    norm_path    = os.path.join(model_dir, "norm_stats.json")
    actions_path = os.path.join(model_dir, "actions.npy")

    print(f"[INFO] Modelo   : {args.model}")
    print(f"[INFO] Run name : {run_name}")
    print(f"[INFO] Dados    : {args.data}")
    print(f"[INFO] Saída    : {model_dir}")

    # ── Carregamento ──────────────────────────────────────────────────────────
    X_raw, y, actions, meta = load_sequences(args.data)
    print(f"[INFO] {len(X_raw)} sequências | {len(actions)} classes")

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

    # ── T e estatísticas de normalização (calculadas SÓ no treino) ────────────
    T_fixed = args.T if args.T else max(int(np.median([xi.shape[0] for xi in X_raw])), 16)
    mu, sd, F = compute_norm_stats(X_train, T_fixed, feature_mode)
    save_norm_stats(mu, sd, T_fixed, F, feature_mode, norm_path)
    np.save(actions_path, actions)
    print(f"[INFO] T={T_fixed} | F={F} | feature_mode={feature_mode}")

    # ── tf.data ───────────────────────────────────────────────────────────────
    ds_train = make_dataset(X_train, y_train, T_fixed, mu, sd,
                            args.batch, training=use_aug, feature_mode=feature_mode)
    ds_test  = make_dataset(X_test,  y_test,  T_fixed, mu, sd,
                            args.batch, training=False,  feature_mode=feature_mode)

    # ── Modelo ────────────────────────────────────────────────────────────────
    steps_per_epoch = max(1, int(np.ceil(len(X_train) / args.batch)))
    lr_schedule     = None
    use_cosine      = args.cosine

    if use_cosine:
        lr_schedule = make_cosine_schedule(cfg.INIT_LR, steps_per_epoch, args.epochs)
        print("[INFO] Usando CosineDecay.")
    else:
        print("[INFO] Usando ReduceLROnPlateau.")

    model = build_model(args.model, (T_fixed, F), len(actions), lr_schedule)

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
    callbacks = make_callbacks(model_path, use_cosine=use_cosine)
    history = model.fit(
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
