# ablation.py
# -*- coding: utf-8 -*-
"""
Estudo de ablação sistemático para o modelo LIBRAS.

Treina múltiplas configurações e gera tabela comparativa + gráficos.

Dimensões avaliadas
-------------------
  1. Arquitetura   : lstm | bilstm | bilstm_attn
  2. Augmentação   : com | sem
  3. Feature mode  : wrist_centered | absolute
  4. Seq. length T : 8 | 16 | 32

Uso:
    python ablation.py                       # todas as dimensões (lento)
    python ablation.py --suite arch          # só arquiteturas
    python ablation.py --suite aug           # só augmentação
    python ablation.py --suite feat          # só feature mode
    python ablation.py --suite seq           # só seq length
    python ablation.py --epochs 60           # épocas por run (padrão 80)
    python ablation.py --skip_existing       # não retreina se resultado existe
    python ablation.py --data dataset        # pasta de dados
"""

import os, sys, json, argparse, random, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Definição das suítes de ablação
# ─────────────────────────────────────────────────────────────────────────────

SUITES = {
    "arch": [
        {"name": "lstm",         "args": ["--model", "lstm"]},
        {"name": "bilstm",       "args": ["--model", "bilstm"]},
        {"name": "bilstm_attn",  "args": ["--model", "bilstm_attn"]},
    ],
    "aug": [
        {"name": "bilstm_attn_aug",    "args": ["--model", "bilstm_attn"]},
        {"name": "bilstm_attn_no_aug", "args": ["--model", "bilstm_attn", "--no_aug"]},
    ],
    "feat": [
        {"name": "bilstm_attn_wrist",    "args": ["--model", "bilstm_attn"]},
        {"name": "bilstm_attn_absolute", "args": ["--model", "bilstm_attn", "--absolute"]},
    ],
    "seq": [
        {"name": "bilstm_attn_T8",  "args": ["--model", "bilstm_attn", "--T",  "8"]},
        {"name": "bilstm_attn_T16", "args": ["--model", "bilstm_attn", "--T", "16"]},
        {"name": "bilstm_attn_T32", "args": ["--model", "bilstm_attn", "--T", "32"]},
    ],
}

SUITE_LABELS = {
    "arch": "Arquitetura",
    "aug":  "Augmentação de dados",
    "feat": "Modo de feature",
    "seq":  "Comprimento da sequência (T)",
}

DISPLAY_NAMES = {
    "lstm":                  "LSTM",
    "bilstm":                "BiLSTM",
    "bilstm_attn":           "BiLSTM + Atenção",
    "bilstm_attn_aug":       "Com augmentação",
    "bilstm_attn_no_aug":    "Sem augmentação",
    "bilstm_attn_wrist":     "Wrist-centered",
    "bilstm_attn_absolute":  "Coordenadas absolutas",
    "bilstm_attn_T8":        "T=8 frames",
    "bilstm_attn_T16":       "T=16 frames",
    "bilstm_attn_T32":       "T=32 frames",
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
# Execução de um run
# ─────────────────────────────────────────────────────────────────────────────

def run_config(config, data_dir, epochs, results_base, skip_existing):
    """Executa train.py para uma configuração de ablação."""
    run_name = config["name"]
    result_path = os.path.join(results_base, run_name, "metrics_bootstrap.json")

    if skip_existing and os.path.exists(result_path):
        print(f"\n[SKIP] {run_name} — resultado já existe.")
        return read_metrics(run_name, results_base)

    # Salva modelo e resultados em subpastas separadas por run
    env_overrides = {
        "ABLATION_RUN_NAME": run_name,
        "ABLATION_RESULTS_BASE": results_base,
    }

    cmd = [
        sys.executable, "train.py",
        "--data", data_dir,
        "--epochs", str(epochs),
        "--run_name", run_name,
        "--results_dir", os.path.join(results_base, run_name),
    ] + config["args"]

    print(f"\n{'='*60}")
    print(f"[RUN] {run_name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env.update(env_overrides)

    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        print(f"[WARN] Run '{run_name}' terminou com código {ret.returncode}.")

    return read_metrics(run_name, results_base)


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
    Gráfico de barras com 4 métricas (accuracy, precision, recall, F1)
    para cada run da suíte, com barras de erro (±std bootstrap).
    """
    labels = [DISPLAY_NAMES.get(n, n) for n in run_names]
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Acurácia", "Precisão (macro)", "Recall (macro)", "F1-macro"]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    n_runs   = len(run_names)
    n_metrics = len(metric_keys)
    x = np.arange(n_runs)
    bar_w = 0.18

    fig, ax = plt.subplots(figsize=(max(8, n_runs * 1.8), 6))

    for mi, (mk, ml, color) in enumerate(zip(metric_keys, metric_labels, colors)):
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
        bars = ax.bar(x + offset, means, bar_w,
                      yerr=stds, capsize=4,
                      label=ml, color=color, alpha=0.85)
        # anotar valor
        for rect, mean in zip(bars, means):
            if mean > 0:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.005,
                        f"{mean:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Valor da métrica")
    ax.set_title(f"Ablação — {SUITE_LABELS.get(suite_name, suite_name)}")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fname = os.path.join(outdir, f"ablation_{suite_name}.png")
    _savefig(fname)


def save_table(all_results, outdir):
    """Salva tabela CSV e TXT com todos os resultados."""
    rows = []
    for suite_name, suite_runs in all_results.items():
        for run_name, metrics in suite_runs.items():
            if metrics is None:
                continue
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
            })

    if not rows:
        print("[WARN] Nenhum resultado para tabular.")
        return

    # CSV
    import csv
    csv_path = os.path.join(outdir, "ablation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] CSV: {csv_path}")

    # TXT legível
    txt_path = os.path.join(outdir, "ablation_results.txt")
    col_w = [("Suíte", 8), ("Configuração", 28),
             ("Acc ± std", 15), ("Prec ± std", 15),
             ("Rec ± std", 15), ("F1 ± std", 15)]
    header = "  ".join(f"{h:<{w}}" for h, w in col_w)
    sep    = "-" * len(header)

    lines = [sep, header, sep]
    prev_suite = None
    for r in rows:
        if r["suite"] != prev_suite:
            if prev_suite is not None:
                lines.append("")
            lines.append(f"  [{SUITE_LABELS.get(r['suite'], r['suite'])}]")
            prev_suite = r["suite"]

        acc_s  = f"{r['accuracy']} ± {r['acc_std']}"
        prec_s = f"{r['precision']} ± {r['prec_std']}"
        rec_s  = f"{r['recall']} ± {r['rec_std']}"
        f1_s   = f"{r['f1']} ± {r['f1_std']}"

        row_str = (f"  {r['suite']:<8}  {r['label']:<28}  "
                   f"{acc_s:<15}  {prec_s:<15}  {rec_s:<15}  {f1_s:<15}")
        lines.append(row_str)

    lines.append(sep)
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"[OK] TXT: {txt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Estudo de ablação LIBRAS.")
    p.add_argument("--suite", choices=list(SUITES) + ["all"], default="all",
                   help="Suíte a executar (padrão: all)")
    p.add_argument("--epochs", type=int, default=80,
                   help="Épocas por run (padrão: 80)")
    p.add_argument("--data",   default="dataset",
                   help="Pasta de dados (padrão: dataset)")
    p.add_argument("--out",    default="results_ablation",
                   help="Diretório de saída (padrão: results_ablation)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Pula runs cujo resultado já existe")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    suites_to_run = list(SUITES) if args.suite == "all" else [args.suite]
    all_results   = {}

    for suite_name in suites_to_run:
        print(f"\n{'#'*60}")
        print(f"# SUÍTE: {SUITE_LABELS.get(suite_name, suite_name)}")
        print(f"{'#'*60}")

        suite_results = {}
        run_names     = []

        for config in SUITES[suite_name]:
            run_name = config["name"]
            run_names.append(run_name)
            metrics = run_config(config, args.data, args.epochs,
                                 args.out, args.skip_existing)
            suite_results[run_name] = metrics
            if metrics:
                acc = metrics["accuracy"]
                f1  = metrics["f1"]
                print(f"  → {DISPLAY_NAMES.get(run_name, run_name):<30} "
                      f"acc={acc[0]:.4f}±{acc[1]:.4f}  f1={f1[0]:.4f}±{f1[1]:.4f}")

        all_results[suite_name] = suite_results
        plot_suite(suite_name, run_names, suite_results, args.out)

    save_table(all_results, args.out)
    print(f"\n[OK] Ablação concluída. Resultados em: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
