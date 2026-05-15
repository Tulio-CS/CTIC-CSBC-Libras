# compute_efficiency.py
# -*- coding: utf-8 -*-
"""
Mede eficiência computacional dos modelos LIBRAS:
  - Número de parâmetros (treináveis e totais)
  - Tamanho no disco (.keras) e teórico em FP32
  - GFLOPs por sequência (forward pass batch=1)
  - Latência CPU: só inferência, pipeline completo (web), com warm-up e percentis
  - Simulação do pipeline da aplicação web (Predictor.step)

Uso:
    py -3.11 compute_efficiency.py
    py -3.11 compute_efficiency.py --models lstm bilstm_attn_best
    py -3.11 compute_efficiency.py --runs 500
"""

import os, sys, json, time, argparse, math
import numpy as np
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from data_utils import load_norm_stats, apply_feature_mode

# Suprime logs de compilação do TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

OUT_DIR = "results_paper"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Modelos a medir (padrão)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    "lstm",
    "bilstm_attn",
    "bilstm_attn_best",
    "bilstm_attn_minds",
    "bilstm_attn_combined",
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

def sizeof_fmt(num: float, suffix: str = "B") -> str:
    for unit in ("", "K", "M", "G"):
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f} T{suffix}"


def percentile(data: list, p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] * (hi - k) + s[hi] * (k - lo)


def load_model_safe(path: str):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e1:
        try:
            import keras
            keras.config.enable_unsafe_deserialization()
            from keras.saving import load_model
            return load_model(path, compile=False)
        except Exception as e2:
            raise RuntimeError(f"Falha ao carregar {path}: {e1} / {e2}")


# ─────────────────────────────────────────────────────────────────────────────
# Contagem de FLOPs
# ─────────────────────────────────────────────────────────────────────────────

def count_gflops(model, T: int, F: int) -> float:
    """
    Conta FLOPs de uma forward pass (batch=1) usando o profiler do TF.
    Retorna GFLOPs (bilhões de operações de ponto flutuante).
    O profiler conta multiply-adds como 2 FLOPs cada.
    Retorna float('nan') se não conseguir computar.
    """
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2,
        )

        @tf.function
        def _forward(x):
            return model(x, training=False)

        dummy = tf.constant(np.zeros((1, T, F), dtype=np.float32))
        concrete = _forward.get_concrete_function(dummy)
        frozen   = convert_variables_to_constants_v2(concrete)

        with tf.Graph().as_default() as g:
            tf.graph_util.import_graph_def(frozen.graph.as_graph_def(), name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            prof = tf.compat.v1.profiler.profile(
                g, run_meta=run_meta, cmd="op", options=opts
            )
        return prof.total_float_ops / 1e9

    except Exception as e:
        print(f"    [AVISO] Profiler FLOP falhou ({e}), usando estimativa analítica.")
        return _estimate_gflops_analytical(model, T, F)


def _estimate_gflops_analytical(model, T: int, F: int) -> float:
    """
    Estimativa analítica de GFLOPs baseada nas camadas Keras.
    Multiply-adds contados como 2 operações cada.
    """
    total = 0.0
    prev_shape = (T, F)

    for layer in model.layers:
        cfg_l = layer.get_config()
        name  = layer.__class__.__name__

        if name in ("LSTM", "GRU"):
            units  = cfg_l.get("units", 0)
            in_dim = prev_shape[-1]
            # 4 gates × (input_proj + recurrent_proj) × 2 (mul-add)
            total += 4 * (in_dim * units + units * units) * T * 2
            prev_shape = (T, units)

        elif name == "Bidirectional":
            inner = layer.forward_layer
            units = inner.get_config().get("units", 0)
            in_dim = prev_shape[-1]
            # ×2 for both directions
            total += 2 * 4 * (in_dim * units + units * units) * T * 2
            prev_shape = (T, units * 2)

        elif name == "MultiHeadAttention":
            heads   = cfg_l.get("num_heads", 1)
            key_dim = cfg_l.get("key_dim", 32)
            val_dim = cfg_l.get("value_dim") or key_dim
            embed   = prev_shape[-1]
            seq     = prev_shape[0]
            # Q/K/V projections
            total += 3 * 2 * seq * embed * heads * key_dim
            # Attention scores and weighted sum
            total += 2 * seq * seq * heads * key_dim
            total += 2 * seq * seq * heads * val_dim
            # Output projection
            total += 2 * seq * heads * val_dim * embed
            prev_shape = (seq, embed)

        elif name == "Dense":
            units  = cfg_l.get("units", 0)
            in_dim = prev_shape[-1]
            total += 2 * in_dim * units
            prev_shape = (units,)

        elif name in ("GlobalAveragePooling1D", "GlobalMaxPooling1D"):
            prev_shape = (prev_shape[-1],)

        elif name == "LayerNormalization":
            # ~4 ops per element (mean, variance, normalize, scale+shift)
            total += 4 * math.prod(prev_shape)

    return total / 1e9


# ─────────────────────────────────────────────────────────────────────────────
# Medição de latência de inferência (só model call)
# ─────────────────────────────────────────────────────────────────────────────

def measure_inference_latency(model, T: int, F: int,
                               n_warmup: int = 20, n_runs: int = 200) -> dict:
    """
    Latência da chamada direta ao modelo (batch=1).
    Usa model(x, training=False) — igual ao predictor.py da web.
    """
    dummy = np.zeros((1, T, F), dtype=np.float32)
    x_tf  = tf.constant(dummy)

    # Warm-up (compila o graph TF)
    for _ in range(n_warmup):
        _ = model(x_tf, training=False)

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model(x_tf, training=False).numpy()
        times_ms.append((time.perf_counter() - t0) * 1e3)

    return {
        "mean_ms":  st.mean(times_ms),
        "std_ms":   st.stdev(times_ms),
        "p50_ms":   percentile(times_ms, 50),
        "p95_ms":   percentile(times_ms, 95),
        "p99_ms":   percentile(times_ms, 99),
        "min_ms":   min(times_ms),
        "max_ms":   max(times_ms),
        "n_runs":   n_runs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Medição do pipeline completo da aplicação web
# ─────────────────────────────────────────────────────────────────────────────

def measure_web_pipeline(model, mu: np.ndarray, sd: np.ndarray,
                          T: int, F: int, feature_mode: str,
                          n_warmup: int = 20, n_runs: int = 200) -> dict:
    """
    Simula exatamente o que Predictor.step() faz por frame (buffer cheio):
      1. np.array(raw_landmarks)     ← chegada pelo WebSocket
      2. apply_feature_mode          ← transformação wrist-centered
      3. buffer.append + stack       ← montagem da janela
      4. z-score normalization
      5. model(Xn, training=False)   ← inferência
      6. .numpy()[0]                 ← extração do resultado
      7. EMA update                  ← suavização
    """
    # Simula buffer cheio (T frames de landmarks aleatórios)
    rng    = np.random.RandomState(0)
    buffer = [rng.randn(F).astype(np.float32) for _ in range(T)]
    raw_lm = rng.randn(F).tolist()
    ema    = np.zeros(model.output_shape[-1], dtype=np.float32)
    alpha  = cfg.EMA_ALPHA

    def _step():
        # 1. Parse landmarks
        feat = np.array(raw_lm, dtype=np.float32)
        feat = feat[:F]

        # 2. Feature mode (wrist-centered)
        feat_2d = feat.reshape(1, F)
        feat_2d = apply_feature_mode(feat_2d, feature_mode)
        feat    = feat_2d.reshape(-1)

        # 3. Sliding window (simula buffer cheio: descarta 1º, adiciona novo)
        buf = buffer[1:] + [feat]
        X   = np.stack(buf, axis=0)

        # 4. Z-score
        Xn = (X - mu) / (sd + 1e-8)
        Xn = Xn[np.newaxis, ...].astype(np.float32)

        # 5-6. Inferência
        prob = model(Xn, training=False).numpy()[0]

        # 7. EMA
        ema_new = alpha * prob + (1 - alpha) * ema
        return ema_new

    # Warm-up
    for _ in range(n_warmup):
        _step()

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _step()
        times_ms.append((time.perf_counter() - t0) * 1e3)

    # Desagregação por sub-etapa (em uma run separada com timing interno)
    breakdown = _measure_breakdown(model, buffer, raw_lm, mu, sd, F, feature_mode, alpha, ema)

    return {
        "mean_ms":  st.mean(times_ms),
        "std_ms":   st.stdev(times_ms),
        "p50_ms":   percentile(times_ms, 50),
        "p95_ms":   percentile(times_ms, 95),
        "p99_ms":   percentile(times_ms, 99),
        "n_runs":   n_runs,
        "breakdown": breakdown,
    }


def _measure_breakdown(model, buffer, raw_lm, mu, sd, F, feature_mode, alpha, ema):
    """Mede cada sub-etapa individualmente (50 repetições cada)."""
    N = 50
    t_feat, t_norm, t_infer, t_ema = [], [], [], []

    for _ in range(N):
        # Features
        t0  = time.perf_counter()
        feat = np.array(raw_lm, dtype=np.float32)[:F]
        feat = apply_feature_mode(feat.reshape(1, F), feature_mode).reshape(-1)
        t_feat.append((time.perf_counter() - t0) * 1e3)

        # Stack + normalization
        buf = buffer[1:] + [feat]
        t0  = time.perf_counter()
        X   = np.stack(buf, axis=0)
        Xn  = ((X - mu) / (sd + 1e-8))[np.newaxis, ...].astype(np.float32)
        t_norm.append((time.perf_counter() - t0) * 1e3)

        # Inferência
        t0   = time.perf_counter()
        prob = model(Xn, training=False).numpy()[0]
        t_infer.append((time.perf_counter() - t0) * 1e3)

        # EMA
        t0  = time.perf_counter()
        _   = alpha * prob + (1 - alpha) * ema
        t_ema.append((time.perf_counter() - t0) * 1e3)

    def s(lst): return {"mean_ms": st.mean(lst), "p95_ms": percentile(lst, 95)}
    return {
        "feature_transform": s(t_feat),
        "normalization":     s(t_norm),
        "model_inference":   s(t_infer),
        "ema_update":        s(t_ema),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark de um modelo
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_model(model_name: str, n_runs: int) -> dict | None:
    model_dir  = os.path.join(cfg.MODELS_DIR, model_name)
    model_path = os.path.join(model_dir, "model.keras")
    norm_path  = os.path.join(model_dir, "norm_stats.json")

    if not os.path.exists(model_path):
        print(f"  [SKIP] {model_name} — model.keras não encontrado.")
        return None

    print(f"\n{'─'*55}")
    print(f"  Modelo: {model_name}")
    print(f"{'─'*55}")

    # Carrega modelo e normstat
    t_load = time.perf_counter()
    model  = load_model_safe(model_path)
    t_load = (time.perf_counter() - t_load) * 1e3
    mu, sd, T, F, feature_mode = load_norm_stats(norm_path)
    mu2, sd2 = mu.reshape(-1), sd.reshape(-1)

    # ── Parâmetros ──────────────────────────────────────────────────────────
    total_params     = model.count_params()
    trainable_params = sum(
        int(tf.size(w).numpy()) for w in model.trainable_weights
    )
    nontrainable_params = total_params - trainable_params

    # ── Tamanho ─────────────────────────────────────────────────────────────
    disk_bytes     = os.path.getsize(model_path)
    fp32_bytes_th  = total_params * 4          # teórico: todos FP32

    print(f"  Parâmetros totais   : {total_params:,}")
    print(f"  Parâmetros treináv. : {trainable_params:,}")
    print(f"  Tamanho no disco    : {sizeof_fmt(disk_bytes)}")
    print(f"  Teórico FP32        : {sizeof_fmt(fp32_bytes_th)}")
    print(f"  T={T} | F={F} | feature_mode={feature_mode}")

    # ── GFLOPs ──────────────────────────────────────────────────────────────
    print("  Calculando GFLOPs ...")
    gflops = count_gflops(model, T, F)
    print(f"  GFLOPs / seq        : {gflops:.4f}")

    # ── Latência de inferência (só model call) ───────────────────────────────
    print(f"  Medindo latência de inferência ({n_runs} runs, warm-up=20) ...")
    lat_infer = measure_inference_latency(model, T, F,
                                          n_warmup=20, n_runs=n_runs)
    print(f"  Inferência CPU — mean: {lat_infer['mean_ms']:.2f} ms | "
          f"p50: {lat_infer['p50_ms']:.2f} | "
          f"p95: {lat_infer['p95_ms']:.2f} | "
          f"p99: {lat_infer['p99_ms']:.2f}")

    # ── Pipeline web completo ────────────────────────────────────────────────
    print(f"  Medindo pipeline web ({n_runs} runs, warm-up=20) ...")
    lat_web = measure_web_pipeline(model, mu2, sd2, T, F, feature_mode,
                                    n_warmup=20, n_runs=n_runs)
    print(f"  Pipeline web  — mean: {lat_web['mean_ms']:.2f} ms | "
          f"p50: {lat_web['p50_ms']:.2f} | "
          f"p95: {lat_web['p95_ms']:.2f}")

    bd = lat_web["breakdown"]
    print("  Detalhamento do pipeline:")
    for step, label in [("feature_transform", "  Feature transform"),
                         ("normalization",     "  Normalização     "),
                         ("model_inference",   "  Inferência modelo"),
                         ("ema_update",        "  EMA update       ")]:
        m = bd[step]["mean_ms"]
        p = bd[step]["p95_ms"]
        print(f"    {label}: mean={m:.3f} ms | p95={p:.3f} ms")

    return {
        "model_name":          model_name,
        "T":                   T,
        "F":                   F,
        "feature_mode":        feature_mode,
        "params_total":        total_params,
        "params_trainable":    trainable_params,
        "params_nontrainable": nontrainable_params,
        "disk_size_bytes":     disk_bytes,
        "disk_size_mb":        disk_bytes / 1e6,
        "fp32_size_mb":        fp32_bytes_th / 1e6,
        "gflops":              gflops,
        "load_time_ms":        t_load,
        "inference_latency":   lat_infer,
        "web_pipeline":        lat_web,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Impressão da tabela comparativa
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: list[dict]):
    if not results:
        return

    hdr = (f"{'Modelo':<26} {'Params':>10} {'FP32':>7} "
           f"{'GFLOPs':>8} {'Inf.mean':>9} {'Inf.p95':>8} "
           f"{'Web.mean':>9} {'Web.p95':>8}")
    sep = "─" * len(hdr)

    print(f"\n{'═'*len(hdr)}")
    print("  TABELA COMPARATIVA DE EFICIÊNCIA COMPUTACIONAL")
    print(f"{'═'*len(hdr)}")
    print(hdr)
    print(sep)
    for r in results:
        inf = r["inference_latency"]
        web = r["web_pipeline"]
        print(
            f"{r['model_name']:<26} "
            f"{r['params_total']:>10,} "
            f"{r['fp32_size_mb']:>6.1f}M "
            f"{r['gflops']:>8.4f} "
            f"{inf['mean_ms']:>8.2f}ms "
            f"{inf['p95_ms']:>7.2f}ms "
            f"{web['mean_ms']:>8.2f}ms "
            f"{web['p95_ms']:>7.2f}ms"
        )
    print(f"{'═'*len(hdr)}")
    print("  Inf. = latência da chamada ao modelo (batch=1, CPU)")
    print("  Web  = pipeline completo (transform → norm → model → EMA)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Salvamento dos resultados
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: list[dict]):
    # JSON completo
    json_path = os.path.join(OUT_DIR, "efficiency_results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"  -> {json_path}")

    # CSV resumido
    import csv as csvmod
    csv_path = os.path.join(OUT_DIR, "efficiency_results.csv")
    fields = ["model_name", "params_total", "fp32_size_mb", "gflops",
              "inf_mean_ms", "inf_p50_ms", "inf_p95_ms", "inf_p99_ms",
              "web_mean_ms", "web_p50_ms", "web_p95_ms"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csvmod.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            inf = r["inference_latency"]
            web = r["web_pipeline"]
            w.writerow({
                "model_name":   r["model_name"],
                "params_total": r["params_total"],
                "fp32_size_mb": f"{r['fp32_size_mb']:.2f}",
                "gflops":       f"{r['gflops']:.4f}",
                "inf_mean_ms":  f"{inf['mean_ms']:.3f}",
                "inf_p50_ms":   f"{inf['p50_ms']:.3f}",
                "inf_p95_ms":   f"{inf['p95_ms']:.3f}",
                "inf_p99_ms":   f"{inf['p99_ms']:.3f}",
                "web_mean_ms":  f"{web['mean_ms']:.3f}",
                "web_p50_ms":   f"{web['p50_ms']:.3f}",
                "web_p95_ms":   f"{web['p95_ms']:.3f}",
            })
    print(f"  -> {csv_path}")

    # Tabela Markdown para colar no RESULTS.md
    md_path = os.path.join(OUT_DIR, "efficiency_table.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("## Eficiência Computacional\n\n")
        fh.write("Medido em CPU (sem GPU), batch=1, 200 runs com 20 warm-up.\n\n")
        fh.write("| Modelo | Parâmetros | Tamanho FP32 | GFLOPs/seq "
                 "| Latência média | Latência p95 | Pipeline web |\n")
        fh.write("|--------|-----------|-------------|-----------|"
                 "--------------|-------------|-------------|\n")
        for r in results:
            inf = r["inference_latency"]
            web = r["web_pipeline"]
            fh.write(
                f"| `{r['model_name']}` "
                f"| {r['params_total']:,} "
                f"| {r['fp32_size_mb']:.1f} MB "
                f"| {r['gflops']:.4f} "
                f"| {inf['mean_ms']:.2f} ms "
                f"| {inf['p95_ms']:.2f} ms "
                f"| {web['mean_ms']:.2f} ms |\n"
            )
        fh.write("\n_Pipeline web = feature transform + normalização + modelo + EMA._\n")
    print(f"  -> {md_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark de eficiência computacional.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="Modelos a medir (nomes das pastas em models/)")
    p.add_argument("--runs",   type=int, default=200,
                   help="Número de runs para latência (padrão: 200)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  BENCHMARK DE EFICIÊNCIA COMPUTACIONAL")
    print(f"  TF {tf.__version__} | dispositivos: {tf.config.list_physical_devices()}")
    print("=" * 60)

    all_results = []
    for model_name in args.models:
        r = benchmark_model(model_name, n_runs=args.runs)
        if r is not None:
            all_results.append(r)

    if all_results:
        print_table(all_results)
        print("[Salvando resultados ...]")
        save_results(all_results)
    else:
        print("[AVISO] Nenhum modelo encontrado.")

    print("Concluído.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
