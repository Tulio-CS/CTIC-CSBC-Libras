# benchmark_libras.py
# pip install tensorflow numpy

import os, time, statistics, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------- Config --------
MODEL_PATH      = "best_model.keras"   # ajuste
SEQ_LEN         = 16                               # janela típica
N_FEATURES      = 126                              # 2 mãos * 21 * 3
WARMUP          = 20
RUNS            = 100

# Se quiser estimar FLOPs de LSTM/BiLSTM, descreva sua pilha aqui:
# Exemplo: duas camadas BiLSTM de 128 + densa final
ARCH = {
    "type": "bilstm",   # "lstm" ou "bilstm" (para estimativa de FLOPs)
    "layers": [
        {"units":128, "input_dim":N_FEATURES},  # primeira camada
        {"units":128, "input_dim":128}          # segunda camada (usa out da anterior)
    ]
}
# ------------------------

def bytes_per_dtype(dtype):
    return {"fp32":4, "fp16":2, "int8":1}[dtype]

def size_mb(n_params, dtype="fp32"):
    return n_params * bytes_per_dtype(dtype) / 1e6

def lstm_flops_per_step(units, input_dim):
    # 4 gates, cada gate ~ (input*units + units*units + bias) * 2 (mul+add)
    # Regra prática consolidada ≈ 8 * units * (input_dim + units + 1)
    return 8 * units * (input_dim + units + 1)

def estimate_sequence_flops(arch, seq_len):
    if arch is None: 
        return None
    total = 0
    if arch["type"] == "lstm":
        for layer in arch["layers"]:
            total += lstm_flops_per_step(layer["units"], layer["input_dim"]) * seq_len
    elif arch["type"] == "bilstm":
        for layer in arch["layers"]:
            total += 2 * lstm_flops_per_step(layer["units"], layer["input_dim"]) * seq_len
    return total  # FLOPs por sequência (aprox)

def main():
    print(f"Carregando modelo: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
    model.summary()

    n_params = model.count_params()
    print(f"\nParâmetros: {n_params:,}")
    print(f"Tamanho estimado:")
    print(f"  FP32: ~{size_mb(n_params,'fp32'):.2f} MB")
    print(f"  FP16: ~{size_mb(n_params,'fp16'):.2f} MB")
    print(f"  INT8: ~{size_mb(n_params,'int8'):.2f} MB (após quantização)")

    # FLOPs (estimativa p/ LSTM/BiLSTM)
    seq_flops = estimate_sequence_flops(ARCH, SEQ_LEN)
    if seq_flops:
        gflops = seq_flops / 1e9
        print(f"\nFLOPs (aprox) por sequência [{SEQ_LEN}×{N_FEATURES}]: ~{gflops:.3f} GFLOPs")

    # Latência (TF/Keras, FP32)
    dummy = np.random.randn(1, SEQ_LEN, N_FEATURES).astype(np.float32)
    infer = tf.function(model, jit_compile=False)
    # warmup
    for _ in range(WARMUP):
        _ = infer(dummy)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _ = infer(dummy)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    p50 = statistics.median(times)
    p95 = np.percentile(times, 95)
    fps = 1000.0 / p50
    print(f"\nKeras FP32 latência (batch=1): p50={p50:.2f} ms | p95={p95:.2f} ms | ~{fps:.1f} FPS")

    # (Opcional) Converter e medir TFLite FP16
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        open("model_fp16.tflite","wb").write(tflite_model)
        print("Gerado: model_fp16.tflite")

        # Rodar TFLite local
        import tensorflow.lite as tflite
        interp = tflite.Interpreter(model_path="model_fp16.tflite")
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]['index']
        out = interp.get_output_details()[0]['index']
        # warmup
        for _ in range(WARMUP):
            interp.set_tensor(inp, dummy)
            interp.invoke()
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            interp.set_tensor(inp, dummy)
            interp.invoke()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        p50 = statistics.median(times); p95 = np.percentile(times,95); fps = 1000.0/p50
        print(f"TFLite FP16 latência: p50={p50:.2f} ms | p95={p95:.2f} ms | ~{fps:.1f} FPS")
    except Exception as e:
        print(f"(TFLite FP16 opcional) Falhou/ignorado: {e}")

    # (Opcional) INT8 precisa de calibração: forneça amostras reais
    # Veja depois se quiser que eu te mande a rotina de calibração.

if __name__ == "__main__":
    main()
