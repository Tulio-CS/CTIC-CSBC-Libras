# benchmark_live.py
# -*- coding: utf-8 -*-
"""
Benchmark ao vivo para LIBRAS:
- Mede FPS, latências por etapa, tamanhos (modelo + mediapipe), e estatísticas p50/p95.
- Pressione 'S' para imprimir um resumo no console; 'Q' para sair.

Requisitos:
  pip install opencv-python mediapipe tensorflow numpy
  # opcional:
  pip install psutil
"""

import os
import sys
import json
import time
import csv
import math
import statistics as stats
from collections import deque, defaultdict

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    import psutil
except Exception:
    psutil = None

import mediapipe as mp

# ----------------- Config -----------------
MODEL_PATHS = ["best_model.keras", "libras_recognition_model_v2.keras"]
ACTIONS_PATH = "libras_actions.npy"
NORM_STATS_PATH = "results_v3/norm_stats.json"

CAM_INDEX = 0
CONF_THRESH = 0.30
FONT = cv2.FONT_HERSHEY_SIMPLEX

EMA_ALPHA = 0.6                 # suavização exponencial de prob (opcional)
WINDOW_FPS = 30                 # janelas para FPS instantâneo
MAJORITY_K = 8                  # voto majoritário (opcional)
SHOW_OVERLAY = True
DRAW_HANDS = True
LOG_TO_CSV = True
CSV_PATH = "live_benchmark.csv"

# -------------- Utils --------------
def load_first_existing(paths):
    for p in paths if isinstance(paths, (list, tuple)) else [paths]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Nenhum dos caminhos existe: {paths}")

def load_norm_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    mu = np.array(js["mu"], dtype=np.float32)
    sd = np.array(js["sd"], dtype=np.float32)
    return mu, sd

def norm_apply(x, mu, sd):
    return (x - mu) / (sd + 1e-8)

def ensure_len(vec, target_len):
    F = vec.shape[-1]
    if F == target_len:
        return vec
    if F > target_len:
        return vec[..., :target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:F] = vec
    return out

def sizeof_fmt(num, suffix="B"):
    for unit in ["","K","M","G","T","P","E","Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"

def dir_size_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total

def estimate_mediapipe_install_size():
    # Tenta inferir o diretório do mediapipe via __file__
    try:
        import mediapipe as _mp
        base = os.path.dirname(_mp.__file__)
        return dir_size_bytes(base), base
    except Exception:
        return None, None

def percentile(arr, p):
    if not arr:
        return None
    arr_sorted = sorted(arr)
    k = (len(arr_sorted) - 1) * (p / 100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c:
        return arr_sorted[int(k)]
    d0 = arr_sorted[f] * (c - k)
    d1 = arr_sorted[c] * (k - f)
    return d0 + d1

class ProbEMA:
    def __init__(self, n_classes, alpha=0.6):
        self.alpha = alpha
        self.state = np.zeros(n_classes, dtype=np.float32)
        self.init = False

    def update(self, p):
        if not self.init:
            self.state = p.astype(np.float32).copy()
            self.init = True
        else:
            self.state = self.alpha * p + (1 - self.alpha) * self.state
        return self.state

# ============== MediaPipe setup ===============
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_two_hands_features(results):
    """
    Vetor de 126 floats (21*3*2): [Rhand(63), Lhand(63)], centrado no punho.
    Se faltar mão, bloco = zeros.
    """
    def hand_vec(hand_landmarks):
        pts = []
        for lm in hand_landmarks.landmark:
            pts.extend([lm.x, lm.y, lm.z])
        arr = np.array(pts, dtype=np.float32)  # (63,)
        wrist = np.array(arr[:3], dtype=np.float32)
        arr = arr.reshape(-1, 3) - wrist
        return arr.reshape(-1)

    right = np.zeros(63, dtype=np.float32)
    left  = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label.lower()
            vec = hand_vec(hand_lms)
            if label.startswith("right"):
                right = vec
            else:
                left = vec
    return np.concatenate([right, left], axis=0)  # 126

# -------------- Main --------------
def main():
    # ---- Info de HW/TF ----
    devices = tf.config.list_physical_devices()
    print("[INFO] Dispositivos TF:", devices)

    # ---- Modelo e metadados ----
    model_path = load_first_existing(MODEL_PATHS)
    actions = np.load(ACTIONS_PATH).astype(str)
    mu, sd = load_norm_stats(NORM_STATS_PATH)

    print(f"[INFO] Carregando modelo: {model_path}")
    t0 = time.perf_counter()
    model = load_model(model_path)
    t_model_load = time.perf_counter() - t0

    _, T, F = model.input_shape
    print(f"[INFO] Janela esperada T={T}, F={F}")
    print(f"[INFO] Classes ({len(actions)}): {actions.tolist()}")
    print(f"[INFO] Tempo p/ carregar modelo: {t_model_load*1000:.1f} ms")

    # ---- Tamanhos: modelo e mediapipe ----
    try:
        model_size = os.path.getsize(model_path)
    except OSError:
        model_size = None
    mp_size, mp_base = estimate_mediapipe_install_size()

    print("[INFO] Tamanhos aproximados:")
    if model_size is not None:
        print(f"  - Modelo: {sizeof_fmt(model_size)} ({model_path})")
    else:
        print("  - Modelo: N/D")
    if mp_size is not None:
        print(f"  - MediaPipe instalado: {sizeof_fmt(mp_size)} (dir: {mp_base})")
    else:
        print("  - MediaPipe: N/D")

    # ---- CSV ----
    if LOG_TO_CSV and not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts", "fps_instant",
                "t_capture_ms", "t_mp_ms", "t_feat_ms", "t_norm_ms",
                "t_pred_ms", "t_render_ms", "hands_detected", "pred", "conf"
            ])

    # ---- Webcam ----
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir a webcam.")

    # ---- Buffers e métrica ----
    ema = ProbEMA(n_classes=len(actions), alpha=EMA_ALPHA)
    seq = deque(maxlen=T)
    vote = deque(maxlen=MAJORITY_K)
    times = defaultdict(list)   # guarda latências em ms por etapa
    fps_window = deque(maxlen=WINDOW_FPS)
    frames_total = 0
    frames_no_hands = 0
    t_start = time.perf_counter()

    print("[INFO] Pressione 'S' para resumo, 'Q' para sair.")
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            t_frame0 = time.perf_counter()
            # ---- captura
            tc0 = time.perf_counter()
            ok, frame = cap.read()
            t_capture = (time.perf_counter() - tc0) * 1000.0
            if not ok:
                print("[WARN] Frame perdido.")
                break

            frame = cv2.flip(frame, 1)

            # ---- mediapipe
            tm0 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True
            t_mp = (time.perf_counter() - tm0) * 1000.0

            hands_detected = 1 if res.multi_hand_landmarks else 0
            if DRAW_HANDS and res.multi_hand_landmarks:
                for lms in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

            if not hands_detected:
                frames_no_hands += 1

            # ---- features
            tf0 = time.perf_counter()
            feat = extract_two_hands_features(res)  # 126
            feat = ensure_len(feat, F)
            t_feat = (time.perf_counter() - tf0) * 1000.0

            # ---- normalização
            tn0 = time.perf_counter()
            seq.append(feat)
            pred_label = "?"
            conf = 0.0

            t_norm = (time.perf_counter() - tn0) * 1000.0

            # ---- predição (se janela cheia)
            tp_ms = 0.0
            if len(seq) == T:
                X = np.array(seq, dtype=np.float32)   # (T, F)
                Xn = norm_apply(X, mu, sd)
                Xn = np.expand_dims(Xn, axis=0)       # (1, T, F)

                tp0 = time.perf_counter()
                prob = model.predict(Xn, verbose=0)[0]
                tp_ms = (time.perf_counter() - tp0) * 1000.0

                prob = ema.update(prob)
                cls = int(np.argmax(prob))
                conf = float(prob[cls])
                vote.append(cls)
                vote_cls = max(set(vote), key=vote.count) if vote else cls

                if conf >= CONF_THRESH and vote.count(vote_cls) >= max(2, MAJORITY_K // 2):
                    pred_label = actions[vote_cls]
                else:
                    pred_label = "?"

            # ---- render / overlay
            tr0 = time.perf_counter()
            if SHOW_OVERLAY:
                now = time.perf_counter()
                fps_window.append(now)
                fps_inst = None
                if len(fps_window) >= 2:
                    elapsed = fps_window[-1] - fps_window[0]
                    if elapsed > 0:
                        fps_inst = (len(fps_window) - 1) / elapsed

                overlay_lines = [
                    f"Pred: {pred_label} ({conf:.2f})",
                    f"FPS: {fps_inst:.1f}" if fps_inst else "FPS: ...",
                    f"cap:{t_capture:.1f} | mp:{t_mp:.1f} | feat:{t_feat:.1f} | norm:{t_norm:.1f} | pred:{tp_ms:.1f} ms"
                ]
                y = 24
                cv2.rectangle(frame, (0, 0), (frame.shape[1], y + 8), (0, 0, 0), -1)
                cv2.putText(frame, overlay_lines[0], (10, y), FONT, 0.7, (255, 255, 255), 2)
                y += 22
                cv2.putText(frame, overlay_lines[1], (10, y), FONT, 0.6, (255, 255, 255), 1)
                y += 20
                cv2.putText(frame, overlay_lines[2], (10, y), FONT, 0.5, (255, 255, 255), 1)

            cv2.imshow("LIBRAS Live - Benchmark", frame)
            t_render = (time.perf_counter() - tr0) * 1000.0

            # ---- armazenar tempos
            times["capture_ms"].append(t_capture)
            times["mediapipe_ms"].append(t_mp)
            times["features_ms"].append(t_feat)
            times["normalize_ms"].append(t_norm)
            if tp_ms > 0:
                times["predict_ms"].append(tp_ms)
            times["render_ms"].append(t_render)

            # ---- CSV
            if LOG_TO_CSV:
                try:
                    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        inst_fps_val = None
                        if len(fps_window) >= 2:
                            elapsed = fps_window[-1] - fps_window[0]
                            inst_fps_val = (len(fps_window) - 1) / elapsed if elapsed > 0 else None
                        w.writerow([
                            time.time(),
                            f"{inst_fps_val:.3f}" if inst_fps_val else "",
                            f"{t_capture:.3f}", f"{t_mp:.3f}", f"{t_feat:.3f}", f"{t_norm:.3f}",
                            f"{tp_ms:.3f}", f"{t_render:.3f}", hands_detected, pred_label, f"{conf:.4f}"
                        ])
                except Exception:
                    pass

            frames_total += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('s'), ord('S')):
                print_summary(times, frames_total, frames_no_hands, t_start)

    cap.release()
    cv2.destroyAllWindows()
    # resumo final
    print_summary(times, frames_total, frames_no_hands, t_start)

def fmt_stats(name, arr):
    if not arr:
        return f"{name:14s} -> n=0"
    return (f"{name:14s} -> n={len(arr):5d} | "
            f"mean={stats.mean(arr):7.2f} ms | "
            f"min={min(arr):6.2f} | p50={percentile(arr,50):6.2f} | "
            f"p95={percentile(arr,95):6.2f} | max={max(arr):6.2f}")

def print_summary(times, frames_total, frames_no_hands, t_start):
    elapsed = time.perf_counter() - t_start
    fps_avg = frames_total / elapsed if elapsed > 0 else 0.0
    no_hands_pct = (frames_no_hands / frames_total * 100.0) if frames_total else 0.0

    print("\n========== RESUMO ==========")
    print(f"Frames totais: {frames_total} | Tempo: {elapsed:.2f}s | FPS médio: {fps_avg:.2f}")
    print(f"Frames sem mãos: {frames_no_hands} ({no_hands_pct:.1f}%)")
    print(fmt_stats("capture",   times["capture_ms"]))
    print(fmt_stats("mediapipe", times["mediapipe_ms"]))
    print(fmt_stats("features",  times["features_ms"]))
    print(fmt_stats("normalize", times["normalize_ms"]))
    print(fmt_stats("predict",   times["predict_ms"]))
    print(fmt_stats("render",    times["render_ms"]))

    if psutil:
        p = psutil.Process(os.getpid())
        mem = p.memory_info().rss
        cpu = psutil.cpu_percent(interval=0.1)
        print(f"Mem RSS (proc): {sizeof_fmt(mem)} | CPU% (amostra): {cpu:.1f}%")
    else:
        print("(psutil não instalado; CPU/RAM omitidos)")
    print("============================\n")

if __name__ == "__main__":
    main()
