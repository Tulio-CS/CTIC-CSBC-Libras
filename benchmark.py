# benchmark.py
# -*- coding: utf-8 -*-
"""
Benchmark de performance ao vivo: FPS, latências por etapa, tamanho do modelo.

Uso:
    python benchmark.py
    python benchmark.py --model bilstm --cam 0

Teclas:
    S — imprime resumo no console
    Q — encerra e imprime resumo final
"""

import os, sys, json, time, math, csv, argparse
import statistics as stats
from collections import deque, defaultdict

import cv2
import numpy as np

import config as cfg
from data_utils import load_norm_stats

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark de performance LIBRAS.")
    p.add_argument("--model", default=cfg.DEFAULT_MODEL_NAME)
    p.add_argument("--cam",   type=int, default=cfg.CAM_INDEX)
    p.add_argument("--no_log", action="store_true")
    return p.parse_args()


def sizeof_fmt(num):
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def percentile(arr, p):
    if not arr:
        return float("nan")
    arr_s = sorted(arr)
    k = (len(arr_s) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return arr_s[int(k)]
    return arr_s[f] * (c - k) + arr_s[c] * (k - f)


def extract_landmarks(results):
    right = np.zeros(63, dtype=np.float32)
    left  = np.zeros(63, dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for lms, handedness in zip(results.multi_hand_landmarks,
                                   results.multi_handedness):
            label = handedness.classification[0].label.lower()
            pts   = np.array([[lm.x, lm.y, lm.z]
                              for lm in lms.landmark], dtype=np.float32)
            pts  -= pts[0:1, :]
            vec   = pts.reshape(-1)
            if label.startswith("right"):
                right = vec
            else:
                left = vec
    return np.concatenate([right, left])


class ProbEMA:
    def __init__(self, n, alpha=0.6):
        self.alpha = alpha
        self.state = np.zeros(n, dtype=np.float32)
        self._init  = False

    def update(self, p):
        if not self._init:
            self.state = p.copy(); self._init = True
        else:
            self.state = self.alpha * p + (1 - self.alpha) * self.state
        return self.state


def print_summary(times, frames_total, frames_no_hands, t_start, model_info):
    elapsed    = time.perf_counter() - t_start
    fps_avg    = frames_total / elapsed if elapsed > 0 else 0.0
    no_hnd_pct = frames_no_hands / max(frames_total, 1) * 100

    def row(name, arr):
        if not arr:
            return f"  {name:<14} : n=0"
        return (f"  {name:<14} : n={len(arr):5d} | "
                f"mean={stats.mean(arr):6.2f} ms | "
                f"p50={percentile(arr,50):6.2f} | "
                f"p95={percentile(arr,95):6.2f} | "
                f"max={max(arr):6.2f}")

    print("\n" + "="*65)
    print(f"  Modelo           : {model_info}")
    print(f"  Frames totais    : {frames_total}  |  Tempo: {elapsed:.1f}s  |  FPS médio: {fps_avg:.2f}")
    print(f"  Frames sem mãos  : {frames_no_hands} ({no_hnd_pct:.1f}%)")
    print("─"*65)
    for key, label in [("capture_ms",  "Captura"),
                        ("mediapipe_ms","MediaPipe"),
                        ("features_ms", "Features"),
                        ("normalize_ms","Normaliz."),
                        ("predict_ms",  "Predição"),
                        ("render_ms",   "Render")]:
        print(row(label, times.get(key, [])))
    if HAS_PSUTIL:
        p   = psutil.Process(os.getpid())
        mem = p.memory_info().rss
        cpu = psutil.cpu_percent(interval=0.1)
        print(f"\n  Memória RSS      : {sizeof_fmt(mem)}")
        print(f"  CPU (amostra)    : {cpu:.1f}%")
    print("="*65 + "\n")


def main():
    args = parse_args()

    model_dir    = os.path.join(cfg.MODELS_DIR, args.model)
    model_path   = os.path.join(model_dir, "model.keras")
    norm_path    = os.path.join(model_dir, "norm_stats.json")
    actions_path = os.path.join(model_dir, "actions.npy")

    for p in [model_path, norm_path, actions_path]:
        if not os.path.exists(p):
            print(f"[ERRO] {p} não encontrado.")
            print(f"       Treine: python train.py --model {args.model}")
            sys.exit(1)

    import tensorflow as tf
    import mediapipe as mp
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    print(f"[INFO] TF devices: {tf.config.list_physical_devices()}")

    t0 = time.perf_counter()
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"[INFO] Modelo carregado em {(time.perf_counter()-t0)*1000:.1f} ms")
    print(f"[INFO] Tamanho do modelo: {sizeof_fmt(os.path.getsize(model_path))}")

    actions = np.load(actions_path).astype(str)
    mu, sd, T, F, _ = load_norm_stats(norm_path)
    mu = mu.reshape(-1).astype(np.float32)
    sd = sd.reshape(-1).astype(np.float32)

    # CSV de log
    log_csv  = not args.no_log
    log_file = None
    if log_csv:
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/benchmark_{args.model}.csv"
        log_file = open(log_path, "w", newline="", encoding="utf-8")
        lw = csv.writer(log_file)
        lw.writerow(["ts","fps","t_cap","t_mp","t_feat","t_norm","t_pred","t_render",
                     "hands","pred","conf"])

    seq        = deque(maxlen=T)
    vote       = deque(maxlen=cfg.MAJORITY_K)
    ema        = ProbEMA(len(actions))
    times      = defaultdict(list)
    fps_window = deque(maxlen=30)
    frames_total = frames_no_hands = 0
    t_start    = time.perf_counter()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERRO] Câmera {args.cam} indisponível.")
        sys.exit(1)

    print("[INFO] S = resumo  |  Q = sair\n")

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:

        while True:
            t_frame = time.perf_counter()

            # Captura
            tc = time.perf_counter()
            ok, frame = cap.read()
            t_cap = (time.perf_counter() - tc) * 1e3
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            # MediaPipe
            tm = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True
            t_mp = (time.perf_counter() - tm) * 1e3

            det = bool(results.multi_hand_landmarks)
            if not det:
                frames_no_hands += 1
            if det:
                for lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

            # Features
            tf_ = time.perf_counter()
            feat = extract_landmarks(results)
            feat = feat[:F] if len(feat) >= F else np.pad(feat, (0, F - len(feat)))
            t_feat = (time.perf_counter() - tf_) * 1e3

            # Normalização
            tn = time.perf_counter()
            seq.append(feat)
            t_norm = (time.perf_counter() - tn) * 1e3

            # Predição
            pred_label = "?"; conf = 0.0; tp_ms = 0.0
            if len(seq) == T:
                Xn = (np.array(seq, dtype=np.float32) - mu) / (sd + 1e-8)
                Xn = np.expand_dims(Xn, 0)
                tp = time.perf_counter()
                prob = model.predict(Xn, verbose=0)[0]
                tp_ms = (time.perf_counter() - tp) * 1e3
                prob  = ema.update(prob)
                cls   = int(np.argmax(prob))
                conf  = float(prob[cls])
                vote.append(cls)
                vote_cls = max(set(vote), key=vote.count)
                if conf >= cfg.CONF_THRESH and vote.count(vote_cls) >= max(2, cfg.MAJORITY_K//2):
                    pred_label = actions[vote_cls]

            # Render
            tr = time.perf_counter()
            now = time.perf_counter()
            fps_window.append(now)
            fps_inst = (len(fps_window)-1)/(fps_window[-1]-fps_window[0]) \
                       if len(fps_window) >= 2 and fps_window[-1]-fps_window[0] > 0 else 0

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0,0), (w, 70), (20,20,20), -1)
            cv2.putText(frame, f"{pred_label} ({conf:.2f})", (10,28), FONT, 0.85, (0,230,0), 2)
            cv2.putText(frame, f"FPS: {fps_inst:.1f}", (10,52), FONT, 0.6, (200,200,200), 1)
            cv2.putText(frame,
                        f"cap:{t_cap:.0f} mp:{t_mp:.0f} feat:{t_feat:.0f} pred:{tp_ms:.0f} ms",
                        (10,68), FONT, 0.4, (180,180,180), 1)
            cv2.imshow("LIBRAS — Benchmark", frame)
            t_render = (time.perf_counter() - tr) * 1e3

            # Registra tempos
            times["capture_ms"].append(t_cap)
            times["mediapipe_ms"].append(t_mp)
            times["features_ms"].append(t_feat)
            times["normalize_ms"].append(t_norm)
            if tp_ms > 0:
                times["predict_ms"].append(tp_ms)
            times["render_ms"].append(t_render)

            if log_csv:
                lw.writerow([f"{time.time():.3f}", f"{fps_inst:.2f}",
                             f"{t_cap:.2f}", f"{t_mp:.2f}", f"{t_feat:.2f}",
                             f"{t_norm:.2f}", f"{tp_ms:.2f}", f"{t_render:.2f}",
                             int(det), pred_label, f"{conf:.4f}"])

            frames_total += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('s'), ord('S')):
                print_summary(times, frames_total, frames_no_hands,
                              t_start, args.model)

    cap.release()
    cv2.destroyAllWindows()
    if log_file:
        log_file.close()
    print_summary(times, frames_total, frames_no_hands, t_start, args.model)


if __name__ == "__main__":
    main()
