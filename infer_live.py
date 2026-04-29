# infer_live.py
# -*- coding: utf-8 -*-
"""
Inferência ao vivo para LIBRAS com suavização EMA + majority vote.

Uso:
    python infer_live.py                        # modelo padrão (bilstm_attn)
    python infer_live.py --model bilstm
    python infer_live.py --model bilstm_attn --cam 1
    python infer_live.py --conf 0.5             # limiar de confiança mais alto

Teclas:
    Q — sair
    R — resetar buffers (útil se a predição "travar")
"""

import os, sys, json, csv, time, argparse
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

import config as cfg
from data_utils import load_norm_stats

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles
FONT       = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────────────
# Argumentos
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inferência LIBRAS ao vivo.")
    p.add_argument("--model", default=cfg.DEFAULT_MODEL_NAME,
                   help="Nome do modelo (pasta em models/)")
    p.add_argument("--cam",   type=int, default=cfg.CAM_INDEX)
    p.add_argument("--conf",  type=float, default=cfg.CONF_THRESH,
                   help="Limiar mínimo de confiança para aceitar predição")
    p.add_argument("--no_log", action="store_true",
                   help="Não salva log CSV")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Extração de features — MESMA lógica do collect_data.py
# Ordem: [right(63), left(63)]
# ─────────────────────────────────────────────────────────────────────────────

def extract_landmarks(results) -> np.ndarray:
    right = np.zeros(63, dtype=np.float32)
    left  = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lms, handedness in zip(results.multi_hand_landmarks,
                                   results.multi_handedness):
            label = handedness.classification[0].label.lower()
            pts   = np.array([[lm.x, lm.y, lm.z]
                              for lm in lms.landmark], dtype=np.float32)
            pts  -= pts[0:1, :]           # centraliza no pulso
            vec   = pts.reshape(-1)
            if label.startswith("right"):
                right = vec
            else:
                left = vec

    return np.concatenate([right, left])


def ensure_len(vec, target_len):
    F = vec.shape[-1]
    if F == target_len:
        return vec
    if F > target_len:
        return vec[:target_len]
    out      = np.zeros(target_len, dtype=np.float32)
    out[:F]  = vec
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Suavização EMA
# ─────────────────────────────────────────────────────────────────────────────

class ProbEMA:
    def __init__(self, n_classes, alpha=cfg.EMA_ALPHA):
        self.alpha = alpha
        self.state = np.zeros(n_classes, dtype=np.float32)
        self._init  = False

    def update(self, p):
        if not self._init:
            self.state = p.copy()
            self._init = True
        else:
            self.state = self.alpha * p + (1 - self.alpha) * self.state
        return self.state

    def reset(self):
        self.state[:] = 0
        self._init    = False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Carrega artefatos do modelo ──────────────────────────────────────────
    model_dir    = os.path.join(cfg.MODELS_DIR, args.model)
    model_path   = os.path.join(model_dir, "model.keras")
    norm_path    = os.path.join(model_dir, "norm_stats.json")
    actions_path = os.path.join(model_dir, "actions.npy")

    for p in [model_path, norm_path, actions_path]:
        if not os.path.exists(p):
            print(f"[ERRO] Arquivo não encontrado: {p}")
            print(f"       Treine primeiro: python train.py --model {args.model}")
            sys.exit(1)

    import tensorflow as tf
    print(f"[INFO] Carregando modelo: {model_path}")
    model   = tf.keras.models.load_model(model_path, compile=False)
    actions = np.load(actions_path).astype(str)
    mu, sd, T, F, feature_mode = load_norm_stats(norm_path)
    mu  = mu.reshape(-1).astype(np.float32)   # (F,)
    sd  = sd.reshape(-1).astype(np.float32)

    print(f"[INFO] Classes ({len(actions)}): {actions.tolist()}")
    print(f"[INFO] Janela T={T} | F={F} | feature_mode={feature_mode}")
    print(f"[INFO] Limiar de confiança: {args.conf:.2f}")
    print(f"[INFO] Pressione Q para sair, R para resetar buffers.\n")

    # ── Log CSV ──────────────────────────────────────────────────────────────
    log_csv = cfg.LOG_LIVE_CSV and not args.no_log
    if log_csv:
        os.makedirs(os.path.dirname(cfg.LIVE_CSV_PATH), exist_ok=True)
        log_file = open(cfg.LIVE_CSV_PATH, "a", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        if os.path.getsize(cfg.LIVE_CSV_PATH) == 0:
            log_writer.writerow(["timestamp", "model", "pred", "confidence"])

    # ── Buffers ──────────────────────────────────────────────────────────────
    seq  = deque(maxlen=T)
    vote = deque(maxlen=cfg.MAJORITY_K)
    ema  = ProbEMA(len(actions))

    # ── Câmera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERRO] Câmera {args.cam} não disponível.")
        sys.exit(1)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame perdido.")
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            # Desenha landmarks
            if results.multi_hand_landmarks:
                for lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, lms, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            # Features
            feat = extract_landmarks(results)
            feat = ensure_len(feat, F)
            seq.append(feat)

            label_text = "..."
            conf       = 0.0

            if len(seq) == T:
                X  = np.array(seq, dtype=np.float32)  # (T, F)
                Xn = (X - mu) / (sd + 1e-8)
                Xn = np.expand_dims(Xn, 0)            # (1, T, F)

                prob = model.predict(Xn, verbose=0)[0]
                prob = ema.update(prob)

                cls      = int(np.argmax(prob))
                conf     = float(prob[cls])
                vote.append(cls)

                vote_cls = max(set(vote), key=vote.count) if vote else cls
                enough   = vote.count(vote_cls) >= max(2, cfg.MAJORITY_K // 2)

                if conf >= args.conf and enough:
                    label_text = f"{actions[vote_cls]}  ({conf:.2f})"
                    if log_csv:
                        log_writer.writerow([
                            time.time(), args.model,
                            actions[vote_cls], f"{conf:.4f}"
                        ])
                else:
                    label_text = f"?  ({conf:.2f})"

            # ── Overlay ───────────────────────────────────────────────────────
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 48), (30, 30, 30), -1)
            cv2.putText(frame, label_text, (12, 33), FONT, 0.95,
                        (0, 230, 0) if conf >= args.conf else (150, 150, 150), 2)

            cv2.imshow("LIBRAS — Inferência ao vivo", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('r'), ord('R')):
                seq.clear(); vote.clear(); ema.reset()

    cap.release()
    cv2.destroyAllWindows()
    if log_csv:
        log_file.close()
    print("[INFO] Encerrado.")


if __name__ == "__main__":
    main()
