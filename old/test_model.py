# infer_live.py
# -*- coding: utf-8 -*-
"""
Inferência ao vivo para LIBRAS com suavização temporal, normalização consistente e rejeição por baixa confiança.

Requisitos:
  pip install opencv-python mediapipe tensorflow numpy

Arquivos esperados no diretório atual:
  - libras_actions.npy
  - libras_recognition_model_v2.keras  (ou best_model.keras)
  - norm_stats.json                    (gerado no treino v2)
"""

import os
import json
import time
import csv
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import mediapipe as mp

# ----------------- Config -----------------
MODEL_PATHS = ["best_model.keras", "libras_recognition_model_v2.keras"]
ACTIONS_PATH = "libras_actions.npy"
NORM_STATS_PATH = "results_v3/norm_stats.json"

CAM_INDEX = 0
CONF_THRESH = 0.30           # confiança mínima para aceitar a predição
EMA_ALPHA = 0.6              # suavização exponencial das probabilidades
MAJORITY_K = 8               # janelas recentes para majority vote
FONT = cv2.FONT_HERSHEY_SIMPLEX

LOG_TO_CSV = True
CSV_PATH = "live_predictions.csv"

# ----------------- Utils ------------------
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
    # x shape: (T, F) | mu/sd: (F,)
    return (x - mu) / (sd + 1e-8)

def ensure_len(vec, target_len):
    """Garante comprimento do vetor de features F; preenche com zeros ou corta."""
    F = vec.shape[-1]
    if F == target_len: 
        return vec
    if F > target_len:
        return vec[..., :target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:F] = vec
    return out

# ============== MediaPipe setup ===============
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_two_hands_features(results, image_w, image_h):
    """
    Retorna um vetor de 126 floats (21*3*2): [Rhand(63), Lhand(63)].
    Se faltar mão, preenche bloco com zeros.
    Coordenadas normalizadas [0,1] e centradas no punho.
    """
    # --- helper para 1 mão (21 pontos) ---
    def hand_vec(hand_landmarks):
        pts = []
        for lm in hand_landmarks.landmark:
            # coords normalizadas já vêm [0..1] do MediaPipe (x,y); z é relativo
            pts.extend([lm.x, lm.y, lm.z])
        arr = np.array(pts, dtype=np.float32)  # (63,)
        # centraliza pelo pulso (ponto 0)
        wrist = np.array(arr[:3], dtype=np.float32)
        arr = arr.reshape(-1, 3) - wrist  # 21x3 menos wrist
        return arr.reshape(-1)            # 63

    # Inicializa com zeros
    right = np.zeros(63, dtype=np.float32)
    left  = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label.lower()  # 'Left'/'Right'
            vec = hand_vec(hand_lms)
            if label.startswith("right"):
                right = vec
            else:
                left = vec
    # Concatena (direita primeiro para bater com treino, ajuste se for o oposto)
    return np.concatenate([right, left], axis=0)  # 126

# -------------- Smoothing helpers --------------
class ProbEMA:
    def __init__(self, n_classes, alpha=0.6):
        self.alpha = alpha
        self.state = np.zeros(n_classes, dtype=np.float32)

    def update(self, p):
        if self.state.sum() == 0:
            self.state = p.copy()
        else:
            self.state = self.alpha * p + (1 - self.alpha) * self.state
        return self.state

# -------------- Main live loop --------------
def main():
    # Carrega modelo e metadados
    model_path = load_first_existing(MODEL_PATHS)
    print(f"[INFO] Carregando modelo: {model_path}")
    model = load_model(model_path)

    actions = np.load(ACTIONS_PATH)
    actions = actions.astype(str)
    n_classes = len(actions)
    print(f"[INFO] Classes: {actions.tolist()}")

    mu, sd = load_norm_stats(NORM_STATS_PATH)
    mu = mu.astype(np.float32); sd = sd.astype(np.float32)

    # Descobre (T, F) esperados pelo modelo
    _, T, F = model.input_shape
    print(f"[INFO] Modelo espera janela T={T}, features F={F}")

    if LOG_TO_CSV and not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "pred", "confidence", "probs_json"])

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir a webcam.")

    # Buffers
    seq = deque(maxlen=T)
    vote = deque(maxlen=MAJORITY_K)
    ema = ProbEMA(n_classes, alpha=EMA_ALPHA)

    print("[INFO] Pressione 'q' para sair.")
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame perdido.")
                break
            
            frame = cv2.flip(frame, 1)
            # BGR->RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = hands.process(img)
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Desenho opcional
            if results.multi_hand_landmarks:
                for lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, lms, mp_hands.HAND_CONNECTIONS)

            # === VETOR DE FEATURES ===
            feat = extract_two_hands_features(results, img.shape[1], img.shape[0])  # 126
            feat = ensure_len(feat, F)  # garante F correto (caso você mude o extrator)
            seq.append(feat)

            label_text = "..."
            conf = 0.0

            if len(seq) == T:
                X = np.array(seq, dtype=np.float32)  # (T, F)
                # Normalização como no treino
                Xn = norm_apply(X, mu, sd)
                Xn = np.expand_dims(Xn, axis=0)     # (1, T, F)

                # Probabilidades
                prob = model.predict(Xn, verbose=0)[0]  # (C,)
                prob = ema.update(prob)                 # suavização EMA

                cls = int(np.argmax(prob))
                conf = float(prob[cls])

                # Majority vote (discretiza)
                vote.append(cls)
                vote_cls = max(set(vote), key=vote.count) if vote else cls

                if conf >= CONF_THRESH and vote.count(vote_cls) >= max(2, MAJORITY_K//2):
                    label_text = f"{actions[vote_cls]} ({conf:.2f})"
                else:
                    label_text = f"? ({conf:.2f})"

                # Log em CSV
                if LOG_TO_CSV:
                    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow([time.time(), actions[cls], f"{conf:.4f}",
                                    json.dumps(prob.astype(float).tolist())])

            # Overlay
            cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(img, f"Pred: {label_text}", (10, 28), FONT, 0.8, (255, 255, 255), 2)

            cv2.imshow("LIBRAS Live", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Encerrado.")

if __name__ == "__main__":
    main()
