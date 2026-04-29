# web/predictor.py
# -*- coding: utf-8 -*-
"""
Carrega o modelo Keras e gerencia sessões de inferência por WebSocket.
Cada conexão recebe uma sessão isolada com buffer, EMA e majority-vote próprios.
"""

import os, sys
import numpy as np
from collections import deque

# Adiciona a raiz do projeto ao path para importar config e data_utils
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as cfg
from data_utils import load_norm_stats, apply_feature_mode


# ─────────────────────────────────────────────────────────────────────────────
# Sessão por conexão WebSocket
# ─────────────────────────────────────────────────────────────────────────────

class InferenceSession:
    def __init__(self, T: int, n_classes: int,
                 ema_alpha: float = 0.6, majority_k: int = 8):
        self.buffer     = deque(maxlen=T)
        self.vote       = deque(maxlen=majority_k)
        self.ema        = np.zeros(n_classes, dtype=np.float32)
        self._ema_init  = False
        self.ema_alpha  = ema_alpha

    def update_ema(self, prob: np.ndarray) -> np.ndarray:
        if not self._ema_init:
            self.ema       = prob.copy()
            self._ema_init = True
        else:
            self.ema = self.ema_alpha * prob + (1 - self.ema_alpha) * self.ema
        return self.ema

    def reset(self):
        self.buffer.clear()
        self.vote.clear()
        self.ema[:]    = 0
        self._ema_init = False


# ─────────────────────────────────────────────────────────────────────────────
# Predictor (singleton — carregado uma vez na inicialização do servidor)
# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    def __init__(self, model_name: str | None = None):
        import tensorflow as tf

        model_name = model_name or cfg.DEFAULT_MODEL_NAME
        model_dir  = os.path.join(ROOT, cfg.MODELS_DIR, model_name)

        model_path   = os.path.join(model_dir, "model.keras")
        norm_path    = os.path.join(model_dir, "norm_stats.json")
        actions_path = os.path.join(model_dir, "actions.npy")

        for p in [model_path, norm_path, actions_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Arquivo necessário não encontrado: {p}\n"
                    f"Treine primeiro com: py -3.11 train.py --model {model_name}"
                )

        print(f"[Predictor] Carregando modelo: {model_path}")
        self.model   = tf.keras.models.load_model(model_path, compile=False)
        self.actions = np.load(actions_path).astype(str)
        mu, sd, self.T, self.F, self.feature_mode = load_norm_stats(norm_path)

        # (F,) — para operações vetorizadas rápidas
        self.mu = mu.reshape(-1).astype(np.float32)
        self.sd = sd.reshape(-1).astype(np.float32)

        self.n_classes = len(self.actions)
        print(f"[Predictor] Pronto — {self.n_classes} classes | T={self.T} | F={self.F}")

        # Warm-up para evitar latência na primeira predição
        dummy = np.zeros((1, self.T, self.F), dtype=np.float32)
        self.model.predict(dummy, verbose=0)
        print("[Predictor] Warm-up concluído.")

    # ── API pública ──────────────────────────────────────────────────────────

    def new_session(self) -> InferenceSession:
        return InferenceSession(
            T=self.T,
            n_classes=self.n_classes,
            ema_alpha=cfg.EMA_ALPHA,
            majority_k=cfg.MAJORITY_K,
        )

    def step(self, session: InferenceSession, raw_landmarks: list) -> dict:
        """
        Recebe um vetor raw de 126 landmarks (sem centering aplicado pelo cliente),
        adiciona ao buffer da sessão e retorna a predição corrente.

        raw_landmarks: lista de 126 floats [right_raw(63), left_raw(63)]
        """
        feat = np.array(raw_landmarks, dtype=np.float32)

        # Garante comprimento correto (padding ou truncamento)
        if len(feat) < self.F:
            feat = np.pad(feat, (0, self.F - len(feat)))
        feat = feat[: self.F]

        # Aplica a mesma feature_mode do treinamento (ex: wrist_centered)
        feat_2d = feat.reshape(1, self.F)            # (1, F) para apply_feature_mode
        feat_2d = apply_feature_mode(feat_2d, self.feature_mode)
        feat    = feat_2d.reshape(-1)

        session.buffer.append(feat)

        result: dict = {
            "pred":        None,
            "conf":        0.0,
            "top3":        {},
            "buffer_fill": len(session.buffer),
            "buffer_need": self.T,
        }

        if len(session.buffer) < self.T:
            return result

        # Janela completa → inferência
        X  = np.stack(session.buffer, axis=0)        # (T, F)
        Xn = (X - self.mu) / (self.sd + 1e-8)       # z-score
        Xn = Xn[np.newaxis, ...]                     # (1, T, F)

        prob = self.model.predict(Xn, verbose=0)[0]  # (n_classes,)
        prob = session.update_ema(prob)               # EMA suavização

        cls  = int(np.argmax(prob))
        conf = float(prob[cls])

        # Majority vote
        session.vote.append(cls)
        vote_cls = max(set(session.vote), key=session.vote.count)
        enough   = session.vote.count(vote_cls) >= max(2, cfg.MAJORITY_K // 2)

        # Top-3 predições para exibir no frontend
        top3_idx = np.argsort(prob)[::-1][:3]
        top3     = {self.actions[i]: round(float(prob[i]), 4) for i in top3_idx}

        result["pred"] = self.actions[vote_cls] if (conf >= cfg.CONF_THRESH and enough) else "?"
        result["conf"] = round(conf, 4)
        result["top3"] = top3

        return result
