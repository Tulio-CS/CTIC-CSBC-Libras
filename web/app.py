# web/app.py
# -*- coding: utf-8 -*-
"""
Backend FastAPI para inferência LIBRAS ao vivo via WebSocket.

Protocolo WebSocket:
  Cliente → Servidor  (JSON por frame):
    { "landmarks": [126 floats], "reset": false }

  Servidor → Cliente  (JSON por frame):
    { "pred": "Ola", "conf": 0.95, "top3": {"Ola": 0.95, ...},
      "buffer_fill": 12, "buffer_need": 15 }

Inicialização:
  cd D:/LSB_Object_Detection
  uvicorn web.app:app --host 0.0.0.0 --port 8000

  # ou na VPS (após deploy.sh):
  uvicorn web.app:app --host 0.0.0.0 --port 8000 --workers 1
"""

import os, json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from predictor import Predictor

# ─────────────────────────────────────────────────────────────────────────────
# Inicialização do modelo (uma vez só, ao subir o servidor)
# ─────────────────────────────────────────────────────────────────────────────

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    model_name = os.getenv("LIBRAS_MODEL", "bilstm_attn")
    predictor  = Predictor(model_name=model_name)
    yield
    # cleanup (se necessário)


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="LIBRAS Live API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints REST
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    if predictor is None:
        return {"status": "loading"}
    return {
        "status":    "ok",
        "classes":   predictor.n_classes,
        "T":         predictor.T,
        "F":         predictor.F,
        "model":     os.getenv("LIBRAS_MODEL", "bilstm_attn"),
        "actions":   predictor.actions.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — uma sessão por conexão
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    if predictor is None:
        await ws.send_json({"error": "Modelo ainda carregando. Tente novamente."})
        await ws.close()
        return

    session = predictor.new_session()

    try:
        while True:
            raw  = await ws.receive_text()
            msg  = json.loads(raw)

            # Reset pedido pelo cliente (ex: botão "Resetar")
            if msg.get("reset"):
                session.reset()
                await ws.send_json({"reset": True})
                continue

            landmarks = msg.get("landmarks", [])
            if not landmarks:
                continue

            result = predictor.step(session, landmarks)
            await ws.send_json(result)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Serve frontend estático (index.html)
# ─────────────────────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
