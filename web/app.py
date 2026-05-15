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
"""

import os, json, asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from predictor import Predictor

# ─────────────────────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────────────────────

predictor: Predictor | None = None

# Thread pool de 1 worker: garante que o modelo TF não é chamado em paralelo
# e libera o event loop do asyncio durante a inferência (que é bloqueante).
_executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    model_name = os.getenv("LIBRAS_MODEL", "bilstm_attn")
    predictor  = Predictor(model_name=model_name)
    yield


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
# REST
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    if predictor is None:
        return {"status": "loading"}
    return {
        "status":  "ok",
        "classes": predictor.n_classes,
        "T":       predictor.T,
        "F":       predictor.F,
        "model":   os.getenv("LIBRAS_MODEL", "bilstm_attn"),
        "actions": predictor.actions.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    if predictor is None:
        await ws.send_json({"error": "Modelo ainda carregando. Tente novamente."})
        await ws.close()
        return

    session = predictor.new_session()
    loop    = asyncio.get_event_loop()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("reset"):
                session.reset()
                await ws.send_json({"reset": True})
                continue

            landmarks = msg.get("landmarks", [])
            if not landmarks:
                continue

            # Roda predictor.step() em thread pool para não bloquear o asyncio.
            # Isso é crítico no mobile: sem isso, o servidor trava enquanto o
            # modelo TF processa, e frames se acumulam na fila.
            result = await loop.run_in_executor(
                _executor, predictor.step, session, landmarks
            )
            await ws.send_json(result)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Frontend estático
# ─────────────────────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
