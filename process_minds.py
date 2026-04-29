# process_minds.py
# -*- coding: utf-8 -*-
"""
Processa vídeos do dataset MINDS e extrai sequências .npy compatíveis
com o pipeline de treinamento deste projeto.

IMPORTANTE — ordenação das mãos:
    Este script usa a mesma convenção de collect_data.py e infer_live.py:
        out = [mão_direita (0..62), mão_esquerda (63..125)]
    Isso garante que um modelo treinado no dataset próprio funcione
    diretamente com dados do MINDS sem retreinamento.

Uso:
    python process_minds.py                       # padrão: MINDS/ -> libras_data_minds/
    python process_minds.py --input MINDS --output libras_data_minds
    python process_minds.py --seq_len 15 --dry_run   # simula sem salvar
"""

import os, re, sys, uuid, glob, argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    print("[ERRO] mediapipe não instalado. Execute: pip install mediapipe")
    sys.exit(1)

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Argumentos
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Processa vídeos MINDS para .npy.")
    p.add_argument("--input",   default="MINDS",
                   help="Pasta com vídeos .mp4")
    p.add_argument("--output",  default=cfg.MINDS_DATA_DIR,
                   help=f"Pasta de saída (padrão: {cfg.MINDS_DATA_DIR})")
    p.add_argument("--seq_len", type=int, default=cfg.COLLECT_SEQUENCE_LENGTH,
                   help=f"Frames por sequência (padrão: {cfg.COLLECT_SEQUENCE_LENGTH})")
    p.add_argument("--dry_run", action="store_true",
                   help="Simula o processamento sem salvar arquivos")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Extração de label do nome do arquivo MINDS
# ─────────────────────────────────────────────────────────────────────────────

def extract_label(filename: str) -> str:
    """
    Extrai o nome do sinal a partir do padrão MINDS:
        [num]NomeSinalizador[num][-sufixo].mp4
    Exemplo: '01AlunoSinalizador08-1.mp4' → 'Aluno'
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    m = re.match(r'^\d*([A-Za-zÀ-ÿ]+)Sinalizador\d+(?:[-_].*)?$', base)
    if m:
        return m.group(1)

    # Fallback: parte antes de "Sinalizador", sem dígitos iniciais
    if "Sinalizador" in base:
        left = re.sub(r'^\d+', '', base.split("Sinalizador")[0])
        left = re.sub(r'[^A-Za-zÀ-ÿ]', '', left)
        if left:
            return left

    raise ValueError(f"Não foi possível extrair label de: {filename!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Amostragem uniforme de frames
# ─────────────────────────────────────────────────────────────────────────────

def uniform_frame_indices(n_frames: int, seq_len: int) -> np.ndarray:
    """Índices uniformemente espaçados em [0, n_frames-1]."""
    if n_frames <= 0:
        return np.array([], dtype=int)
    if n_frames < seq_len:
        idx = np.arange(n_frames, dtype=int)
        return np.pad(idx, (0, seq_len - n_frames), mode="edge")
    return np.linspace(0, n_frames - 1, seq_len, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Extração de landmarks — MESMA convenção de collect_data.py
# Ordem: [right(63), left(63)]
# ─────────────────────────────────────────────────────────────────────────────

def extract_landmarks(results) -> np.ndarray:
    """
    Retorna vetor (126,) com [right_hand, left_hand].
    Convenção idêntica a collect_data.py / infer_live.py para compatibilidade
    direta com modelos treinados no dataset próprio.
    """
    right = np.zeros(63, dtype=np.float32)
    left  = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lms, handedness in zip(results.multi_hand_landmarks,
                                   results.multi_handedness):
            label = handedness.classification[0].label.lower()
            pts   = np.array([[lm.x, lm.y, lm.z]
                              for lm in lms.landmark], dtype=np.float32)
            pts  -= pts[0:1, :]          # centraliza no pulso
            vec   = pts.reshape(-1)
            if label.startswith("right"):
                right = vec
            else:
                left = vec

    return np.concatenate([right, left])  # (126,) — right primeiro


# ─────────────────────────────────────────────────────────────────────────────
# Processamento de um vídeo
# ─────────────────────────────────────────────────────────────────────────────

def process_video(video_path: str, label: str, out_dir: Path,
                  seq_len: int, hands, dry_run: bool) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERRO] Não abriu: {video_path}")
        return False

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs     = uniform_frame_indices(n_frames, seq_len)

    if idxs.size == 0:
        print(f"  [ERRO] Vídeo sem frames: {video_path}")
        cap.release()
        return False

    sequence = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            sequence.append(np.zeros(126, dtype=np.float32))
            continue
        # Espelha para ficar no mesmo ponto de vista da webcam (selfie)
        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        sequence.append(extract_landmarks(results))

    cap.release()

    arr = np.stack(sequence, axis=0)   # (seq_len, 126)

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{label}-{uuid.uuid4()}.npy"
        np.save(str(out_dir / fname), arr)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    mp_hands = mp.solutions.hands
    video_paths = sorted(glob.glob(
        os.path.join(args.input, "**", "*.mp4"), recursive=True
    ))

    if not video_paths:
        print(f"[ERRO] Nenhum .mp4 encontrado em '{args.input}'.")
        sys.exit(1)

    print(f"[INFO] Vídeos encontrados : {len(video_paths)}")
    print(f"[INFO] Saída              : {args.output}")
    print(f"[INFO] Frames por seq     : {args.seq_len}")
    if args.dry_run:
        print("[INFO] Modo DRY RUN — nenhum arquivo será salvo.")

    ok = fail = skip = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        for vp in video_paths:
            try:
                label = extract_label(vp)
            except ValueError as e:
                print(f"  [SKIP] {e}")
                skip += 1
                continue

            out_dir = Path(args.output) / label
            success = process_video(vp, label, out_dir,
                                    args.seq_len, hands, args.dry_run)
            if success:
                ok += 1
                print(f"  [OK]   {os.path.basename(vp)} → {label}")
            else:
                fail += 1
                print(f"  [FAIL] {os.path.basename(vp)}")

    print(f"\n[OK]   {ok} processados")
    print(f"[FAIL] {fail} falhas")
    print(f"[SKIP] {skip} ignorados (label não extraído)")
    print(f"\nTotal em '{args.output}': {ok} sequências")


if __name__ == "__main__":
    main()
