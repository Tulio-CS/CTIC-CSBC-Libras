# extract_minds_to_numpy.py
# -*- coding: utf-8 -*-
import os
import re
import uuid
import glob
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# ============ CONFIG ============
INPUT_DIR = "MINDS"             # pasta com os vídeos .mp4
OUTPUT_DIR = "libras_data_minds"
SEQUENCE_LENGTH = 15            # nº de frames por sequência (compatível com seu coletor)
MAX_HANDS = 2                   # até duas mãos
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5
# ================================

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRK_CONF,
)

def extract_label(filename: str) -> str:
    """
    Extrai o nome do sinal a partir do padrão:
      Numsinal Nomesinal Num sinalizador
    Ex.: '01AlunoSinalizador10.mp4' -> 'Aluno'
    Também lida com sufixos como '-1' ou '_2' antes da extensão.
    """
    base = os.path.basename(filename)
    # Remove extensão
    base_noext = os.path.splitext(base)[0]

    # Regex: [opcional números][nome][Sinalizador][números][opcional -_qualquercoisa]
    m = re.match(r'^\s*\d*([A-Za-zÀ-ÿ]+)Sinalizador\d+(?:[-_].*)?$', base_noext)
    if m:
        return m.group(1)

    # Fallback: tente dividir por "Sinalizador" e pegar o bloco antes,
    # removendo dígitos do começo
    if "Sinalizador" in base_noext:
        left = base_noext.split("Sinalizador")[0]
        # remove números iniciais
        left = re.sub(r'^\d+', '', left)
        # mantém apenas letras (inclui acentos)
        left = re.sub(r'[^A-Za-zÀ-ÿ]', '', left)
        if left:
            return left

    raise ValueError(f"Não foi possível extrair o label de: {filename}")

def frame_indices_uniform(n_frames: int, seq_len: int) -> np.ndarray:
    """Indices (int) uniformemente espaçados no intervalo [0, n_frames-1]."""
    if n_frames <= 0:
        return np.array([], dtype=int)
    if n_frames < seq_len:
        # repete alguns índices para alcançar o tamanho desejado
        idx = np.linspace(0, n_frames - 1, n_frames, dtype=int)
        rep = np.pad(idx, (0, seq_len - n_frames), mode='edge')
        return rep
    return np.linspace(0, n_frames - 1, seq_len, dtype=int)

def landmarks_from_results(results) -> np.ndarray:
    """
    Retorna um vetor (126,) com [x,y,z]*21 por mão (Left, depois Right).
    Se só houver uma mão, a outra é preenchida com zeros.
    Se nenhuma mão, retorna zeros.
    """
    vec_per_hand = 21 * 3
    out = np.zeros(vec_per_hand * 2, dtype=np.float32)  # 126

    if not results.multi_hand_landmarks:
        return out

    # Mapear índice -> label ('Left'/'Right')
    hand_labels = []
    if results.multi_handedness:
        for i, h in enumerate(results.multi_handedness):
            try:
                hand_labels.append((i, h.classification[0].label))
            except Exception:
                hand_labels.append((i, "Unknown"))
    else:
        # Se não vier handedness, use ordem original
        hand_labels = [(i, "Unknown") for i in range(len(results.multi_hand_landmarks))]

    # Ordena Left -> Right
    def sort_key(p):
        # Left primeiro, depois Right, depois Unknown
        return {"Left": 0, "Right": 1}.get(p[1], 2)

    hand_labels.sort(key=sort_key)

    # Preenche até 2 mãos
    write_offset = 0
    for (orig_idx, _label) in hand_labels[:2]:
        hlm = results.multi_hand_landmarks[orig_idx]
        coords = []
        for lm in hlm.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        coords = np.array(coords, dtype=np.float32)

        out[write_offset: write_offset + vec_per_hand] = coords
        write_offset += vec_per_hand

    return out  # shape (126,)

def process_video(video_path: str, label: str) -> bool:
    """
    Processa 1 vídeo e salva uma sequência .npy em OUTPUT_DIR/label.
    Retorna True/False para sucesso/falha.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRO] Não abriu: {video_path}")
        return False

    # nº de frames (nem sempre é perfeito, mas serve p/ amostrar)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = frame_indices_uniform(n_frames, SEQUENCE_LENGTH)
    if idxs.size == 0:
        print(f"[ERRO] Vídeo sem frames: {video_path}")
        cap.release()
        return False

    sequence = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Se falhou, preenche zeros para manter tamanho
            sequence.append(np.zeros(126, dtype=np.float32))
            continue

        # espelha p/ ficar como sua coleta (selfie)
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        vec = landmarks_from_results(results)
        sequence.append(vec)

    cap.release()

    sequence = np.stack(sequence, axis=0)  # (SEQUENCE_LENGTH, 126)

    # Salvar
    out_dir = Path(OUTPUT_DIR) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{label}-{uuid.uuid4()}.npy"
    np.save(str(out_dir / fname), sequence)
    return True

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    video_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*.mp4"), recursive=True))
    if not video_paths:
        print(f"Nenhum .mp4 encontrado em '{INPUT_DIR}'.")
        return

    total, ok = 0, 0
    for vp in video_paths:
        total += 1
        try:
            label = extract_label(vp)
        except ValueError as e:
            print(f"[SKIP] {e}")
            continue

        if process_video(vp, label):
            ok += 1
            print(f"[OK] {vp} -> label='{label}'")
        else:
            print(f"[FAIL] {vp}")

    print(f"\nConcluído: {ok}/{total} vídeos processados. Saída em '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
