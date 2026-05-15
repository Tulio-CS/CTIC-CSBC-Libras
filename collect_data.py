# collect_data.py
# -*- coding: utf-8 -*-
"""
Coleta de dados de gestos LIBRAS via webcam.

Uso:
    python collect_data.py --sign Ola
    python collect_data.py --sign Obrigado --sequences 40 --length 15
    python collect_data.py --sign a --cam 1

Teclas durante a coleta:
    SPACE  — inicia a próxima sequência manualmente
    Q      — encerra a coleta
    R      — descarta e regrava a última sequência
"""

import os, sys, argparse, uuid
import cv2
import numpy as np
import mediapipe as mp

import config as cfg

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


# ─────────────────────────────────────────────────────────────────────────────
# Argumentos
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Coleta de dados LIBRAS.")
    p.add_argument("--sign",      required=True,
                   help="Nome do sinal/gesto (ex: 'Ola', 'a', 'por favor')")
    p.add_argument("--sequences", type=int, default=cfg.COLLECT_NUM_SEQUENCES,
                   help=f"Número de sequências (padrão: {cfg.COLLECT_NUM_SEQUENCES})")
    p.add_argument("--length",    type=int, default=cfg.COLLECT_SEQUENCE_LENGTH,
                   help=f"Frames por sequência (padrão: {cfg.COLLECT_SEQUENCE_LENGTH})")
    p.add_argument("--output",    default=cfg.COLLECT_OUTPUT_DIR,
                   help=f"Diretório de saída (padrão: {cfg.COLLECT_OUTPUT_DIR})")
    p.add_argument("--cam",       type=int, default=cfg.CAM_INDEX,
                   help="Índice da câmera")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Extração de features (consistente com infer_live.py)
# Ordem: [mão_direita (63), mão_esquerda (63)]
# ─────────────────────────────────────────────────────────────────────────────

def extract_landmarks(results) -> np.ndarray:
    """
    Extrai vetor de 126 floats [right_hand(63), left_hand(63)].
    Se uma mão não for detectada, seu bloco é preenchido com zeros.
    Coordenadas normalizadas [0..1], centradas no pulso (landmark 0).
    """
    right = np.zeros(63, dtype=np.float32)
    left  = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lms, handedness in zip(results.multi_hand_landmarks,
                                   results.multi_handedness):
            label = handedness.classification[0].label.lower()
            # Salva coordenadas RAW (absolutas) — a centralização no pulso
            # é aplicada pelo data_utils.py durante o treinamento.
            vec = np.array([c for lm in lms.landmark
                            for c in (lm.x, lm.y, lm.z)], dtype=np.float32)
            if label.startswith("right"):
                right = vec
            else:
                left = vec

    return np.concatenate([right, left])   # (126,) — right primeiro


# ─────────────────────────────────────────────────────────────────────────────
# Overlay de UI
# ─────────────────────────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_overlay(frame, sign, seq_idx, total_seqs, frame_idx, total_frames,
                 phase, hands_detected, quality):
    h, w = frame.shape[:2]

    # Faixa superior
    cv2.rectangle(frame, (0, 0), (w, 55), (30, 30, 30), -1)

    # Status de detecção (verde = detectada, vermelho = não detectada)
    det_color = (0, 220, 0) if hands_detected else (0, 0, 220)
    det_text  = "MAO OK" if hands_detected else "SEM MAO"
    cv2.putText(frame, det_text, (w - 120, 22), FONT, 0.6, det_color, 2)

    # Informações gerais
    cv2.putText(frame, f"Sinal: {sign}", (10, 22), FONT, 0.7, (255, 255, 255), 2)
    cv2.putText(frame,
                f"Seq: {seq_idx}/{total_seqs}  Frame: {frame_idx}/{total_frames}",
                (10, 45), FONT, 0.55, (200, 200, 200), 1)

    # Fase
    if phase == "waiting":
        msg   = "Pressione SPACE para iniciar"
        color = (0, 200, 255)
    elif phase == "recording":
        msg   = "GRAVANDO..."
        color = (0, 60, 220) if frame_idx % 10 < 5 else (0, 100, 255)
    else:  # done
        msg   = "Sequencia salva!"
        color = (0, 200, 0)

    cv2.putText(frame, msg, (10, h - 15), FONT, 0.7, color, 2)

    # Barra de progresso da sequência
    if phase == "recording" and total_frames > 0:
        bar_w = int((frame_idx / total_frames) * (w - 20))
        cv2.rectangle(frame, (10, h - 10), (w - 10, h - 4), (80, 80, 80), -1)
        cv2.rectangle(frame, (10, h - 10), (10 + bar_w, h - 4), (0, 200, 0), -1)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    out_dir = os.path.join(args.output, args.sign)
    os.makedirs(out_dir, exist_ok=True)

    # Conta sequências já existentes
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    print(f"[INFO] Sinal       : {args.sign}")
    print(f"[INFO] Sequências  : {args.sequences} novas (já existem: {existing})")
    print(f"[INFO] Frames/seq  : {args.length}")
    print(f"[INFO] Saída       : {out_dir}")
    print(f"\nControles: SPACE = iniciar | Q = sair | R = regravar última")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir câmera {args.cam}.")
        sys.exit(1)

    collected   = 0
    last_seq    = None   # guarda última sequência para R (regravar)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while collected < args.sequences:
            seq_num = existing + collected + 1
            print(f"\n--- Sequência {collected + 1}/{args.sequences} ---")

            # ── Fase de espera ────────────────────────────────────────────
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame   = cv2.flip(frame, 1)
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                det     = bool(results.multi_hand_landmarks)

                if det:
                    for lms in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, lms, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )

                draw_overlay(frame, args.sign, seq_num,
                             existing + args.sequences, 0,
                             args.length, "waiting", det, None)
                cv2.imshow("Coleta LIBRAS", frame)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                if key == ord(' '):   # SPACE
                    break
            else:
                break  # câmera perdida

            if key == ord('q'):
                print("[INFO] Coleta encerrada pelo usuário.")
                break

            # ── Fase de gravação ──────────────────────────────────────────
            sequence       = []
            frames_no_hand = 0

            for fi in range(args.length):
                ok, frame = cap.read()
                if not ok:
                    break
                frame   = cv2.flip(frame, 1)
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                det     = bool(results.multi_hand_landmarks)

                if det:
                    for lms in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, lms, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                else:
                    frames_no_hand += 1

                vec = extract_landmarks(results)
                sequence.append(vec)

                draw_overlay(frame, args.sign, seq_num,
                             existing + args.sequences,
                             fi + 1, args.length, "recording", det, None)
                cv2.imshow("Coleta LIBRAS", frame)
                cv2.waitKey(1)

            # ── Aviso se muitos frames sem mão ────────────────────────────
            pct_no_hand = frames_no_hand / max(args.length, 1) * 100
            if pct_no_hand > 30:
                print(f"[AVISO] {pct_no_hand:.0f}% dos frames sem mão detectada. "
                      f"Considere regravar (tecla R).")

            # ── Salva sequência ───────────────────────────────────────────
            arr       = np.array(sequence, dtype=np.float32)  # (T, 126)
            fname     = f"{args.sign}-{uuid.uuid4()}.npy"
            fpath     = os.path.join(out_dir, fname)
            np.save(fpath, arr)
            last_seq  = fpath
            collected += 1

            # ── Feedback de "salvo" ───────────────────────────────────────
            ok2, frame2 = cap.read()
            if ok2:
                frame2 = cv2.flip(frame2, 1)
                draw_overlay(frame2, args.sign, seq_num,
                             existing + args.sequences,
                             args.length, args.length, "done", True, None)
                cv2.imshow("Coleta LIBRAS", frame2)
                key2 = cv2.waitKey(600) & 0xFF

                # R = regravar a última (remove o arquivo salvo)
                if key2 == ord('r') and last_seq and os.path.exists(last_seq):
                    os.remove(last_seq)
                    collected -= 1
                    print("[INFO] Última sequência descartada — regravando.")

    cap.release()
    cv2.destroyAllWindows()

    final = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    print(f"\n[OK] Coleta concluída. Total de sequências em '{out_dir}': {final}")


if __name__ == "__main__":
    main()
