# hands_process_video.py
# -*- coding: utf-8 -*-
"""
Read an MP4, detect hands with MediaPipe, draw landmarks, and export an MP4.
"""

import sys
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    sys.stderr.write(
        "ERROR: mediapipe not found. Install with:\n"
        "  pip install mediapipe opencv-python\n"
    )
    raise


def main():
    # ======= DEFINA SEUS PARÂMETROS AQUI =======
    input_path = "MINDS/07BancoSinalizador08-1.mp4"     # caminho do vídeo de entrada
    output_path = "mindsTESTS/Qual.mp4"   # caminho do vídeo de saída
    max_hands = 2
    min_detect = 0.5
    min_track = 0.5
    no_draw = False
    # ===========================================

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.stderr.write(f"ERROR: could not open input: {input_path}\n")
        sys.exit(1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0  # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MP4-compatible codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        sys.stderr.write(f"ERROR: could not open output: {output_path}\n")
        sys.exit(1)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Hands detection
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=min_detect,
        min_tracking_confidence=min_track,
        model_complexity=1
    )

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = hands.process(frame_rgb)

            if not no_draw and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                    )

            out.write(frame_bgr)
            frame_idx += 1

    finally:
        hands.close()
        cap.release()
        out.release()

    print(f"Done. Wrote {frame_idx} frames to: {output_path} (FPS={fps:.2f}, {width}x{height})")


if __name__ == "__main__":
    main()
