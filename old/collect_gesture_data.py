import cv2
import mediapipe as mp
import numpy as np
import os
import uuid

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

DATA_DIR = 'libras_data'
SIGN_NAME = 'Vontade' # Este será o nome do gesto
NUM_SEQUENCES = 30
SEQUENCE_LENGTH = 15

os.makedirs(os.path.join(DATA_DIR, SIGN_NAME), exist_ok=True)

cap = cv2.VideoCapture(0)

for sequence in range(NUM_SEQUENCES):
    sequence_data = []
    print(f"Coletando dados para o sinal '{SIGN_NAME}', sequência {sequence+1}/{NUM_SEQUENCES}")
    
    for _ in range(30):
        ret, frame = cap.read()
        if not ret: break
        frame_flipped = cv2.flip(frame, 1)
        cv2.putText(frame_flipped, 'Preparar...', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow('Coleta de Dados', frame_flipped)
        cv2.waitKey(10)
        
    for frame_num in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        if not ret: break

        frame_flipped = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        current_frame_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_flipped,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                for landmark in hand_landmarks.landmark:
                    current_frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        if len(current_frame_landmarks) < 126:
            current_frame_landmarks.extend([0.0] * (126 - len(current_frame_landmarks)))
            
        sequence_data.append(current_frame_landmarks[:126])
        
        cv2.putText(
            frame_flipped,
            f'Sinal: {SIGN_NAME} Sequencia: {sequence+1} Frame: {frame_num+1}',
            (15, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )
        cv2.imshow('Coleta de Dados', frame_flipped)
        cv2.waitKey(10)
        
    # Gerar um UUID único para o nome do arquivo
    file_name = f'{SIGN_NAME}-{uuid.uuid4()}.npy'
    np.save(os.path.join(DATA_DIR, SIGN_NAME, file_name), np.array(sequence_data))

cap.release()
cv2.destroyAllWindows()
print("Coleta de dados concluída.")