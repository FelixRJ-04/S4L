import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
from collections import deque

# Cargar el modelo entrenado
model = load_model('Modelo_Prueba_LSTM.h5')

# Cargar las etiquetas únicas
etiquetas_unicas = np.load('etiquetas_unicas.npy', allow_pickle=True)

# Inicializar MediaPipe para la detección de cuerpo, cara y manos
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Configurar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Crear un buffer para almacenar secuencias de 30 frames (o los timesteps que utilizaste)
sequence_length = 30  # Asegúrate de que coincide con el entrenamiento
keypoints_sequence = deque(maxlen=sequence_length)

# Definir el número de características esperadas por el modelo (1665 en este caso para x, y, z de 555 puntos en total)
n_features_expected = 1662

with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen a RGB para Mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(image_rgb)

        # Inicializar listas vacías para los keypoints de cada parte
        keypoints = []

        # Extraer keypoints de cada parte si están disponibles
        if result.pose_landmarks:
            keypoints_pose = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark]).flatten()
            keypoints.append(keypoints_pose)

        if result.face_landmarks:
            keypoints_face = np.array([[lm.x, lm.y, lm.z] for lm in result.face_landmarks.landmark]).flatten()
            keypoints.append(keypoints_face)

        if result.left_hand_landmarks:
            keypoints_left_hand = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark]).flatten()
            keypoints.append(keypoints_left_hand)

        if result.right_hand_landmarks:
            keypoints_right_hand = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark]).flatten()
            keypoints.append(keypoints_right_hand)

        # Concatenar todos los keypoints y aplanar la lista
        if len(keypoints) > 0:
            keypoints = np.concatenate(keypoints).flatten()

            # Rellenar con ceros si es necesario para igualar el número de características esperadas
            if len(keypoints) < n_features_expected:
                padding = np.zeros(n_features_expected - len(keypoints))
                keypoints = np.concatenate([keypoints, padding])

            # Añadir los keypoints al buffer de la secuencia
            keypoints_sequence.append(keypoints)

        # Verificar que tenemos suficientes frames en la secuencia
        if len(keypoints_sequence) == sequence_length:
            # Convertir la secuencia en un array de numpy con la forma correcta (1, 30, 1665)
            input_data = np.array(keypoints_sequence)  # (30, 1665)
            input_data = np.expand_dims(input_data, axis=0)  # (1, 30, 1665)

            # Hacer la predicción
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)

            # Mostrar el nombre de la seña correspondiente
            palabra = etiquetas_unicas[predicted_class]
            cv2.putText(frame, f"Seña: {palabra}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Dibujar los puntos en el cuerpo, cara y manos con tamaños más pequeños
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
        if result.face_landmarks:
            mp_drawing.draw_landmarks(frame, result.face_landmarks, None,  # No hay conexiones para la cara
                                      mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1))
        if result.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2))
        if result.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2))

        # Mostrar la imagen con las predicciones
        cv2.imshow('Detección de Senas', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Presionar ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
