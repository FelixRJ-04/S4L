import socket
import json
import cv2 as cv
import base64
import time
import numpy as np
from keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic
from Dibuja import *
from Constantes import *
from Muestras import *

HOST = '127.0.0.1'
PORT = 12345
FPS_LIMIT = 120  

with open('diccionario.json', 'r') as f:
    diccionario = json.load(f)
model = load_model('./Modelo_Kevin.h5')

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Esperando conexión...")

    conn, addr = s.accept()
    with conn:
        print(f"Conectado con {addr}")
        with Holistic() as holistic_model:
            video = cv.VideoCapture(0)
            last_send_time = time.time()

            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break

                resultados = DeteccionMediapipe(frame, holistic_model)
                if hayMano(resultados):
                    # Controlar la tasa de envío de frames
                    if time.time() - last_send_time >= 1 / FPS_LIMIT:
                        kp_frame = extraeKeypoints(resultados)
                        kp_normalized = normalizar_keypoints([kp_frame], int(FramesF))
                        res = model.predict(np.expand_dims(kp_normalized, axis=0))[0]
                        palabra = diccionario[np.argmax(res)]

                        DibujaKeypoints(frame, resultados)

                        # Cambiar la calidad de imagen a 40
                        _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 40])
                        frame_data = base64.b64encode(buffer.tobytes()).decode('utf-8')

                        try:
                            data = json.dumps({
                                "prediccion": palabra,
                                "frame": frame_data
                            })
                            conn.sendall((data + "\n").encode('utf-8'))  
                            last_send_time = time.time()
                        except (BrokenPipeError, ConnectionResetError):
                            print("Conexión cerrada por el cliente. Finalizando...")
                            break

                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

            video.release()



    