import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from Dibuja import *
from Constantes  import *
from Muestras import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gtts import gTTS
import os
import pygame
from time import sleep
    
def evaluar_model(src=None, threshold=0.8, margin_frame=1, delay_frames=3):
    kp_seq, sentence = [], []
    contenido=os.listdir('./data_5M/keypoints')
    contenido = np.array(contenido, dtype=str)
    word_ids =[]
    for NumPalabra, Palabra in enumerate(contenido):
        Palabra = contenido[NumPalabra]
        Palabra_limpia = Palabra.replace(".h5", "").replace("1", "").replace("2", "").replace("3", "")
        word_ids.append(Palabra_limpia)
    model = load_model('./Modelo_Felix.h5')
    count_frame, fix_frames, recording= 0, 0, False
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(src or 0)
        while video.isOpened():
            ret, frame = video.read()
            if not ret: break
            results = DeteccionMediapipe(frame, holistic_model)
            if hayMano(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    kp_frame = extraeKeypoints(results)
                    kp_seq.append(kp_frame)
            else:
                if count_frame >= LongMinFrames + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    kp_seq = kp_seq[: - (margin_frame + delay_frames)]
                    kp_normalized = normalizar_keypoints(kp_seq, int(FramesF))
                    res = model.predict(np.expand_dims(kp_normalized, axis=0))[0]
                    print(np.argmax(res), f"({res[np.argmax(res)] * 100:.2f}%)")
                    if res[np.argmax(res)] > threshold:
                        sent = word_ids[np.argmax(res)]
                        sentence.insert(0, sent)
                        text_to_speech(sent)
                recording, fix_frames, count_frame, kp_seq = False, 0, 0, []
            if not src:
                cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
                cv2.putText(frame, ' | '.join(sentence), PosicionFuente, Fuente, TamFuente, (255, 255, 255))
                DibujaKeypoints(frame, results)
                cv2.imshow('Traductor LSM', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        video.release()
        cv2.destroyAllWindows()
        return sentence
if __name__ == "__main__":
    evaluar_model()