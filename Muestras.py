import cv2 as cv
import numpy as np
import os 

from mediapipe.python.solutions.holistic import Holistic
from datetime import datetime
from Dibuja import draw_keypoints, mediapipe_detection, guardaFrames
from typing import NamedTuple
from Constantes import *

FONT = cv.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 1662
MODEL_FRAMES = 15


def creaCarpeta(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def hayMano(results: NamedTuple) -> bool:
    return results.left_hand_landmarks or results.right_hand_landmarks


def muestras(path, margenFrame =1, maxFrames=29, delayFrames=3):
    creaCarpeta(path)
    frames=[]
    framesComp=0
    cuentaFrame=0
    recording=False
    
    with Holistic() as holistic_model:
        camera=cv.VideoCapture(0)
        
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret: 
                break
            
            img =frame.copy()
            results= mediapipe_detection(frame, holistic_model)
            
            if hayMano(results) or recording:
                recording=False
                cuentaFrame += 1
                if cuentaFrame > margenFrame:
                     cv.putText(img, 'Almacenando', FONT_POS, FONT, FONT_SIZE, (255, 255, 255), thickness=1)
                     frames.append(np.asarray(frame))
                     c_frames=len(frames)
                     print(f"La cantidad de frames es: {c_frames}")
            else:
                if len(frames) >= maxFrames + margenFrame:
                    framesComp +=1 
                    if framesComp < delayFrames:
                        recording =True
                        continue
                    frames_i= frames[0:(margenFrame+ maxFrames)]
                    cv.imshow('Img', frames[0])
                    today = datetime.now().strftime('%y%m%d%H%M%S%f')
                    carpetaSalida = os.path.join(path, f"Muestra_{today}")
                    creaCarpeta(carpetaSalida)
                    guardaFrames(frames_i, carpetaSalida)
                        
                recording, framesComp = False, 0
                frames, cuentaFrame=[], 0
                cv.putText(img, 'Listo para capturar de nuevo...', FONT_POS, FONT, FONT_SIZE, (0, 177, 90))
                    
            draw_keypoints(img, results)
            cv.imshow(f'Toma de muestras para "{os.path.basename(path)}"', img)
            if cv.waitKey(10)&0xFF ==ord('q'):
                break
            
        camera.release()
        cv.destroyAllWindows()
        
if __name__ == "__main__":
    word_name = "Pan"
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    muestras(word_path)
