import cv2 as cv
import numpy as np
import os 

from mediapipe.python.solutions.holistic import Holistic
from datetime import datetime
from Dibuja import *
from typing import NamedTuple
from Constantes import *

def creaCarpeta(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)
        
def hayMano(results: NamedTuple) -> bool:
    return results.left_hand_landmarks or results.right_hand_landmarks


def muestras(ruta, margenFrame =1, maxFrames=29, delayFrames=3):
    creaCarpeta(ruta)
    frames=[]
    framesComp=0
    cuentaFrame=0
    recording=False
    cantidadMuestras=0 
    
    with Holistic() as holistic_model:
        camera=cv.VideoCapture(0)
        
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret: 
                break
            
            img =frame.copy()
            resultados= DeteccionMediapipe(frame, holistic_model)
            
            if hayMano(resultados) or recording:
                recording=False
                cuentaFrame += 1
                if cuentaFrame > margenFrame:
                     cv.putText(img, 'Almacenando', FONT_POS, FONT, FONT_SIZE, (255, 255, 255), thickness=1)
                     frames.append(np.asarray(frame))
                     cFrames=len(frames)
                     print(f"La cantidad de frames es: {cFrames}")
            else:
                if len(frames) >= maxFrames + margenFrame:
                    framesComp +=1 
                    if framesComp < delayFrames:
                        recording =True
                        continue
                    framesIndice= frames[0:(margenFrame+ maxFrames)]
                    fecha = datetime.now().strftime('%y%m%d%H%M%S%f')
                    carpetaSalida = os.path.join(ruta, f"Muestra_{fecha}")
                    creaCarpeta(carpetaSalida)
                    guardaFrames(framesIndice, carpetaSalida)
                    contenido=os.path.join('./frame_actions_5', f"{Palabra}")
                    contenidoD= os.listdir(contenido)
                    contenidoL = np.array(contenidoD, dtype=str)
                    cantidadMuestras=len(contenidoL)
                    cv.imshow('Img', frames[0])
                    print(f"La cantidad de muestras en la carpeta {Palabra} es {cantidadMuestras}")
                        
                recording, framesComp = False, 0
                frames, cuentaFrame=[], 0
                cv.putText(img, 'Listo para capturar de nuevo...', FONT_POS, FONT, FONT_SIZE, (0, 177, 90))
                    
            DibujaKeypoints(img, resultados)
            cv.imshow(f'Toma de muestras para "{os.path.basename(ruta)}"', img)
            if cv.waitKey(10)&0xFF ==ord('q'):
                break
            
        camera.release()
        cv.destroyAllWindows()
        
if __name__ == "__main__":
    Palabra = "x"
    rutaPalabra = os.path.join(RutaRaiz, RutaFrameActions, Palabra)
    muestras(rutaPalabra)
