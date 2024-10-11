import os
import cv2 as cv
import numpy as np
import pandas as pd
import pygame
from gtts import gTTS
from time import sleep
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec

def DibujaKeypoints(imagen, resultados):
    draw_landmarks(
        imagen,
        resultados.face_landmarks,
        FACEMESH_CONTOURS,
        DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    #CONEXIONES DE POSE
    draw_landmarks(
        imagen,
        resultados.pose_landmarks,
        POSE_CONNECTIONS,
        DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    #CONEXIONES DE MANO IZQUIERDA
    draw_landmarks(
        imagen,
        resultados.left_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    # CONEXIONES DE MANO DERECHA
    draw_landmarks(
        imagen,
        resultados.right_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )

def guardaFrames(frames, OutputFolder):
    for numFrame, frame in enumerate(frames):
        rutaFrame = os.path.join(OutputFolder, f"{numFrame + 1}.jpg")
        cv.imwrite(rutaFrame, cv.cvtColor(frame, cv.COLOR_BGR2BGRA))
        
def extraeKeypoints(resultados):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in resultados.pose_landmarks.landmark]).flatten() if resultados.pose_landmarks else np.zeros(33*4)
    cara = np.array([[res.x, res.y, res.z] for res in resultados.face_landmarks.landmark]).flatten() if resultados.face_landmarks else np.zeros(468*3)
    mI = np.array([[res.x, res.y, res.z] for res in resultados.left_hand_landmarks.landmark]).flatten() if resultados.left_hand_landmarks else np.zeros(21*3)
    mD = np.array([[res.x, res.y, res.z] for res in resultados.right_hand_landmarks.landmark]).flatten() if resultados.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, cara, mI, mD])

def ObtenKeypoints(modelo, rutaMuestra):
    kp_seq = np.array([])
    for nomImg in os.listdir(rutaMuestra):
        rutaImg = os.path.join(rutaMuestra, nomImg)
        frame = cv.imread(rutaImg)
        results = DeteccionMediapipe(frame, modelo)
        FrameKeypoints = extraeKeypoints(results)
        SecKeypopints = np.concatenate([kp_seq, [FrameKeypoints]] if kp_seq.size > 0 else [[FrameKeypoints]])
    return SecKeypopints

def InsertaSecuenciaKeypoints(df, NumMuestra:int, SecKeypoints):
    for frame, keypoints in enumerate(SecKeypoints):
        data = {'sample': NumMuestra, 'frame': frame + 1, 'keypoints': [keypoints]}
        dfKeypoints = pd.DataFrame(data)
        df = pd.concat([df, dfKeypoints])
    return df

def DeteccionMediapipe(imagen, model):
    imagen = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
    imagen.flags.writeable = False
    resultados = model.process(imagen)
    return resultados

def text_to_speech(text):
    tts = gTTS(text=text, lang='es')
    filename = "speech.mp3"
    tts.save(filename)
    
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        sleep(1)
    pygame.mixer.quit()
    pygame.quit()
    os.remove(filename)
    
def interpolar_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints
    indices = np.linspace(0, current_length - 1, target_length)
    interpolated_keypoints = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        if lower_idx == upper_idx:
            interpolated_keypoints.append(keypoints[lower_idx])
        else:
            interpolated_point = (1 - weight) * np.array(keypoints[lower_idx]) + weight * np.array(keypoints[upper_idx])
            interpolated_keypoints.append(interpolated_point.tolist())
    return interpolated_keypoints

def normalizar_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length < target_length:
        return interpolar_keypoints(keypoints, target_length)
    elif current_length > target_length:
        step = current_length / target_length
        indices = np.arange(0, current_length, step).astype(int)[:target_length]
        return [keypoints[i] for i in indices]
    else:
        return keypoints