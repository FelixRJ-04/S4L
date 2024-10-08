import os
import cv2 as cv
import numpy as np
import pandas as pd
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