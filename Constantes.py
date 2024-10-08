import os
import cv2 as cv

Fuente = cv.FONT_HERSHEY_PLAIN
TamFuente = 1.5
PosicionFuente = (5, 30)

# SETTINGS
LongMinFrames = 5
LongKeypoints = 1662
FramesF = 30

# PATHS
RutaRaiz = os.getcwd()
RutaFrameActions = os.path.join(RutaRaiz, "frame_actions_5")
RutaDatos = os.path.join(RutaRaiz, "data_5M")
RutaKeypoints = os.path.join(RutaDatos, "keypoints")