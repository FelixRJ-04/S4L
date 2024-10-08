import os
import cv2 as cv

FONT = cv.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

# SETTINGS
MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 1662
MODEL_FRAMES = 30

# PATHS
RutaRaiz = os.getcwd()
RutaFrameActions = os.path.join(RutaRaiz, "frame_actions_5")
RutaDatos = os.path.join(RutaRaiz, "data_5M")
RutaKeypoints = os.path.join(RutaDatos, "keypoints")