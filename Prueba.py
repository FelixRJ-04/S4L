import cv2 as cv
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2 
import numpy as np

camera = cv.VideoCapture(0)

while camera.isOpened():
    foto, ret = camera.read()
    cv.imshow('Ready', ret)
    key= cv.waitKey(1)
    if key == 27:
        break
        
camera.release()
cv.destroyAllWindows()
