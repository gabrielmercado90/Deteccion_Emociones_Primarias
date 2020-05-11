import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import EmotionClassifier
import cv2

detector = EmotionClassifier('modelo/model.json')
detector.load_weights('modelo/pesos_08.05.2020-17_06_23.H5')

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, frame = video.read()
    if frame is not None:
        output = detector.predict(frame)
        #mostrar
        cv2.imshow('capture', output) 

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break


video.release()
cv2.destroyAllWindows()