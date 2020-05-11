import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import EmotionClassifier


#leer imagen en carpeta
img = cv2.imread('im_pruebas/prueba.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detector = EmotionClassifier('modelo/model.json')
detector.load_weights('modelo/pesos_08.05.2020-17_06_23.H5')
output = detector.predict(img)

plt.imshow(output), plt.show()