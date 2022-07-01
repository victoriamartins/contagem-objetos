import cv2
import imutils
import numpy as np


path = 'imgs/5.jpeg'
imagem = cv2.imread(path, 0)

val, otsu = cv2.threshold(imagem, 100, 255, cv2.THRESH_OTSU)

kernel = np.ones((5, 1))
er = cv2.erode(otsu, kernel, iterations=4)

kernel = np.ones((2, 2))
gradient = cv2.morphologyEx(er, cv2.MORPH_GRADIENT, kernel, iterations=1)

cnts = cv2.findContours(gradient.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
objects = str(len(cnts))
print(objects)

cv2.imshow('Contornos', gradient)
cv2.waitKey(0)
