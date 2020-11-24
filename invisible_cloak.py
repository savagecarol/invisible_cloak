import cv2
import numpy as np

cap = cv2.VideoCapture(0)
back = cv2.imread("image.jpg")


while cap.isOpened():
    #take each frame
    ret, frame = cap.read()
    if ret:
        
