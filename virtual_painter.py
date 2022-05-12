import fingertrackingmodule as ftm
import cv2
import os
import numpy as np
import time

DEBUG = True
FOLDER_PATH = "Header"

listHeader = os.listdir(FOLDER_PATH)
if DEBUG:
    print(listHeader)

cap = cv2.VideoCapture(0)
width = 1920
height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

cv2.namedWindow("Painter", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Painter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
prevTime = time.time()
fps = 0
while cap.isOpened():
        success, image = cap.read()
        if not success:
            print('Не удалось получить кадр с web-камеры')
            continue
        image = cv2.flip(image, 1)
        if DEBUG:
            cv2.putText(image, f'FPS: {int(fps)}', (20,50) ,cv2.FONT_HERSHEY_PLAIN, 3,(0,100,250), 5 )
        cv2.imshow('Painter', image)
        curretTime = time.time()
        fps = 1 / (curretTime - prevTime)
        prevTime = curretTime
        if cv2.waitKey(1) & 0xFF == 27:
            break
