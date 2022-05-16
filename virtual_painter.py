from turtle import distance

from pyparsing import ParseExpression
import fingertrackingmodule as ftm
import cv2
import os
import numpy as np
import time

BRUSH_THICKNESS = 25
ERASER_THICK0NESS = 100
DRAW = False
ERASE = False

DEBUG = True
drawColor = np.zeros(1, np.uint8)
FOLDER_PATH = "Header"

drawColor = [0, 0, 0]
headers_img = []               
listHeader = os.listdir(FOLDER_PATH)
if DEBUG:
    print(listHeader)
for imgPath in listHeader:
    image = cv2.imread(FOLDER_PATH+'/'+imgPath)
    headers_img.append(image)
header = headers_img[-1]


cap = cv2.VideoCapture(0)
width = 1920
height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
imgCanvas = np.zeros((height, width, 3), np.uint8)

cv2.namedWindow("Painter", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Painter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
prevTime = time.time()
fps = 0
p1 = 4 
p2 = 8
xp, yp = 0, 0
detector = ftm.fingerDetector()
while cap.isOpened():
        success, image = cap.read()
        if not success:
            print('Не удалось получить кадр с web-камеры')
            continue
        image = cv2.flip(image, 1)
        if DEBUG:
            cv2.putText(image, f'FPS: {int(fps)}', (20,50) ,cv2.FONT_HERSHEY_PLAIN, 3,(0,100,250), 5 )
        
        h, w, c = header.shape
        detector.findHands(image)
        mhl = detector.result.multi_hand_landmarks
        if mhl:
            handCount = len(mhl)
            for i in range(handCount):
                detector.findPosition(image, i)
                x1, y1 = detector.handList[i][p1][0],detector.handList[i][p1][1]
                x2, y2 = detector.handList[i][p2][0],detector.handList[i][p2][1]
                cx, cy = (x1 + x2) // 2, (y1 + y2) //2
                cv2.circle(image, (cx, cy), 15, drawColor, cv2.FILLED)

                distance = detector.findDistance(p1, p2, i)
                if distance < 50:
                    if cy <= h:
                        if 205 <= cx <= 500:
                            header = headers_img[0]
                            DRAW = True
                            ERASE = False
                            drawColor = (0, 0, 255)
                        elif 560 <= cx <= 847:
                            header = headers_img[1]
                            DRAW = True
                            ERASE = False
                            drawColor = (255, 0, 0)
                        elif 900 <= cx <= 1190:
                            header = headers_img[2]
                            DRAW = True
                            ERASE = False
                            drawColor = (0, 255, 0)
            
                        elif 1152 <= cx <= 1495:
                            header = headers_img[3]
                            DRAW = False
                            ERASE = True
                            drawColor = (0, 0, 0)
                        elif 1600 <= cx <= 2000:
                            header = headers_img[4]
                            DRAW = False
                            ERASE = False
                    elif DRAW:
                        if xp == 0 and yp == 0:
                            xp, yp = cx, cy 
                        else:
                            cv2.line(image, (xp, yp), (cx, cy), drawColor, BRUSH_THICKNESS)
                            cv2.line(imgCanvas, (xp, yp), (cx, cy), drawColor, BRUSH_THICKNESS)
                    elif ERASE:
                        if xp == 0 and yp == 0:
                            xp, yp = cx, cy 
                        else:
                            cv2.line(image, (xp, yp), (cx, cy), drawColor, ERASER_THICK0NESS)
                            cv2.line(imgCanvas, (xp, yp), (cx, cy), drawColor, ERASER_THICK0NESS)
                xp, yp = cx, cy
                        
                        
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)  
        _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR) 
        image = cv2.bitwise_and(image, imgInv)
        image = cv2.bitwise_or(image, imgCanvas)


        image[0:h, 0:w] = header
        cv2.imshow('Painter', image)
        curretTime = time.time()
        fps = 1 / (curretTime - prevTime)
        prevTime = curretTime
        if cv2.waitKey(1) & 0xFF == 27:
            break
