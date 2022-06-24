from sre_constants import SUCCESS
from unittest import result
import cv2
import mediapipe as mp 
import time 

#this part run the webcam 

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDrow = mp.solutions.drawing_utils

pTime = 0 
cTime = 0

while True:
    SUCCESS, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
   # print (result.multi_hand_landmarks)
   # lm = landmarks :the point on the hand
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                #print(id,lm)

                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)

                print(id, cx , cy )
                #if id ==0:
                cv2.circle(img, (cx,cy), 10, ( 201, 18, 24), cv2.FILLED)

            mpDrow.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,3,(69, 150, 255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
