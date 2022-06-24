import cv2
import mediapipe as mp 
import time 
import handTrackingModule as htm
pTime   = 0 
cTime   = 0
cap     = cv2.VideoCapture(1)
detector= htm.handDetector()
while True:
    SUCCESS, img = cap.read()
    img   = detector.findHands(img)
    lmList= detector.findPosition(img)
    if len(lmList) !=0:
        print([4])
    cTime = time.time()
    fps   = 1/(cTime-pTime)
    pTime = cTime


    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,3,(69, 150, 255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)