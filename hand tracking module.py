from sre_constants import SUCCESS
from unittest import result
import cv2
import mediapipe as mp 
import time 

#this part run the webcam 

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,TrackCon=0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.TrackCon = TrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.detectionCon,self.TrackCon)
        self.mpDrow = mp.solutions.drawing_utils

    def findHands(self,img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print (result.multi_hand_landmarks)
        #lm = landmarks : the point on hand 
        if self.results.multi_hand_landmarks:
          for handLms in self.results.multi_hand_landmarks:
            if draw:
                self.mpDrow.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img


    def findPosition(self, img, handNo=8, drow=True):

        lmList = [ ]
        if self.results.multi_hand_landmarks:
           myHand=self.results.multi_hand_landmarks[handNo]
           for id,lm in enumerate(myHand.landmark):
                    #print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                    #print(id, cx , cy )
                lmList.append([id,cx,cy])
                if drow:
                  cv2.circle(img, (cx,cy), 10, ( 201, 18, 24), cv2.FILLED)
        return lmList


def main():
    pTime   = 0 
    cTime   = 0
    cap     = cv2.VideoCapture(1)
    detector = handDetector()

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


if __name__ == '__main__':
    main()
