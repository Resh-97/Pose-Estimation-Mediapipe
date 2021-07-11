import cv2
import mediapipe as mp
import time
from PoseEstimationModule import poseDetector

cam = cv2.VideoCapture('PoseVideos/3.mp4')

prevTime = 0
currTime = 0
detector = poseDetector()

while True:
    success, img = cam.read()
    img = cv2.resize(img, (640,480))
    img = detector.findPose(img)
    lmList = detector.getPosition(img, draw=False)
    if len(lmList) != 0 :
        print(lmList[14])
        cv2.circle(img,(lmList[14][1],lmList[14][2]),15, (0,0,255), cv2.FILLED)
    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3 )
    cv2.imshow("Image",img)
    cv2.waitKey(1)
