import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
##Static mode if true then the model will always detect. If False: When the confidence
## is more than than 0.5 then it goes for tracking and when tracking is more
## than 0.5 it does to detection
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cam = cv2.VideoCapture('PoseVideos/3.mp4')
prevTime = 0
currTime = 0
while True:
    success, img = cam.read()
    img = cv2.resize(img,(640,480))
    imgRGB =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            x_center, y_center = int(lm.x * width), int(lm.y * height)
            print(id,x_center, y_center)
            cv2.circle(img,(x_center,y_center),3, (255,0,0), cv2.FILLED)

    currTime =time.time()
    fps = 1/ (currTime - prevTime)
    prevTime = currTime


    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
