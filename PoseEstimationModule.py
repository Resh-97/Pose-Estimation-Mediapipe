import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture('PoseVideos/3.mp4')

class poseDetector():

    def __init__(self, mode =False, model_complexity = 1, smooth =True,detectionConfidence = 0.5, trackConfidence = 0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.model_complexity,self.smooth,self.detectionConfidence,self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self,img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self,img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                x_center, y_center = int(lm.x * width), int(lm.y * height)
                lmList.append([id,x_center, y_center])
                if draw:
                    cv2.circle(img,(x_center,y_center),3, (255,0,0), cv2.FILLED)
        return lmList

def main():
    prevTime = 0
    currTime = 0
    detector = poseDetector()

    while True:
        success, img = cam.read()
        #img = cv2.resize(img, (640,480))
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











if __name__ == '__main__':
    main()
