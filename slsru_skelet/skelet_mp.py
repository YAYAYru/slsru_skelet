import mediapipe as mp
import cv2
import numpy as np

class SkeletAllMediapipe:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHolistic = mp.solutions.holistic
        #self.holistic = self.mpHolistic.Holistic(self.mode, self.maxHands,self.detectionCon, self.trackCon)
        self.holistic = self.mpHolistic.Holistic(min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_skelet(self, img, pose_draw=True, hand_draw=True):
        #imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #self.results = self.holistic.process(imgRGB)
        self.results = self.holistic.process(img)
        #self.mpDraw.draw_landmarks(img, self.results.face_landmarks, self.mpHolistic.FACEMESH_TESSELATION)
        if pose_draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpHolistic.POSE_CONNECTIONS)
        if hand_draw:
            self.mpDraw.draw_landmarks(img, self.results.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)
            self.mpDraw.draw_landmarks(img, self.results.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)

            
            #for handLms in self.results.pose_landmarks:
                #print(self.results.pose_landmarks)
                #if draw:
                    #self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def extract_keypoints(self):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in self.results.pose_landmarks.landmark]).flatten() if self.results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in self.results.face_landmarks.landmark]).flatten() if self.results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in self.results.left_hand_landmarks.landmark]).flatten() if self.results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in self.results.right_hand_landmarks.landmark]).flatten() if self.results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

class SkeletPoseMediapipe:
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)
    def find_skelet(self, img, draw=True):
        #imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #self.results = self.pose.process(imgRGB)
        self.results = self.pose.process(img)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
                
        return img 

    def findPosition(self, img, pose_draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if pose_draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    def extract_keypoints(self):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in self.results.pose_landmarks.landmark]).flatten() if self.results.pose_landmarks else np.zeros(33*4)
        return np.concatenate([pose])

class SkeletHandMediapipe:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_skelet(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
        
