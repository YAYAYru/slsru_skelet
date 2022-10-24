import cv2
import math
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import mediapipe as mp

@dataclass
class ViewOpenCV:
    sequence = []
    sentence = []
    colors = [(245,117,16), (117,245,16), (16,117,245)]

    def __init__(self,path):
        self.cap = cv2.VideoCapture(path)
        


    def part1_process(self):
        success, img = self.cap.read()
        img = cv2.resize(img, (1920, 1080))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def prob_viz(self, res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)        
        return output_frame        

    def paint_1tv_process(self, img, labels=["bbb","aaa","ccc"], res=[0.9,0.05,0.05], threshold=0.8):

        #3. Viz logic
        if res[np.argmax(res)] > threshold: 
            if len(self.sentence) > 0: 
                if labels[np.argmax(res)] != self.sentence[-1]:
                    self.sentence.append(labels[np.argmax(res)])
            else:
                 self.sentence.append(labels[np.argmax(res)])

        if len(labels) > 5: 
            sentence = labels[-5:]
        

        img = self.prob_viz(res, labels, img, self.colors)

        cv2.rectangle(img, (0,0), (640, 40), (245, 117, 16), -1)
        #cv2.putText(img, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, ' '.join(self.sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return img

    def part2_process(self, img):
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    def key_q(self):        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            return True

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()




        

    
    def show(self, ):
        if self.path_file == "":
            print("Input path to file")
            exit()
        cap = cv2.VideoCapture(self.path_file)
        while cap.isOpened():
            ret, frame = cap.read()

            cv2.imshow('OpenCV Feed', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


def draw_point(points, img, width, height):
    i = 0
    for f in range(points.shape[0]):
        i += 1
        if i == 1:
            if not np.isnan(points[f]):
                cv2.circle(img, (int(points[f] * width), int(points[f + 1] * height)), 1, (255, 255, 255), -1)
            i += 1
        if i == 5:
            i = 0

def show_from_csv(df: pd.DataFrame, output_type="hands+pose", where_left="left"):
    mp_drawing = mp.solutions.drawing_utils
    if output_type == "full":
        width = int(df["resolution_width"][0])
        height = int(df["resolution_height"][0])
        rows = int(df.shape[0])
        fps = df["fps"][0]
        face_p = df.loc[0:rows, "face_x0":"face_p467"].values
        pose_p = df.loc[0:rows, "pose_x0":"pose_p31"].values
        lhand_p = df.loc[0:rows, "lhand_x0":"lhand_p20"].values
        rhand_p = df.loc[0:rows, "rhand_x0":"rhand_p20"].values
        points_arr = [face_p, pose_p, lhand_p, rhand_p]
        print(face_p.shape)
        print(pose_p.shape)
        print(lhand_p.shape)
        print(rhand_p.shape)
        mp_holistic = mp.solutions.holistic
        if output_type == "full":
            i = 0
            for i in range(rows):

                black = np.zeros((height, width, 3))
                draw_point(points_arr[0][i], black, width, height)
                draw_point(points_arr[1][i], black, width, height)
                draw_point(points_arr[2][i], black, width, height)
                draw_point(points_arr[3][i], black, width, height)
                # black = cv2.resize(black, (400,400))
                black = cv2.resize(black, (1280, 720))
                if where_left == "right":
                    cv2.imshow("black_full", cv2.flip(black, 1))
                if where_left == "left":
                    cv2.imshow("black_full", black)
                key = cv2.waitKey(1)
                time.sleep(1 / fps)
                if key == ord("q"):
                    break
    if output_type == "hands+pose":
        width = int(df["resolution_width"][0])
        height = int(df["resolution_height"][0])
        rows = int(df.shape[0])
        fps = df["fps"][0]
        pose_p = df.loc[0:rows, "pose_x0":"pose_p31"].values
        lhand_p = df.loc[0:rows, "lhand_x0":"lhand_p20"].values
        rhand_p = df.loc[0:rows, "rhand_x0":"rhand_p20"].values
        points_arr = [pose_p, lhand_p, rhand_p]
        # print(face_p.shape)
        print(pose_p.shape)
        print(lhand_p.shape)
        print(rhand_p.shape)
        mp_holistic = mp.solutions.holistic
        if output_type == "hands+pose":
            i = 0
            for i in range(rows):

                black = np.zeros((height, width, 3))
                draw_point(points_arr[0][i], black, width, height)
                draw_point(points_arr[1][i], black, width, height)
                draw_point(points_arr[2][i], black, width, height)
                # black = cv2.resize(black, (400,400))
                black = cv2.resize(black, (1280, 720))
                if where_left == "right":
                    cv2.imshow("black_full", cv2.flip(black, 1))
                if where_left == "left":
                    cv2.imshow("black_full", black)
                key = cv2.waitKey(1)
                time.sleep(1 / fps)
                if key == ord("q"):
                    break
cv2.destroyAllWindows()

if __name__=="__main__":
    PATH_CSV = ""
    df = pd.read_csv("data/csv/12_s10020_9_1.mp4.csv")
    show_from_csv(df)