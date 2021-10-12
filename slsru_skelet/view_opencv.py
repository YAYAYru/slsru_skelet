import cv2
import math
from dataclasses import dataclass
import numpy as np

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
