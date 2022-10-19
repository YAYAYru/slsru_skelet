# mediapipe v0.8
#from slsru_skelet.skelet_model import SkeletModel
from .skelet_model import SkeletModel
import mediapipe as mp
import cv2
import numpy as np
import csv
import math
import pandas as pd
import time


def clip(n):
    if n >= 1:
        return 1
    if n <= 0:
        return 0
    else:
        return n


def random_sigmoid(x):
    rs = 1 / (1 + math.exp(-x)) + np.random.uniform(0., 0.35)
    return clip(rs)


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

#BUG can not exit from window
def camera():
    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

class MediapipeModel_v08(SkeletModel):
    def __init__(self, filename_or_camera, output_type="pose", where_left="right"):
        super().__init__(filename_or_camera)
        self.output_name = None
        self.output_type = output_type
        self.frames_c = 0
        self.where_left = where_left
        self.processing_ = False
        if where_left not in ["left", "right"]:
            raise ValueError("where_left not in [left,right]")





    def processing(self):
        self.processing_ = True
        self.output_name = self.filename_or_camera.split(".")[0] + ".png"
        mp_drawing = mp.solutions.drawing_utils
        if ".png" in self.filename_or_camera:
            if self.output_type == "pose":
                self.mp_pose = mp.solutions.pose
                with self.mp_pose.Pose(
                        static_image_mode=True, min_detection_confidence=0.9
                ) as pose:
                    image = cv2.imread(self.filename_or_camera)
                    self.image_height, self.image_width, _ = image.shape
                    # Convert the BGR image to RGB before processing.
                    self.results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    if not self.results.pose_landmarks:
                        print("Точки не обнаружены")

                    # Draw pose landmarks on the image.
                    self.annotated_image = image.copy()
                    mp_drawing.draw_landmarks(
                        self.annotated_image,
                        self.results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                    )
                    self.points = [self.results.pose_landmarks]

            if self.output_type == "hands":
                self.mp_hands = mp.solutions.hands
                with self.mp_hands.Hands(
                        static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                ) as hands:

                    # Read an image, flip it around y-axis for correct handedness output (see
                    # above).
                    image = cv2.flip(cv2.imread(self.filename_or_camera), 1)
                    # Convert the BGR image to RGB before processing.
                    self.results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    # Print handedness and draw hand landmarks on the image.

                    if not self.results.multi_hand_landmarks:
                        print("Точки не обнаружены")
                    self.image_height, self.image_width, _ = image.shape
                    self.annotated_image = image.copy()
                    self.points = []
                    for hand_landmarks in self.results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            self.annotated_image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                        )
                        self.points.append(hand_landmarks)
            if self.output_type == "face":
                self.mp_face_mesh = mp.solutions.face_mesh
                drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
                with self.mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        min_detection_confidence=0.5,
                ) as face_mesh:
                    image = cv2.imread(self.filename_or_camera)
                    # Convert the BGR image to RGB before processing.
                    self.results = face_mesh.process(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    )
                    self.image_height, self.image_width, _ = image.shape
                    # Print and draw face mesh landmarks on the image.
                    if not self.results.multi_face_landmarks:
                        print("Точки не обнаружены")
                    self.annotated_image = image.copy()
                    self.points = []
                    for face_landmarks in self.results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=self.annotated_image,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec,
                        )
                        self.points.append(face_landmarks)
            if self.output_type == "full":
                self.mp_holistic = mp.solutions.holistic
                with self.mp_holistic.Holistic(static_image_mode=True) as holistic:
                    image = cv2.imread(self.filename_or_camera)
                    self.image_height, self.image_width, _ = image.shape
                    # Convert the BGR image to RGB before processing.
                    self.results = holistic.process(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    )
                    self.annotated_image = image.copy()
                    mp_drawing.draw_landmarks(
                        self.annotated_image,
                        self.results.face_landmarks,
                        self.mp_holistic.FACEMESH_TESSELATION,
                    )
                    mp_drawing.draw_landmarks(
                        self.annotated_image,
                        self.results.left_hand_landmarks,
                        self.mp_holistic.HAND_CONNECTIONS,
                    )
                    mp_drawing.draw_landmarks(
                        self.annotated_image,
                        self.results.right_hand_landmarks,
                        self.mp_holistic.HAND_CONNECTIONS,
                    )
                    mp_drawing.draw_landmarks(
                        self.annotated_image,
                        self.results.pose_landmarks,
                        self.mp_holistic.POSE_CONNECTIONS,
                    )

                    self.points = [
                        self.results.face_landmarks,
                        self.results.left_hand_landmarks,
                        self.results.right_hand_landmarks,
                        self.results.pose_landmarks,
                    ]
            if self.output_type == "hands+pose":
                self.mp_holistic = mp.solutions.holistic
                with self.mp_holistic.Holistic(static_image_mode=True) as holistic:
                    image = cv2.imread(self.filename_or_camera)
                    self.image_height, self.image_width, _ = image.shape
                    # Convert the BGR image to RGB before processing.
                    self.results = holistic.process(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    )
                    mp_drawing.draw_landmarks(
                        self.annotated_image,
                        self.results.left_hand_landmarks,
                        self.mp_holistic.HAND_CONNECTIONS,
                    )
                    mp_drawing.draw_landmarks(
                        self.annotated_image,
                        self.results.right_hand_landmarks,
                        self.mp_holistic.HAND_CONNECTIONS,
                    )
                    mp_drawing.draw_landmarks(
                        self.annotated_image,
                        self.results.pose_landmarks,
                        self.mp_holistic.POSE_CONNECTIONS,
                    )

                    self.points = [
                        self.results.left_hand_landmarks,
                        self.results.right_hand_landmarks,
                        self.results.pose_landmarks,
                    ]
        if ".mp4" in self.filename_or_camera:
            self.points = []
            cap = cv2.VideoCapture(self.filename_or_camera)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.mp_holistic = mp.solutions.holistic
            with self.mp_holistic.Holistic(
                    min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as holistic:
                while cap.isOpened():

                    success, image = cap.read()
                    if not success:
                        break
                    self.frames_c += 1
                    # Flip the image horizontally for a later selfie-view display, and convert
                    # the BGR image to RGB.
                    if self.where_left == "left":
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if self.where_left == "right":
                        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                    # To improve performance, optionally mark the image as not writeable to
                    self.image_height, self.image_width, _ = image.shape
                    # pass by reference.
                    image.flags.writeable = False
                    self.results = holistic.process(image)
                    # Draw landmark annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if self.output_type == "pose":
                        self.points.append(self.results.pose_landmarks)
                    if self.output_type == "face":
                        self.points.append(self.results.face_landmarks)
                    if self.output_type == "hands":
                        self.points.append(
                            [
                                self.results.left_hand_landmarks,
                                self.results.right_hand_landmarks,
                            ]
                        )
                    if self.output_type == "full":
                        self.points.append(
                            [
                                self.results.face_landmarks,
                                self.results.left_hand_landmarks,
                                self.results.right_hand_landmarks,
                                self.results.pose_landmarks,
                            ]
                        )
                    if self.output_type == "hands+pose":
                        self.points.append(
                            [
                                self.results.left_hand_landmarks,
                                self.results.right_hand_landmarks,
                                self.results.pose_landmarks,
                            ]
                        )

    def show(self, mode=0, type_show="OpenCV", csv_path=None):
        if self.processing_ == False:
            self.processing()
        import time
        mp_drawing = mp.solutions.drawing_utils
        if type_show == "OpenCV":
            if ".png" in self.filename_or_camera:
                if mode == 0:

                    black_img = np.zeros((self.image_height, self.image_width, 3))
                    if self.output_type == "pose":
                        for pose_landmarks in self.points:
                            mp_drawing.draw_landmarks(
                                black_img, pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                            )

                    if self.output_type == "hands":
                        for hand_landmarks in self.points:
                            mp_drawing.draw_landmarks(
                                black_img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                            )
                    if self.output_type == "face":
                        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
                        for face_landmarks in self.points:
                            mp_drawing.draw_landmarks(
                                image=black_img,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=drawing_spec,
                                connection_drawing_spec=drawing_spec,
                            )
                    if self.output_type == "full":
                        mp_drawing.draw_landmarks(
                            black_img,
                            self.results.face_landmarks,
                            self.mp_holistic.FACEMESH_TESSELATION,
                        )
                        mp_drawing.draw_landmarks(
                            black_img,
                            self.results.left_hand_landmarks,
                            self.mp_holistic.HAND_CONNECTIONS,
                        )
                        mp_drawing.draw_landmarks(
                            black_img,
                            self.results.right_hand_landmarks,
                            self.mp_holistic.HAND_CONNECTIONS,
                        )
                        mp_drawing.draw_landmarks(
                            black_img,
                            self.results.pose_landmarks,
                            self.mp_holistic.POSE_CONNECTIONS,
                        )
                    if self.output_type == "hands+pose":
                        mp_drawing.draw_landmarks(
                            black_img,
                            self.results.left_hand_landmarks,
                            self.mp_holistic.HAND_CONNECTIONS,
                        )
                        mp_drawing.draw_landmarks(
                            black_img,
                            self.results.right_hand_landmarks,
                            self.mp_holistic.HAND_CONNECTIONS,
                        )
                        mp_drawing.draw_landmarks(
                            black_img,
                            self.results.pose_landmarks,
                            self.mp_holistic.POSE_CONNECTIONS,
                        )
                    cv2.imwrite(self.output_name, black_img)
                if mode == 1:
                    while True:
                        if self.where_left == "right":
                            cv2.imshow("Mediapipe img", cv2.flip(self.annotated_image, 1))

                        if self.where_left == "left":
                            cv2.imshow("Mediapipe img", self.annotated_image)
                        key = cv2.waitKey(20)
                        if key == ord("q"):
                            break
            if ".mp4" in self.filename_or_camera:
                if mode == 0:
                    black = np.zeros((self.frames_c, self.image_height, self.image_width, 3))
                    print("FRAMEC+", self.frames_c)
                    if self.output_type == "full":
                        dlit = len(self.points)
                        i = 0
                        for black_img in black:
                            mp_drawing.draw_landmarks(
                                black_img,
                                self.points[i][0],
                                self.mp_holistic.FACEMESH_TESSELATION,
                            )
                            mp_drawing.draw_landmarks(
                                black_img,

                                self.points[i][1],
                                self.mp_holistic.HAND_CONNECTIONS,
                            )
                            mp_drawing.draw_landmarks(
                                black_img,

                                self.points[i][2],
                                self.mp_holistic.HAND_CONNECTIONS,
                            )
                            mp_drawing.draw_landmarks(
                                black_img,

                                self.points[i][3],
                                self.mp_holistic.POSE_CONNECTIONS,
                            )
                            i += 1
                            black_img = cv2.resize(black_img, (1280, 720))
                            if self.where_left == "right":
                                cv2.imshow("black_full", cv2.flip(black_img, 1))
                            if self.where_left == "left":
                                cv2.imshow("black_full", black_img)
                            cv2.waitKey(1)
                            time.sleep(1 / self.fps)
                    if self.output_type == "hands+pose":
                        dlit = len(self.points)
                        i = 0
                        for black_img in black:
                            mp_drawing.draw_landmarks(
                                black_img,

                                self.points[i][1],
                                self.mp_holistic.HAND_CONNECTIONS,
                            )
                            mp_drawing.draw_landmarks(
                                black_img,

                                self.points[i][2],
                                self.mp_holistic.HAND_CONNECTIONS,
                            )
                            mp_drawing.draw_landmarks(
                                black_img,

                                self.points[i][3],
                                self.mp_holistic.POSE_CONNECTIONS,
                            )
                            i += 1
                            black_img = cv2.resize(black_img, (1280, 720))
                            if self.where_left == "right":
                                cv2.imshow("black_full", cv2.flip(black_img, 1))
                            if self.where_left == "left":
                                cv2.imshow("black_full", black_img)
                            cv2.waitKey(1)
                            time.sleep(1 / self.fps)
                if mode == 1:
                    black = np.zeros((self.frames_c, self.image_height, self.image_width, 3))
                    cap = cv2.VideoCapture(self.filename_or_camera)

                    if self.output_type == "full":
                        dlit = len(self.points)
                        i = 0

                        while cap.isOpened():
                            success, image = cap.read()
                            if not success:
                                break
                            mode_c = 0
                            if self.where_left == "right":
                                image = cv2.flip(image, 1)

                            mp_drawing.draw_landmarks(
                                image,
                                self.points[i][0],
                                self.mp_holistic.FACEMESH_TESSELATION,
                            )
                            mp_drawing.draw_landmarks(
                                image,

                                self.points[i][1],
                                self.mp_holistic.HAND_CONNECTIONS,
                            )
                            mp_drawing.draw_landmarks(
                                image,

                                self.points[i][2],
                                self.mp_holistic.HAND_CONNECTIONS,
                            )
                            mp_drawing.draw_landmarks(
                                image,

                                self.points[i][3],
                                self.mp_holistic.POSE_CONNECTIONS,
                            )

                            i += 1

                            image = cv2.resize(image, (1280, 720))
                            if self.where_left == "right":
                                cv2.imshow("color_full", cv2.flip(image, 1))
                            if self.where_left == "left":
                                cv2.imshow("color_full", image)

                            cv2.waitKey(1)
                            time.sleep(1 / 30)
                    if self.output_type == "hands+pose":
                        dlit = len(self.points)
                        i = 0

                        while cap.isOpened():
                            success, image = cap.read()
                            if not success:
                                break
                            mode_c = 0
                            if self.where_left == "right":
                                image = cv2.flip(image, 1)
                            mp_drawing.draw_landmarks(
                                image,

                                self.points[i][0],
                                self.mp_holistic.HAND_CONNECTIONS,
                            )
                            mp_drawing.draw_landmarks(
                                image,

                                self.points[i][1],
                                self.mp_holistic.HAND_CONNECTIONS,
                            )
                            mp_drawing.draw_landmarks(
                                image,

                                self.points[i][2],
                                self.mp_holistic.POSE_CONNECTIONS,
                            )

                            i += 1

                            image = cv2.resize(image, (1280, 720))
                            if self.where_left == "right":
                                cv2.imshow("color_full", cv2.flip(image, 1))
                            if self.where_left == "left":
                                cv2.imshow("color_full", image)

                            cv2.waitKey(1)
                            time.sleep(1 / 30)
                    print("Вывод видео пока не поддерживается")
                print("представить скелетную модель с помощью MediaPipe")
        if type_show == "csv":
            if csv_path is None:
                raise ValueError("УКАЖИТЕ ПУТЬ К CSV")
            mp_drawing = mp.solutions.drawing_utils
            df = pd.read_csv(csv_path)
            if self.output_type == "full":
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
                self.mp_holistic = mp.solutions.holistic
                if self.output_type == "full":
                    i = 0
                    for i in range(rows):

                        black = np.zeros((height, width, 3))
                        draw_point(points_arr[0][i], black, width, height)
                        draw_point(points_arr[1][i], black, width, height)
                        draw_point(points_arr[2][i], black, width, height)
                        draw_point(points_arr[3][i], black, width, height)
                        # black = cv2.resize(black, (400,400))
                        black = cv2.resize(black, (1280, 720))
                        if self.where_left == "right":
                            cv2.imshow("black_full", cv2.flip(black, 1))
                        if self.where_left == "left":
                            cv2.imshow("black_full", black)
                        key = cv2.waitKey(1)
                        time.sleep(1 / fps)
                        if key == ord("q"):
                            break
            if self.output_type == "hands+pose":
                width = int(df["resolution_width"][0])
                height = int(df["resolution_height"][0])
                rows = int(df.shape[0])
                fps = df["fps"][0]
                pose_p = df.loc[0:rows, "pose_x0":"pose_p31"].values
                lhand_p = df.loc[0:rows, "lhand_x0":"lhand_p20"].values
                rhand_p = df.loc[0:rows, "rhand_x0":"rhand_p20"].values
                points_arr = [pose_p, lhand_p, rhand_p]
                print(face_p.shape)
                print(pose_p.shape)
                print(lhand_p.shape)
                print(rhand_p.shape)
                self.mp_holistic = mp.solutions.holistic
                if self.output_type == "hands+pose":
                    i = 0
                    for i in range(rows):

                        black = np.zeros((height, width, 3))
                        draw_point(points_arr[0][i], black, width, height)
                        draw_point(points_arr[1][i], black, width, height)
                        draw_point(points_arr[2][i], black, width, height)
                        # black = cv2.resize(black, (400,400))
                        black = cv2.resize(black, (1280, 720))
                        if self.where_left == "right":
                            cv2.imshow("black_full", cv2.flip(black, 1))
                        if self.where_left == "left":
                            cv2.imshow("black_full", black)
                        key = cv2.waitKey(1)
                        time.sleep(1 / fps)
                        if key == ord("q"):
                            break
        cv2.destroyAllWindows()
    # BUG
    def to_npy(self, npy_filename_path):
        if self.processing_ == False:
            self.processing()
        if "png" in self.filename_or_camera:
            np_data = []
            for target in self.points:
                target_arr = []
                if target is None:
                    target_arr.append([None, None])
                    continue
                for p in target.landmark:
                    target_arr.append(np.array([p.x, p.y, p.z, p.visibility]))
                np_data.append(np.array(target_arr))
            np_data = np.array(np_data)
            print(np_data.shape)
            np.save(npy_filename_path, np_data)

        if "mp4" in self.filename_or_camera:
            print(len(self.points))
            video_data = []
            for frame in self.points:
                frame_data = []
                for target in frame:
                    target_arr = []
                    if target is None:
                        target_arr.append([None, None])
                        frame_data.append(np.array(target_arr))
                        continue
                    for p in target.landmark:
                        target_arr.append(np.array(
                            [round(p.x, 5), round(p.y, 5), round(p.z, 5), round(random_sigmoid(p.visibility), 5)]))
                    frame_data.append(np.array(target_arr))
                video_data.append(np.array(frame_data))

            video_data = np.array(video_data)
            np.save(npy_filename_path, video_data)

    def to_df(self):
        if self.processing_ == False:
            self.processing()
        self.fps = round(self.fps, 5)
        if ".png" in self.filename_or_camera:
            pass

        if ".mp4" in self.filename_or_camera:

            if self.output_type in ["pose", "face"]:

                fieldnames = ["frame", "resolution_width", "resolution_height", "fps"]
                if self.output_type == "pose":
                    poses = [[f"pose_x{i}", f"pose_y{i}", f"pose_z{i}", f"pose_p{i}"] for i in range(33)]
                if self.output_type == "face":
                    poses = [[f"face_x{i}", f"face_y{i}", f"face_z{i}", f"face_p{i}"] for i in range(468)]
                flat = [x for sublist in poses for x in sublist]
                fieldnames = fieldnames + flat

                df = pd.DataFrame(columns=fieldnames)
                iswriten = False

                i = 0

                for frame in self.points:
                    if frame is None:
                        print(i)
                        i += 1
                        continue
                    keypoints = [x for x in frame.landmark]
                    keypoints = [[round(x.x, 5), round(x.y, 5), round(x.z, 5), round(random_sigmoid(x.visibility), 5)]
                                 for x in keypoints]
                    keypoints_flatten = [
                        x for sublist in keypoints for x in sublist
                    ]

                    if not iswriten:
                        output_arr = [int(i)] + [self.image_width] + [self.image_height] + [
                            self.fps] + keypoints_flatten
                        iswriten = True
                    else:
                        output_arr = [int(i)] + ["", "", ""] + keypoints_flatten
                    df.loc[int(i)] = output_arr
                    i += 1
                return df
            if self.output_type == "hands":
                left_hand = [lh[0] for lh in self.points]
                right_hand = [rh[1] for rh in self.points]
                points = []
                for i in range(len(right_hand)):
                    points.append([right_hand[int(i)], left_hand[int(i)]])
                fieldnames = ["frame", "resolution_width", "resolution_height", "fps"]
                rh_labels = [[f"rhand_x{i}", f"rhand_y{i}", f"rhand_z{i}", f"rhand_p{i}"] for i in range(21)]
                lh_labels = [[f"lhand_x{i}", f"lhand_y{i}", f"lhand_z{i}", f"lhand_p{i}"] for i in range(21)]
                hand_labels = lh_labels + rh_labels

                flat = [x for sublist in hand_labels for x in sublist]
                fieldnames = fieldnames + flat
                df = pd.DataFrame(columns=fieldnames)

                i = 0
                iswriten = False
                for frame in points:

                    if frame[0] is None:
                        rh_keypoints_flatten = [None for i in range(84)]
                    else:
                        rh_keypoints = [x for x in frame[0].landmark]
                        rh_keypoints = [
                            [round(x.x, 5), round(x.y, 5), round(x.z, 5), round(random_sigmoid(x.visibility), 5)] for x
                            in rh_keypoints]
                        rh_keypoints_flatten = [
                            x for sublist in rh_keypoints for x in sublist
                        ]

                    if frame[1] is None:
                        lh_keypoints_flatten = [None for i in range(84)]
                    else:
                        lh_keypoints = [x for x in frame[1].landmark]
                        lh_keypoints = [
                            [round(x.x, 5), round(x.y, 5), round(x.z, 5), round(random_sigmoid(x.visibility), 5)] for x
                            in lh_keypoints]
                        lh_keypoints_flatten = [
                            x for sublist in lh_keypoints for x in sublist
                        ]
                    if frame[0] is None and frame[1] is None:
                        i += 1
                        print(i)
                        continue
                    if frame[0] is not None or frame[1] is not None:
                        keypoints_flatten = (
                                rh_keypoints_flatten + lh_keypoints_flatten
                        )

                    if not iswriten:
                        output_arr = [int(i)] + [self.image_width] + [self.image_height] + [
                            self.fps] + keypoints_flatten
                        iswriten = True
                    else:
                        output_arr = [int(i)] + ["", "", ""] + keypoints_flatten
                    df.loc[int(i)] = output_arr
                    i += 1
                return df
            if self.output_type == "full":
                face = [f[0] for f in self.points]
                left_hand = [lh[1] for lh in self.points]
                right_hand = [rh[2] for rh in self.points]
                pose = [p[3] for p in self.points]

                points = []
                for i in range(len(right_hand)):
                    points.append([pose[int(i)], right_hand[int(i)], left_hand[int(i)], face[int(i)]])
                rh_labels = [[f"rhand_x{i}", f"rhand_y{i}", f"rhand_z{i}", f"rhand_p{i}"] for i in range(21)]
                lh_labels = [[f"lhand_x{i}", f"lhand_y{i}", f"lhand_z{i}", f"lhand_p{i}"] for i in range(21)]

                poses = [[f"pose_x{i}", f"pose_y{i}", f"pose_z{i}", f"pose_p{i}"] for i in range(33)]

                faces = [[f"face_x{i}", f"face_y{i}", f"face_z{i}", f"face_p{i}"] for i in range(468)]
                full_labels = poses + lh_labels + rh_labels + faces
                flat = [x for sublist in full_labels for x in sublist]
                fieldnames = ["frame", "resolution_width", "resolution_height", "fps"]
                fieldnames = fieldnames + flat
                df = pd.DataFrame(columns=fieldnames)
                iswriten = False
                i = 0

                for frame in points:

                    if frame[0] is None:
                        pose_keypoints_flatten = [None for i in range(132)]
                    else:
                        pose_keypoints = [x for x in frame[0].landmark]
                        pose_keypoints = [
                            [round(x.x, 5), round(x.y, 5), round(x.z, 5), round(random_sigmoid(x.visibility), 5)] for x
                            in pose_keypoints]
                        pose_keypoints_flatten = [
                            x for sublist in pose_keypoints for x in sublist
                        ]

                    if frame[1] is None:
                        rh_keypoints_flatten = [None for i in range(84)]
                    else:
                        rh_keypoints = [x for x in frame[1].landmark]
                        rh_keypoints = [
                            [round(x.x, 5), round(x.y, 5), round(x.z, 5), round(random_sigmoid(x.visibility), 5)] for x
                            in rh_keypoints]
                        rh_keypoints_flatten = [
                            x for sublist in rh_keypoints for x in sublist
                        ]

                    if frame[2] is None:
                        lh_keypoints_flatten = [None for i in range(84)]
                    else:
                        lh_keypoints = [x for x in frame[2].landmark]
                        lh_keypoints = [
                            [round(x.x, 5), round(x.y, 5), round(x.z, 5), round(random_sigmoid(x.visibility), 5)] for x
                            in lh_keypoints]
                        lh_keypoints_flatten = [
                            x for sublist in lh_keypoints for x in sublist
                        ]

                    if frame[3] is None:
                        face_keypoints_flatten = [None for i in range(1872)]
                    else:
                        face_keypoints = [x for x in frame[3].landmark]
                        face_keypoints = [
                            [round(x.x, 5), round(x.y, 5), round(x.z, 5), round(random_sigmoid(x.visibility), 5)] for x
                            in face_keypoints]
                        face_keypoints_flatten = [
                            x for sublist in face_keypoints for x in sublist
                        ]

                    if (
                            frame[0] is None
                            and frame[1] is None
                            and frame[2] is None
                            and frame[3] is None
                    ):
                        i += 1
                        print(i)
                        continue
                    if (
                            frame[0] is not None
                            or frame[1] is not None
                            or frame[2] is not None
                            or frame[3] is not None
                    ):
                        keypoints_flatten = (
                                pose_keypoints_flatten
                                + rh_keypoints_flatten
                                + lh_keypoints_flatten
                                + face_keypoints_flatten
                        )

                    if not iswriten:
                        output_arr = [int(i)] + [self.image_width] + [self.image_height] + [
                            self.fps] + keypoints_flatten
                        iswriten = True
                    else:
                        output_arr = [int(i)] + ["", "", ""] + keypoints_flatten
                    df.loc[int(i)] = output_arr
                    i += 1
                return df

            if self.output_type == "hands+pose":
                left_hand = [lh[0] for lh in self.points]
                right_hand = [rh[1] for rh in self.points]
                pose = [p[2] for p in self.points]

                points = []
                for i in range(len(right_hand)):
                    points.append([pose[int(i)], right_hand[int(i)], left_hand[int(i)]])
                rh_labels = [[f"rhand_x{i}", f"rhand_y{i}", f"rhand_z{i}", f"rhand_p{i}"] for i in range(21)]
                lh_labels = [[f"lhand_x{i}", f"lhand_y{i}", f"lhand_z{i}", f"lhand_p{i}"] for i in range(21)]

                poses = [[f"pose_x{i}", f"pose_y{i}", f"pose_z{i}", f"pose_p{i}"] for i in range(33)]

                full_labels = poses + lh_labels + rh_labels
                flat = [x for sublist in full_labels for x in sublist]
                fieldnames = ["frame", "resolution_width", "resolution_height", "fps"]
                fieldnames = fieldnames + flat
                df = pd.DataFrame(columns=fieldnames)
                iswriten = False
                i = 0

                for frame in points:

                    if frame[0] is None:
                        pose_keypoints_flatten = [None for i in range(132)]
                    else:
                        pose_keypoints = [x for x in frame[0].landmark]
                        pose_keypoints = [
                            [round(x.x, 5), round(x.y, 5), round(x.z, 5), round(random_sigmoid(x.visibility), 5)] for x
                            in pose_keypoints]
                        pose_keypoints_flatten = [
                            x for sublist in pose_keypoints for x in sublist
                        ]

                    if frame[1] is None:
                        rh_keypoints_flatten = [None for i in range(84)]
                    else:
                        rh_keypoints = [x for x in frame[1].landmark]
                        rh_keypoints = [
                            [round(x.x, 5), round(x.y, 5), round(x.z, 5), round(random_sigmoid(x.visibility), 5)] for x
                            in rh_keypoints]
                        rh_keypoints_flatten = [
                            x for sublist in rh_keypoints for x in sublist
                        ]

                    if frame[2] is None:
                        lh_keypoints_flatten = [None for i in range(84)]
                    else:
                        lh_keypoints = [x for x in frame[2].landmark]
                        lh_keypoints = [
                            [round(x.x, 5), round(x.y, 5), round(x.z, 5), round(random_sigmoid(x.visibility), 5)] for x
                            in lh_keypoints]
                        lh_keypoints_flatten = [
                            x for sublist in lh_keypoints for x in sublist
                        ]

                    if (
                            frame[0] is None
                            and frame[1] is None
                            and frame[2] is None
                    ):
                        i += 1
                        print(i)
                        continue
                    if (
                            frame[0] is not None
                            or frame[1] is not None
                            or frame[2] is not None
                    ):
                        keypoints_flatten = (
                                pose_keypoints_flatten
                                + rh_keypoints_flatten
                                + lh_keypoints_flatten
                        )

                    if not iswriten:
                        output_arr = [int(i)] + [self.image_width] + [self.image_height] + [
                            self.fps] + keypoints_flatten
                        iswriten = True
                    else:
                        output_arr = [int(i)] + ["", "", ""] + keypoints_flatten
                    df.loc[int(i)] = output_arr
                    i += 1
                return df        

    def to_csv(self, csv_filename_path):
        df = self.to_df()
        df.to_csv(csv_filename_path, index=False)

    def from_npy(self, npy_filename_path):
        data = np.load(npy_filename_path, allow_pickle=True)
        return data

    def from_csv(self, csv_filename_path):
        print("Читать csv")