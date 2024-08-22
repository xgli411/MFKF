import cv2
import numpy as np
import time
import joblib
from FlashPose.demo.vis import get_pose3D as FlashPose

pose_knn = joblib.load('Model/PoseKeypoint.joblib')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = FlashPose(video_path, output_dir)
prevTime = 0
keyXYZ = [
    "nose_x",
    "nose_y",
    "nose_z",
    "left_eye_x",
    "left_eye_y",
    "left_eye_z",
    "right_eye_x",
    "right_eye_y",
    "right_eye_z",
    "left_ear_x",
    "left_ear_y",
    "left_ear_z",
    "right_ear_x",
    "right_ear_y",
    "right_ear_z",
    "left_shoulder_x",
    "left_shoulder_y",
    "left_shoulder_z",
    "right_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_z",
    "left_elbow_x",
    "left_elbow_y",
    "left_elbow_z",
    "right_elbow_x",
    "right_elbow_y",
    "right_elbow_z",
    "left_wrist_x",
    "left_wrist_y",
    "left_wrist_z",
    "right_wrist_x",
    "right_wrist_y",
    "right_wrist_z",
    "left_hip_x",
    "left_hip_y",
    "left_hip_z",
    "right_hip_x",
    "right_hip_y",
    "right_hip_z",
    "left_knee_x",
    "left_knee_y",
    "left_knee_z",
    "right_knee_x",
    "right_knee_y",
    "right_knee_z",
    "left_ankle_x",
    "left_ankle_y",
    "left_ankle_z",
    "right_ankle_x",
    "right_ankle_y",
    "right_ankle_z",
]
res_point = []
cap = cv2.VideoCapture("")
with mp_pose as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video.")
            break
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            for index, landmarks in enumerate(results.pose_landmarks.landmark):
                print(index, landmarks.x, landmarks.y, landmarks.z)
                #得到跌倒和正常状态下各个坐标点的坐标，并放入到csv文件下中，以供knn学习
                res_point.append(landmarks.x)
                res_point.append(landmarks.y)
                res_point.append(landmarks.z)
            shape1 = int(len(res_point) / len(keyXYZ))
            res_point = np.array(res_point).reshape(shape1, len(keyXYZ))
            pred = pose_knn.predict(res_point)
            res_point = []
            print(pred)
            if pred == 0:
                cv2.putText(image, "Fall", (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)
            else:
                cv2.putText(image, "Normal", (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2)
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)

        cv2.imshow('Real time display', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
