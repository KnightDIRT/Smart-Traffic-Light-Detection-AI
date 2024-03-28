import numpy as np
import cv2
import os
import torch
from ultralytics import YOLO
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

ROOT_DIR = "C:\\Users\\Torenia\\OneDrive\\Desktop\\University files\\Year2Term2\\AI Technology\\Workspace\\Project\\For Demonstration\\"
videoName = "TestVideo"
ranEpochs = 300
model = YOLO(os.path.join(ROOT_DIR, f"best.pt"))
model_RF = joblib.load(os.path.join(ROOT_DIR, "model.joblib"))
class_labels = ["Green", "Red", "Yellow", "Off"]

cap = cv2.VideoCapture(os.path.join(ROOT_DIR, f"{videoName}.mp4"))

output = cv2.VideoWriter(os.path.join(ROOT_DIR, f"{videoName}_Output.mp4"), cv2.VideoWriter_fourcc(*'MP4V'), 10, (1280, 720))

count = 0
while cap.isOpened():
    ret,frame = cap.read()
    results = model(frame, stream=True)

    frame_result = np.copy(frame)
    for result in results:
        for box in result.boxes.cpu().numpy():
            box = np.int_(box.xyxy.tolist()[0])
            frame_result = cv2.rectangle(frame_result, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)

            cropped = frame_result[box[1]:box[3], box[0]:box[2]]
            classVal = model_RF.predict([cv2.resize(cropped, (100, 100)).flatten()])
            cv2.putText(frame_result, class_labels[classVal[0]], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    output.write(frame_result)
    cv2.imshow('window-name', frame_result)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

output.release()
cap.release()
cv2.destroyAllWindows()