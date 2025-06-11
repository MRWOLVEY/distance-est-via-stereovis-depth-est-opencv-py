import cv2 
from ultralytics import YOLO

import torch

cap = cv2.VideoCapture(2)
model = YOLO("yolo11s.pt")

while True:
    if not cap.isOpened():
        print('couldnt open cams')
        break
    ret, frame = cap.read()
    model.to('cuda:0')
    print(torch.cuda.is_available())    

    detections = model(frame)

    print(detections)
    break

