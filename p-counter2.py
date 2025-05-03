import numpy as np
from ultralytics import YOLO
from res import*
import cv2
import cvzone
import math
from sort import *
 
cap = cv2.VideoCapture("VIDS/people.mp4") 
model=YOLO("yolov8n.pt")
mask = cv2.imread("IMGS/stairs.png")
 
t=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
th=[100,200,350,600,500,800]
Up=[]
Down=[]

while True:
    success, img = cap.read()
    zone = cv2.bitwise_and(img, mask)
    results = model(zone, stream=True)
    dets = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = clsN[cls]
            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                dets = np.vstack((dets, currentArray))
    resultsTracker = t.update(dets)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cx, cy = x1 + w // 2, y1 + h // 2
        if th[0] < cx < th[2] and th[1] - 15 < cy < th[1] + 15:
            if Up.count(id) == 0:
                Up.append(id)
        if th[3] < cx < th[5] and th[4] - 15 < cy < th[4] + 15:
            if Down.count(id) == 0:
                Down.append(id)
    
    cvzone.putTextRect(img,f'People Going Up: {str(len(Up))}',(600,50),scale=3,colorR=(0,0,0),colorT=(255,255,255),thickness=3)
    cvzone.putTextRect(img,f'People Going Down: {str(len(Down))}',(600,130),scale=3,colorR=(0,0,0),colorT=(255,255,255),thickness=3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)