import numpy as np
from ultralytics import YOLO
from res import*
import cv2
import cvzone
import math
from sort import *
 
cap = cv2.VideoCapture("VIDS/cars.mp4") 
model=YOLO("yolov8n.pt")
mask = cv2.imread("IMGS/cars.png")
 
t=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
th=[400,297,673]
c=[]
while True:
    success, img = cap.read()
    if not cap.isOpened():
        print("Error: Could not open this video")
        exit()
    if not success or img is None:
        print("Stream ended")
        break
    zone = cv2.bitwise_and(img, mask)
    results = model(zone, stream=True)
    dets = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for b in boxes:
            # Bounding Box
            x1, y1, x2, y2 = b.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((b.conf[0] * 100)) / 100
            # Class Name
            cls = int(b.cls[0])
            detClass = clsN[cls]
            if detClass == "car" or detClass == "truck" or detClass == "bus" or detClass == "motorbike" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                dets = np.vstack((dets, currentArray))
    resTracker = t.update(dets)
    for r in resTracker:
        x1, y1, x2, y2, id = r
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(r)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cx, cy = x1 + w // 2, y1 + h // 2
        if th[0] < cx < th[2] and th[1] - 15 < cy < th[1] + 15:
            if c.count(id) == 0:
                c.append(id)
    cvzone.putTextRect(img,f'Cars Counted: {str(len(c))}',(100,50),scale=3,colorR=(0,0,0),colorT=(255,255,255),thickness=3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break