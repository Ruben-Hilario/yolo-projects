from ultralytics import YOLO
from res import*
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
model=YOLO("C-V/YOLO/yolov10n.pt")
while(True):
    success,img=cap.read()
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for b in boxes:
            x1,y1,x2,y2=b.xyxy[0]
            x1,x2,y1,y2=ret(x1,y1,x2,y2)
            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            conf=math.ceil((b.conf[0]*100)/100)
            print(conf)
            cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)))

    cv2.imshow("Image",img)
    cv2.waitKey(1)