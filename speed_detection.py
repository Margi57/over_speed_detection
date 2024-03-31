
import cv2
import pandas as pd
from ultralytics import YOLO
# # from tracker import*
import math
import time
# import os

model=YOLO('yolov8s.pt')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# tracker=Tracker()
count=0

cap=cv2.VideoCapture('D:\D_BM_internship\over_speed_detect\highway.mp4')


class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            # print("center point:::::",self.center_points)
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    # print("center point:::::",self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    # print("objects_bbs_ids::::::",objects_bbs_ids)
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids



tracker=Tracker()
count=0
down={}
up={}

counter_down=[]
counter_up=[]
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
    # print(";;;;;::::::::::",results)
    a=results[0].boxes.data
    a = a.detach().cpu().numpy() 
    px=pd.DataFrame(a).astype("float")
    # print(px)

    
    list=[]
    
    for index,row in px.iterrows():
    #   print(row)
      x1=int(row[0])
      y1=int(row[1])
      x2=int(row[2])
      y2=int(row[3])
      d=int(row[5])
      c=class_list[d]
      if 'car' in c:
          list.append([x1,y1,x2,y2])
        #   print("::::::",list)
      

    bbox_id=tracker.update(list)
    # print("id is::",bbox_id)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        # print("ididididididid",id)
        
        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
        cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        
        red_line_y=198
        blue_line_y=268
        offset=6

    

        if red_line_y<(cy+offset) and red_line_y >(cy - offset):
            # this if condition is putting the id and the circle when the center of the object toched the red line
            # down[id]=cy #cy is current possition.saving the ids of the cars which are touching the red line first
            down[id]=time.time()
            
        if id in down:
            if blue_line_y<(cy+offset) and blue_line_y >(cy - offset):
                total_time=time.time() - down[id]
                if counter_down.count(id)==0:
                    counter_down.append(id)
                    distance=10  #2 line between distance
                    a_speed_ms=distance / total_time
                    a_speed_kh=a_speed_ms * 3.6
                    # print(":::::::::::::::",a_speed_kh)
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.rectangle(frame,(x3, y3),(x4, y4),(0,255,0),2)
                    cv2.putText(frame,str(id),(x3, y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)
                    # print(":::::::::::::::",int(a_speed_kh))
        
        if blue_line_y<(cy+offset) and blue_line_y >(cy - offset):
            # this if condition is putting the id and the circle when the center of the object toched the red line
            # down[id]=cy #cy is current possition.saving the ids of the cars which are touching the red line first
            up[id]=time.time()
            
        if id in up:
            if red_line_y<(cy+offset) and red_line_y >(cy - offset):
                total1_time=time.time()-up[id]
                if counter_up.count(id)==0:
                    counter_up.append(id)
                    distance1=10  #2 line between distance
                    a_speed_ms1=distance1/total1_time
                    a_speed_kh1=a_speed_ms1 * 3.6
                    
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,x4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)
                   
        
        cv2.line(frame,(172,198),(774,198),(0,0,255),3)
        # cv2.pitText(frame,('red ine'))
        cv2.line(frame,(8,268),(927,268),(255,0,0),3)
        
        cv2.putText(frame,('Going_Down - ' + str(len(counter_down))),(10,30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,('Going_up - ' + str(len(counter_up))),(10,60),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),1,cv2.LINE_AA)
        
        
    cv2.imshow("frames", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()









































