import cv2
# import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
model=YOLO('yolov8m.pt')
from deep_sort.deep_sort import DeepSort

deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)
area1=[(332,380),(279,392),(454,469),(508,450)]

area2=[(279,392),(232,403),(388,489),(454,469)]
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('videos/peoplecount1.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
people_entering = {}
people_exiting={}
peple_complete_exiting={}
peple_complete_entering={}
complete_entering=0
frames = []
i = 0
counter, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()
unique_track_ids = set()

if (cap.isOpened() == False):
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (1020, 500)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
out = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
# out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
while True:    
    ret,frame = cap.read()
    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))

    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                   'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                   'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    results=model.predict(frame,device="cpu", classes=0, conf=0.8)
    list=[]
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        probs = result.probs  # Class probabilities for classification outputs
        cls = boxes.cls.tolist()  # Convert tensor to list
        xyxy = boxes.xyxy
        conf = boxes.conf
        xywh = boxes.xywh  # box with xywh format, (N, 4)
        for class_index in cls:
            class_name = class_names[int(class_index)]
            # print("Class:", class_name)

    pred_cls = np.array(cls)
    conf = conf.detach().cpu().numpy()
    xyxy = xyxy.detach().cpu().numpy()
    bboxes_xywh = xywh
    bboxes_xywh = xywh.cpu().numpy()
    bboxes_xywh = np.array(bboxes_xywh, dtype=float)

    tracks = tracker.update(bboxes_xywh, conf, frame)

    for track in tracker.tracker.tracks:
        track_id = track.track_id
        hits = track.hits
        x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
        w = x2 - x1  # Calculate width
        h = y2 - y1  # Calculate height
        # 226, 56, 69
        color = (69, 56, 226)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        cv2.putText(frame, str(class_name), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
        unique_track_ids.add(track_id)

        results = cv2.pointPolygonTest(np.array(area2, np.int32), (int(x2), int(y2)), False)
        if results >=0:
            people_entering[track_id] = (int(x2), int(y2))
        if track_id in people_entering:
            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (int(x2), int(y2)), False)
            if results1 >= 0:
                peple_complete_entering[track_id] = (int(x2), int(y2))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # cv2.circle(frame, (int(x1 + w), int(y1 + h)), 5, (255, 0, 255), -1) 247, 171, 14
                cv2.putText(frame, str(class_name), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, (0.5),
                            (255, 255, 255), 1)

        results_exit = cv2.pointPolygonTest(np.array(area1, np.int32), (int(x2), int(y2)), False)
        if results_exit >= 0:
            people_exiting[track_id] = (int(x2), int(y2))

        if track_id in people_exiting:
            results2 = cv2.pointPolygonTest(np.array(area2, np.int32), (int(x2), int(y2)), False)
            if results2 >= 0:
                peple_complete_exiting[track_id] = (int(x2), int(y2))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # cv2.circle(frame, (int(x1 + w), int(y1 + h)), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(class_name), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)

        # Add the track_id to the set of unique track IDs

    # Update the person count based on the number of unique track IDs
    person_count = len(unique_track_ids)
    # Update FPS and place on frame
    current_time = time.perf_counter()
    elapsed = (current_time - start_time)
    counter += 1
    if elapsed > 1:
        fps = counter / elapsed
        counter = 0
        start_time = current_time


    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('1'),(504,471),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('2'),(466,485),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    text1=f"People count : {person_count}"
    text2=f"People Enter : {len(peple_complete_entering)}"
    text3=f"People Exit : {len(peple_complete_exiting)}"
    cv2.putText(frame,str(text1),(55,30),cv2.FONT_HERSHEY_COMPLEX,(0.7),(255,255,255),1)
    cv2.putText(frame,str(text2),(300,30),cv2.FONT_HERSHEY_COMPLEX,(0.7),(255,255,255),1)
    cv2.putText(frame,str(text3),(550,30),cv2.FONT_HERSHEY_COMPLEX,(0.7),(255,255,255),1)



    print(len(peple_complete_exiting),"peple_complete_exiting")
    print(person_count,"person count")
    print(len(peple_complete_entering),"peple_entered")
    print(fps,"fps")
    # Append the frame to the list
    # frames.append(frame)

    # Write the frame to the output video file
    out.write(frame)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
