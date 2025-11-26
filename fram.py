import os
import sys
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# ตั้งค่าโมเดลและวิดีโอ
# -----------------------------
model_path = 'my_model.pt'      # ใส่ชื่อไฟล์โมเดลของคุณ
img_source = 0     # ใส่ 0 หรือ คลิปวิดิโอก็ได้
min_thresh = 0.5               # ความมั่นใจขั้นต่ำ
user_res = "640x800"           # ความละเอียดสำหรับแสดงผล (หรือ None)
record = True                  # บันทึกวิดีโอผลลัพธ์

# -----------------------------
# โหลดโมเดล
# -----------------------------
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

# -----------------------------
# ตั้งค่าความละเอียด
# -----------------------------
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# -----------------------------
# ตั้งค่าบันทึกวิดีโอ
# -----------------------------
if record:
    if not resize:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# -----------------------------
# โหลดวิดีโอ
# -----------------------------
cap = cv2.VideoCapture(img_source)
if user_res:
    cap.set(3, resW)
    cap.set(4, resH)

# -----------------------------
# ตั้งค่าสีกรอบ bbox
# -----------------------------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# -----------------------------
# ตัวแปรควบคุม FPS
# -----------------------------
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200

# -----------------------------
# วนลูปตรวจจับ
# -----------------------------
while True:
    t_start = time.perf_counter()
    
    ret, frame = cap.read()
    if not ret or frame is None:
        print('Reached end of video or cannot read frames. Exiting.')
        break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            object_count += 1

    cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow('YOLO Detection', frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

    # คำนวณ FPS
    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# -----------------------------
# ปิดการทำงาน
# -----------------------------
print(f'Average FPS: {avg_frame_rate:.2f}')
cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
