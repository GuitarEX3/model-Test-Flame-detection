import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
import requests

# -----------------------------
# ตั้งค่า Telegram
# -----------------------------
BOT_TOKEN = "8314750392:AAGGXY3HwEYgkJwXfMBxoFJ1Kd1U89CFWPc"
CHAT_ID = "7707514933"  # ต้องเป็น string

def tg_text(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

def tg_photo(path, caption=""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {"photo": open(path, "rb")}
    data = {"chat_id": CHAT_ID, "caption": caption}
    requests.post(url, files=files, data=data)

# -----------------------------
# ตั้งค่า YOLO
# -----------------------------
model_path = 'my_model.pt'      # ไฟล์โมเดล
img_source = "video/v.mp4"      # ไฟล์วิดีโอ
min_thresh = 0.5                # threshold

user_res = "640x800"
record = False

if not os.path.exists(model_path):
    print('ERROR: Model path invalid.')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

cap = cv2.VideoCapture(img_source)
if user_res:
    cap.set(3, resW)
    cap.set(4, resH)

bbox_colors = [(164,120,87), (68,148,228), (93,97,209),
               (178,182,133), (88,159,106), (96,202,231),
               (159,124,168), (169,162,241), (98,118,150),
               (172,176,184)]

# -----------------------------
# ตัวแปรกัน spam แจ้งเตือน
# -----------------------------
last_alert = 0
cooldown = 5  # ส่งได้ทุก 5 วินาที

# -----------------------------
# เริ่มตรวจจับ
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    fire_detected = False

    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy

        classidx = int(det.cls)
        conf = float(det.conf)
        classname = labels[classidx]

        # <<< แก้ชื่อผิด "Frame" → "Fire" >>>
        if classname.lower() == "frame":
            classname = "fire"

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{classname} {int(conf*100)}%"
            cv2.putText(frame, label, (xmin, ymin-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # เช็คเจอไฟไหม้
            if classname.lower() in ["fire", "flame", "burn"]:
                fire_detected = True

    # -----------------------------
    # ส่งแจ้งเตือน Telegram (รูป + ข้อความ)
    # -----------------------------
    now = time.time()
    if fire_detected and (now - last_alert > cooldown):
        img_path = "fire.jpg"
        cv2.imwrite(img_path, frame)

        tg_text("🔥 พบไฟไหม้จ้าาาาา!")
        tg_photo(img_path, caption="🔥 ตรวจพบไฟไหม้จากกล้อง")

        print("ส่งแจ้งเตือน Telegram แล้ว!")
        last_alert = now

    cv2.imshow("Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
