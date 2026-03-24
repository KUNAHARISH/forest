import cv2
import time
from ultralytics import YOLO
import face_recognition
import os
import numpy as np
import requests
import threading

TELEGRAM_BOT_TOKEN = "8671286798:AAFMIBGgcZkZ-3WlHxbxopwyalzliPJmOTY"
TELEGRAM_CHAT_ID = "1806569163"

MESSAGE_COOLDOWN = 15  
last_sent_times = {}

def send_telegram_alert(message_text):
    if not TELEGRAM_BOT_TOKEN:
        return
        
    current_time = time.time()
    if message_text in last_sent_times:
        if (current_time - last_sent_times[message_text]) < MESSAGE_COOLDOWN:
            return  
            
    last_sent_times[message_text] = current_time
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": f"{message_text}"}
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            print(f"\nTelegram Alert Sent: {message_text}")
        else:
            print(f"\n Telegram Failed! (Check Chat ID): {res.text}")
    except Exception as e:
        print(f"\nTelegram Exception: {e}")


try:
    import RPi.GPIO as GPIO  # type: ignore
    GPIO.setmode(GPIO.BCM)
    SENSOR_PIN = 17
    GPIO.setup(SENSOR_PIN, GPIO.IN)
    PI_MODE = True
except:
    print("Running on PC (GPIO disabled)")
    PI_MODE = False


import glob
print("Loading custom trained YOLO model for special animals (lions, etc)...")
list_of_files = glob.glob('runs/**/weights/best.pt', recursive=True)
if list_of_files:
    latest_model_path = max(list_of_files, key=os.path.getctime)
    print(f"Loading custom trained model from {latest_model_path}...")
    model = YOLO(latest_model_path)
else:
    print("Loading default YOLO model...")
    model = YOLO("yolov8n.pt")
model.to("cpu")
print("YOLO model loaded\n")


known_encodings = []
known_names = []

known_faces_dir = "dataset/known_faces"
os.makedirs(known_faces_dir, exist_ok=True)

print("Loading face dataset...")

for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(known_faces_dir, filename)
        try:
            img = face_recognition.load_image_file(path)
           
            enc = face_recognition.face_encodings(img, num_jitters=10)

            if len(enc) > 0:
                known_encodings.append(enc[0])
                
                
                name_from_file = os.path.splitext(filename)[0].replace("_", " ")
                known_names.append(name_from_file)
                print(f"Loaded face for: {name_from_file}")
            else:
                print(f"No faces found in {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

print("Face dataset ready\n")


def detect_motion(prev_frame, curr_frame):

    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    motion_score = cv2.countNonZero(thresh)

    
    if motion_score > 5000:
        return True
    else:
        return False


def run_system():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print(" Camera not detected!")
            return

    cap.set(3, 640)
    cap.set(4, 480)

    print("Camera ON → Monitoring Started\n")

    ret, prev_frame = cap.read()
    frame_count = 0
    no_motion_count = 0  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1


        cv_motion = detect_motion(prev_frame, frame)

        if PI_MODE:
            ir_motion = GPIO.input(SENSOR_PIN)
        else:
            ir_motion = True  

        motion = cv_motion or ir_motion

        if motion:
            no_motion_count = 0
            message = "Motion Detected"
        else:
            no_motion_count += 1
            message = "No Motion"
            if no_motion_count > 100:  
                print("\n Object left area. Shutting down camera...\n")
                break

        if motion:

           
            results = model(frame, imgsz=416, conf=0.65)

            for r in results:
                
                if len(r.boxes) > 0:
                    frame = r.plot()

                for box in r.boxes:
                    cls = int(box.cls[0])
                    name = model.names[cls]

                
                    if name.lower() == "person":
                        message = " Person Detected"
                    else:
                        message = f" {name.capitalize()} Detected"

           
            if frame_count % 4 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                faces = face_recognition.face_locations(rgb, number_of_times_to_upsample=2)

                if len(faces) > 0:
                    try:
                       
                        encodings = face_recognition.face_encodings(rgb, faces, num_jitters=1)
                    except:
                        encodings = []

                    for enc in encodings:
                        face_distances = face_recognition.face_distance(known_encodings, enc)
                        
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            
                            if face_distances[best_match_index] < 0.55:
                              
                                message = f"{known_names[best_match_index]}"
                            else:
                                message = "Unknown Person"
                        else:
                            message = " Unknown Person"

                   
                    for (top, right, bottom, left) in faces:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        prev_frame = frame.copy()

        
        if message and message not in ["No Motion", "Motion Detected"]:
            threading.Thread(target=send_telegram_alert, args=(message,), daemon=True).start()

       
        cv2.putText(frame, message, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2)

        cv2.imshow("Smart Forest Monitoring System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping system...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera OFF\n")


print("System Started...\n")

while True:

    if PI_MODE:
        motion = GPIO.input(SENSOR_PIN)
    else:
        motion = True

    if motion:
        print("Motion Trigger → Starting System")

        run_system()

        time.sleep(2)

    else:
        print("Waiting for motion...")
        time.sleep(1)