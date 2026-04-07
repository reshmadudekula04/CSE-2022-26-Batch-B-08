import os
import cv2
import sqlite3
import threading
import time
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, request, render_template, redirect, url_for, session, flash, Response, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from PIL import Image
import io

# ---------- YOLO CONFIG FIX ----------
os.environ['YOLO_CONFIG_DIR'] = os.path.join(os.getcwd(), 'yolo_config')
os.makedirs(os.environ['YOLO_CONFIG_DIR'], exist_ok=True)
# ------------------------------------

# ---------- CONFIGURATION ----------
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
MODEL_PATH = "best.pt"
DATABASE = 'users.db'
ALERT_SOUND = os.path.join(os.path.dirname(__file__), "alert.wav")
WEAPON_CLASSES = ['Grenade', 'Gun', 'Handgun', 'Knife']

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- GLOBAL VARIABLES ----------
camera = None
live_detection_active = False
last_detection_time = 0
last_alert_time = 0
alert_cooldown = 6
detection_interval = 0.1
current_detections = []
frame_lock = threading.Lock()

# ---------- DATABASE ----------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        detection_type TEXT,
        confidence REAL,
        image_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# ---------- MODEL ----------
try:
    MODEL = YOLO(MODEL_PATH)
    CLASS_NAMES = MODEL.names
    print("YOLOv8 model loaded:", CLASS_NAMES)
except Exception as e:
    MODEL = None
    CLASS_NAMES = []
    print("Error loading YOLO model:", e)

# ---------- HELPERS ----------
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'jfif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def play_alert():
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time < alert_cooldown:
        return False
    last_alert_time = current_time
    print("ALERT! Weapon detected!")
    return True

def detect_in_frame(frame):
    if MODEL is None:
        return frame, [], False
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = MODEL(rgb_frame, conf=0.65)
        result = results[0]

        detections = []
        weapon_detected = False

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = MODEL.names[class_id]

                detections.append({'label': class_name, 'confidence': confidence})
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"{class_name}: {confidence:.2f}"
                (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if class_name in WEAPON_CLASSES and confidence >= 0.75:
                    weapon_detected = True
        return frame, detections, weapon_detected
    except Exception as e:
        print("Detection error:", e)
        return frame, [], False

def generate_frames():
    global camera, live_detection_active, last_detection_time, current_detections
    while True:
        if camera is None or not live_detection_active:
            break
        success, frame = camera.read()
        if not success:
            break

        current_time = time.time()
        if current_time - last_detection_time > detection_interval:
            processed_frame, detections, weapon_detected = detect_in_frame(frame.copy())
            with frame_lock:
                current_detections = detections
            last_detection_time = current_time
            if weapon_detected:
                play_alert()
            frame = processed_frame

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ---------- ROUTES ----------
@app.route('/')
def index():
    return render_template('index.html')

# (Include all other routes: about, login, register, home, predict, live, video_feed, start_live, stop_live, logout)
# ... [Keep your existing route implementations here] ...

# ---------- MAIN ----------
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 10000))  # Render port
    print(f"Flask app running on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
