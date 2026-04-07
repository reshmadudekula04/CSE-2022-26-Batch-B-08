import os
import cv2
import sqlite3
import threading
import time
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, session, flash, Response, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import numpy as np

# ---------- CONFIGURATION ----------
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
MODEL_PATH = "best.pt"
DATABASE = 'users.db'
ALERT_SOUND = os.path.join(os.path.dirname(__file__), "alert.wav")
WEAPON_CLASSES = ['Grenade', 'Gun', 'Handgun', 'Knife']  # Weapon labels

# ---------- APP SETUP ----------
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

# ---------- DATABASE SETUP ----------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            detection_type TEXT,
            confidence REAL,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# ---------- MODEL SETUP ----------
try:
    MODEL = YOLO(MODEL_PATH)
    CLASS_NAMES = MODEL.names
    print("YOLOv8 model loaded:", CLASS_NAMES)
except Exception as e:
    MODEL = None
    CLASS_NAMES = []
    print("Error loading YOLO model:", e)

# ---------- HELPER FUNCTIONS ----------
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'jfif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def play_alert():
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time < alert_cooldown:
        return False
    last_alert_time = current_time
    # Placeholder for sound
    # Uncomment if using winsound on Windows
    # import winsound
    # winsound.PlaySound(ALERT_SOUND, winsound.SND_ASYNC)
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

                detections.append({
                    'label': class_name,
                    'confidence': confidence,
                    'bbox': box.xyxy[0].tolist()
                })

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

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['c_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters!', 'error')
            return render_template('register.html')

        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, hashed_password)
            )
            conn.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
            return render_template('register.html')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        flash('Invalid username or password!', 'error')
    return render_template('login.html')

@app.route('/home')
def home():
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file selected!', 'error')
            return redirect(request.url)

        file = request.files['image']
        if not file or file.filename == '':
            flash('No file selected!', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not filename:
                filename = f"upload_{int(time.time())}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            results = MODEL(image_path, conf=0.75)
            result = results[0]

            detections = []
            weapon_detected = False

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = MODEL.names[class_id]
                    detections.append({'label': class_name, 'confidence': confidence})
                    if class_name in WEAPON_CLASSES and confidence >= 0.65:
                        weapon_detected = True

            if weapon_detected:
                play_alert()
                result_text = "Weapon detected! Alert triggered!"
                flash('Weapon detected! Security alert!', 'warning')
            else:
                result_text = "No weapons detected."
                flash('Scan completed. No weapons found.', 'success')

            output_filename = f"detected_{filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            cv2.imwrite(output_path, result.plot())
            return render_template('prediction.html', result=result_text,
                                   image_url=url_for('static', filename=f'outputs/{output_filename}'),
                                   detections=detections)
        flash('Invalid file type!', 'error')
        return redirect(request.url)
    return render_template('prediction.html', result=None, image_url=None, detections=[])

@app.route('/live')
def live():
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return Response('Unauthorized', status=401)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_live', methods=['POST'])
def start_live():
    global camera, live_detection_active, last_alert_time, current_detections
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    last_alert_time = 0
    current_detections = []
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        return jsonify({'error': 'Cannot open camera'}), 500
    live_detection_active = True
    return jsonify({'status': 'success', 'message': 'Live detection started'})

@app.route('/stop_live', methods=['POST'])
def stop_live():
    global camera, live_detection_active, current_detections
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    live_detection_active = False
    current_detections = []
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'success', 'message': 'Live detection stopped'})

@app.route('/logout')
def logout():
    global camera, live_detection_active, current_detections
    if camera is not None:
        camera.release()
        camera = None
    live_detection_active = False
    current_detections = []
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

# ---------- MAIN ----------
if __name__ == '__main__':
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
