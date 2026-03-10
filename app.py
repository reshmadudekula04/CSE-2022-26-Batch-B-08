
import os
import cv2
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, request, render_template, redirect, url_for, session, flash, Response, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import numpy as np
from PIL import Image
import winsound
import base64
import io

# ---------- CONFIGURATION ----------
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
MODEL_PATH = "best.pt"  # Your YOLOv8s model
DATABASE = 'users.db'
ALERT_SOUND = os.path.join(os.path.dirname(__file__), "alert.wav")

# Class names based on your dataset
CLASS_NAMES = []
WEAPON_CLASSES = ['Grenade', 'Gun', 'Handgun', 'Knife']  # All are weapons

# ---------- APP SETUP ----------
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global variables for live detection
camera = None
live_detection_active = False
last_detection_time = 0
last_alert_time = 0
alert_cooldown = 6  # 10 seconds cooldown between alerts
detection_interval = 0.1  # Process 2 frames per second
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
def load_model():
    """Load YOLOv8 model"""
    try:
        model = YOLO(MODEL_PATH)
        print("YOLOv8 model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
MODEL = load_model()

# Load model at startup
if MODEL is not None:
    CLASS_NAMES = MODEL.names
    print("Model Classes:", CLASS_NAMES)

def play_alert():
    """Play alert sound when weapon is detected"""
    global last_alert_time
    
    current_time = time.time()
    if current_time - last_alert_time < alert_cooldown:
        return False  # Still in cooldown
    
    if os.path.exists(ALERT_SOUND):
        try:
            winsound.PlaySound(ALERT_SOUND, winsound.SND_ASYNC)
            last_alert_time = current_time
            return True
        except Exception as e:
            print(f"Error playing alert sound: {e}")
            
            return True
    else:
        
        return True
    return False

def detect_in_frame(frame):
    """Detect weapons in a frame with proper bounding boxes"""
    if MODEL is None:
        return frame, []
    
    try:
        # Convert frame to RGB (OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference with higher confidence threshold
        results = MODEL(rgb_frame, conf=0.65)  # Confidence threshold 50%
        result = results[0]
        
        detections = []
        weapon_detected = False
        
        # Process detections
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Get class name from model
                class_name = MODEL.names[class_id]

                # Debug print (this shows what the model actually detects)
                print("Detected:", class_name, "Confidence:", confidence)
                
                # Add to detections list
                detections.append({
                    'label': class_name,
                    'confidence': confidence,
                    'bbox': box.xyxy[0].tolist() if hasattr(box.xyxy[0], 'tolist') else box.xyxy[0]
                })
                
                # Draw bounding box with better visibility
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw thicker bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Draw filled background for label
                label = f"{class_name}: {confidence:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 0, 255), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Check if weapon is detected
                if class_name in WEAPON_CLASSES and confidence >= 0.65:
                    weapon_detected = True
        
        return frame, detections, weapon_detected
        
    except Exception as e:
        print(f"Detection error: {e}")
        return frame, [], False

def generate_frames():
    """Generate frames for live stream with proper detection and alert timing"""
    global camera, live_detection_active, last_detection_time, current_detections
    
    while True:
        if camera is None or not live_detection_active:
            break
            
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame for detection at intervals
        current_time = time.time()
        if current_time - last_detection_time > detection_interval:
            processed_frame, detections, weapon_detected = detect_in_frame(frame.copy())
            
            with frame_lock:
                current_detections = detections
            
            last_detection_time = current_time
            
            # Play alert if weapon detected (with cooldown)
            if weapon_detected:
                alert_played = play_alert()
                if alert_played:
                    print(f"Weapon detected! Alert played at {datetime.now()}")
            
            frame = processed_frame
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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
        
        # Basic validation
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, hashed_password)
            )
            conn.commit()
            flash('Registration successful! Please login.', 'success')
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
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
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
        # Debug: Print request data
        print("Files in request:", request.files)
        print("Form data:", request.form)
        
        if 'image' not in request.files:
            flash('No file selected!', 'error')
            print("No 'image' field in request.files")
            return redirect(request.url)
        
        file = request.files['image']
        print("File received:", file)
        print("Filename:", file.filename)
        print("Content type:", file.content_type)
        
        # Check if file is selected
        if not file or file.filename == '':
            flash('No file selected! Please choose an image.', 'error')
            print("File is empty or no filename")
            return redirect(request.url)
        
        # Check if file has a valid extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print("Secure filename:", filename)
            
            # Ensure filename is not empty
            if not filename:
                filename = f"upload_{int(time.time())}.jpg"
                print("Generated filename:", filename)
            
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("Saving to:", image_path)
            
            try:
                # Save the file
                file.save(image_path)
                print("File saved successfully")
                
                # Check if file was actually saved
                if not os.path.exists(image_path):
                    flash('Failed to save uploaded file.', 'error')
                    return redirect(request.url)
                
                # Perform detection
                try:
                    # Run YOLOv8 inference
                    print("Running detection...")
                    results = MODEL(image_path)
                    result = results[0]
                    
                    detections = []
                    weapon_detected = False
                    
                    # Process detections
                    if result.boxes is not None:
                        print(f"Found {len(result.boxes)} detections")
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = MODEL.names[class_id]
                            
                            detections.append({
                                'label': class_name,
                                'confidence': confidence
                            })
                            
                            # Check if weapon is detected
                            if class_name in WEAPON_CLASSES and confidence >= 0.65:
                                weapon_detected = True
                    
                    print(f"Weapon detected: {weapon_detected}")
                    print(f"Detections: {detections}")
                    
                    # Play alert if weapon detected
                    if weapon_detected:
                        play_alert()
                        result_text = "Weapon detected! Alert triggered!"
                        flash('Weapon detected! Security alert!', 'warning')
                    else:
                        result_text = "No weapons detected."
                        flash('Scan completed. No weapons found.', 'success')
                    
                    # Save output image with detections
                    output_filename = f"detected_{filename}"
                    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                    
                    # Plot results on image
                    plotted_image = result.plot()
                    cv2.imwrite(output_path, plotted_image)
                    
                    return render_template('prediction.html',
                                        result=result_text,
                                        image_url=url_for('static', filename=f'outputs/{output_filename}'),
                                        detections=detections)
                
                except Exception as e:
                    print(f"Error during detection: {str(e)}")
                    flash(f'Error during detection: {str(e)}', 'error')
                    return redirect(request.url)
            
            except Exception as e:
                print(f"Error saving file: {str(e)}")
                flash(f'Error saving file: {str(e)}', 'error')
                return redirect(request.url)
        
        else:
            print(f"Invalid file type: {file.filename}")
            flash('Invalid file type! Please upload JPG, PNG, GIF, or BMP image.', 'error')
            return redirect(request.url)
    
    # GET request - show upload form
    return render_template('prediction.html', result=None, image_url=None, detections=[])

# Live Detection Routes
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
    
    try:
        # Reset alert cooldown
        last_alert_time = 0
        current_detections = []
        
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not camera.isOpened():
            return jsonify({'error': 'Cannot open camera'}), 500
        
        live_detection_active = True
        return jsonify({'status': 'success', 'message': 'Live detection started'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_live', methods=['POST'])
def stop_live():
    global camera, live_detection_active, current_detections
    
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        live_detection_active = False
        current_detections = []
        
        if camera is not None:
            camera.release()
            camera = None
        
        return jsonify({'status': 'success', 'message': 'Live detection stopped'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_detections', methods=['GET'])
def get_detections():
    """Get current detections from live feed"""
    global current_detections, last_alert_time
    
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    with frame_lock:
        detections = current_detections.copy()
    
    # Check if any weapons are detected
    weapon_detected = any(
        d['label'] in WEAPON_CLASSES and d['confidence'] >= 0.65
        for d in detections
   )
    
    # Check alert cooldown
    current_time = time.time()
    can_alert = current_time - last_alert_time >= alert_cooldown
    
    return jsonify({
        'status': 'success',
        'detections': detections,
        'weapon_detected': weapon_detected,
        'can_alert': can_alert,
        'alert_cooldown_remaining': max(0, alert_cooldown - (current_time - last_alert_time))
    })

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    global camera
    if camera is None:
        return jsonify({'error': 'Camera not active'}), 400
    
    try:
        success, frame = camera.read()
        if not success:
            return jsonify({'error': 'Failed to capture frame'}), 500
        
        # Process the captured frame
        processed_frame, detections, weapon_detected = detect_in_frame(frame.copy())
        
        # Save the captured frame
        timestamp = int(time.time())
        filename = f"capture_{timestamp}.jpg"
        capture_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(capture_path, processed_frame)
        
        return jsonify({
            'status': 'success',
            'weapon_detected': weapon_detected,
            'detections': detections,
            'image_url': url_for('static', filename=f'uploads/{filename}')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_alert_status', methods=['GET'])
def get_alert_status():
    """Get current alert cooldown status"""
    global last_alert_time
    
    current_time = time.time()
    time_since_last_alert = current_time - last_alert_time
    cooldown_remaining = max(0, alert_cooldown - time_since_last_alert)
    
    return jsonify({
        'status': 'success',
        'last_alert_time': last_alert_time,
        'time_since_last_alert': time_since_last_alert,
        'cooldown_remaining': cooldown_remaining,
        'can_alert': cooldown_remaining <= 0
    })

@app.route('/logout')
def logout():
    # Stop live detection if active
    global camera, live_detection_active, current_detections
    if camera is not None:
        camera.release()
        camera = None
    live_detection_active = False
    current_detections = []
    
    session.clear()
    flash('You have been logged out successfully!', 'success')
    return redirect(url_for('index'))

# ---------- HELPER FUNCTIONS ----------
def allowed_file(filename):
    """Check if the file has an allowed extension"""
    if not filename:
        return False
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'jfif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- MAIN ----------
if __name__ == '__main__':
    init_db()
    if MODEL is None:
        print("Warning: Model failed to load. Please check the model file.")
    
    # Cleanup on exit
    import atexit
    @atexit.register
    def cleanup():
        global camera
        if camera is not None:
            camera.release()
    
    # Run the app
    print("Starting Flask server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Alert cooldown: {alert_cooldown} seconds")
    
    # Ensure directories exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        print(f"Created upload folder: {UPLOAD_FOLDER}")
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        print(f"Created output folder: {OUTPUT_FOLDER}")
    
    app.run(debug=True, threaded=True, port=5000)