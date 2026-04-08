import os

# ✅ YOLO CONFIG FIX (Render writable dir)
os.environ['YOLO_CONFIG_DIR'] = os.path.join(os.getcwd(), 'yolo_config')
os.makedirs(os.environ['YOLO_CONFIG_DIR'], exist_ok=True)

import cv2
import sqlite3
from ultralytics import YOLO
from flask import Flask, request, render_template, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# ---------- CONFIG ----------
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
MODEL_PATH = "best.pt"
DATABASE = 'users.db'
WEAPON_CLASSES = ['Grenade', 'Gun', 'Handgun', 'Knife']

app = Flask(__name__)
app.secret_key = 'super_secret_key_change_this'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- GLOBAL ----------
MODEL = None

# ---------- DATABASE ----------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# ---------- LOAD MODEL ----------
def load_model():
    global MODEL
    try:
        if not os.path.exists(MODEL_PATH):
            print("❌ best.pt NOT FOUND")
            return

        MODEL = YOLO(MODEL_PATH)
        print("✅ YOLO model loaded")

    except Exception as e:
        print("❌ Model load failed:", e)

load_model()

# ---------- HELPERS ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {
        'png','jpg','jpeg','gif','bmp','jfif'
    }

def detect_image(image_path):
    if MODEL is None:
        return None, [], False

    try:
        results = MODEL(image_path, conf=0.65)
        result = results[0]

        detections = []
        weapon_detected = False

        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = MODEL.names[cls]

                detections.append({
                    'label': label,
                    'confidence': round(conf, 2)
                })

                if label in WEAPON_CLASSES and conf >= 0.65:
                    weapon_detected = True

        return result, detections, weapon_detected

    except Exception as e:
        print("❌ Detection error:", e)
        return None, [], False

# ---------- ROUTES ----------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE username=?",
            (request.form['username'],)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], request.form['password']):
            session['user_id'] = user['id']
            return redirect('/home')

        flash("Invalid login", "error")

    return render_template('login.html')


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        try:
            conn = get_db_connection()
            conn.execute(
                "INSERT INTO users (username,email,password) VALUES (?,?,?)",
                (
                    request.form['username'],
                    request.form['email'],
                    generate_password_hash(request.form['password'])
                )
            )
            conn.commit()
            conn.close()

            flash("Registration successful!", "success")
            return redirect('/login')

        except Exception as e:
            print("❌ Register error:", e)
            flash("User already exists", "error")

    return render_template('register.html')


@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect('/login')

    return render_template('home.html')


# ✅ FIXED LIVE ROUTE (was missing before)
@app.route('/live')
def live():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('live.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'user_id' not in session:
        return redirect('/login')

    if request.method == 'POST':

        if 'image' not in request.files:
            flash("No file uploaded", "error")
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            flash("No file selected", "error")
            return redirect(request.url)

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result, detections, weapon_detected = detect_image(filepath)

            if result is None:
                flash("Model not loaded or detection failed", "error")
                return redirect(request.url)

            try:
                output_name = "detected_" + filename
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name)

                plotted_img = result.plot()

                if plotted_img is not None:
                    cv2.imwrite(output_path, plotted_img)
                else:
                    flash("Image processing failed", "error")
                    return redirect(request.url)

            except Exception as e:
                print("❌ Image save error:", e)
                flash("Error processing image", "error")
                return redirect(request.url)

            message = "🚨 Weapon detected!" if weapon_detected else "✅ No weapon detected"

            return render_template(
                'prediction.html',
                result=message,
                image_url=url_for('static', filename=f'outputs/{output_name}'),
                detections=detections
            )

    return render_template('prediction.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


# ---------- INIT ----------
init_db()
