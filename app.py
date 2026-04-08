import os

# ---------- FIX FOR RENDER ----------
os.environ['YOLO_CONFIG_DIR'] = os.path.join(os.getcwd(), 'yolo_config')
os.makedirs(os.environ['YOLO_CONFIG_DIR'], exist_ok=True)

import sqlite3
import cv2
from flask import Flask, request, render_template, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# TRY IMPORT YOLO SAFELY
try:
    from ultralytics import YOLO
except:
    YOLO = None

# ---------- CONFIG ----------
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
DATABASE = os.path.join(os.getcwd(), 'users.db')  # FIXED
MODEL_PATH = "best.pt"

app = Flask(__name__)
app.secret_key = "secret_key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODEL = None

# ---------- DATABASE ----------
def init_db():
    conn = sqlite3.connect(DATABASE)
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT
    )''')
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# ---------- FILE CHECK ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png','jpg','jpeg']

# ---------- DETECTION ----------
def detect_image(image_path):
    global MODEL

    # LOAD MODEL ONLY WHEN NEEDED (IMPORTANT FIX)
    if MODEL is None:
        try:
            if YOLO is None or not os.path.exists(MODEL_PATH):
                print("⚠️ Model not available, skipping detection")
                return None, [], False

            print("⏳ Loading model...")
            MODEL = YOLO(MODEL_PATH)
            print("✅ Model loaded")

        except Exception as e:
            print("❌ Model error:", e)
            return None, [], False

    try:
        results = MODEL(image_path)
        result = results[0]

        plotted = result.plot()
        output_name = "out.jpg"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name)

        cv2.imwrite(output_path, plotted)

        return output_name, [], True

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

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        try:
            db = get_db()
            db.execute(
                "INSERT INTO users (username,email,password) VALUES (?,?,?)",
                (
                    request.form['username'],
                    request.form['email'],
                    generate_password_hash(request.form['password'])
                )
            )
            db.commit()
            db.close()
            return redirect('/login')

        except:
            flash("User already exists")

    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username=?",
            (request.form['username'],)
        ).fetchone()
        db.close()

        if user and check_password_hash(user['password'], request.form['password']):
            session['user'] = user['id']
            return redirect('/home')

        flash("Invalid login")

    return render_template('login.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect('/login')
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'user' not in session:
        return redirect('/login')

    if request.method == 'POST':
        file = request.files.get('image')

        if not file or file.filename == '':
            flash("No file")
            return redirect(request.url)

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            output, _, _ = detect_image(path)

            if output:
                return render_template(
                    'prediction.html',
                    image_url=url_for('static', filename='outputs/' + output),
                    result="Detection Done"
                )

            else:
                return render_template(
                    'prediction.html',
                    result="⚠️ Model not working on server"
                )

    return render_template('prediction.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ---------- START ----------
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
