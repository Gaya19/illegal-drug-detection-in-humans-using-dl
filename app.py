from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.applications.xception import preprocess_input  #kotha add
from werkzeug.utils import secure_filename
import os
import json
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import SeparableConv2D
import cv2   # Camera support
USERS_FILE = "users.json" #news

# Load users from file if exists
if os.path.exists(USERS_FILE):
    with open(USERS_FILE, "r") as f:
        users = json.load(f)
else:
    users = {}


# ===============================
# Import eye verification function
# ===============================
from utils import contains_eye

# ===============================
# Paths
# ===============================
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static", "uploads")
MODEL_PATH = os.path.join(APP_ROOT, "models", "xception_model.h5")
CLASS_INDICES_PATH = os.path.join(APP_ROOT, "models", "class_indices.json")

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# Flask App Config
# ===============================
app = Flask(__name__)
app.secret_key = "change-this-secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # 15MB max upload

# ===============================
# Load Model + Class Labels
# ===============================
model = None
class_labels = {}
input_size = (299, 299)

try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH,
    custom_objects={'SeparableConv2D': SeparableConv2D})

        if hasattr(model, "input_shape") and len(model.input_shape) == 4:
            _, h, w, c = model.input_shape
            if h and w and c == 3:
                input_size = (h, w)
        print(f"✅ Model loaded successfully. Expected input size: {input_size}")

        if os.path.exists(CLASS_INDICES_PATH):
            with open(CLASS_INDICES_PATH, "r") as f:
                indices = json.load(f)
            class_labels = {v: k for k, v in indices.items()}
            print("✅ Class labels loaded:", class_labels)
        else:
            class_labels = {0: "drug", 1: "infected", 2: "normal"}
            print("⚠ Using default class labels:", class_labels)
    else:
        print(f"⚠ Model not found at {MODEL_PATH}")

except Exception as e:
    print("❌ Error loading model:", e)

# ===============================
# Recommendations
# ===============================
recommendations = {
    "drug": [
        "* Seek professional medical evaluation immediately.",
        "* Stay hydrated and maintain a nutrient-rich diet.",
        "*Avoid taking harmful drugs or alcohol.",
        "Drink enough water and keep yourself hydrated.",
        "* Get proper sleep and rest your body.",
        "* If you feel unwell, consult a doctor immediately."
    ],
    "normal": [
        "* Your eyes look healthy — maintain this lifestyle.",
        "* Eat leafy greens, carrots, and fish oil for strong eye health.",
        "* Stay hydrated and ensure proper sleep.",
        "* Limit continuous screen exposure and take breaks.",
        "* Schedule routine eye exams once a year."
    ],
    "infected": [
        "* Consult an eye doctor for proper treatment.",
        "* Do not rub or touch your eyes frequently.",
        "* Wash your hands regularly and maintain eye hygiene.",
        "* Avoid sharing towels, pillows, or cosmetics.",
        "* Use clean water or saline to gently rinse the eyes if irritated.",
        "* Avoid wearing contact lenses until recovery."
    ]
}

# ===============================
# Helper for Camera Frames
# ===============================
def preprocess_frame(frame, input_size):
    frame = cv2.resize(frame, input_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype("float32")   # delete /255.0
    frame = preprocess_input(frame)  #kotha add
    frame = np.expand_dims(frame, axis=0)
    return frame

# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if email in users:
            flash("Email already exists! Please login.")
            return redirect(url_for("login"))

        users[email] = password  # store in dictionary

        # Save users dictionary to JSON file
        with open(USERS_FILE, "w") as f:
            json.dump(users, f)

        flash("Account created successfully! Please login.")
        return redirect(url_for("login"))

    return render_template("signup.html")



@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        # Reload users from file every time
        with open(USERS_FILE, "r") as f:
            users = json.load(f)

        if email in users and users[email] == password:
            flash("Login Successful 🎉")
            return redirect(url_for("home"))
        else:
            flash("Invalid login ❌")
            return redirect(url_for("login"))

    return render_template("login.html")

   #new


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("upload_page"))

    f = request.files["file"]
    if f.filename == "":
        flash("No selected file")
        return redirect(url_for("upload_page"))

    filename = secure_filename(f.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(save_path)

    # ===============================
    # ✅ EARLY EYE VERIFICATION ADDED
    # ===============================
    if not contains_eye(save_path):
        return render_template(
            "result.html",
            prediction="❌ The uploaded image is NOT an eye. Please upload a valid eye image.",
            confidence="--",
            img_path=url_for("static", filename=f"uploads/{filename}"),
            suggestions=["Please upload a clear human eye image."]
        )
    # ===============================

    if model is None:
        flash("Model not loaded.")
        return redirect(url_for("upload_page"))

    try:
        img = image.load_img(save_path, target_size=input_size)
        img_array = image.img_to_array(img)  #/255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # kotha add

        preds = model.predict(img_array)[0]
        class_idx = int(np.argmax(preds))
        result = class_labels.get(class_idx, "unknown")
        confidence = float(preds[class_idx])

        suggested = recommendations.get(result, ["No recommendations available."])

        return render_template(
            "result.html",
            prediction=result,
            confidence=f"{confidence*100:.2f}%",
            img_path=url_for("static", filename=f"uploads/{filename}"),
            suggestions=suggested
        )

    except Exception as e:
        print("❌ Prediction error:", e)
        flash(f"Prediction failed: {e}")
        return redirect(url_for("upload_page"))


# ===============================
# Camera Route
# ===============================
@app.route("/camera")
def camera_predict():
    if model is None:
        return "❌ Model not loaded."

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "❌ Cannot access camera."

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "❌ Failed to capture frame."

    try:
        img_array = preprocess_frame(frame, input_size)

        preds = model.predict(img_array)[0]
        class_idx = int(np.argmax(preds))
        result = class_labels.get(class_idx, "unknown")
        confidence = float(preds[class_idx])

        suggested = recommendations.get(result, ["No recommendations available."])

        filename = "camera_capture.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(save_path, frame)

        return render_template(
            "result.html",
            prediction=result,
            confidence=f"{confidence*100:.2f}%",
            img_path=url_for("static", filename=f"uploads/{filename}"),
            suggestions=suggested
        )

    except Exception as e:
        print("❌ Camera prediction error:", e)
        return f"Prediction failed: {e}"

# ===============================
# Run App
# ===============================
if __name__ == "__main__":

    app.run(debug=True)
