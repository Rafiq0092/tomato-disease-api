import os
import io
import time
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import gdown

# ==============================
# MODEL DOWNLOAD CONFIG
# ==============================
MODEL_URL = "https://drive.google.com/uc?export=download&id=1mWcEHDMa2WOVcF1FxkcOk-6HkpRGJfAO"
MODEL_PATH = "models/model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("⬇ Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully")

download_model()

# ==============================
# CONFIG
# ==============================
IMG_SIZE = (220, 220)
STAGE1_BEST_PATH = MODEL_PATH

CLASS_NAMES = ["Early_blight", "Healthy", "Late_blight"]

FRONTEND_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

CONFIDENCE_THRESHOLD = 0.60

# ==============================
# DISEASE KB
# ==============================
DISEASE_KB = {
    "Early_blight": {
        "pathogen": "Alternaria solani (Fungus)",
        "medicine_spray": [
            "Mancozeb 75% WP",
            "Chlorothalonil 75% WP",
            "Copper Oxychloride 50% WP",
            "Propineb 70% WP"
        ],
        "remedy": [
            "Remove infected leaves",
            "Spray recommended fungicide at 7–10 day interval",
            "Improve plant nutrition and reduce plant stress"
        ],
        "prevention": [
            "Crop rotation (2–3 years)",
            "Avoid overhead irrigation",
            "Mulching to prevent soil splash",
            "Maintain proper plant spacing",
            "Remove old plant debris"
        ]
    },
    "Late_blight": {
        "pathogen": "Phytophthora infestans (Oomycete)",
        "medicine_spray": [
            "Metalaxyl + Mancozeb",
            "Cymoxanil + Mancozeb",
            "Fluopicolide",
            "Dimethomorph"
        ],
        "remedy": [
            "Remove severely infected plants immediately",
            "Apply systemic fungicide at 5–7 day interval",
            "Stop overhead irrigation and keep foliage dry"
        ],
        "prevention": [
            "Use certified disease-free seedlings",
            "Ensure proper field drainage",
            "Avoid planting during cool and wet seasons",
            "Increase air circulation",
            "Regular field monitoring and early detection"
        ]
    },
    "Healthy": {
        "pathogen": None,
        "medicine_spray": [],
        "remedy": [
            "Continue monitoring (2–3 times/week)",
            "Maintain balanced nutrition",
            "Avoid overhead irrigation",
            "Improve airflow"
        ],
        "prevention": [
            "Keep field clean",
            "Ensure proper spacing",
            "Use healthy seedlings",
            "Monitor regularly"
        ]
    }
}

# ==============================
# FLASK APP
# ==============================
app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

MODEL = None

# ==============================
# MODEL LOADER (LAZY LOAD)
# ==============================
def load_model_once():
    global MODEL
    if MODEL is None:
        print("⚡ Loading model...")
        MODEL = tf.keras.models.load_model(STAGE1_BEST_PATH, compile=False)
        print("✅ Model loaded")
    return MODEL

# ==============================
# PREPROCESS
# ==============================
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x

# ==============================
# PREDICT
# ==============================
def predict_one(model, x):
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])

# ==============================
# ROUTES
# ==============================

# 🔥 FIXED: health does NOT load model
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "API running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model = load_model_once()

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_bytes = file.read()

        x = preprocess_image(file_bytes)

        start = time.time()
        pred_class, confidence = predict_one(model, x)
        latency = (time.time() - start) * 1000

        return jsonify({
            "predicted_class": pred_class,
            "confidence": confidence,
            "latency_ms": round(latency, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Tomato API is running"


# ==============================
# MAIN (IMPORTANT FIX)
# ==============================
if __name__ == "__main__":
    # ❌ REMOVE THIS (IMPORTANT)
    # load_model_once()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)