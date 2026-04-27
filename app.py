import os
import io
import time
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf

import gdown

MODEL_URL = "https://drive.google.com/uc?export=download&id=1mWcEHDMa2WOVcF1FxkcOk-6HkpRGJfAO"
MODEL_PATH = "models/model.h5"


def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("⬇ Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully")


download_model()

IMG_SIZE = (220 , 220)
STAGE1_BEST_PATH = MODEL_PATH

#classes
CLASS_NAMES = ["Early_blight", "Healthy", "Late_blight"]

 #Live Server runs at 5500
FRONTEND_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

CONFIDENCE_THRESHOLD = 0.60

# about Disease Knowledge 

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
            "Maintain balanced nutrition (avoid excess nitrogen)",
            "Avoid overhead irrigation; keep foliage dry",
            "Remove weeds and improve airflow"
        ],
        "prevention": [
            "Keep field clean and remove debris",
            "Ensure proper spacing and airflow",
            "Use healthy seed/seedlings",
            "Monitor regularly for early symptoms"
        ]
    }
}

# Flask App

app = Flask(__name__)

# allow frontend to call /health and /predict
CORS(
    app,
    resources={r"/*": {"origins": FRONTEND_ORIGINS}},
    supports_credentials=False
)

MODEL = None


def load_model_once():
    global MODEL
    if MODEL is None:
        if not os.path.exists(STAGE1_BEST_PATH):
            raise FileNotFoundError(f"Model not found at: {STAGE1_BEST_PATH}")
        MODEL = tf.keras.models.load_model(STAGE1_BEST_PATH, compile=False)
    return MODEL


def preprocess_image(file_bytes: bytes, img_size=(192, 192)) -> np.ndarray:
  
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(img_size)
    x = np.asarray(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def advisory_for(pred_class: str) -> dict:
    
    info = DISEASE_KB.get(pred_class, None)
    if info is None:
        return {
            "pathogen": None,
            "medicine_spray": [],
            "remedy": ["Advisory not available. Please consult an expert."],
            "prevention": []
        }
    # Ensure keys exist
    return {
        "pathogen": info.get("pathogen", None),
        "medicine_spray": info.get("medicine_spray", []),
        "remedy": info.get("remedy", []),
        "prevention": info.get("prevention", [])
    }


def predict_one(model, x: np.ndarray):
    probs = model.predict(x, verbose=0)[0].astype(np.float32)
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    return probs, pred_idx, pred_class, confidence


# Routes (for frontend)
@app.route("/health", methods=["GET"])
def health():
    try:
        _ = load_model_once()
        return jsonify({
            "status": "ok",
            "classes": CLASS_NAMES,
            "img_size": list(IMG_SIZE),
            "model_path": STAGE1_BEST_PATH
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model = load_model_once()

        # Frontend sends
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Use form-data key='file'."}), 400

        f = request.files["file"]
        if not f or f.filename.strip() == "":
            return jsonify({"error": "Empty file."}), 400

        file_bytes = f.read()
        if not file_bytes:
            return jsonify({"error": "Empty file bytes."}), 400

        x = preprocess_image(file_bytes, IMG_SIZE)

        t0 = time.time()
        probs, pred_idx, pred_class, confidence = predict_one(model, x)
        latency_ms = (time.time() - t0) * 1000.0

        
        advisory = advisory_for(pred_class)

        # optional warning
        warning = None
        if confidence < CONFIDENCE_THRESHOLD:
            warning = (
                "Capture a clearer image: single leaf, good light, no blur, "
                "minimal background, close-up symptoms visible."
            )

        # JSON response schema
        return jsonify({
            "filename": f.filename,
            "predicted_class": pred_class,
            "latency_ms": round(latency_ms, 2),
            "advisory": advisory,
            "warning": warning
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reload", methods=["POST"])
def reload_model():
    global MODEL
    try:
        MODEL = None
        tf.keras.backend.clear_session()
        load_model_once()
        return jsonify({"status": "reloaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    load_model_once()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)