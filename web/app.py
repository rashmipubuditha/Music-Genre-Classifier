# ============================================================
#  Music Genre Classifier — Flask Backend  (Step 7)
#  Run from the project root:
#    python web/app.py
#
#  Requires:
#    pip install flask pyspark
#    The model must already be trained and saved to model/genre_classifier
# ============================================================

import os
import sys
import logging

# ── Force PySpark to use THIS Python executable for all worker processes ──────
# Must be set BEFORE importing pyspark, so workers inherit the venv Python.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from flask import Flask, request, jsonify, send_from_directory
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# ── Paths ────────────────────────────────────────────────────
# All paths are relative to the project root (one level up from web/)
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR     = os.path.join(BASE_DIR, "web")
MODEL_PATH  = os.path.join(BASE_DIR, "model", "genre_classifier")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")

# ── Flask app ────────────────────────────────────────────────
app = Flask(__name__, static_folder=WEB_DIR)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Spark session (singleton — created once at startup) ──────
log.info(f"Starting PySpark with Python: {sys.executable}")
log.info("Initialising Spark session ...")
spark = (SparkSession.builder
    .appName("MusicGenreClassifier-API")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
log.info("Spark session ready.")

# ── Load model & labels once at startup ──────────────────────
log.info(f"Loading pipeline model from {MODEL_PATH} …")
pipeline_model = PipelineModel.load(MODEL_PATH)
log.info("Model loaded.")

with open(LABELS_PATH, "r", encoding="utf-8") as fh:
    genre_labels = [line.strip() for line in fh if line.strip()]
log.info(f"Genre labels: {genre_labels}")


# ─────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page frontend."""
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body : { "lyrics": "<song text>" }
    Returns: { "pop": 0.12, "rock": 0.45, … }
    """
    # ── Input validation ──────────────────────────────────────
    body = request.get_json(silent=True)
    if not body or "lyrics" not in body:
        return jsonify({"error": "Request body must contain a 'lyrics' field."}), 400

    lyrics = str(body["lyrics"]).strip()
    if not lyrics:
        return jsonify({"error": "The 'lyrics' field must not be empty."}), 400

    if len(lyrics) > 50_000:
        return jsonify({"error": "Lyrics too long (max 50 000 characters)."}), 400

    # ── Run prediction ────────────────────────────────────────
    try:
        input_df = spark.createDataFrame([(lyrics,)], ["lyrics"])
        result   = pipeline_model.transform(input_df)
        prob_vec = result.select("probability").first()["probability"]
    except Exception as exc:
        log.exception("Prediction failed")
        return jsonify({"error": f"Prediction error: {str(exc)}"}), 500

    # ── Build response ────────────────────────────────────────
    probabilities = {
        genre_labels[i]: round(float(prob_vec[i]), 4)
        for i in range(len(genre_labels))
    }
    return jsonify(probabilities)


@app.route("/health")
def health():
    """Simple liveness check — useful for run.bat to poll readiness."""
    return jsonify({"status": "ok", "genres": genre_labels})


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
