# 🎵 Music Genre Classifier

A full-stack music genre classification system that predicts song genres from lyrics using **Apache Spark MLlib**. The project combines a TF-IDF + Logistic Regression pipeline trained on ~28,000 songs with a Flask REST API and a modern web interface.

> **MSc Big Data Analysis — 2026**
> **M Rashmi Pubuditha — 268383V**

---

## 📌 Features

- **8-Genre Classification** — Pop, Rock, Blues, Jazz, Country, Reggae, Hip Hop, Soul
- **Spark MLlib Pipeline** — Tokenizer → StopWordsRemover → HashingTF → IDF → Logistic Regression
- **Real-time Web Interface** — Paste lyrics, get instant predictions with confidence scores
- **Interactive Visualization** — Horizontal bar chart showing probability distribution across all genres
- **One-Click Launcher** — `run.bat` handles environment validation, model training, and server startup
- **REST API** — JSON-based `/predict` endpoint for programmatic access

---

## 🛠️ Tech Stack

| Layer                  | Technology                                           |
| ---------------------- | ---------------------------------------------------- |
| **ML Training**        | Apache Spark MLlib (Scala)                           |
| **Feature Extraction** | TF-IDF (65,536-dim vectors)                          |
| **Classification**     | Logistic Regression (multinomial, L2 regularization) |
| **Backend API**        | Flask + PySpark                                      |
| **Frontend**           | HTML5, CSS3, JavaScript, Chart.js                    |
| **Data Processing**    | Pandas, NumPy                                        |
| **Runtime**            | Java JDK 17, Python 3.10, Apache Spark 4.x           |

---

## 📁 Project Structure

```
MusicGenreClassifier_V2/
├── run.bat                        # One-click launcher script
├── README.md
├── dataset/
│   ├── music_dataset.csv          # Raw Mendeley dataset (7 genres)
│   ├── Mendeley_cleaned.csv       # Cleaned Mendeley dataset
│   ├── Student_dataset.csv        # Soul genre dataset (110 songs)
│   ├── Merged_dataset.csv         # Combined 8-genre training data
│   ├── dataset.py                 # Mendeley CSV cleaner
│   ├── create_student_dataset.py  # Soul dataset generator
│   └── merge_datasets.py          # Dataset merger
├── model/
│   ├── train_model.scala          # Spark MLlib training pipeline
│   ├── predict.scala              # Prediction function (Scala)
│   ├── labels.txt                 # Genre label index mapping
│   └── genre_classifier/          # Saved PipelineModel
└── web/
    ├── app.py                     # Flask REST API
    ├── index.html                 # Web frontend (SPA)
    └── venv/                      # Python virtual environment
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement           | Version                       |
| --------------------- | ----------------------------- |
| Java JDK              | 11 or later                   |
| Apache Spark          | 3.x / 4.x                     |
| Python                | 3.8+                          |
| Environment Variables | `SPARK_HOME`, `JAVA_HOME` set |

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/MusicGenreClassifier.git
   cd MusicGenreClassifier
   ```

2. **Set up the Python virtual environment**

   ```bash
   cd web
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install flask pyspark
   cd ..
   ```

3. **Run the application**
   ```bash
   run.bat
   ```
   This will:
   - Validate your Spark/Java/Python environment
   - Train the model via `spark-shell` (first run only, ~5–10 min)
   - Start the Flask server on `http://localhost:5000`
   - Open your browser automatically

---

## 🤖 ML Pipeline

### Pipeline Architecture

```
Raw Lyrics (String)
  │
  ├─ [Stage 0] Tokenizer ──────────────── words (Array)
  ├─ [Stage 1] StopWordsRemover ────────── filtered_words (Array)
  ├─ [Stage 2] HashingTF (65,536 features) ── raw_features (Vector)
  ├─ [Stage 3] IDF (minDocFreq=2) ──────── features (Vector)
  ├─ [Stage 4] StringIndexer ───────────── label (Double)
  └─ [Stage 5] LogisticRegression ──────── prediction + probability
```

### Hyperparameters

| Parameter         | Value  | Description                        |
| ----------------- | ------ | ---------------------------------- |
| TF Num Features   | 65,536 | Hashing trick vector size          |
| IDF Min Doc Freq  | 2      | Ignore terms appearing in < 2 docs |
| LR Max Iterations | 100    | Training iterations                |
| LR Reg Param (λ)  | 0.01   | L2 (Ridge) regularization          |
| Elastic Net       | 0.0    | Pure L2 regularization             |
| Train/Test Split  | 80/20  | Seed = 42 for reproducibility      |

### Evaluation

The model is evaluated on a held-out 20% test set with:

- **Overall Accuracy**
- **Weighted F1 Score**
- **Confusion Matrix** (8×8)
- **Per-Class Precision, Recall, F1**

---

## 🌐 API Reference

### `POST /predict`

Classify song lyrics into one of 8 genres.

**Request:**

```json
{
  "lyrics": "I heard it through the grapevine, not much longer would you be mine..."
}
```

**Response:**

```json
{
  "soul": 0.4523,
  "pop": 0.2102,
  "blues": 0.1403,
  "rock": 0.0912,
  "jazz": 0.0498,
  "country": 0.0289,
  "hip hop": 0.0187,
  "reggae": 0.0086
}
```

### `GET /health`

Liveness check.

```json
{
  "status": "ok",
  "genres": [
    "pop",
    "country",
    "blues",
    "rock",
    "jazz",
    "reggae",
    "hip hop",
    "soul"
  ]
}
```

---

## 🎨 Web Interface

The frontend features a **dark navy + gold** theme with:

- Lyrics input textarea with validation
- One-click classification button
- Predicted genre badge with confidence bar
- Interactive horizontal bar chart (Chart.js) showing all genre probabilities
- Loading spinner during Spark inference
- Responsive layout for desktop and mobile

---

## 📊 Dataset

| Source                | Genres                                               | Songs       | Description                |
| --------------------- | ---------------------------------------------------- | ----------- | -------------------------- |
| Mendeley              | 7 (pop, country, blues, rock, jazz, reggae, hip hop) | ~28,000     | Public lyrics dataset      |
| Student Dataset       | 1 (soul)                                             | 110         | Curated classic soul songs |
| **Merged (Training)** | **8**                                                | **~28,100** | **Combined dataset**       |

The soul dataset includes iconic artists: Aretha Franklin, Marvin Gaye, Stevie Wonder, Otis Redding, Sam Cooke, James Brown, Al Green, and more.

---

## 📝 Usage Examples

### Via Web Interface

1. Open `http://localhost:5000` in your browser
2. Paste song lyrics into the text area
3. Click **Classify Genre**
4. View the predicted genre and probability distribution

### Via Spark Shell

```bash
spark-shell
:load model/predict.scala
showPrediction("baby i love you every moment every day of my life")
```

### Via cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"lyrics\": \"I got sunshine on a cloudy day\"}"
```

---

## ⚙️ Configuration

| Setting             | Location                  | Default                  |
| ------------------- | ------------------------- | ------------------------ |
| Spark Driver Memory | `web/app.py`              | 2 GB                     |
| Spark Master        | `web/app.py`              | `local[2]`               |
| Flask Port          | `web/app.py`              | 5000                     |
| Max Lyrics Length   | `web/app.py`              | 50,000 chars             |
| Model Save Path     | `model/train_model.scala` | `model/genre_classifier` |

---

## 👤 Author

**M Rashmi Pubuditha**
