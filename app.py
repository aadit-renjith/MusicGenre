"""
Music Genre Classification API
FastAPI backend that serves the trained SVM model.
Upload a .wav or .mp3 file and get the predicted genre.
"""

import os
import io
import tempfile
import numpy as np
import joblib
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ── Feature extraction (mirrors src/extract_features.py) ──────────────

def extract_features(y, sr, n_mfcc=40):
    """Extract the same 87-dim feature vector used during training."""
    # MFCCs (40 means + 40 stds = 80 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Additional features (7 features)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma)
    chroma_std = np.std(chroma)

    return np.concatenate([
        mfcc_mean, mfcc_std,
        [zcr, rms, spectral_centroid, spectral_bandwidth,
         spectral_rolloff, chroma_mean, chroma_std]
    ])

# ── Load model artifacts at startup ───────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

model = None
scaler = None
label_encoder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, label_encoder
    model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print("[OK] Model, scaler, and label encoder loaded.")
    yield

# ── FastAPI app ───────────────────────────────────────────────────────

app = FastAPI(
    title="Music Genre Classifier",
    description="Upload an audio file to predict its music genre.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "online", "message": "Music Genre Classifier API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_genre(file: UploadFile = File(...)):
    """Accept an audio file and return the predicted genre with confidence."""

    # Validate file type
    allowed = (".wav", ".mp3", ".ogg", ".flac")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Accepted: {', '.join(allowed)}"
        )

    # Save upload to temp file (librosa needs a file path)
    try:
        contents = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    try:
        # Load audio
        y, sr = librosa.load(tmp_path, sr=22050, duration=30)

        # Extract features
        features = extract_features(y, sr)

        # Handle NaN / Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Predict
        predicted_index = model.predict(features_scaled)[0]
        predicted_genre = label_encoder.inverse_transform([predicted_index])[0]

        # Confidence (SVM decision_function → softmax-like probabilities)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = {
                label_encoder.inverse_transform([i])[0]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            }
        else:
            confidence = {predicted_genre: 1.0}

        return {
            "predicted_genre": predicted_genre,
            "confidence": confidence,
            "filename": file.filename,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        os.unlink(tmp_path)
