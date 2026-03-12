import librosa
import numpy as np
import os
from tqdm import tqdm

DATA_DIR = "Data/genres_original"
SAMPLE_RATE = 22050
DURATION = 30  # seconds

def load_audio_files():
    data = []
    labels = []
    genres = os.listdir(DATA_DIR)
    for genre in tqdm(genres, desc="Loading audio files"):
        genre_path = os.path.join(DATA_DIR, genre)
        if not os.path.isdir(genre_path):
            continue
        for filename in os.listdir(genre_path):
            if filename.endswith(".wav"):
                path = os.path.join(genre_path, filename)
                y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
                data.append(y)
                labels.append(genre)
    return np.array(data, dtype=object), np.array(labels)
