import pandas as pd
from tqdm import tqdm
import librosa
import os
from extract_features import extract_full_feature_vector

DATA_DIR = "Data/genres_original"
SAMPLE_RATE = 22050

def build_feature_dataset(output_csv="Data/features_extracted.csv"):
    rows = []
    genres = os.listdir(DATA_DIR)
    for genre in tqdm(genres, desc="Extracting features"):
        genre_path = os.path.join(DATA_DIR, genre)
        if not os.path.isdir(genre_path):
            continue
        for file in os.listdir(genre_path):
            if file.endswith(".wav"):
                path = os.path.join(genre_path, file)
                try:
                    y, sr = librosa.load(path, sr=SAMPLE_RATE)
                    feats, extra_keys = extract_full_feature_vector(y, sr)
                    row = list(feats) + [genre]
                    rows.append(row)
                except Exception as e:
                    print(f"Skipping {file}: {e}")
    columns = [f"mfcc_mean_{i}" for i in range(40)] + \
              [f"mfcc_std_{i}" for i in range(40)] + extra_keys + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"[OK] Features saved to {output_csv}")

if __name__ == "__main__":
    build_feature_dataset()
