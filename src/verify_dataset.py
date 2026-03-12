import os
import librosa

DATA_DIR = "Data/genres_original"

def verify_audio_files(data_dir):
    for genre in os.listdir(data_dir):
        genre_path = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_path):
            continue
        count = 0
        for file in os.listdir(genre_path):
            if file.endswith(".wav"):
                try:
                    y, sr = librosa.load(os.path.join(genre_path, file))
                    count += 1
                except Exception as e:
                    print(f"❌ Error reading {file}: {e}")
        print(f"✅ {genre}: {count} valid audio files")

if __name__ == "__main__":
    verify_audio_files(DATA_DIR)
