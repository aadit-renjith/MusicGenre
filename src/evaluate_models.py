import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf

print("Loading test data and models...")
DATA_SPLIT_PATH = "Data/train_test_split.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

try:
    _, X_test, _, y_test = joblib.load(DATA_SPLIT_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print(f"Loaded test data: {X_test.shape[0]} samples")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.pkl') or f.endswith('.h5')]
if not model_files:
    print("No trained models found in 'models/' directory.")
    exit(1)

class_names = label_encoder.classes_

for model_file in model_files:
    model_name = model_file.replace('_model.pkl', '').replace('.h5', '').upper()
    print(f"\n--- Evaluating {model_name} ---")
    
    model_path = os.path.join(MODELS_DIR, model_file)
    if model_file.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)
        # Reshape explicitly for CNNs
        X_test_cnn = np.expand_dims(X_test, axis=2)
        y_pred_probs = model.predict(X_test_cnn)
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Save the report
    report_path = os.path.join(OUTPUTS_DIR, f"{model_name.lower()}_evaluation.txt")
    with open(report_path, "w") as f:
        f.write(f"--- {model_name} Evaluation ---\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    print(f"[OK] Evaluation report saved to {report_path}")

print("\n[SUCCESS] All models evaluated successfully!")
