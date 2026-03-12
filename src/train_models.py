import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

print("Loading preprocessed data...")
DATA_SPLIT_PATH = "Data/train_test_split.pkl"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

try:
    X_train, X_test, y_train, y_test = joblib.load(DATA_SPLIT_PATH)
    print(f"Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
except Exception as e:
    print(f"Error loading {DATA_SPLIT_PATH}: {e}")
    exit(1)

# Initialize models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
}

trained_models = {}

# Train Models
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Save the model
    model_path = os.path.join(MODELS_DIR, f"{name.lower()}_model.pkl")
    joblib.dump(model, model_path)
    print(f"[OK] {name} trained and saved to {model_path}")

print("\n[SUCCESS] All models trained successfully!")
