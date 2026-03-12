import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(input_csv="Data/features_extracted.csv",
                    output_split="Data/train_test_split.pkl",
                    scaler_path="models/feature_scaler.pkl",
                    label_encoder_path="models/label_encoder.pkl"):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 1. Handle missing/infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # 2. Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    # Save label encoder
    os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Label encoder saved to {label_encoder_path}")
    
    # 3. Separate features and labels
    X = df.drop(columns=['label', 'label_encoded'])
    y = df['label_encoded']
    
    # 4. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Feature scaler saved to {scaler_path}")
    
    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Save split
    os.makedirs(os.path.dirname(output_split), exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), output_split)
    print(f"Data split saved to {output_split}")
    
    print("Preprocessing complete!")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
