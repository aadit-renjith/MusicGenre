# Music Genre Classification

This project implements an end-to-end machine learning pipeline to classify music audio files into genres based on extracted features (like MFCCs).

## Project Structure

```text
Music_Genre/
│
├── Data/                   # Directory containing data (original audio, extracted CSVs, train/test split)
├── models/                 # Saved machine learning models, scalers, and label encoders
├── outputs/                # Evaluation results and metrics
├── src/                    # Source code directory
│   ├── extract_features.py       # Functions to extract MFCCs and other audio features
│   ├── build_feature_dataset.py  # Script to process all audio and save features to CSV
│   ├── preprocess_data.py        # Scale features and create train/test splits
│   ├── train_models.py           # Train RandomForest, SVM, and XGBoost models
│   ├── evaluate_models.py        # Evaluate trained models and save metrics
│   ├── verify_dataset.py         # Helper to check audio file validity
│   └── load_audio.py             # Basic audio loading utilities
├── test.py                 # Script to test and inspect the extracted dataset
├── requirements.txt        # Python dependencies
└── .gitignore              # Git ignore rules
```

## Setup Instructions

1. **Install Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline end-to-end

Follow these steps to generate features, train the models, and evaluate them:

### 1. Verification & Feature Extraction
Ensure your `Data/genres_original/` folder contains the audio files. Then verify the files and build the feature dataset:
```bash
python src/verify_dataset.py
python src/build_feature_dataset.py
```
*Note: This will create `Data/features_extracted.csv`.*

### 2. Preprocessing
Clean the data, scale features, encode labels, and create standard train/test splits:
```bash
python src/preprocess_data.py
```

### 3. Model Training
Train multiple classifiers (RandomForest, SVM, XGBoost):
```bash
python src/train_models.py
```
*Models are saved to the `models/` directory.*

### 4. Model Evaluation
Evaluate the models and generate performance reports:
```bash
python src/evaluate_models.py
```
*The metrics will be printed to the console and saved in `outputs/`.*

## Model Performance Summary

After training and testing, the Support Vector Machine (SVM) currently achieves the highest accuracy at around **72%**. XGBoost (69.5%) and Random Forest (66.5%) follow closely.
