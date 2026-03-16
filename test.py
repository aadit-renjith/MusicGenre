import pandas as pd

# Load your feature CSV safely
df = pd.read_csv("Data/features_extracted.csv")


print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")

# Inspect first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum().sum(), "missing values in total")

# Check label distribution
print(df['label'].value_counts())
