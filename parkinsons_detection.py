# Parkinson's Disease Detection - Full Implementation with Kaggle Download

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# ------------------- Paths -------------------
DATA_DIR = "data"
VIS_DIR = "visualizations"
CSV_FILE = "parkinsons.csv"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# ------------------- Step 1: Download dataset from Kaggle -------------------
if not os.path.exists(os.path.join(DATA_DIR, CSV_FILE)):
    print("Downloading latest dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("thedevastator/unlocking-clues-to-parkinson-s-disease-progressi")
    print(f"Dataset downloaded at {dataset_path}")

    # Find CSV inside downloaded dataset
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in Kaggle dataset!")

    # Copy CSV to local data folder
    shutil.copy(os.path.join(dataset_path, csv_files[0]), os.path.join(DATA_DIR, CSV_FILE))
    print(f"CSV file copied to {DATA_DIR}/{CSV_FILE}")

# ------------------- Step 2: Load dataset -------------------
df = pd.read_csv(os.path.join(DATA_DIR, CSV_FILE))
print(f"Dataset loaded: {CSV_FILE} ({df.shape[0]} rows, {df.shape[1]} columns)")

# ------------------- Step 3: Preprocessing -------------------
# Drop irrelevant columns
df.columns = df.columns.str.strip().str.lower()  # lowercase & strip spaces
X = df.drop(['index', 'subject#', 'total_updrs'], axis=1)
y = df['total_updrs']  # target for regression

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------- Step 4: Train-test split -------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------- Step 5: Train model -------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------- Step 6: Prediction & Evaluation -------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# ------------------- Step 7: Visualization -------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Total UPDRS")
plt.ylabel("Predicted Total UPDRS")
plt.title("Random Forest Regression: Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # diagonal line
plt.savefig(os.path.join(VIS_DIR, "plots.png"))
plt.show()
print(f"Plot saved to {VIS_DIR}/plots.png")
