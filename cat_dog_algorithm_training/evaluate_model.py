import os
import joblib
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load trained model and scaler
model = joblib.load("cat_dog_model.pkl")
scaler = joblib.load("scaler.pkl")

# Dataset path (change to test folder)
DATASET_PATH = "Cat_Dog_Dataset/test"
IMG_SIZE = (128, 128)  # Match training size

# Load dataset (Must match training format)
def load_dataset():
    X, y = [], []
    for label, category in enumerate(["cat", "dog"]):
        folder_path = os.path.join(DATASET_PATH, category)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)  # Load in RGB
            if img is not None:
                img = cv2.resize(img, IMG_SIZE).flatten()  # Resize & Flatten
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

# Load dataset
print("Loading test dataset...")
X, y = load_dataset()
print(f"Test Dataset Loaded: {X.shape}, Labels: {y.shape}")

# Normalize using the same scaler
X_scaled = scaler.transform(X)  # Now dimensions will match

# Predict and evaluate
y_pred = model.predict(X_scaled)

accuracy = accuracy_score(y, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
