import os
import cv2
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Set dataset path
DATASET_PATH = "Cat_Dog_Dataset/train"

# New Image size (Increased from 64x64 to 128x128)
IMG_SIZE = (128, 128)

# Load dataset with RGB images instead of grayscale
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
print("Loading dataset...")
X, y = load_dataset()
print(f"Dataset Loaded: {X.shape}, Labels: {y.shape}")

# Normalize pixel values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression
print("Training Logistic Regression...")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_pred)

# Train KNN with optimized parameters
print("Training Optimized KNN...")
knn_model = KNeighborsClassifier(n_neighbors=7, weights='distance')  # Increased k and used 'distance' weights
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

# Train an SVM Model (New Model)
print("Training SVM Model...")
svm_model = SVC(kernel='rbf', C=1.0)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Compare Model Performance
models = {
    "Logistic Regression": log_accuracy,
    "KNN": knn_accuracy,
    "SVM": svm_accuracy
}

best_model_name = max(models, key=models.get)
best_accuracy = models[best_model_name]

if best_model_name == "Logistic Regression":
    best_model = log_model
elif best_model_name == "KNN":
    best_model = knn_model
else:
    best_model = svm_model

# Save best model and scaler
joblib.dump(best_model, "cat_dog_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save evaluation report
report = classification_report(y_test, best_model.predict(X_test))
with open("evaluation_report.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Accuracy: {best_accuracy * 100:.2f}%\n")
    f.write("\nClassification Report:\n" + report)

# Print results
print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {best_accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)
