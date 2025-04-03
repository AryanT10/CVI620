import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load MNIST dataset
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

# Split data into features (X) and labels (y)
X_train = train_df.iloc[:, 1:].values  # Pixel values
y_train = train_df.iloc[:, 0].values   # Labels (digits)
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# Normalize pixel values (scaling to 0-1 range)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

# Train a logistic regression model
model = LogisticRegression(solver='lbfgs', max_iter=1000, n_jobs=-1)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "mnist_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

# Print evaluation metrics
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)

# Save report
with open("evaluation_report.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")
    f.write("\nClassification Report:\n" + report)
