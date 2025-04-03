import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
print("Loading model...")
model = joblib.load("mnist_model.pkl")
scaler = joblib.load("scaler.pkl")
print("Model loaded successfully!")

def predict_digit(image_array):
    """ Predicts the digit for a given image array """
    print("Processing image for prediction...")
    image_array = np.array(image_array).reshape(1, -1)  # Reshape to match model input
    image_array = scaler.transform(image_array)  # Normalize
    prediction = model.predict(image_array)
    print(f"Prediction Complete: {prediction[0]}")
    return prediction[0]

# Load test dataset
print("Loading test dataset...")
test_df = pd.read_csv("mnist_test.csv")
print("Dataset loaded successfully!")

# Select a sample image
sample_index = 0  # Change index to test different images
print(f"Selecting image at index {sample_index}...")
sample_image = test_df.iloc[sample_index, 1:].values  # Extract pixel values
print("Image extracted!")

# Make prediction
predicted_digit = predict_digit(sample_image)

# Display result
print(f"\n🟢 Predicted Digit: {predicted_digit}")
