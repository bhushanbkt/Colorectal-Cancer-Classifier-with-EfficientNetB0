import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import os

# Define the path to the .h5 weights file
weights_path = "colorectal_cancer_model.h5"

# Define class labels (adjust to match your training classes)
class_labels = ['Adenocarcinoma', 'High-grade IN', 'Low-grade IN', 'Normal', 'Polyp', 'Serrated adenoma']

# Define model architecture (reconstruct the architecture as per the model saved)
def build_model():
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_labels), activation='softmax')
    ])
    return model

# Load weights into the model
try:
    model = build_model()
    model.load_weights(weights_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to preprocess and predict
def preprocess_image(image, img_size=(224, 224)):
    """Resize and normalize the image for prediction."""
    image = image.resize(img_size)
    image = img_to_array(image) / 255
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

CONFIDENCE_THRESHOLD = 0.8  # Adjust this threshold to control sensitivity

def predict_category(image):
    """Predict the category of the image using the loaded model."""
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)[0]  # Get prediction probabilities for classes

    # Apply temperature scaling for better probability distribution
    temperature = 0.5 # Adjust this value as necessary to control "sharpness" of the probability distribution
    predictions = np.exp(predictions / temperature) / np.sum(np.exp(predictions / temperature))

    predicted_class_index = np.argmax(predictions)
    confidence = predictions[predicted_class_index]

    # If confidence is below threshold, mark as uncertain
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_label = "Uncertain"
    else:
        predicted_label = class_labels[predicted_class_index]
        
    return predicted_label, confidence

# Streamlit UI
st.title("Colorectal Cancer Classification")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        if 'model' in globals():  # Check if model is loaded
            label, confidence = predict_category(image)
            if label == "Uncertain":
                st.warning("The model is uncertain about this prediction.")
            else:
                st.success(f"This image is classified as: {label} with confidence {confidence * 100:.2f}%")
        else:
            st.error("Model could not be loaded. Please check the model file.")
