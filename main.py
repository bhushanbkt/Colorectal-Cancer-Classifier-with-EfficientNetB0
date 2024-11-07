import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define the path to the .h5 weights file
weights_path = "colorectal_cancer_model.h5"

# Define class labels (adjust to match your training classes)
class_labels = ['Low-grade IN', 'Adenocarcinoma', 'High-grade IN', 'Normal', 'Polyp', 'Serrated adenoma']

# Information about the stages and precautions
stage_info = {
    'Low-grade IN': {
        'description': "Low-grade intraepithelial neoplasia (IN) refers to mildly abnormal cells that are not cancerous.",
        'precautions': "Regular check-ups and early monitoring can help prevent progression."
    },
    'Adenocarcinoma': {
        'description': "Adenocarcinoma is a type of cancer that begins in the glandular cells of the colon or rectum.",
        'precautions': "Treatment options include surgery, chemotherapy, and radiation. Consult a medical professional for further guidance."
    },
    'High-grade IN': {
        'description': "High-grade intraepithelial neoplasia refers to more abnormal cells, which have a higher risk of becoming cancerous.",
        'precautions': "Close monitoring and potential intervention may be required to prevent cancer development."
    },
    'Normal': {
        'description': "No signs of cancer or precancerous conditions are observed. Regular screening is recommended.",
        'precautions': "Maintain a healthy lifestyle, eat a balanced diet, and undergo regular screenings."
    },
    'Polyp': {
        'description': "Polyps are growths on the inner lining of the colon or rectum that can become cancerous over time.",
        'precautions': "Polyp removal is recommended to prevent cancer. Regular colonoscopies are advised."
    },
    'Serrated adenoma': {
        'description': "Serrated adenomas are abnormal growths in the colon that have the potential to turn cancerous.",
        'precautions': "Removal of serrated adenomas is necessary to prevent cancer. Follow-up care is important."
    }
}

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

CONFIDENCE_THRESHOLD = 0.9  # Lower this threshold to accept lower confidence values

def predict_category(image):
    """Predict the category of the image using the loaded model."""
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)[0]  # Get prediction probabilities for classes

    # Get the class with the highest and second highest probabilities
    sorted_indices = np.argsort(predictions)[::-1]  # Sort predictions in descending order
    first_class_index = sorted_indices[0]
    second_class_index = sorted_indices[1]

    first_class = class_labels[first_class_index]
    second_class = class_labels[second_class_index]
    first_confidence = predictions[first_class_index]
    second_confidence = predictions[second_class_index]

    # Return the predictions and their confidences
    return first_class, first_confidence, second_class, second_confidence

# Streamlit UI
st.title("Colorectal Cancer Classification", anchor="colorectal-cancer")
st.image('img/images.png', use_column_width=True)

# UI for image upload and prediction
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        if 'model' in globals():  # Check if model is loaded
            first_class, first_confidence, second_class, second_confidence = predict_category(image)
            
            # Display the top predictions
            st.success(f"Primary Prediction: {first_class} with confidence {first_confidence * 100:.2f}%")
            st.info(f"Secondary Prediction: {second_class} with confidence {second_confidence * 100:.2f}%")
            
            # Show additional information about the predicted class
            st.subheader("About this Stage")
            st.markdown(f"**Description:** {stage_info[first_class]['description']}")
            st.markdown(f"**Precautions:** {stage_info[first_class]['precautions']}")

            # Show information about all stages
            st.subheader("Information About All Stages")
            for class_label in class_labels:
                st.markdown(f"### {class_label}")
                st.markdown(f"**Description:** {stage_info[class_label]['description']}")
                st.markdown(f"**Precautions:** {stage_info[class_label]['precautions']}")
                st.markdown("---")
            
            # Optionally, show images for specific stages if available
            if first_class == "Adenocarcinoma":
                st.image("adenocarcinoma_image.jpg", caption="Adenocarcinoma Image", use_column_width=True)
            elif first_class == "Polyp":
                st.image("polyp_image.jpg", caption="Polyp Image", use_column_width=True)
            # Add images for other stages if necessary

        else:
            st.error("Model could not be loaded. Please check the model file.")
