import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pickle

# Load the trained model and preprocessing pipeline using pickle
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('preprocessing_pipeline.pkl', 'rb') as pipeline_file:
    preprocessing_pipeline = pickle.load(pipeline_file)

categories = ["cloudy", "desert", "green_area", "water"]

def augment_image(image):
    angle = np.random.uniform(-20, 20)
    image = image.rotate(angle)
    if np.random.rand() > 0.5:
        image = ImageOps.mirror(image)
    return image

def convert_to_grayscale(image):
    return ImageOps.grayscale(image)

def preprocess_image(image):
    image = image.resize((128, 128))
    image = convert_to_grayscale(image)
    image_array = np.array(image).flatten()
    image_array = preprocessing_pipeline.transform([image_array])
    return image_array

# Streamlit app
st.title("Satellite Image Classification")
st.write("Upload a satellite image to classify it into one of the categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    image = augment_image(image)
    image_array = preprocess_image(image)
    prediction = rf_model.predict(image_array)
    category = categories[prediction[0]]
    st.write(f"The image is classified as: {category}")

st.write("""
## About
This app classifies satellite images into four categories: cloudy, desert, green area, and water. It uses a Random Forest model trained on augmented and preprocessed images.
""")
