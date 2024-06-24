import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pickle

# Load the model and preprocessing pipeline
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('preprocessing_pipeline.pkl', 'rb') as pipeline_file:
    preprocessing_pipeline = pickle.load(pipeline_file)

categories = ["cloudy", "desert", "green_area", "water"]

def preprocess_image(image):
    image = image.resize((128, 128))
    image = ImageOps.grayscale(image)
    img_array = np.array(image).flatten().reshape(1, -1)
    img_array = preprocessing_pipeline.transform(img_array)
    return img_array

# Streamlit app
st.title("Satellite Image Classification App")
st.write("Upload an image, and the model will classify it into one of the following categories: Cloudy, Desert, Green Area, Water")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        img_array = preprocess_image(image)
        prediction = rf_model.predict(img_array)
        category = categories[prediction[0]]
        st.write(f"The image is classified as: {category}")
    except Exception as e:
        st.write(f"An error occurred: {e}")
