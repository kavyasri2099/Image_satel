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

def augment_image(image):
    angle = np.random.uniform(-20, 20)
    image = image.rotate(angle)
    if np.random.rand() > 0.5:
        image = ImageOps.mirror(image)
    return image

def preprocess_image(image):
    image = image.resize((128, 128))
    image = augment_image(image)
    image = ImageOps.grayscale(image)
    image_array = np.array(image).flatten().reshape(1, -1)
    image_array = preprocessing_pipeline.transform(image_array)
    return image_array

st.title("Satellite Image Classification App")
st.write("Upload an image to classify it into one of the following categories: cloudy, desert, green_area, water.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    image_array = preprocess_image(image)
    prediction = rf_model.predict(image_array)
    category = categories[prediction[0]]
    
    st.write(f"The image is classified as: {category}")
