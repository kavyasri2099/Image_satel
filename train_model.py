import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle

# Path to dataset
data_path = "data"
categories = ["cloudy", "desert", "green_area", "water"]

def augment_image(image):
    angle = np.random.uniform(-20, 20)
    image = image.rotate(angle)
    if np.random.rand() > 0.5:
        image = ImageOps.mirror(image)
    return image

def convert_to_grayscale(image):
    return ImageOps.grayscale(image)

def process_and_save_images(data_path, categories):
    images = []
    labels = []

    for category in categories:
        category_path = os.path.join(data_path, category)
        label = categories.index(category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = Image.open(img_path)
            img = img.resize((128, 128))
            img = augment_image(img)
            img = convert_to_grayscale(img)
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

images, labels = process_and_save_images(data_path, categories)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

X_train = preprocessing_pipeline.fit_transform(X_train)
X_test = preprocessing_pipeline.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_features='sqrt')
rf_model.fit(X_train, y_train)

# Save the model and preprocessing pipeline
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('preprocessing_pipeline.pkl', 'wb') as pipeline_file:
    pickle.dump(preprocessing_pipeline, pipeline_file)
