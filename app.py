# app.py

import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image

# Load model once
model = MobileNetV2(weights='imagenet')

# Load calorie mapping
calorie_df = pd.read_csv("calories.csv")

def load_image(img_path):
    """Loads and preprocesses image for model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_food(img_path):
    """Uses MobileNetV2 to predict the food item."""
    processed_img = load_image(img_path)
    preds = model.predict(processed_img)
    decoded_preds = decode_predictions(preds, top=3)[0]
    return decoded_preds

def get_calorie_info(food_name):
    """Maps predicted food to calorie count."""
    match = calorie_df[calorie_df['Food'].str.lower() == food_name.lower()]
    if not match.empty:
        return int(match['Calories'].values[0])
    else:
        return "Unknown"

# Streamlit UI
st.title("🍱 Food Calorie Estimator")
uploaded_file = st.file_uploader("Upload a food image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_path = "temp.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    preds = predict_food(img_path)
    top_pred = preds[0][1]  # Most likely class
    calories = get_calorie_info(top_pred)

    st.subheader(f"🍽️ Predicted Food: {top_pred}")
    st.subheader(f"🔥 Estimated Calories: {calories}")
