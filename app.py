import streamlit as st
import pandas as pd
import pickle

# --- Page Config --- (This should be the first Streamlit command)
st.set_page_config(page_title="House Price Estimator 🏡", layout="centered")

# --- Load Model and Encoders ---
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# --- Title and Description ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏡 Smart House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter the property details below to estimate its market value instantly.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Input Layout ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("📐 Area (sq ft)", min_value=500, max_value=10000, value=1000)
        bedrooms = st.selectbox("🛏️ Bedrooms", [1, 2, 3, 4, 5])
        bathrooms = st.selectbox("🛁 Bathrooms", [1, 2, 3, 4])
        stories = st.selectbox("🏢 Stories", [1, 2, 3, 4])
        parking = st.selectbox("🚗 Parking Spaces", [0, 1, 2, 3])

    with col2:
        mainroad = st.selectbox("🛣️ Main Road Access", ['yes', 'no'])
        guestroom = st.selectbox("🛋️ Guest Room", ['yes', 'no'])
        basement = st.selectbox("🏚️ Basement", ['yes', 'no'])
        hotwaterheating = st.selectbox("♨️ Hot Water Heating", ['yes', 'no'])
        airconditioning = st.selectbox("❄️ Air Conditioning", ['yes', 'no'])
        prefarea = st.selectbox("📍 Preferred Area", ['yes', 'no'])
        furnishingstatus = st.selectbox("🪑 Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

    st.markdown(" ")
    submitted = st.form_submit_button("💰 Predict Price")

# --- Data Preparation ---
if submitted:
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [encoders['mainroad'].transform([mainroad])[0]],
        'guestroom': [encoders['guestroom'].transform([guestroom])[0]],
        'basement': [encoders['basement'].transform([basement])[0]],
        'hotwaterheating': [encoders['hotwaterheating'].transform([hotwaterheating])[0]],
        'airconditioning': [encoders['airconditioning'].transform([airconditioning])[0]],
        'parking': [parking],
        'prefarea': [encoders['prefarea'].transform([prefarea])[0]],
        'furnishingstatus': [encoders['furnishingstatus'].transform([furnishingstatus])[0]]
    })

    # --- Prediction ---
    prediction = model.predict(input_data)[0]
    formatted_price = f"₹ {int(prediction):,}"

    st.markdown("---")
    st.success(f"🏷️ **Estimated House Price**: {formatted_price}")
    st.balloons()
