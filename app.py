import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Car Price Predictor", layout="centered")

# Load data and model
data = pd.read_csv('car_data_cleaned.csv')
with open('car_price_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# --- UI Header ---
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color:#4CAF50;">🚘 Car Price Prediction</h1>
        <p style="font-size:18px;">Enter your car details to get an accurate price estimate.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Layout Columns for Input ---
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("🏢 Select Car Company", sorted(data['company'].dropna().unique()))

# Filter car names based on company using full_name column
filtered_names = data[data['full_name'].str.contains(company, case=False, na=False)]['name'].unique()
with col2:
    name = st.selectbox("🚗 Car Name", sorted(filtered_names))

with col1:
    km_driven = st.number_input("📍 Kilometers Driven", min_value=0, step=100)

with col2:
    mileage = st.number_input("⛽ Mileage (km/ltr/kg)", min_value=0.0, step=0.1, format="%.2f")

with col1:
    engine = st.number_input("🛠️ Engine (CC)", min_value=0.0, step=10.0)

with col2:
    max_power = st.number_input("⚡ Max Power (bhp)", min_value=0.0, step=1.0)

with col1:
    seats = st.selectbox("🪑 Number of Seats", sorted(data['seats'].dropna().unique()))

with col2:
    car_age = st.number_input("📅 Car Age (Years)", min_value=0, step=1)

with col1:
    fuel = st.selectbox("⛽ Fuel Type", sorted(data['fuel'].dropna().unique()))

with col2:
    seller_type = st.selectbox("🧍 Seller Type", sorted(data['seller_type'].dropna().unique()))

with col1:
    transmission = st.selectbox("🔁 Transmission", sorted(data['transmission'].dropna().unique()))

with col2:
    owner = st.selectbox("👤 Owner Type", sorted(data['owner'].dropna().unique()))

st.markdown("---")

# --- Predict Button ---
if st.button("🔮 Predict Car Price"):
    input_data = pd.DataFrame({
        'km_driven': [km_driven],
        'mileage(km/ltr/kg)': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats],
        'car_age': [car_age],
        'name': [name],
        'company': [company],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner]
    })

    prediction = model.predict(input_data)[0]
    predicted_price = np.expm1(prediction)  # reverse of log1p if applied during training

    st.success(f"💰 **Estimated Car Price:** ₹ {round(predicted_price, 2)} Lakh")

    st.balloons()
