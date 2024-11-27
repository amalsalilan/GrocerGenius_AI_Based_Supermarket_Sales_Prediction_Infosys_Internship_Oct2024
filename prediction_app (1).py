import streamlit as st
import numpy as np
from joblib import load


xg_model = load("xgboost_model.joblib")


st.set_page_config(page_title="Supermarket Sales Prediction", layout="wide")

# Centered Header and Image, can be removed if team aggreeddddddd
st.markdown("<h1 style='text-align: center; color: white;'>Supermarket Sales Prediction</h1>", unsafe_allow_html=True)

# image for the top idk 
st.image("https://via.placeholder.com/800x200?text=Supermarket+Sales+Prediction+Dashboard", caption="Optimize your store's performance with predictive analytics!", use_column_width=True)

# two clossses 
left_column, right_column = st.columns([2, 2], gap="large")

with left_column:
    st.header("Product Details")
    item_type = st.selectbox("Item Type", ["Fruits and Vegetables", "Dairy Products", "Beverages", "Household"])
    item_fat_content = st.radio("Item Fat Content", ["Low Fat", "Regular", "Non-Edible"])
    item_weight = st.number_input("Item Weight (kg)", min_value=0.0, max_value=100.0, step=0.1, value=0.0)
    item_visibility = st.slider("Item Visibility (%)", min_value=0.0, max_value=100.0, step=0.1)
    item_mrp = st.number_input("Item MRP (₹)", min_value=0.0, max_value=500.0, step=0.1, value=0.0)

with right_column:
    st.header("Outlet Details")
    outlet_identifier = st.selectbox("Outlet Identifier", ["OUT013", "OUT017", "OUT018", "OUT019"])
    establishment_year = st.number_input("Establishment Year", min_value=1900, max_value=2024, step=1, value=2000)
    outlet_size = st.radio("Outlet Size", ["Small", "Medium", "High"])
    location_type = st.selectbox("Location Type", ["Tier 1", "Tier 2", "Tier 3"])
    outlet_type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])

# predection button not erking rght now idk whyy
st.markdown("<div style='display: flex; justify-content: center; margin-top: 40px;'>", unsafe_allow_html=True)
if st.button("Predict Sales", key="centered_button"):
    # Placeholder logic for converting user input into a prediction-friendly format
    user_data = np.array([item_weight, item_visibility, item_mrp]).reshape(1, -1)  #we can add more fields here
    prediction = xg_model.predict(user_data)
    st.success(f"Predicted Sales: ₹{prediction[0]:.2f}")
st.markdown("</div>", unsafe_allow_html=True)

#custom CSS for color and spacing adjustments picking blue for now
st.markdown(
    """
    <style>
        body {
            background-color: #0a1931;
            color: white;
        }
        .stButton button {
            width: 300px;
            height: 50px;
            font-size: 18px;
            background-color: #0b6e4f;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #0a5c3b;
        }
        .css-1q8dd3e {
            width: 80% !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
