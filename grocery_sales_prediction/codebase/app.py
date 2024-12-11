# codebase/app.py

import os
import time
import streamlit as st
import pandas as pd
from utils import run_inference

# Set the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.title("Grocery Sales Prediction")

col1,col2=st.columns(2)




# Load unique values for Item_Identifier (optional)
item_identifiers = ['FDA15', 'DRC01', 'FDN15', 'FDX07', 'NCD19']  # Replace with actual identifiers if needed

# Dropdown options for categorical columns (excluding Item_Identifier)
dropdown_options = {
    "Item_Fat_Content": ["Low Fat", "Regular"],
    "Item_Type": [
        "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables",
        "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
        "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
        "Breads", "Starchy Foods", "Others", "Seafood"
    ],
    "Outlet_Identifier": ["OUT049", "OUT018", "OUT010", "OUT013", "OUT027", "OUT045", "OUT017", "OUT046", "OUT035", "OUT019"],
    "Outlet_Size": ["Small", "Medium", "High"],
    "Outlet_Location_Type": ["Tier 1", "Tier 2", "Tier 3"],
    "Outlet_Type": [
        "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"
    ],
}

# Numerical ranges for inputs
value_ranges = {
    "Item_Visibility": (0.0, 0.328391),  # Min, Max from data
    "Item_Weight": (4.555, 21.35),
    "Item_MRP": (31.29, 266.8884),
}

# Collect user inputs
user_inputs = {}

# Dropdown for Item Identifier (if needed)
user_inputs["Item_Identifier"] = st.selectbox(
    "Select Item Identifier:", options=item_identifiers
)

# Dropdowns for categorical inputs
for key, options in dropdown_options.items():
    user_inputs[key] = st.selectbox(f"Select {key}:", options)

# Number input for Item Weight
user_inputs["Item_Weight"] = st.number_input(
    "Enter Item Weight:", min_value=value_ranges["Item_Weight"][0], max_value=value_ranges["Item_Weight"][1],
    value=(value_ranges["Item_Weight"][0] + value_ranges["Item_Weight"][1]) / 2, step=0.01
)

# Number input for Item MRP
user_inputs["Item_MRP"] = st.number_input(
    "Enter Item MRP:", min_value=value_ranges["Item_MRP"][0], max_value=value_ranges["Item_MRP"][1],
    value=(value_ranges["Item_MRP"][0] + value_ranges["Item_MRP"][1]) / 2, step=0.01
)

# Number input for Item Visibility
user_inputs["Item_Visibility"] = st.number_input(
    "Enter Item Visibility:", min_value=value_ranges["Item_Visibility"][0], max_value=value_ranges["Item_Visibility"][1],
    value=(value_ranges["Item_Visibility"][0] + value_ranges["Item_Visibility"][1]) / 2, step=0.0001
)

# Selectbox for Outlet Establishment Year
years = list(range(1985, 2010))  # Years from 1985 to 2009
user_inputs["Outlet_Establishment_Year"] = st.selectbox(
    "Select Outlet Establishment Year:", options=years
)
# progress bar start----------------------------------->
progress_text = "Operation in progress"
progress_placeholder = st.empty()  # Create a placeholder for the progress bar
my_bar = progress_placeholder.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete+1, text=progress_text)

time.sleep(1)
progress_placeholder.empty()  # Remove the progress bar after completion
# progress bar end-------------------------------------------------->

def cook_breakfast():
    msg = st.toast("Hang tight! Crunching the numbers for your sales prediction…")
    time.sleep(1)
    msg.toast("Generating sales forecast—just a moment please...")
    time.sleep(1)
    msg.toast("Hold on while we estimate your item outlet sales…")


# When the user clicks the "Predict" button
if st.button("Predict",cook_breakfast()):
    # Convert user_inputs dictionary to DataFrame
    user_input_df = pd.DataFrame([user_inputs])

    # Run inference
    try:
        predictions = run_inference(user_input_df)
        predicted_sales = predictions[0]
        st.success(f"Predicted Item Outlet Sales: ${predicted_sales:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")