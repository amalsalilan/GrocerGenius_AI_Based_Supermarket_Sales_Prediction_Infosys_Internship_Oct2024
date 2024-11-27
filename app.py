import streamlit as st
import pandas as pd
import sys
import os
import pickle
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import necessary functions from the src module
from model import load_model  # Function to load the trained model

def main():
    # Set page configuration for better UX
    st.set_page_config(
        page_title="Grocer Genius: Sales Prediction App",
        page_icon="🛒",
        layout="wide",
    )

    # Sidebar with title and description
    st.sidebar.title("🌟 Grocer Genius")
    st.sidebar.write(
        "Welcome to the **Grocery Sales Prediction App**! Fill in the details below to predict the sales of a grocery item."
    )

    # Sidebar navigation
    st.sidebar.header("🔧 Features")
    st.sidebar.write("Adjust the input features in the main interface to get predictions.")
    
    st.sidebar.header("📊 About")
    st.sidebar.write(
        "This application uses a **machine learning model** to predict sales based on input features such as product details and store attributes."
    )

    # Main interface
    st.title("🛒 Grocer Genius: Sales Prediction App")
    st.markdown("#### Empower your business with AI-driven sales predictions!")

    # Input section in tabs
    tab1, tab2 = st.tabs(["📦 Product Information", "🏬 Store Information"])

    with tab1:
        st.subheader("Product Information")
        col1, col2 = st.columns(2)

        with col1:
            item_identifier = st.text_input(
                "🔑 Item Identifier",
                value="FDA15",
                help="Unique identifier for the product."
            )
            item_weight = st.number_input(
                "⚖️ Item Weight (in kg)",
                min_value=0.0,
                max_value=100.0,
                value=9.3,
                help="Weight of the product."
            )
            item_fat_content = st.selectbox(
                "🧈 Item Fat Content",
                options=["Low Fat", "Regular"],
                index=0,
                help="Indicates the fat content of the product."
            )

        with col2:
            item_visibility = st.slider(
                "👁️ Item Visibility",
                min_value=0.0,
                max_value=0.25,
                value=0.016,
                step=0.001,
                help="Percentage of display area allocated to this product."
            )
            item_type = st.selectbox(
                "📂 Item Type",
                options=[
                    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
                    "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast",
                    "Health and Hygiene", "Hard Drinks", "Canned", "Breads",
                    "Starchy Foods", "Others", "Seafood"
                ],
                index=4,
                help="The category to which the product belongs."
            )
            item_mrp = st.number_input(
                "💰 Item MRP",
                min_value=0.0,
                max_value=500.0,
                value=249.81,
                step=0.01,
                help="Maximum Retail Price of the product."
            )

    with tab2:
        st.subheader("Store Information")
        col1, col2 = st.columns(2)

        with col1:
            outlet_identifier = st.selectbox(
                "🏬 Outlet Identifier",
                options=[
                    "OUT049", "OUT018", "OUT010", "OUT013", "OUT027",
                    "OUT045", "OUT017", "OUT046", "OUT035", "OUT019"
                ],
                index=7,
                help="Unique identifier for the store."
            )
            outlet_establishment_year = st.number_input(
                "📅 Outlet Establishment Year",
                min_value=1980,
                max_value=2020,
                value=1999,
                step=1,
                help="Year the store was established."
            )

        with col2:
            outlet_size = st.selectbox(
                "📏 Outlet Size",
                options=["Small", "Medium", "High"],
                index=1,
                help="The size of the store."
            )
            outlet_location_type = st.selectbox(
                "📍 Outlet Location Type",
                options=["Tier 1", "Tier 2", "Tier 3"],
                index=0,
                help="City type where the store is located."
            )
            outlet_type = st.selectbox(
                "🏪 Outlet Type",
                options=[
                    "Supermarket Type1", "Supermarket Type2",
                    "Supermarket Type3", "Grocery Store"
                ],
                index=3,
                help="Type of the store."
            )

    # Centered Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🎯 Predict Sales")

    if predict_button:
        # Organize input data into a DataFrame
        input_data = pd.DataFrame({
            "Item_Identifier": [item_identifier],
            "Item_Weight": [item_weight],
            "Item_Fat_Content": [item_fat_content],
            "Item_Visibility": [item_visibility],
            "Item_Type": [item_type],
            "Item_MRP": [item_mrp],
            "Outlet_Identifier": [outlet_identifier],
            "Outlet_Establishment_Year": [outlet_establishment_year],
            "Outlet_Size": [outlet_size],
            "Outlet_Location_Type": [outlet_location_type],
            "Outlet_Type": [outlet_type]
        })

        try:
            # Load the trained model and preprocessor
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "lasso_model.pkl")
            preprocessor_path = os.path.join(base_dir, "models", "preprocessor.pkl")

            model = load_model(model_path)

            # Load preprocessor
            with open(preprocessor_path, "rb") as preprocessor_file:
                preprocessor = pickle.load(preprocessor_file)

            # Transform input data
            processed_data = preprocessor.transform(input_data)

            # Predict sales
            predictions = model.predict(processed_data)

            # Display prediction
            st.success(f"🔮 Predicted Sales: **${predictions[0]:,.2f}**")

        except Exception as e:
            st.error(f"🚨 An error occurred: {e}")

    else:
        st.info("👈 Adjust inputs and click **Predict Sales** to see the result.")

    # Footer
    st.markdown(
        "---\n**Note:** This app is powered by machine learning to provide an estimation based on the input features. Results are for informational purposes."
    )

if __name__ == "__main__":
    main()
