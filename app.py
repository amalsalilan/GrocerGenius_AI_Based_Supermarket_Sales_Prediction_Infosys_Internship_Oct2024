import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Supermarket Sales Prediction",
    layout="wide"
)

# Custom CSS with complete pink theme
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 1rem;
    }
    .section-header {
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 2rem;
        background-color: #FFE6F3;
        border-radius: 0.5rem;
        margin-top: 2rem;
        border: 2px solid #FF69B4;
    }
    /* Custom button color */
    .stButton>button {
        background-color: #FF69B4 !important;
        color: white !important;
    }
    .stButton>button:hover {
        background-color: #FF1493 !important;
    }
    /* Header colors */
    h1, h2, h3 {
        color: #FF1493 !important;
    }
    /* Slider track */
    .stSlider input[type="range"] {
        background: #FF69B4 !important;
    }
    /* Slider thumb */
    .stSlider input[type="range"]::-webkit-slider-thumb {
        background: #FF1493 !important;
    }
    /* Slider progress */
    .stSlider > div > div > div > div {
        background-color: #FF69B4 !important;
    }
    /* Radio button when selected */
    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child > div {
        background-color: #FF69B4 !important;
    }
    /* Radio button hover */
    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:hover > div:first-child {
        border-color: #FF69B4 !important;
    }
    /* Radio button checked ring */
    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child > div:last-child {
        background-color: #FF69B4 !important;
    }
    /* Custom selectbox color */
    .stSelectbox>div>div>div {
        background-color: white;
        border: 2px solid #FF69B4;
    }
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #FF69B4 !important;
    }
    /* Select box focus */
    .stSelectbox > div[data-baseweb="select"] > div:focus {
        background-color: #FFE6F3 !important;
        border-color: #FF69B4 !important;
    }
    /* All other focus states */
    *:focus {
        border-color: #FF69B4 !important;
        box-shadow: 0 0 0 2px #FFE6F3 !important;
    }
    /* Selected option in dropdowns */
    .stSelectbox > div[data-baseweb="select"] > div > div > div[aria-selected="true"] {
        background-color: #FFE6F3 !important;
        color: #FF1493 !important;
    }
    /* Hover states */
    .stSelectbox > div[data-baseweb="select"]:hover,
    .stRadio > div[role="radiogroup"] > label:hover,
    .stCheckbox > label:hover {
        background-color: #FFE6F3 !important;
    }
    /* Override any remaining orange elements */
    .element-container button {
        background-color: #FF69B4 !important;
        border-color: #FF69B4 !important;
    }
    .stProgress .st-bo {
        background-color: #FF69B4 !important;
    }
    /* Alert colors */
    .stAlert {
        background-color: #FFE6F3 !important;
        border-color: #FF69B4 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model_and_encoders():
    """
    Load the trained model and any necessary encoders.
    Returns:
        model: The loaded model object
    """
    try:
        model = joblib.load(r'C:\Users\yanvi\OneDrive\Desktop\UI\best_xgb_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_input(data_dict):
    """
    Preprocess the input data to match the model's expected format.
    Args:
        data_dict: Dictionary containing the raw input data
    Returns:
        numpy.ndarray: Processed features ready for model prediction
    """
    # Initialize an array of 33 features with default values (e.g., 0)
    total_features = 33
    features_array = np.zeros(total_features)

    # Mapping dictionaries for categorical variables
    item_fat_content_map = {'Low Fat': 1, 'Regular': 0, 'Non-Edible': 2}
    item_type_map = {'Fruits and Vegetables': 0, 'Snacks': 1, 'Household': 2, 'Frozen Foods': 3,
                     'Dairy': 4, 'Canned': 5, 'Baking Goods': 6, 'Health and Hygiene': 7,
                     'Soft Drinks': 8, 'Meat': 9, 'Breads': 10, 'Hard Drinks': 11, 'Others': 12,
                     'Starchy Foods': 13, 'Breakfast': 14, 'Seafood': 15}
    outlet_size_map = {'Small': 0, 'Medium': 1, 'High': 2}
    location_type_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
    outlet_type_map = {'Supermarket Type1': 0, 'Supermarket Type2': 1, 
                       'Supermarket Type3': 2, 'Grocery Store': 3}
    outlet_identifier_map = {'OUT013': 0, 'OUT017': 1, 'OUT018': 2, 'OUT019': 3, 
                             'OUT027': 4, 'OUT035': 5, 'OUT045': 6, 'OUT046': 7, 'OUT049': 8}

    # Map 10 selected features to their positions in the 33-feature array
    features_mapping = {
        0: data_dict['Item_Weight'],  # Example: Feature position 0 is Item_Weight
        1: data_dict['Item_Visibility'],
        2: data_dict['Item_MRP'],
        3: item_fat_content_map[data_dict['Item_Fat_Content']],
        4: item_type_map[data_dict['Item_Type']],
        5: outlet_identifier_map[data_dict['Outlet_Identifier']],
        6: data_dict['Outlet_Establishment_Year'],
        7: outlet_size_map[data_dict['Outlet_Size']],
        8: location_type_map[data_dict['Outlet_Location_Type']],
        9: outlet_type_map[data_dict['Outlet_Type']]
    }

    # Populate the features_array with the mapped values
    for idx, value in features_mapping.items():
        features_array[idx] = value

    return features_array.reshape(1, -1)

def main():
    st.title("Supermarket Sales Prediction")
    
    # Create two columns for Product Details and Outlet Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Product Details")
        
        input_data = {}
        
        # Product Details Section
        input_data['Item_Type'] = st.selectbox(
            "Item Type",
            ['Fruits and Vegetables', 'Snacks', 'Household', 'Frozen Foods', 'Dairy', 'Canned',
             'Baking Goods', 'Health and Hygiene', 'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks',
             'Others', 'Starchy Foods', 'Breakfast', 'Seafood']
        )
        
        input_data['Item_Fat_Content'] = st.radio(
            "Item Fat Content",
            ['Low Fat', 'Regular', 'Non-Edible']
        )
        
        input_data['Item_Weight'] = st.number_input(
            "Item Weight (kg)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1
        )
        
        input_data['Item_Visibility'] = st.slider(
            "Item Visibility (%)",
            min_value=0.0,
            max_value=100.0,
            value=37.24,
            step=0.01
        ) / 100.0  # Convert percentage to decimal
        
        input_data['Item_MRP'] = st.number_input(
            "Item MRP (₹)",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

    with col2:
        st.header("Outlet Details")
        
        # Outlet Details Section
        input_data['Outlet_Identifier'] = st.selectbox(
            "Outlet Identifier",
            ['OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'],
            index=0
        )
        
        input_data['Outlet_Establishment_Year'] = st.number_input(
            "Establishment Year",
            min_value=1900,
            max_value=2024,
            value=1998
        )
        
        input_data['Outlet_Size'] = st.radio(
            "Outlet Size",
            ['Small', 'Medium', 'High']
        )
        
        input_data['Outlet_Location_Type'] = st.selectbox(
            "Location Type",
            ['Tier 1', 'Tier 2', 'Tier 3']
        )
        
        input_data['Outlet_Type'] = st.selectbox(
            "Outlet Type",
            ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store']
        )

    # Review Section (collapsible)
    with st.expander("Review Your Inputs (Click to Expand)"):
        st.json(input_data)

    # Prediction Button
    if st.button("Predict Sales", type="primary"):
        model = load_model_and_encoders()
        if model is not None:
            try:
                # Preprocess input
                processed_input = preprocess_input(input_data)
                
                # Make prediction
                prediction = model.predict(processed_input)
                
                # Display prediction in a nice format with pink theme
                st.markdown("""
                    <div class="prediction-box">
                        <h3>Predicted Sales</h3>
                        <h2 style="color: #FF1493;">₹ {:.2f}</h2>
                    </div>
                    """.format(prediction[0]), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()