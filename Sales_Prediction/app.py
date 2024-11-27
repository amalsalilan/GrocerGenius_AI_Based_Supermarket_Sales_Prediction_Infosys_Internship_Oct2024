import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the saved model and pre-fitted StandardScaler
model = joblib.load('trained_model.pkl')
scaler = joblib.load('standard_scaler.pkl')

# Label encoder mappings (used for encoding categorical features)
label_encoders = {
    'Item_Fat_Content': {'Low Fat': 0, 'Regular': 1},
    'Item_Type': {
        'Dairy': 0, 'Soft Drinks': 1, 'Meat': 2, 'Fruits and Vegetables': 3, 'Household': 4,
        'Baking Goods': 5, 'Snack Foods': 6, 'Frozen Foods': 7, 'Breakfast': 8, 'Health and Hygiene': 9,
        'Hard Drinks': 10, 'Canned': 11, 'Breads': 12, 'Starchy Foods': 13, 'Others': 14, 'Seafood': 15
    },
    'Outlet_Identifier': {
        'OUT049': 0, 'OUT018': 1, 'OUT010': 2, 'OUT013': 3, 'OUT027': 4,
        'OUT045': 5, 'OUT017': 6, 'OUT046': 7, 'OUT035': 8, 'OUT019': 9
    },
    'Outlet_Size': {'Small': 0, 'Medium': 1, 'High': 2},
    'Outlet_Location_Type': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2},
    'Outlet_Type': {'Supermarket Type1': 0, 'Supermarket Type2': 1, 'Supermarket Type3': 2, 'Grocery Store': 3}
}

# Check the feature names that the model expects
def get_feature_names(model):
    try:
        # If it's a tree-based model (e.g., XGBoost, RandomForest), try accessing the feature names
        return model.feature_names_in_
    except AttributeError:
        # Otherwise, return an empty list
        return []

# Function to preprocess the input data
def preprocess_data(data):
    feature_names = get_feature_names(model)
    required_columns = list(feature_names)  # Use list instead of set for reordering columns

    # Apply label encoding for categorical columns using pre-defined mappings
    for column, mapping in label_encoders.items():
        if column in data.columns:
            data[column] = data[column].map(mapping)
    
    # Ensure 'Item_Identifier' is treated as categorical and encoded
    if 'Item_Identifier' in data.columns:
        # Convert 'Item_Identifier' to categorical (numeric)
        data['Item_Identifier'] = data['Item_Identifier'].astype('category').cat.codes

    # Handle numerical columns and apply the pre-fitted StandardScaler for scaling
    numerical_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
    # Apply the pre-fitted StandardScaler
    if 'Item_MRP' in data.columns and 'Item_Weight' in data.columns:
        data[numerical_columns] = scaler.transform(data[numerical_columns])

    # Create missing features based on model expectations
    # Calculate sales_per_mrp: Assuming it's a ratio of MRP to Weight
    if 'sales_per_mrp' not in data.columns:
        data['sales_per_mrp'] = data['Item_MRP'] / data['Item_Weight']
    
    # Calculate Outlet_age: Age of the outlet in years
    if 'Outlet_age' not in data.columns:
        current_year = 2024
        data['Outlet_age'] = current_year - data['Outlet_Establishment_Year']

    # One-hot encode missing features if necessary
    if 'Item_Type_Household' not in data.columns:
        data['Item_Type_Household'] = (data['Item_Type'] == 'Household').astype(int)
    if 'Item_Fat_Content_Regular' not in data.columns:
        data['Item_Fat_Content_Regular'] = (data['Item_Fat_Content'] == 'Regular').astype(int)

    # Add missing columns with default values (if missing in model's training dataset)
    for col in required_columns:
        if col not in data.columns:
            if col == 'Item_Fat_Content':
                data[col] = 'Low Fat'  # Default category
            elif col == 'Item_Type':
                data[col] = 'Dairy'  # Default category
            elif col == 'Outlet_Size':
                data[col] = 'Small'  # Default category
            elif col == 'Outlet_Location_Type':
                data[col] = 'Tier 1'  # Default category
            elif col == 'Outlet_Type':
                data[col] = 'Supermarket Type1'  # Default category
            else:
                data[col] = 0  # Default numeric value

    # Reorder columns to match the model's expected order
    data = data[required_columns]
    
    return data

# Streamlit App
def main():
    # Set page configuration
    st.set_page_config(
        page_title='Grocery Sales Prediction App',
        page_icon='🛒',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.title('🛒 Grocery Sales Prediction App')
    st.write('Fill in the details below to predict the sales of a grocery item.')

    # Input sections
    col1, col2 = st.columns(2)

    with col1:
        st.header('📦 Product Information')
        item_identifier = st.text_input('Item Identifier', value='FDA15')
        item_weight = st.number_input('Item Weight (in kg)', min_value=0.0, max_value=100.0, value=9.3)
        item_fat_content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular'], index=0)
        item_visibility = st.slider('Item Visibility', min_value=0.0, max_value=0.25, value=0.016, step=0.001)
        item_type = st.selectbox('Item Type', sorted([  # Alphabetical sorting of types
            'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
            'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
            'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads',
            'Starchy Foods', 'Others', 'Seafood'
        ]), index=4)
        item_mrp = st.number_input('Item MRP', min_value=0.0, max_value=500.0, value=249.81, step=0.01)

    with col2:
        st.header('🏬 Store Information')
        outlet_identifier = st.selectbox('Outlet Identifier', sorted([  # Alphabetical sorting
            'OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027',
            'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019'
        ]), index=7)
        outlet_establishment_year = st.number_input('Outlet Establishment Year', min_value=1980, max_value=2020, value=1999, step=1)
        outlet_size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'], index=1)
        outlet_location_type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'], index=0)
        outlet_type = st.selectbox('Outlet Type', ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'], index=3)

    # Button to trigger prediction
    predict_button = st.button('Predict Sales')

    if predict_button:
        input_data = pd.DataFrame({
            'Item_Identifier': [item_identifier],
            'Item_Weight': [item_weight],
            'Item_Fat_Content': [item_fat_content],
            'Item_Visibility': [item_visibility],
            'Item_Type': [item_type],
            'Item_MRP': [item_mrp],
            'Outlet_Identifier': [outlet_identifier],
            'Outlet_Establishment_Year': [outlet_establishment_year],
            'Outlet_Size': [outlet_size],
            'Outlet_Location_Type': [outlet_location_type],
            'Outlet_Type': [outlet_type]
        })

        try:
            # Preprocess the input data before making predictions
            preprocessed_data = preprocess_data(input_data)

            # Make predictions with the model
            predictions = model.predict(preprocessed_data)
            st.success(f'🔮 Predicted Sales: **${predictions[0]:,.2f}**')
        except Exception as e:
            st.error(f'An error occurred during prediction: {e}')

if __name__ == '__main__':
    main()
