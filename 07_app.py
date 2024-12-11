import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import r2_score

# Load the saved encoders and model
ordinal_encoder = joblib.load('app_files/ordinal_encoder.pkl')
onehot_encoder = joblib.load('app_files/onehot_encoder.pkl')
scaler = joblib.load('app_files/standard_scaler.pkl')

# Load label encoders
item_identifier_encoder = joblib.load('app_files/Item_Identifier_label_encoder.pkl')
outlet_identifier_encoder = joblib.load('app_files/Outlet_Identifier_label_encoder.pkl')

# Load the saved Gradient Boosting model
best_gb_model = joblib.load('app_files/xgboost_model.pkl')

# Define preprocessing function for inference
def data_preprocessing(input_data):
    # Handling missing values (same logic as training phase)
    input_data['Item_Weight'] = input_data['Item_Weight'].fillna(input_data.groupby('Item_Type')['Item_Weight'].transform('mean'))
    input_data['Outlet_Size'] = input_data['Outlet_Size'].fillna(input_data['Outlet_Size'].mode()[0])

    # Feature derivation (same as training phase)
    input_data['Outlet_Age'] = 2024 - input_data['Outlet_Establishment_Year']
    input_data['Price_Per_Unit_Weight'] = input_data['Item_MRP'] / input_data['Item_Weight']

    # Simplify Item_Fat_Content (same as training)
    input_data['Item_Fat_Content'] = input_data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

    # Log Transformation for Item Visibility
    input_data['Item_Visibility_Log'] = np.log1p(input_data['Item_Visibility'])

    # MRP Categorization (same as training)
    min_value = input_data['Item_MRP'].min()
    max_value = input_data['Item_MRP'].max()
    range_value = max_value - min_value
    input_data['MRP_Tier'] = input_data['Item_MRP'].apply(lambda x: 'Low' if x <= min_value + 0.33 * range_value else
                                                          'Medium' if x <= min_value + 0.66 * range_value else 'High')

    # Encoding the data (scaling and transforming)
    numeric_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age', 'Price_Per_Unit_Weight']
    ordinal_features = ['Outlet_Size', 'MRP_Tier']
    nominal_features = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Type']
    
    # Scale numeric features
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    
    # Encode ordinal features
    if ordinal_features:
        input_data[ordinal_features] = ordinal_encoder.transform(input_data[ordinal_features])

    # One-hot encode nominal features
    if nominal_features:
        nominal_encoded = onehot_encoder.transform(input_data[nominal_features])
        nominal_cols = onehot_encoder.get_feature_names_out(nominal_features)
        nominal_df = pd.DataFrame(nominal_encoded, columns=nominal_cols)
        input_data = pd.concat([input_data, nominal_df], axis=1)
        input_data.drop(columns=nominal_features, inplace=True)

    # Label encode ID features (if necessary)
    input_data['Item_Identifier'] = item_identifier_encoder.transform(input_data['Item_Identifier'])
    input_data['Outlet_Identifier'] = outlet_identifier_encoder.transform(input_data['Outlet_Identifier'])

    return input_data

# Streamlit UI for user input
def main():
    # Set page configuration for better UX
    st.set_page_config(
        page_title='Grocery Sales Prediction App',
        page_icon='🛒',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # Title and description
    st.title('🛒 Grocery Sales Prediction App')
    st.write('Welcome! Fill in the details below to predict the sales of a grocery item.')

    # Split the page into two equal columns
    col1, col2 = st.columns(2)

    with col1:
        st.header('📦 Product Information')
        # Product Information Inputs
        item_identifier = st.text_input('Item Identifier', value='FDA15', help='Unique identifier for the product.')
        item_weight = st.number_input('Item Weight (in kg)', min_value=0.0, max_value=100.0, value=9.3, help='Weight of the product.')
        item_fat_content = st.selectbox('Item Fat Content', options=['Low Fat', 'Regular'], index=0, help='Fat content of the product.')
        item_visibility = st.number_input('Item Visibility', min_value=0.0, max_value=0.33, value=0.016, step=0.001, help='Visibility of the product in the store')
        item_type = st.selectbox('Item Type', options=sorted(['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood']), index=4, help='Category of the product.')
        item_mrp = st.number_input('Item MRP', min_value=0.0, max_value=500.0, value=249.81, step=0.01, help='Maximum Retail Price of the product.')

    with col2:
        st.header('🏬 Store Information')
        outlet_identifier = st.selectbox('Outlet Identifier', options=sorted(['OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019']), index=7, help='Store identifier.')
        outlet_establishment_year = st.number_input('Outlet Establishment Year', min_value=1980, max_value=2020, value=1999, step=1, help='Year the store was established.')
        outlet_size = st.selectbox('Outlet Size', options=['Small', 'Medium', 'High'], index=1, help='Size of the store.')
        outlet_location_type = st.selectbox('Outlet Location Type', options=['Tier 1', 'Tier 2', 'Tier 3'], index=0, help='Location type of the store.')
        outlet_type = st.selectbox('Outlet Type', options=['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'], index=0, help='Type of store.')

    # Prediction Button
    if st.button('Predict Sales'):
        # Prepare input data as a DataFrame
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

        # Preprocess the data
        processed_data = data_preprocessing(input_data)

        # Make a prediction
        prediction = best_gb_model.predict(processed_data)

        # Display the result
        st.write(f'### Predicted Sales: ₹ {prediction[0]:.2f}')

if __name__ == "__main__":
    main()
