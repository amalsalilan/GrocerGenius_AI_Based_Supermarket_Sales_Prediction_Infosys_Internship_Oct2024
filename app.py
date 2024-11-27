import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and encoders
model = joblib.load('best_xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
standard_scaler = joblib.load('standard_scaler.pkl')
yeo_johnson = joblib.load('yeo_johnson.pkl')
one_hot_encoder = joblib.load('one_hot_encoder.pkl')
ordinal_encoder = joblib.load('ordinal_Encoder.pkl')

# Load additional pre-computed values
item_weight_medians = joblib.load('item_weight_medians.pkl')
outlet_size_modes = joblib.load('outlet_size_modes.pkl')
median_visibility = joblib.load('median_visibility.pkl')
loo_encodings = joblib.load('loo_encodings.pkl')


def data_preprocessing(data):
    # Copy the input data to avoid modifying original
    data = data.copy()

    # Handle missing Item_Weight
    data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Type'].map(item_weight_medians))

    # Handle missing Outlet_Size
    data['Outlet_Size'] = data['Outlet_Size'].fillna(data['Outlet_Type'].map(outlet_size_modes))

    # Handle Item_Visibility: Replace zero visibility with median visibility
    data['Item_Visibility'] = data['Item_Visibility'].replace(0, median_visibility)

    # Mapping variations to standardized values
    data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)

    # Outlier capping for continuous columns
    continuous_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
    z_threshold = 3
    for col in continuous_columns:
        upper_bound = data[col].mean() + z_threshold * data[col].std()
        lower_bound = data[col].mean() - z_threshold * data[col].std()
        data[col] = np.clip(data[col], lower_bound, upper_bound)

    # Apply Yeo-Johnson transformation to skewed columns
    skewed_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
    data[skewed_columns] = yeo_johnson.transform(data[skewed_columns])

    # Apply standard scaling
    data[['Item_Weight', 'Item_Visibility']] = standard_scaler.transform(data[['Item_Weight', 'Item_Visibility']])

    # Apply MinMax scaling to 'Item_MRP'
    data[['Item_MRP']] = scaler.transform(data[['Item_MRP']])

    # Encode ordinal features
    ordinal_features = ['Outlet_Size', 'Outlet_Location_Type']
    data[ordinal_features] = ordinal_encoder.transform(data[ordinal_features])

    # One-hot encode nominal features
    nominal_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Type']
    encoded_nominal = one_hot_encoder.transform(data[nominal_features])
    encoded_nominal_df = pd.DataFrame(
        encoded_nominal,
        columns=one_hot_encoder.get_feature_names_out(nominal_features),
        index=data.index
    )
    data = pd.concat([data.drop(columns=nominal_features), encoded_nominal_df], axis=1)

    # LOO encoding for high cardinality features
    high_cardinality_features = ['Outlet_Identifier']
    for feature in high_cardinality_features:
        if feature in data.columns:
            data[f'{feature}'] = data[feature].map(loo_encodings.get(feature, {})).fillna(0)

    data.drop(columns=high_cardinality_features, inplace=True)

    # Add new feature engineering
    data['Outlet_Age'] = 2024 - data['Outlet_Establishment_Year']
    data['Visibility_Percentage'] = data['Item_Visibility'] / (data['Item_Visibility'].sum() + 1e-5)
    data['Price_Per_Weight'] = data['Item_MRP'] / (data['Item_Weight'] + 1e-5)
    data['Visibility_to_MRP_Ratio'] = data['Item_Visibility'] / (data['Item_MRP'] + 1e-5)
    data['Discount_Potential'] = data['Item_MRP'] / (data['Item_Visibility'] + 1e-5)

    # Remove spaces in column names
    data.columns = data.columns.str.replace(' ', '_')

    # Drop unnecessary columns
    data.drop(columns=['Item_Identifier', 'Outlet_Establishment_Year'], inplace=True)

    return data


# Streamlit App
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
        item_identifier = st.text_input(
            'Item Identifier',
            value='FDA15',
            help='Unique identifier for the product.'
        )

        item_weight = st.number_input(
            'Item Weight (in kg)',
            min_value=0.0,
            max_value=100.0,
            value=9.3,
            help='Weight of the product.'
        )

        item_fat_content_options = ['Low Fat', 'Regular']
        item_fat_content = st.selectbox(
            'Item Fat Content',
            options=item_fat_content_options,
            index=0,
            help='Indicates the fat content of the product.'
        )

        item_visibility = st.slider(
            'Item Visibility',
            min_value=0.0,
            max_value=0.25,
            value=0.016,
            step=0.001,
            help='The percentage of total display area allocated to this product in the store.'
        )

        item_type_options = [
            'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
            'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
            'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads',
            'Starchy Foods', 'Others', 'Seafood'
        ]
        item_type = st.selectbox(
            'Item Type',
            options=sorted(item_type_options),
            index=4,
            help='The category to which the product belongs.'
        )

        item_mrp = st.number_input(
            'Item MRP',
            min_value=0.0,
            max_value=500.0,
            value=249.81,
            step=0.01,
            help='Maximum Retail Price (list price) of the product.'
        )

    with col2:
        st.header('🏬 Store Information')
        # Store Information Inputs
        outlet_identifier_options = [
            'OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027',
            'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019'
        ]
        outlet_identifier = st.selectbox(
            'Outlet Identifier',
            options=sorted(outlet_identifier_options),
            index=7,
            help='Unique identifier for the store.'
        )

        outlet_establishment_year = st.number_input(
            'Outlet Establishment Year',
            min_value=1980,
            max_value=2020,
            value=1999,
            step=1,
            help='The year in which the store was established.'
        )

        outlet_size_options = ['Small', 'Medium', 'High']
        outlet_size = st.selectbox(
            'Outlet Size',
            options=outlet_size_options,
            index=1,
            help='The size of the store.'
        )

        outlet_location_type_options = ['Tier 1', 'Tier 2', 'Tier 3']
        outlet_location_type = st.selectbox(
            'Outlet Location Type',
            options=outlet_location_type_options,
            index=0,
            help='The type of city in which the store is located.'
        )

        outlet_type_options = [
            'Supermarket Type1', 'Supermarket Type2',
            'Supermarket Type3', 'Grocery Store'
        ]
        outlet_type = st.selectbox(
            'Outlet Type',
            options=outlet_type_options,
            index=0,
            help='The type of store.'
        )

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
        prediction = model.predict(processed_data)

        # Display the result
        st.write(f'### Predicted Sales: ₹ {prediction[0]:.2f}')


if __name__ == "__main__":
    main()
