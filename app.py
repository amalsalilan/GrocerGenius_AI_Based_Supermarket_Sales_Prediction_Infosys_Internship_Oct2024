import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved XGBoost model and encoders
xgb_model = joblib.load('xgb_model.pkl')
ohe_encoder = joblib.load('ohe_encoder.pkl')
loo_encoder = joblib.load('loo_encoder.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')

# Function for preprocessing the new data
def preprocess_data(data):
    # Handle outliers
    for column in data.select_dtypes(include='number').columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = data[column].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))

    # Fill missing values
    data['Item_Weight'] = data.groupby('Item_Type')['Item_Weight'].transform(lambda x: x.fillna(x.median()))
    data['Outlet_Size'] = data.groupby('Outlet_Type')['Outlet_Size'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'Medium')

    # Data transformations
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

    # Ensure that there are at least two distinct bin edges for pd.cut()
    item_visibility_max = data['Item_Visibility'].max()
    if item_visibility_max <= 0.05:
        bins = [-0.01, item_visibility_max + 0.01, item_visibility_max + 0.02]
        labels = ['Low', 'High']
    elif item_visibility_max <= 0.15:
        bins = [-0.01, 0.05, item_visibility_max + 0.01]
        labels = ['Low', 'Medium']
    else:
        bins = [-0.01, 0.05, 0.15, item_visibility_max]
        labels = ['Low', 'Medium', 'High']

    # Apply pd.cut with the adjusted bins and labels
    data['Item_Visibility_Bins'] = pd.cut(data['Item_Visibility'], bins=bins, labels=labels)
    data['Years_Since_Establishment'] = 2024 - data['Outlet_Establishment_Year']

    # One-Hot Encoding for nominal features
    nominal_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Type']
    encoded_nominal = ohe_encoder.transform(data[nominal_columns])
    encoded_nominal_df = pd.DataFrame(encoded_nominal, columns=ohe_encoder.get_feature_names_out(nominal_columns))
    data = pd.concat([data.reset_index(drop=True), encoded_nominal_df.reset_index(drop=True)], axis=1)
    data.drop(nominal_columns, axis=1, inplace=True)

    # Leave-One-Out Encoding for high cardinality features
    data[['Outlet_Identifier']] = loo_encoder.transform(data[['Outlet_Identifier']])

    # Ordinal Encoding for ordinal features
    ordinal_columns = ['Item_Visibility_Bins', 'Outlet_Size', 'Outlet_Location_Type']
    data[ordinal_columns] = ordinal_encoder.transform(data[ordinal_columns])

    # Drop unnecessary columns
    data.drop(columns=['Item_Identifier', 'Outlet_Establishment_Year'], inplace=True)
    if 'Item_Outlet_Sales' in data.columns:
        data.drop(columns=['Item_Outlet_Sales'], inplace=True)

    # Log transformation to reduce skewness
    data['Item_Visibility_Log'] = np.log1p(data['Item_Visibility'])

    # Final feature selection
    X = data
    return X

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
            options=sorted(outlet_type_options),
            index=3,
            help='The type of store.'
        )

    # Place the Predict Sales button in a more convenient place
    # Center the button below the input sections
    st.markdown("<br>", unsafe_allow_html=True)  # Add some vertical space

    # Create columns for centering the button
    _, center_col, _ = st.columns([1, 1, 1])  # Equal width columns for centering

    with center_col:
        predict_button = st.button('Predict Sales')

    if predict_button:
        # Organize input data into a DataFrame
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

        # Preprocess the input data
        X_new = preprocess_data(input_data.copy())

        try:
            # Make predictions using the loaded XGBoost model
            predictions = xgb_model.predict(X_new)

            # Display the prediction result
            st.success(f'🔮 Predicted Sales: **${predictions[0]:,.2f}**')

            # Display only the Outlet Identifier and Item Identifier
            st.header('🛍️ Prediction Details')
            st.write(f"**Outlet Identifier**: {outlet_identifier}")
            st.write(f"**Item Identifier**: {item_identifier}")

        except Exception as e:
            st.error(f'An error occurred during prediction: {e}')
    else:
        st.info('👈 Adjust the input features and click **Predict Sales** to see the result.')

    # Footer with additional information
    st.markdown("""
    ---
    **Note:** This app uses a machine learning model to predict sales based on the input features. The result is for informational purposes only.
    """)

if __name__ == '__main__':
    main()