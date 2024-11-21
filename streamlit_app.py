import streamlit as st
import pandas as pd
import requests

# Flask API URL
API_URL = "https://grocergenius-ai-based-supermarket-sales.onrender.com/predict"

# Streamlit App
def main():
    st.set_page_config(
        page_title='Grocery Sales Prediction App',
        page_icon='🛒',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.title('🛒 Grocery Sales Prediction App')
    st.write('Fill in the details below to predict the sales of a grocery item.')

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        item_identifier = st.text_input('Item Identifier', value='FDA15')
        item_weight = st.number_input('Item Weight (in kg)', 0.0, 100.0, 9.3)
        item_fat_content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular'])
        item_visibility = st.slider('Item Visibility', 0.0, 0.25, 0.016, step=0.001)
        item_type = st.selectbox('Item Type', sorted([
            'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
            'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
            'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads',
            'Starchy Foods', 'Others', 'Seafood'
        ]))
        item_mrp = st.number_input('Item MRP', 0.0, 500.0, 249.81, step=0.01)

    with col2:
        outlet_identifier = st.selectbox('Outlet Identifier', sorted([
            'OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027',
            'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019'
        ]))
        outlet_establishment_year = st.number_input('Outlet Establishment Year', 1980, 2020, 1999, step=1)
        outlet_size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'])
        outlet_location_type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])
        outlet_type = st.selectbox('Outlet Type', sorted([
            'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'
        ]))

    if st.button('Predict Sales'):
        # Create input data dictionary
        input_data = [{
            'Item_Identifier': item_identifier,
            'Item_Weight': item_weight,
            'Item_Fat_Content': item_fat_content,
            'Item_Visibility': item_visibility,
            'Item_Type': item_type,
            'Item_MRP': item_mrp,
            'Outlet_Identifier': outlet_identifier,
            'Outlet_Establishment_Year': outlet_establishment_year,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location_type,
            'Outlet_Type': outlet_type
        }]

        try:
            # Call the Flask API
            response = requests.post(API_URL, json=input_data)

            if response.status_code == 200:
                predictions = response.json()['predictions']
                st.success(f'🔮 Predicted Sales: **${predictions[0]:,.2f}**')
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
