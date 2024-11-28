import streamlit as st
import pandas as pd
import requests

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

    col1, col2 = st.columns(2)

    with col1:
        item_identifier = st.text_input('Item Identifier', value='FDA15')
        item_weight = st.number_input('Item Weight (in kg)', min_value=0.0, max_value=1000.0, value=9.3)
        item_fat_content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular'])
        item_visibility = st.slider('Item Visibility', 0.0, 0.25, 0.016, step=0.001)
        item_type = st.selectbox(
            'Item Type',
            sorted([
                'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
                'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
                'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads',
                'Starchy Foods', 'Others', 'Seafood'
            ])
        )
        item_mrp = st.number_input('Item MRP', min_value=0.0, max_value=5000.0, value=249.81, step=0.01)

    with col2:
        outlet_identifier = st.selectbox(
            'Outlet Identifier',
            ['OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019']
        )
        outlet_establishment_year = st.number_input('Outlet Establishment Year', min_value=1980, max_value=2024, value=1999, step=1)
        outlet_size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'])
        outlet_location_type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])
        outlet_type = st.selectbox('Outlet Type', ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

    if st.button('Predict Sales'):
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
        api_url = "http://127.0.0.1:5000/predict"
        try:
            response = requests.post(api_url, json=input_data)
            if response.status_code == 200:
                prediction = response.json()['predictions'][0]
                st.write(f"### Predicted Sales: ₹{prediction:.2f}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

if __name__ == "__main__":
    main()
