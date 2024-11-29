import requests.exceptions
import streamlit as st
import pandas as pd


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
    st.title('📊 Grocery Sales Prediction App')
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
            'Outlet_Type': [outlet_type],

        })
        api_url = "http://127.0.0.1:5000/predict"
        try:
            response = requests.post(api_url, json=input_data.to_dict(orient='records'))
            if response.status_code == 200:
                prediction = response.json()['predictions'][0]
                # Display the prediction result
                st.success(f'🎯 Predicted Sales: **₹{prediction :,.2f}**')
                st.success(f'Thank you for using the Grocery Sales Prediction App!')
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

    else:
        st.info('👈 Adjust the input features and click **Predict Sales** to see the result.')

    # Inject custom CSS to style the sidebar and main content
    st.markdown(
        """
        <style>
             /* Footer styling */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f0f8ff !important; /* Light pastel blue background */
            color: #333 !important; /* Dark gray text for contrast */
            text-align: center;
            padding: 12px 0 !important;
            font-size: 14px !important;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1) !important;
        }

        .footer a {
            color: #ff69b4 !important; /* Light pink link color */
            text-decoration: none !important;
            font-weight: bold !important;
        }

        .footer a:hover {
            text-decoration: underline !important;
        }
        </style>
    """,
    unsafe_allow_html=True
)
    # Footer
    st.markdown("""
        <div class="footer">
            Powered by <a href="https://springboard.com" target="_blank">Infosys Springboard</a>
        </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
