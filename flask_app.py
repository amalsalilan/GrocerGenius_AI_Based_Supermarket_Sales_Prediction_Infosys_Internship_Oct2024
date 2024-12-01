from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib


# Initialize Flask app
app = Flask(__name__)

# Load the saved model and encoders
try:
    model = joblib.load('best_xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    standard_scaler = joblib.load('standard_scaler.pkl')
    yeo_johnson = joblib.load('yeo_johnson.pkl')
    one_hot_encoder = joblib.load('one_hot_encoder.pkl')
    ordinal_encoder = joblib.load('ordinal_Encoder.pkl')

    item_weight_medians = joblib.load('item_weight_medians.pkl')
    outlet_size_modes = joblib.load('outlet_size_modes.pkl')
    median_visibility = joblib.load('median_visibility.pkl')
    loo_encodings = joblib.load('loo_encodings.pkl')
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessing files: {e}")

def data_preprocessing(data):
    try:
        # Copy input to avoid modifying original
        data = data.copy()

        # Handle missing values
        data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Type'].map(item_weight_medians))
        data['Outlet_Size'] = data['Outlet_Size'].fillna(data['Outlet_Type'].map(outlet_size_modes))
        data['Item_Visibility'] = data['Item_Visibility'].replace(0, median_visibility)

        # Standardize Item_Fat_Content
        data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)

        # Outlier capping for continuous variables
        continuous_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
        z_threshold = 3
        for col in continuous_columns:
            upper_bound = data[col].mean() + z_threshold * data[col].std()
            lower_bound = data[col].mean() - z_threshold * data[col].std()
            data[col] = np.clip(data[col], lower_bound, upper_bound)

        # Yeo-Johnson transformation
        skewed_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
        data[skewed_columns] = yeo_johnson.transform(data[skewed_columns])

        # Apply standard scaling
        data[['Item_Weight', 'Item_Visibility']] = standard_scaler.transform(data[['Item_Weight', 'Item_Visibility']])
        data[['Item_MRP']] = scaler.transform(data[['Item_MRP']])

        # Ordinal and one-hot encoding
        ordinal_features = ['Outlet_Size', 'Outlet_Location_Type']
        data[ordinal_features] = ordinal_encoder.transform(data[ordinal_features])

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
            data[f'{feature}'] = data[feature].map(loo_encodings.get(feature, {})).fillna(0)
        data.drop(columns=high_cardinality_features, inplace=True)

        # Additional features
        data['Outlet_Age'] = 2024 - data['Outlet_Establishment_Year']
        data['Visibility_Percentage'] = data['Item_Visibility'] / (data['Item_Visibility'].sum() + 1e-5)
        data['Price_Per_Weight'] = data['Item_MRP'] / (data['Item_Weight'] + 1e-5)
        data['Visibility_to_MRP_Ratio'] = data['Item_Visibility'] / (data['Item_MRP'] + 1e-5)
        data['Discount_Potential'] = data['Item_MRP'] / (data['Item_Visibility'] + 1e-5)

        # Drop unnecessary columns
        data.drop(columns=['Item_Identifier', 'Outlet_Establishment_Year'], inplace=True)

        return data
    except Exception as e:
        raise ValueError(f"Error in data preprocessing: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        input_data = request.get_json()
        data = pd.DataFrame(input_data)

        # Preprocess the data
        processed_data = data_preprocessing(data)

        # Make predictions
        predictions = model.predict(processed_data)

        # Return predictions
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
