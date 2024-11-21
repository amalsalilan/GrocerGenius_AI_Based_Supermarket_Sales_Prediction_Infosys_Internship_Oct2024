from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

# Load the saved models and encoders
xgb_model = joblib.load('xgb_model.pkl')
ohe_encoder = joblib.load('ohe_encoder.pkl')
loo_encoder = joblib.load('loo_encoder.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define preprocessing function
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming JSON data
        input_data = request.get_json()
        data = pd.DataFrame(input_data)

        # Preprocess the data
        X_new = preprocess_data(data)

        # Make predictions
        predictions = xgb_model.predict(X_new)

        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
