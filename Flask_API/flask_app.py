from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import joblib


try:
    #Load the saved model and encoders
    xgb_model = joblib.load(r'C:/Users/HP/PycharmProjects/Gocergenius/model/in_best_model.pkl')
    ohe_encoder = joblib.load(r'C:/Users/HP/PycharmProjects/Gocergenius/model/in_onehot_encoder.pkl')
    loo_encoder = joblib.load(r'C:/Users/HP/PycharmProjects/Gocergenius/model/in_label_encoder_outlet.pkl')
    ordinal_encoder = joblib.load(r'C:/Users/HP/PycharmProjects/Gocergenius/model/in_ordinal_encoder.pkl')
    loi_encoder = joblib.load(r'C:/Users/HP/PycharmProjects/Gocergenius/model/in_label_encoder_item.pkl')
    feature_names = joblib.load(r'C:/Users/HP/PycharmProjects/Gocergenius/model/in_feature_names.pkl')  # Load saved feature names
except  Exception as e:
    raise RuntimeError(f"Error loading model or preprocessing files : {e}")

#initialize flask app
app=Flask(__name__)

# Function for preprocessing the new data
def preprocess_data(data):
    try:
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

        # Create bins for Item Visibility
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
        data['MRP_Visibility'] = data['Item_MRP'] * data['Item_Visibility']
        data['sales_per_mrp'] = data['Item_Visibility'] / data['Item_MRP']

         # One-Hot Encoding for nominal features
        ohe_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Type']
        ohe_encoded = ohe_encoder.transform(data[ohe_columns])
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_encoder.get_feature_names_out(ohe_columns))
        data = pd.concat([data.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
        data.drop(ohe_columns, axis=1, inplace=True)

        # Leave-One-Out Encoding for identifiers
        data[['Outlet_Identifier']] = loo_encoder.transform(data[['Outlet_Identifier']])
        data[['Item_Identifier']] = loi_encoder.transform(data[['Item_Identifier']])

        # Ordinal Encoding for ordinal features
        ordinal_columns = ['Item_Visibility_Bins', 'Outlet_Size', 'Outlet_Location_Type']
        data[ordinal_columns] = ordinal_encoder.transform(data[ordinal_columns])

        # Drop unnecessary columns
        if 'Item_Outlet_Sales' in data.columns:
            data.drop(columns=['Item_Outlet_Sales'], inplace=True)

        # Log transformation to reduce skewness
        data['Item_Visibility_Log'] = np.log1p(data['Item_Visibility'])

        # Ensure feature order matches training
        for col in feature_names:
            if col not in data.columns:
                data[col] = 0  # Handle missing columns with a default value

        data = data[feature_names]  # Reorder columns to match model

        return data
    except Exception as e:
        raise RuntimeError(f"Error loading model or preprocessing  : {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        input_data = request.get_json()
        data = pd.DataFrame(input_data)

        # Preprocess the data
        processed_data = preprocess_data(data)

        # Make predictions
        predictions = xgb_model.predict(processed_data)

        # Return predictions
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)