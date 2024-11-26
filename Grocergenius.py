import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

def data_processing (data):
    # Handling Missing Values
    non_zero_mean = data.loc[data['Item_Visibility'] > 0, 'Item_Visibility'].mean()
    data['Item_Visibility'] = data['Item_Visibility'].replace(0, non_zero_mean)
    data['Item_Weight'] = data['Item_Weight'].fillna(data.groupby('Item_Type')['Item_Weight'].transform('median'))
    data['Outlet_Size'] = data['Outlet_Size'].fillna(data.groupby('Outlet_Type')['Outlet_Size'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'))
    data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)

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
    current_year = datetime.now().year
    data['Years_Since_Establishment'] = current_year - data['Outlet_Establishment_Year']
    data['Outlet_Establishment_Year'] = data['Outlet_Establishment_Year'].replace(0, 1)

    # Handle outliers
    for column in data.select_dtypes(include='number').columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = data[column].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))

    # Feature Scaling
    min_max_scaler = MinMaxScaler()
    data[['Item_MRP', 'Item_Visibility','Outlet_Establishment_Year']] = min_max_scaler.fit_transform(data[['Item_MRP', 'Item_Visibility','Outlet_Establishment_Year']])

    # Log transformation to reduce skewness
    data['Item_Visibility_Log'] = np.log1p(data['Item_Visibility'])

    with open('model/in_min_max_scaler.pkl', 'wb') as file:
        pickle.dump(min_max_scaler, file)

    return data

# Loading the dataset
data = pd.read_csv('Train.csv')

# Splitting into training and testing sets
training, testing = train_test_split(data, test_size=0.2, random_state=42)

# Preprocessing / passing the function the data
training_data_processed = data_processing(training)
testing_data_processed = data_processing(testing)

def encode_and_scale(train, test):
    # Define mappings for columns that require encoding
    map_Outlet_Size = ['Medium', 'High', 'Small']

    # Initialize label encoders for Item_Identifier and Outlet_Identifier
    label_encoder_item = LabelEncoder()
    label_encoder_outlet = LabelEncoder()

    # Combine Item_Identifiers from train and test to fit the encoder
    combined_item_ids = pd.concat([train['Item_Identifier'], test['Item_Identifier']])
    combined_outlet_ids = pd.concat([train['Outlet_Identifier'], test['Outlet_Identifier']])

    # Fit encoders
    label_encoder_item.fit(combined_item_ids)
    label_encoder_outlet.fit(combined_outlet_ids)

    # Apply label encoding for Item_Identifier and Outlet_Identifier
    train['Item_Identifier'] = label_encoder_item.transform(train['Item_Identifier'])
    test['Item_Identifier'] = label_encoder_item.transform(test['Item_Identifier'])

    train['Outlet_Identifier'] = label_encoder_outlet.transform(train['Outlet_Identifier'])
    test['Outlet_Identifier'] = label_encoder_outlet.transform(test['Outlet_Identifier'])


    # OneHotEncoder for other categorical features
    nominal_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Type']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit the OneHotEncoder on the training data
    ohe.fit(train[nominal_columns])

    # Transform both train and test data
    ohe_train = ohe.transform(train[nominal_columns])
    ohe_test = ohe.transform(test[nominal_columns])

    # Convert the OneHotEncoded arrays to DataFrames for easier merging later
    ohe_train_df = pd.DataFrame(ohe_train, columns=ohe.get_feature_names_out(nominal_columns), index=train.index)
    ohe_test_df = pd.DataFrame(ohe_test, columns=ohe.get_feature_names_out(nominal_columns), index=test.index)

    # Combine OneHotEncoded features with other columns
    train_encoded = pd.concat([ohe_train_df, train.drop(nominal_columns, axis=1)], axis=1)
    test_encoded = pd.concat([ohe_test_df, test.drop(nominal_columns, axis=1)], axis=1)

    # OrdinalEncoding for 'Outlet_Size'
    ordinal_columns = ['Item_Visibility_Bins', 'Outlet_Size', 'Outlet_Location_Type']
    ode = OrdinalEncoder()
    ode.fit(train[ordinal_columns])

    # Apply Ordinal Encoding
    ode_train = ode.transform(train[ordinal_columns])
    ode_test = ode.transform(test[ordinal_columns])

    # Convert the OrdinalEncoded arrays to DataFrames
    ode_train_df = pd.DataFrame(ode_train, columns=ordinal_columns, index=train.index)
    ode_test_df = pd.DataFrame(ode_test, columns=ordinal_columns, index=test.index)

    # Combine OrdinalEncoded features with other columns
    train_encoded = pd.concat([ode_train_df, train_encoded.drop(ordinal_columns, axis=1)], axis=1)
    test_encoded = pd.concat([ode_test_df, test_encoded.drop(ordinal_columns, axis=1)], axis=1)


    # Save encoder models for later use
    with open('model/in_label_encoder_item.pkl', 'wb') as file:
        pickle.dump(label_encoder_item, file)
    with open('model/in_label_encoder_outlet.pkl', 'wb') as file:
        pickle.dump(label_encoder_outlet, file)
    with open('model/in_onehot_encoder.pkl', 'wb') as file:
        pickle.dump(ohe, file)
    with open('model/in_ordinal_encoder.pkl', 'wb') as file:
        pickle.dump(ode, file)

    return train_encoded, test_encoded

# Apply the encoding and scaling function
train_processed, test_processed = encode_and_scale(training_data_processed, testing_data_processed)

# Prepare data for training
x_train = train_processed.drop(['Item_Outlet_Sales'], axis=1)
y_train = train_processed['Item_Outlet_Sales']
x_test = test_processed.drop(['Item_Outlet_Sales'], axis=1)
y_test = test_processed['Item_Outlet_Sales']

# Defining models to train
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'KNN': KNeighborsRegressor(),
    "Gradient Boosting ":GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(),
}

r2_scores = {'Model': [], 'R² score of Testing': [], 'R² score of Training': []}
best_model_name = None
best_test_r2 = float('-inf')  # Start with the lowest possible R² score

# Iterate over models
for model_name, model in models.items():
    model.fit(x_train, y_train)

    # Applying Predict method
    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)

    # Finding the R² score
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    # Append results to the dictionary
    r2_scores['Model'].append(model_name)
    r2_scores['R² score of Testing'].append(test_r2)
    r2_scores['R² score of Training'].append(train_r2)

    # Update the best model if the current one has a higher test R²
    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_model_name = model_name
        best_model = model

# Create DataFrame for scores
r2_scores_df = pd.DataFrame(r2_scores)

# Print the R² scores for all models
print("R² Scores for all models:")
print(r2_scores_df)

# Print the best model
print(f"\nBest Model: {best_model_name}")
print(f"R² Score on Testing Data: {best_test_r2:.4f}")

# Hyperparameter Tuning with RandomizedSearchCV
## Define the parameter distribution for RandomForestRegressor
param_dist = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [3, 5, 10],  # Maximum tree depth
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples at a leaf node

}

# Initialize the RandomForestRegressor
rfr = GradientBoostingRegressor(random_state=42)

# Use RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(
    estimator=rfr,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    scoring='r2',
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all available processors
    random_state=42,  # Ensures reproducible results
    verbose=True,
    refit=True

)

# Fit the model to the training data
random_search.fit(x_train, y_train)
best_random_params = random_search.best_params_

# best parameters from RandomizedSearchCV
print(f"Best Parameters from RandomizedSearchCV: {best_random_params}")

# Training final model with the best parameters
final_model = GradientBoostingRegressor(**best_random_params,random_state=42)
final_model.fit(x_train, y_train)

# Evaluating final model
train_final_preds = final_model.predict(x_train)
test_final_preds = final_model.predict(x_test)

final_train_r2 = r2_score(y_train, train_final_preds)
final_test_r2 = r2_score(y_test, test_final_preds)

print(f"Final Training R²: {final_train_r2}")
print(f"Final Testing R²: {final_test_r2}")

with open('model/in_best_model.pkl', 'wb') as file:
    pickle.dump(final_model, file)

# After training the model
feature_names = x_train.columns.tolist()
joblib.dump(feature_names, 'model/in_feature_names.pkl')



