# **GrocerGenius: AI-Based Supermarket Sales Prediction**

> **Objective**  
> Develop an AI model to accurately forecast supermarket sales using historical data, paired with an intuitive user interface.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Project Workflow](#project-workflow)
3. [Detailed Workflow](#detailed-workflow)
   - Data Collection & Exploration
   - Exploratory Data Analysis (EDA)
   - Data Preprocessing
   - Model Building & Evaluation
   - Deployment & Documentation
4. [Technologies Used](#technologies-used)
5. [Getting Started](#getting-started)
6. [Results](#results)
7. [Future Enhancements](#future-enhancements)
8. [Contributing](#contributing)

---

### **Project Overview**
GrocerGenius is an AI-driven solution aimed at helping supermarkets forecast future sales based on historical sales data. Through predictive modeling, GrocerGenius empowers retail managers to optimize inventory, improve sales strategies, and enhance customer satisfaction.

---

### **Project Workflow**
- **Data Collection & Exploration**: Define the prediction task and understand dataset features.
- **Exploratory Data Analysis (EDA)**: Visualize feature distributions and explore relationships.
- **Data Preprocessing**: Handle missing values, perform feature engineering, and encode categorical features.
- **Model Building & Evaluation**: Train models with cross-validation, optimize hyperparameters, and evaluate performance.
- **Deployment & Documentation**: Deploy the model on a web platform, create an API, and document the process.

---

### **Detailed Workflow**

#### 1. **Data Collection & Exploration**
   - Define the problem statement and objectives.
   - Load and inspect the dataset for initial insights into the structure and content.
   - Assess missing values and initial distributions of key features.

#### 2. **Exploratory Data Analysis (EDA)**
   - **Visualizations**:
      - **Sales Distribution**: Distribution plot to understand sales patterns.
      - **Product Category Sales**: Bar chart of sales across different categories.
      - **Seasonality Analysis**: Line chart to track monthly or quarterly trends.
   - **Correlations**:
      - Identify relationships between features and the target variable.

#### 3. **Data Preprocessing**
   - **Handling Missing Values**: Impute missing values using appropriate methods.
   - **Feature Engineering**: Create new features (e.g., 'Day of the Week', 'Holiday Period') to improve model performance.
   - **Encoding**: Encode categorical features with methods like One-Hot Encoding or Target Encoding.

#### 4. **Model Building & Evaluation**
   - **Model Selection**: Implement algorithms such as Linear Regression, Decision Trees, and XGBoost.
   - **Cross-Validation**: Use cross-validation techniques to ensure model robustness.
   - **Hyperparameter Tuning**: Fine-tune model parameters using Grid Search or Random Search.
   - **Performance Metrics**: Evaluate models based on RMSE, MAE, or R-squared.

#### 5. **Deployment & Documentation**
   - **Deployment**: Deploy the model using a web framework like Flask or Django.
   - **API Creation**: Create REST APIs to interface with the model.
   - **Documentation**: Document the entire process, code, and model usage for ease of replication.

---

### **Technologies Used**
- **Programming Languages**: Python
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Modeling**: Scikit-Learn, XGBoost
- **Deployment**: Flask, REST API
- **Version Control**: Git
- **Project Management**: Jupyter Notebooks for workflow and analysis

---

### **Getting Started**
1. **Prerequisites**:
   - Python 3.8+
   - Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `flask`
2. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Running the Project**:
   - Jupyter Notebooks are available in the `notebooks/` folder for each phase.
   - To deploy the API:
     ```bash
     python app.py
     ```
4. **Testing**: Run the test suite in `tests/` to validate model accuracy and functionality.

---

### **Results**
- **Model Performance**:
- Best Model: XGBoost
   - RMSE: 0.9786834142881268
   - MAE: 64177.4278813146

---

### **Acknowledgements**
- This project is part of **Infosys Internship Program 2024**.
- Special thanks to the mentor for the guide.

---
