# **GrocerGenius: AI-Based Supermarket Sales Prediction**  

## **Overview**  
GrocerGenius is a data-driven solution aimed at transforming supermarket operations by accurately forecasting sales. By analyzing historical sales data and outlet characteristics, the project empowers retail managers to make informed decisions, optimize inventory, and boost customer satisfaction. This repository contains the complete implementation of the project, including data preprocessing, exploratory analysis, machine learning models, and deployment via an API.

---

## **Features**  
- Predict future sales for different supermarket products and outlets.  
- Analyze sales trends and uncover insights through visualizations.  
- Optimize inventory management with accurate forecasts.  
- Easily deploy the model through a Flask-based REST API.  

---

## **Project Workflow**  
1. **Data Collection & Exploration**  
   - Define objectives and inspect the dataset.  
   - Address missing values and data inconsistencies.  

2. **Exploratory Data Analysis (EDA)**  
   - Visualize sales patterns, seasonal trends, and relationships between features.  

3. **Data Preprocessing**  
   - Impute missing values.  
   - Perform feature engineering and encode categorical features.  

4. **Model Building & Evaluation**  
   - Train multiple models like Random Forest, XGBoost, and Linear Regression.  
   - Optimize performance using cross-validation and hyperparameter tuning.  

5. **Deployment**  
   - Build a REST API using Flask for real-time predictions.  

---

## **Technologies Used**  
- **Programming Languages:** Python  
- **Libraries for Analysis & Modeling:** Pandas, NumPy, Scikit-Learn, XGBoost  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Streamlit  
- **Version Control:** Git  

---

## **Dataset Details**  
- **Size:** 8,523 rows, 12 columns  
- **Columns:** Includes features such as:  
  - `Item_Identifier`  
  - `Item_Weight`  
  - `Item_Fat_Content`  
  - `Item_Visibility`  
  - `Item_Type`  
  - `Item_MRP`  
  - `Outlet_Identifier`  
  - `Outlet_Establishment_Year`  
  - `Outlet_Size`  
  - `Outlet_Location_Type`  
  - `Outlet_Type`  
  - `Item_Outlet_Sales` (target variable)  
- **Strengths:** Detailed features about products and outlets for predictive modeling.  
- **Challenges:** Missing values in critical fields.  

---

## **Installation & Usage**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/amalsalilan/GrocerGenius_AI_Based_Supermarket_Sales_Prediction_Infosys_Internship_Oct2024.git
cd GrocerGenius_AI_Based_Supermarket_Sales_Prediction_Infosys_Internship_Oct2024
```

### **2. Install Dependencies**  
Make sure you have Python installed. Install the required libraries using:  
```bash
pip install -r requirements.txt
```

### **3. Run the Project**  
- **Data Analysis:** Use Jupyter Notebooks in the repository to explore the data and build models.  
- **API Deployment:** Navigate to the Streamlit app:  
  ```bash
  streamlit run app.py
  ```
---

## **Results & Insights**  
- Achieved high accuracy in predicting sales using XGBoost.  
- Discovered significant patterns in sales trends, item types, and outlet characteristics.  
- Improved operational strategies for inventory management.  

---

## **Future Scope**  
- Incorporate real-time data for dynamic predictions.  
- Expand the model to include external factors like promotions or weather.  
- Develop a dashboard for interactive visualizations and insights.  

---

## **Contributors**  
- [Vrushika K. Panchal](https://github.com/vrushika-k-panchal)  
- [Amal Salilan](https://github.com/amalsalilan)  

---
## **Try out on:**  
https://grocergenius.streamlit.app/

## **License**  
This project is licensed under the [MIT License](LICENSE).  
