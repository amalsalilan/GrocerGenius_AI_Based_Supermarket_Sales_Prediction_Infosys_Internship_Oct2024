---

# GrocerGenius: AI-Based Supermarket Sales Prediction 🛒🤖

Welcome to **GrocerGenius** – the AI-powered app designed to predict supermarket sales with precision! Using **machine learning** techniques, GrocerGenius helps supermarkets forecast sales based on historical data, promotions, holidays, and more. With this app, grocery stores can optimize stock levels, improve inventory management, and plan promotions effectively.

---

## 🚀 **Project Overview**

**GrocerGenius** uses advanced **AI and machine learning** algorithms to predict supermarket sales for any given day, helping businesses manage inventory, minimize waste, and ensure product availability. The application uses **Streamlit** for an intuitive and interactive frontend, while **Flask** powers the backend, ensuring robust data handling and machine learning predictions.

### Key Features:
- **AI-Powered Predictions**: Accurate sales forecasts using machine learning models.
- **Streamlit Frontend**: Interactive interface for easy input and results visualization.
- **Flask Backend**: Efficiently serves the trained models and handles user requests.
- **Performance Metrics**: Displays key metrics like accuracy, confidence intervals, and model evaluation.

---

## ⚙️ **Technologies Used**

- **Python**: Programming language for backend logic and machine learning.
- **Streamlit**: Framework for creating a user-friendly frontend interface.
- **Flask**: Web framework for building the backend and serving predictions.
- **Scikit-Learn**: Machine learning library for training and evaluating models.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations and array operations.
- **Matplotlib & Seaborn**: Data visualization for better insights.
- **Jupyter Notebooks**: For exploratory data analysis and model prototyping.

---

## 📊 **How It Works**

### 1. **Data Collection**:
   Historical sales data is collected, including key features such as Item Outlet Sales, Product Features, Outlet Types.

### 2. **Data Preprocessing**:
   The raw data is cleaned, formatted, and prepared for analysis by handling missing values, encoding categorical variables, and scaling numerical features.

### 3. **Model Training**:
   Machine learning algorithms like **Random Forest** and **XGBoost** are trained on the cleaned dataset to predict sales for future dates.

### 4. **Prediction**:
   The trained model is deployed through the **Flask backend**, where predictions can be made based on user inputs such as promotion status, Product MRP, and Product Details.

### 5. **Streamlit Interface**:
   The **Streamlit frontend** allows users to input relevant data, trigger sales predictions, and view the results, all within an interactive and easy-to-navigate web app.

---

## 📦 **Installation**

### Prerequisites:

Ensure you have **Python 3.x** installed along with **pip** for package management.

### Steps to Set Up:

1. Clone the repository:

   ```bash
   git clone https://github.com/amalsalilan/GrocerGenius_AI_Based_Supermarket_Sales_Prediction_Infosys_Internship_Oct2024.git
   cd grocergenius
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the **Flask backend**:

   ```bash
   python backend/app.py
   ```

4. Start the **Streamlit frontend**:

   ```bash
   streamlit run frontend/app.py
   ```

5. Open your browser and go to `http://localhost:8501/` to interact with the app.

---

## 💻 **Usage**

- **Input**: Enter parameters such as Item Identifier, Item Wright, or Product information in the Streamlit frontend.
- **Prediction**: After entering the data, the app will predict the sales for the specified date and display the result, along with performance metrics.

---

## 📈 **Model Evaluation**

The machine learning models are evaluated using the following key metrics:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² (Coefficient of Determination)**

These metrics provide insights into the accuracy and reliability of the predictions made by the model.

---

## 🌟 **Contributing**

We welcome contributions! If you’d like to improve the app or fix any issues, feel free to fork the repository, submit pull requests, or open an issue for discussion. Please ensure you follow the guidelines for contributing.

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📑 **Project Structure**

Here’s an overview of the project structure:

```
grocergenius/
│
├── backend/               # Flask backend for model prediction and data handling
│   ├── app.py             # Flask app to handle API requests
│   └── model.py           # Machine learning model and prediction logic
│
├── frontend/              # Streamlit frontend for user interaction
│   └── app.py             # Streamlit app to display inputs, predictions, and results
│
├── data/                  # Folder for raw and processed data
│   ├── raw_data.csv       # Raw historical sales data
│   └── processed_data.csv # Cleaned and preprocessed data for modeling
│
├── requirements.txt       # List of dependencies
├── README.md              # Project documentation
└── LICENSE                # License information
```

---

## 📝 **Acknowledgements**

- A **Special Thank** You to **Amal Sir**, our mentor, whose guidance, expertise, and unwavering support made this project possible. His insights and feedback were instrumental in shaping the development of GrocerGenius, and his encouragement kept us motivated throughout the journey.
- A big shoutout to the entire **team** for their hard work and collaboration in bringing this project to life.
- Thanks to the **Machine Learning** community for providing the tools and resources that helped build this app.
- Thanks to the dataset providers for the invaluable historical data used for training the model.

---

**GrocerGenius** leverages the power of AI to make supermarket sales forecasting smarter, faster, and more accurate. By using this app, supermarkets can optimize inventory and plan effectively, making operations smoother and more efficient.

