# Diabetes Prediction App

## Overview
The **Diabetes Prediction App** is a machine learning-based web application built with **Streamlit**. It allows users to input health parameters and predicts whether they are diabetic or not using trained **SVM** and **Logistic Regression** models.

## Features
- User-friendly web interface built with Streamlit.
- Accepts user inputs such as glucose level, blood pressure, BMI, and more.
- Predicts diabetes using **SVM** and **Logistic Regression** models.
- Displays probability scores for the logistic regression model.
- Lightweight and easy to deploy.

## Installation
### Prerequisites
Ensure you have **Python 3.x** installed on your system.

### Steps to Install & Run:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/diabetes-prediction.git
   cd diabetes-prediction
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure
```
/diabetes-prediction
│── app.py              # Streamlit app file
│── diabetes.py         # Machine learning model training script
│── model.pkl          # Trained SVM model
│── lr_model.pkl       # Trained Logistic Regression model
│── requirements.txt   # List of dependencies
│── README.md          # Project documentation
```

## Usage
1. Open the app in a browser after running `streamlit run app.py`.
2. Enter health-related details such as glucose level, BMI, and blood pressure.
3. Click the **Predict** button.
4. View predictions from different models along with probability estimates.

## Dependencies
- Python 3.x
- Streamlit
- NumPy
- Scikit-learn
- Pandas
- Seaborn



