import streamlit as st
import numpy as np
import pickle

# Load trained models
svm_model = pickle.load(open("model.pkl", "rb"))
lr_model = pickle.load(open("lr_model.pkl", "rb"))

# Streamlit UI
st.title("Diabetes Prediction App")
st.markdown("### Enter the details below to check your diabetes status")

# User Input
no_pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0.0, step=0.1)
bp = st.number_input("Blood Pressure", min_value=0.0, step=0.1)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=0.1)
insulin = st.number_input("Insulin Level", min_value=0.0, step=0.1)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
db_func = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=0, step=1)

# Convert inputs to array
input_features = np.array([no_pregnancies, glucose, bp, skin_thickness, insulin, bmi, db_func, age]).reshape(1, -1)

# Prediction
if st.button("Predict"):
    knn_prediction = svm_model.predict(input_features)
    lr_prediction = lr_model.predict(input_features)
    
    st.subheader("Prediction Results:")
    st.write("SVM Model Prediction:", "Diabetic" if knn_prediction == 1 else "Non-Diabetic")
    st.write("Logistic Regression Prediction:", "Diabetic" if lr_prediction == 1 else "Non-Diabetic")
    
    # Probability using Logistic Regression
    prob = lr_model.predict_proba(input_features)[0]
    if lr_prediction == 0:
        st.write(f"You have {prob[0] * 100:.2f}% chances of being Non-Diabetic")
    else:
        st.write(f"You have {prob[1] * 100:.2f}% chances of being Diabetic")
