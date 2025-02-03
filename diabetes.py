import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

# Load dataset
df = pd.read_csv("E:\\diabetes.csv")

# Data Analysis
df.isnull().sum()
df["Outcome"].value_counts()

# Data Visualization
sns.countplot(x="Outcome", data=df)
sns.countplot(x="Pregnancies", data=df)
sns.countplot(x="Pregnancies", hue="Outcome", data=df)
df.hist(bins=15, figsize=(15, 10))
plt.hist(df["Age"], bins=10, edgecolor="black")
plt.show()

# Prepare data for machine learning
features = df.drop("Outcome", axis=1).values
labels = df["Outcome"].values
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# K-Nearest Neighbors Model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)
prediction_knn = knn.predict(X_test)
print("KNN Accuracy:", knn_score)

# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_score = lr_model.score(X_test, y_test)
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", lr_score)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# Support Vector Machine (SVM) Model
svm_model = SVC(kernel='linear', class_weight="balanced")
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
svm_score = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", svm_score)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# User Input for Prediction
no_pregnancies = int(input("How many pregnancies does the patient have? "))
glucose = float(input("What is glucose level? "))
bp = float(input("Enter blood pressure: "))
skin_thickness = float(input("Enter skin thickness: "))
insulin = float(input("Enter insulin level: "))
bmi = float(input("Enter BMI: "))
db_func = float(input("Enter diabetes pedigree function: "))
age = int(input("Enter age: "))

input_features = np.array([no_pregnancies, glucose, bp, skin_thickness, insulin, bmi, db_func, age]).reshape(1, -1)

# Model Predictions
knn_predict = knn.predict(input_features)
lr_predict = lr_model.predict(input_features)
svm_predict = svm_model.predict(input_features)

print("Your results according to different ML models:")
print("KNN:", "Diabetic" if knn_predict == 1 else "Non-Diabetic")
print("Logistic Regression:", "Diabetic" if lr_predict == 1 else "Non-Diabetic")
print("SVM:", "Diabetic" if svm_predict == 1 else "Non-Diabetic")

# Probability Prediction (Logistic Regression)
prob = lr_model.predict_proba(input_features)[0]
if lr_predict == 0:
    print(f"You have {prob[0] * 100:.2f}% chances of being Non-Diabetic")
else:
    print(f"You have {prob[1] * 100:.2f}% chances of being Diabetic")

# Save Models
pickle.dump(svm_model, open("model.pkl", "wb"))
pickle.dump(lr_model, open("lr_model.pkl", "wb"))