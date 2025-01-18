import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "diabetes_data.xlsx"
data = pd.read_excel(file_path, engine='openpyxl')

# Display the data
st.title("Diabetes Prediction Dashboard")
st.write("## Data Overview")
st.dataframe(data)

# User input for features
st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", int(data["Age"].min()), int(data["Age"].max()), int(data["Age"].mean()))
bmi = st.sidebar.slider("BMI", float(data["BMI"].min()), float(data["BMI"].max()), float(data["BMI"].mean()))
bp = st.sidebar.slider("Blood Pressure", int(data["BloodPressure"].min()), int(data["BloodPressure"].max()), int(data["BloodPressure"].mean()))
glucose = st.sidebar.slider("Glucose", int(data["Glucose"].min()), int(data["Glucose"].max()), int(data["Glucose"].mean()))
insulin = st.sidebar.slider("Insulin", int(data["Insulin"].min()), int(data["Insulin"].max()), int(data["Insulin"].mean()))
dpf = st.sidebar.slider("Diabetes Pedigree Function", float(data["DiabetesPedigreeFunction"].min()), 
                        float(data["DiabetesPedigreeFunction"].max()), float(data["DiabetesPedigreeFunction"].mean()))

# Splitting data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
prediction = model.predict([[age, bmi, bp, glucose, insulin, dpf]])
prediction_proba = model.predict_proba([[age, bmi, bp, glucose, insulin, dpf]])[:, 1]

# Display prediction
st.write("## Prediction")
st.write("Based on the input features, the prediction is:")
st.write("Diabetes" if prediction[0] == 1 else "No Diabetes")
st.write(f"Prediction Probability: {prediction_proba[0]*100:.2f}%")


# Model accuracy
st.write("## Model Accuracy")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy*100:.2f}%")
