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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sidebar controls for user input
st.sidebar.header("Linear Regression Prediction")
selected_features = st.sidebar.multiselect(
    "Select Features for Prediction",
    options=["Age", "BMI", "BloodPressure"],
    default=["Age", "BMI"]
)
predict_target = st.sidebar.selectbox(
    "Select Target Variable",
    options=["Glucose"]
)

# Train Linear Regression model
if selected_features and predict_target:
    X = data[selected_features]
    y = data[predict_target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = linear_model.predict(X_test)
    
    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Display model metrics
    st.write("### Linear Regression Results")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")
    
    # Allow user to input custom values for prediction
    st.write("### Predict Glucose Level")
    user_inputs = {}
    for feature in selected_features:
        user_inputs[feature] = st.sidebar.slider(
            f"Input {feature}",
            float(data[feature].min()),
            float(data[feature].max()),
            float(data[feature].mean())
        )
    
    # Predict based on user inputs
    user_inputs_df = pd.DataFrame([user_inputs])
    prediction = linear_model.predict(user_inputs_df)[0]
    st.write(f"**Predicted {predict_target}:** {prediction:.2f}")


# Model accuracy
st.write("## Model Accuracy")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy*100:.2f}%")
