import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load the dataset
file_path = "perfect_regression_analysis_dataset.xlsx"
data = pd.read_excel(file_path)

# Streamlit App Title
st.title("Diabetes Prediction Dashboard")

# User input for Random Forest Classification
st.sidebar.header("Input Features for Classification")
age =st.sidebar.slider(
    "Age",
    min_value=10,  # Start the slider at 10
    max_value=int(data["Age"].max()),
    value=int(data["Age"].mean()))
bmi = st.sidebar.slider("BMI", float(data["BMI"].min()), float(data["BMI"].max()), float(data["BMI"].mean()))
bp = st.sidebar.slider("Blood Pressure", int(data["BloodPressure"].min()), int(data["BloodPressure"].max()), int(data["BloodPressure"].mean()))
glucose = st.sidebar.slider("Glucose", int(data["Glucose"].min()), int(data["Glucose"].max()), int(data["Glucose"].mean()))
insulin = st.sidebar.slider("Insulin", int(data["Insulin"].min()), int(data["Insulin"].max()), int(data["Insulin"].mean()))
dpf = st.sidebar.slider("Diabetes Pedigree Function", float(data["DiabetesPedigreeFunction"].min()), 
                        float(data["DiabetesPedigreeFunction"].max()), float(data["DiabetesPedigreeFunction"].mean()))

# Splitting data for Random Forest Classification
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training - Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions for classification
classification_prediction = rf_model.predict([[age, bmi, bp, glucose, insulin, dpf]])
classification_proba = rf_model.predict_proba([[age, bmi, bp, glucose, insulin, dpf]])[:, 1]

# Display classification prediction
st.write("## Classification Prediction")
st.write("Based on the input features, the prediction is:")
st.write("Diabetes" if classification_prediction[0] == 1 else "No Diabetes")
st.write(f"Prediction Probability: {classification_proba[0]*100:.2f}%")

# Model accuracy for classification
classification_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
st.write("## Random Forest Model Accuracy")
st.write(f"Model Accuracy: {classification_accuracy*100:.2f}%")

# Sidebar for Regression
st.sidebar.header("Regression Analysis")
regression_features = st.sidebar.multiselect(
    "Select Features for Regression (Glucose Prediction)",
    options=["Age", "BMI", "BloodPressure", "Insulin", "DiabetesPedigreeFunction"],
    default=["Age", "BMI"]
)

# Linear Regression and Multiple Regression
if regression_features:
    # Splitting data for Regression
    X_reg = data[regression_features]
    y_reg = data["Glucose"]
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    # Train Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train_reg, y_train_reg)

    # Predictions for Regression
    y_pred_reg = lr_model.predict(X_test_reg)

    # Model Evaluation for Regression
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)

    # Display Regression Results
    st.write("## Regression Analysis")
    st.write("### Selected Features for Regression:")
    st.write(regression_features)
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # User Input for Regression Prediction
    st.write("### Predict Glucose Level")
    user_inputs = {}
    for feature in regression_features:
        user_inputs[feature] = st.sidebar.slider(
            f"Input {feature}",
            float(data[feature].min()),
            float(data[feature].max()),
            float(data[feature].mean())
        )

    # Make Prediction based on user inputs
    user_inputs_df = pd.DataFrame([user_inputs])
    glucose_prediction = lr_model.predict(user_inputs_df)[0]
    st.write(f"**Predicted Glucose Level:** {glucose_prediction:.2f}")
