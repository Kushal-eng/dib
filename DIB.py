import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Load the dataset
file_path = "diabetes_data.xlsx"
data = pd.read_excel(file_path, engine='openpyxl')

# Display the data
st.title("Diabetes Prediction Dashboard")
st.write("## Data Overview")
st.dataframe(data)

# User input for Random Forest Classifier
st.sidebar.header("Input Features for Random Forest Prediction")
age = st.sidebar.slider("Age", int(data["Age"].min()), int(data["Age"].max()), int(data["Age"].mean()))
bmi = st.sidebar.slider("BMI", float(data["BMI"].min()), float(data["BMI"].max()), float(data["BMI"].mean()))
bp = st.sidebar.slider("Blood Pressure", int(data["BloodPressure"].min()), int(data["BloodPressure"].max()), int(data["BloodPressure"].mean()))
glucose = st.sidebar.slider("Glucose", int(data["Glucose"].min()), int(data["Glucose"].max()), int(data["Glucose"].mean()))
insulin = st.sidebar.slider("Insulin", int(data["Insulin"].min()), int(data["Insulin"].max()), int(data["Insulin"].mean()))
dpf = st.sidebar.slider("Diabetes Pedigree Function", float(data["DiabetesPedigreeFunction"].min()), 
                        float(data["DiabetesPedigreeFunction"].max()), float(data["DiabetesPedigreeFunction"].mean()))

# Splitting data for Random Forest
rf_features = ["Age", "BMI", "BloodPressure", "Glucose", "Insulin", "DiabetesPedigreeFunction"]
X_rf = data[rf_features]
y_rf = data["Outcome"]
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_rf_train, y_rf_train)

# Random Forest Predictions
rf_prediction = rf_model.predict([[age, bmi, bp, glucose, insulin, dpf]])
rf_prediction_proba = rf_model.predict_proba([[age, bmi, bp, glucose, insulin, dpf]])[:, 1]

# Display Random Forest Prediction
st.write("## Random Forest Prediction")
st.write("Based on the input features, the prediction is:")
st.write("Diabetes" if rf_prediction[0] == 1 else "No Diabetes")
st.write(f"Prediction Probability: {rf_prediction_proba[0]*100:.2f}%")

# Model Accuracy for Random Forest
rf_accuracy = accuracy_score(y_rf_test, rf_model.predict(X_rf_test))
st.write("## Random Forest Model Accuracy")
st.write(f"Model Accuracy: {rf_accuracy*100:.2f}%")

# Sidebar controls for Linear Regression
st.sidebar.header("Linear Regression Prediction")
selected_features = st.sidebar.multiselect(
    "Select Features for Prediction (Linear Regression)",
    options=["Age", "BMI", "BloodPressure"],
    default=["Age", "BMI"]
)
predict_target = st.sidebar.selectbox("Select Target Variable (Linear Regression)", options=["Glucose"])

# Train Linear Regression model
if selected_features and predict_target:
    X_lr = data[selected_features]
    y_lr = data[predict_target]
    
    # Split the data for Linear Regression
    X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
    
    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_lr_train, y_lr_train)
    
    # Linear Regression Predictions
    y_lr_pred = lr_model.predict(X_lr_test)
    
    # Model evaluation
    mse = mean_squared_error(y_lr_test, y_lr_pred)
    r2 = r2_score(y_lr_test, y_lr_pred)
    
    # Display model metrics
    st.write("## Linear Regression Results")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")
    
    # Allow user to input custom values for Linear Regression prediction
    st.write("### Predict Target (Linear Regression)")
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
    lr_prediction = lr_model.predict(user_inputs_df)[0]
    st.write(f"**Predicted {predict_target}:** {lr_prediction:.2f}")
