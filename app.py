
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS for responsive design
st.markdown("""
<style>
    .main {background-color: #f0f2f6; padding: 10px;}
    .stButton>button {
        background-color: #4CAF50; 
        color: white; 
        border-radius: 8px; 
        width: 100%; 
        padding: 12px; 
        font-size: 16px;
    }
    .stNumberInput input, .stSelectbox select {
        background-color: #ffffff; 
        border-radius: 5px; 
        padding: 10px; 
        font-size: 16px;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff; 
        padding: 10px; 
        border-radius: 10px;
    }
    .prediction-box {
        padding: 15px; 
        border-radius: 10px; 
        margin: 10px 0; 
        font-size: 16px;
    }
    .success {background-color: #d4edda; color: #155724;}
    .error {background-color: #f8d7da; color: #721c24;}
    /* Responsive adjustments */
    @media (max-width: 600px) {
        .stNumberInput input, .stSelectbox select {font-size: 14px; padding: 8px;}
        .stButton>button {font-size: 14px; padding: 10px;}
        .prediction-box {font-size: 14px; padding: 10px;}
        h1 {font-size: 24px;}
        h2 {font-size: 20px;}
        .stTabs [data-baseweb="tab"] {font-size: 14px;}
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
try:
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Ensure 'heart_disease_model.pkl' and 'scaler.pkl' are in the directory.")
    st.stop()

# Language dictionary (English)
lang = {
    "title": "Heart Disease Prediction",
    "description": "Enter patient details to predict heart disease risk.",
    "header": "Patient Information",
    "age": "Age (years)",
    "sex": "Sex",
    "resting_bp": "Resting Blood Pressure (mm Hg)",
    "cholesterol": "Serum Cholesterol (mg/dl)",
    "max_heart_rate": "Maximum Heart Rate",
    "oldpeak": "Oldpeak (ST depression)",
    "fasting_blood_sugar": "Fasting Blood Sugar > 120 mg/dL",
    "exercise_angina": "Exercise-Induced Angina",
    "chest_pain_type": "Chest Pain Type",
    "resting_ecg": "Resting ECG",
    "st_slope": "ST Slope",
    "predict_button": "Predict Risk",
    "prediction_header": "Prediction Result",
    "recommendations_header": "Health Recommendations",
    "low_risk": "Low risk of heart disease (Probability: {prob:.2%})",
    "high_risk": "High risk of heart disease (Probability: {prob:.2%})",
    "consult": "Consult a healthcare professional for further evaluation.",
    "diet": "Diet Plan",
    "exercise": "Exercise Routine",
    "lifestyle": "Lifestyle Changes",
    "input_error": "Invalid input: Cholesterol and Blood Pressure must be > 0."
}

# App title
st.title(lang["title"])

# Tabs for navigation
tab1, tab2 = st.tabs(["Prediction", "Feature Insights"])

with tab1:
    st.write(lang["description"])
    st.header(lang["header"])

    # Responsive input layout
    col1, col2 = st.columns([1, 1])
    with col1:
        age = st.number_input(lang["age"], min_value=1, max_value=120, value=40, key="age")
        resting_bp = st.number_input(lang["resting_bp"], min_value=50, max_value=250, value=120, key="bp")
        cholesterol = st.number_input(lang["cholesterol"], min_value=50, max_value=600, value=200, key="chol")
        max_heart_rate = st.number_input(lang["max_heart_rate"], min_value=50, max_value=250, value=150, key="hr")
        oldpeak = st.number_input(lang["oldpeak"], min_value=-3.0, max_value=10.0, value=0.0, step=0.1, key="oldpeak")
    with col2:
        sex = st.selectbox(lang["sex"], options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0], key="sex")
        fasting_blood_sugar = st.selectbox(lang["fasting_blood_sugar"], options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="fbs")
        exercise_angina = st.selectbox(lang["exercise_angina"], options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="angina")
        chest_pain_type = st.selectbox(lang["chest_pain_type"], 
                                       options=[("Typical Angina", 1), ("Atypical Angina", 2), 
                                                ("Non-anginal Pain", 3), ("Asymptomatic", 4)], 
                                       format_func=lambda x: x[0], key="cp")
        resting_ecg = st.selectbox(lang["resting_ecg"], 
                                   options=[("Normal", 0), ("ST-T Wave Abnormality", 1), 
                                            ("Left Ventricular Hypertrophy", 2)], 
                                   format_func=lambda x: x[0], key="ecg")
        st_slope = st.selectbox(lang["st_slope"], 
                                options=[("Upward", 1), ("Flat", 2), ("Downward", 3)], 
                                format_func=lambda x: x[0], key="slope")

    # Prediction button
    if st.button(lang["predict_button"], key="predict"):
        if cholesterol <= 0 or resting_bp <= 0:
            st.error(lang["input_error"])
        else:
            try:
                input_data = {
                    'age': age,
                    'sex': sex[1],
                    'resting bp s': resting_bp,
                    'cholesterol': cholesterol,
                    'fasting blood sugar': fasting_blood_sugar[1],
                    'max heart rate': max_heart_rate,
                    'exercise angina': exercise_angina[1],
                    'oldpeak': oldpeak,
                    'chest pain type_2': 1 if chest_pain_type[1] == 2 else 0,
                    'chest pain type_3': 1 if chest_pain_type[1] == 3 else 0,
                    'chest pain type_4': 1 if chest_pain_type[1] == 4 else 0,
                    'resting ecg_1': 1 if resting_ecg[1] == 1 else 0,
                    'resting ecg_2': 1 if resting_ecg[1] == 2 else 0,
                    'ST slope_1': 1 if st_slope[1] == 1 else 0,  # Added missing feature
                    'ST slope_2': 1 if st_slope[1] == 2 else 0,
                    'ST slope_3': 1 if st_slope[1] == 3 else 0
                }
                
                input_df = pd.DataFrame([input_data])
                numeric_cols = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
                input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
                
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                
                st.header(lang["prediction_header"])
                if prediction == 1:
                    st.markdown(f'<div class="prediction-box error">{lang["high_risk"].format(prob=probability)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box success">{lang["low_risk"].format(prob=probability)}</div>', unsafe_allow_html=True)
                st.info(lang["consult"])

                # Recommendations
                st.header(lang["recommendations_header"])
                st.subheader(lang["diet"])
                if cholesterol > 240 or probability > 0.5:
                    st.write("- Reduce saturated fats: Choose grilled or steamed foods.")
                    st.write("- Increase fiber: Eat oats, apples, and broccoli.")
                    st.write("- Limit sodium: Lower salt to manage blood pressure.")
                else:
                    st.write("- Balanced diet: Include whole grains, lean proteins, avocados.")
                    st.write("- Limit sugar: Avoid sugary drinks and snacks.")
                    st.write("- Healthy fats: Use olive oil or nuts moderately.")
                
                st.subheader(lang["exercise"])
                if max_heart_rate < 120 or probability > 0.5:
                    st.write("- Low-intensity cardio: 30 min brisk walk, 5x/week.")
                    st.write("- Consult doctor before intense exercise.")
                    st.write("- Flexibility: Daily yoga or stretching.")
                else:
                    st.write("- Cardio: 150 min/week jogging or cycling.")
                    st.write("- Strength: 2-3 sessions/week for major muscles.")
                    st.write("- Stay active: Walk or take stairs.")
                
                st.subheader(lang["lifestyle"])
                if resting_bp > 140 or probability > 0.5:
                    st.write("- Manage stress: Meditate or practice yoga.")
                    st.write("- Monitor BP: Regular check-ups, low-salt diet.")
                    st.write("- Avoid smoking: Seek support to quit.")
                else:
                    st.write("- Quit smoking: Improve heart health.")
                    st.write("- Sleep: Aim for 7-8 hours nightly.")
                    st.write("- Stay social: Engage in hobbies or community.")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

with tab2:
    st.header("Feature Insights")
    feature_importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(feature_importance)*0.4)))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    plt.title('Feature Importance')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("This chart highlights factors influencing predictions. Higher importance means greater impact.")
