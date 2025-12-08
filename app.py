import streamlit as st
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("credit_scaler.pkl")
label_encoders = joblib.load("credit_label_encoders.pkl")

st.title("Credit Risk Prediction App")
st.write("Enter applicant details to predict credit risk:")

# Input form
age = st.number_input("Age", min_value=18, max_value=100, step=1)
sex = st.selectbox("Sex", ['male', 'female'])
job = st.selectbox("Job", [0, 1, 2, 3])
housing = st.selectbox("Housing", ['own', 'free', 'rent'])
saving_accounts = st.selectbox("Saving Accounts", ['no_info', 'little', 'moderate', 'quite rich', 'rich'])
checking_account = st.selectbox("Checking Account", ['no_info', 'little', 'moderate', 'rich'])
duration = st.slider("Loan Duration (in months)", 4, 72, 12)
purpose = st.selectbox("Purpose", ['radio/TV', 'education', 'furniture/equipment', 'new car', 'used car', 
                                   'domestic appliance', 'business', 'repairs', 'vacation/others'])

if st.button("Predict Credit Risk"):
    input_data = {
        'Age': age,
        'Sex': label_encoders['Sex'].transform([sex])[0],
        'Job': job,
        'Housing': label_encoders['Housing'].transform([housing])[0],
        'Saving_accounts': label_encoders['Saving_accounts'].transform([saving_accounts])[0],
        'Checking_account': label_encoders['Checking_account'].transform([checking_account])[0],
        'Duration': duration,
        'Purpose': label_encoders['Purpose'].transform([purpose])[0]
    }
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    risk_label = "Bad Credit Risk ❌" if prediction == 1 else "Good Credit Risk ✅"
    st.subheader(f"Prediction: {risk_label}")
