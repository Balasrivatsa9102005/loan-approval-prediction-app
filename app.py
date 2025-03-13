import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

import pickle

# Load the Logistic Regression model
logreg_model = pickle.load(open(r'C:\Users\ADMIN\Desktop\Balu\College related\vs code\vs\loanapprovalproject\logreg_loan_approval_model.pkl', 'rb'))

# Load the XGBoost model
xgb_model = pickle.load(open(r'C:\Users\ADMIN\Desktop\Balu\College related\vs code\vs\loanapprovalproject\xgb_loan_approval_model.pkl', 'rb'))

# Load the Scaler
scaler = pickle.load(open(r'C:\Users\ADMIN\Desktop\Balu\College related\vs code\vs\loanapprovalproject\loan_approval_scaler.pkl', 'rb'))

# Streamlit UI setup
st.title('Loan Approval Prediction')
st.title("🎉 Welcome to the Loan Approval Prediction App! 🎉")
st.write("This app will help you predict the likelihood of getting your loan approved based on your inputs. 🏠💵")


# User Input fields using sliders
education = st.selectbox("🎓 Are you a graduate?", ['Yes', 'No'])
self_employed = st.selectbox("💼 Are you self-employed?", ['Yes', 'No'])

# Slider for number of dependents
no_of_dependents = st.slider("👨‍👩‍👧‍👦 Enter the number of dependents:", 0, 10, 0)

# Slider for annual income
income_annum = st.slider("💰 Enter your annual income:", 0.0, 1000000.0, 50000.0)

# Slider for loan amount
loan_amount = st.slider("🏡 Enter your loan amount:", 0.0, 500000.0, 200000.0)

# Slider for loan term (in years)
loan_term = st.slider("📅 Enter loan term (in years):", 1, 30, 10)

# Slider for CIBIL score
cibil_score = st.slider("Enter your CIBIL score (e.g., 600, 750):", 300, 900, 650)

# Slider for assets
assets = st.slider("🏠 Enter your total assets (residential, commercial, luxury, and bank):", 0.0, 5000000.0, 1000000.0)

# Function to make prediction
def predict_loan_approval():
    # Manually encode inputs
    education_encoded = 1 if education.lower() == 'yes' else 0
    self_employed_encoded = 1 if self_employed.lower() == 'yes' else 0
    
    # Prepare the feature array
    input_data = np.array([[education_encoded, self_employed_encoded, no_of_dependents,
                            income_annum, loan_amount, loan_term, cibil_score, assets]])

    # Standardize input data
    input_data_scaled = scaler.transform(input_data)

    # Get the Logistic Regression prediction
    logreg_pred = logreg_model.predict_proba(input_data_scaled)[:, 1]

    # Append the Logistic Regression prediction as an additional feature for XGBoost
    input_data_final = np.column_stack((input_data_scaled, logreg_pred))

    # Get the final loan approval prediction using XGBoost
    loan_approval_probability = xgb_model.predict_proba(input_data_final)[:, 1]

    # Adjusted threshold (e.g., 0.7 for more cautious approval)
    threshold = 0.7
    if loan_approval_probability[0] >= threshold:
        return "🎉 Congratulations, your loan is Approved! 💸"
    else:
        return "❌ Sorry, your loan is Denied. 💔"

# Button to trigger prediction
if st.button('Predict Loan Approval'):
    result = predict_loan_approval()
    st.write(result)
