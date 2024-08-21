import streamlit as st
#import numpy as np
#import pandas as pd
import joblib

import util
from util import Word2VecVectorizer, clean_data, vec_to_df


# Load the model
model_pipeline = joblib.load('best_model.joblib')

# Load the Vectorizer Preprocessor
vec_preprocessor = joblib.load('vec_preprocessor.joblib')

optimal = 0.688
balanced = 0.649
threshold = optimal #Set either optimal or balanced threshold

def make_prediction(data):
    # Predict probability
    predict_proba = model_pipeline.predict_proba(data)[0, 1]
    prediction = 1 if predict_proba >= threshold else 0
    
    # Adjust the display probability to reflect a 0.5 cutoff
    display_proba = (predict_proba - threshold + 0.5) if predict_proba >= threshold else (predict_proba / threshold) * 0.5
    
    return prediction, display_proba

# Streamlit interface
st.title('Loan Default Prediction App')

# Creating columns for organized layout
col1, col2, col3 = st.columns(3)

# Sidebar for inputs
st.sidebar.header("Input Loan Information")

# Section 1: Loan Details
st.sidebar.subheader("Loan Details")
with st.sidebar.container():
    loan_amnt = st.number_input('Requested Loan Amount', min_value=1000, max_value=1000000, value=15000)
    term = st.selectbox('Requested Term', options=['36 months', '60 months'])
    int_rate = st.slider('Loan Interest Rate (%)', 5.00, 40.0, 10.0)
    installment = st.number_input('Expected Installment per month', min_value=50, max_value=9000, value=300)
    grade = st.selectbox('Expected Loan Grade', options=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    sub_grade = st.selectbox('Expected Loan Subgrade', options=[f'{i}' for i in range(1, 6)])

# Section 2: Borrower Information
st.sidebar.subheader("Borrower Information")
with st.sidebar.container():
    emp_title = st.text_input('Borrower Employment Title')
    emp_length = st.selectbox('Length of Employment', options=['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
    home_ownership = st.selectbox('Type of Home Ownership', options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    annual_inc = st.number_input('Borrower Annual Income', min_value=2000, max_value=10000000, value=60000)
    verification_status = st.selectbox('Loan Application Verification Status', options=['Verified', 'Source Verified', 'Not Verified'])
    issue_d = st.text_input('Expected Loan Issue Date (mmm-yyyy)')

# Section 3: Credit Information
st.sidebar.subheader("Credit Information")
with st.sidebar.container():
    purpose = st.text_input('Purpose for Loan')
    title = st.text_input('Loan Title')
    dti = st.slider('Debt to Income Ratio (%)', 0.0, 100.0, 10.0)
    earliest_cr_line = st.text_input('Earliest Credit Line (mmm-yyyy), e.g. Jan-2000')
    open_acc = st.number_input('Open Accounts', min_value=0, max_value=50, value=10)
    pub_rec = st.number_input('Public Records', min_value=0, max_value=10, value=0)
    revol_bal = st.number_input('Revolving Balance', min_value=0, max_value=500000, value=10000)
    revol_util = st.slider('Revolving Utilization (%)', 0.0, 100.0, 30.0)
    total_acc = st.number_input('Total Accounts', min_value=0, max_value=100, value=20)
    initial_list_status = st.selectbox('Initial List Status', options=['w', 'f'])
    application_type = st.selectbox('Application Type', options=['Individual', 'Joint'])
    mort_acc = st.number_input('Number of Mortgage Accounts', min_value=0, max_value=10, value=1)
    pub_rec_bankruptcies = st.number_input('Public Record Bankruptcies', min_value=0, max_value=10, value=0)

# Section 4: Address Information
st.sidebar.subheader("Address Information")
with st.sidebar.container():
    address = st.text_input('Address with 5-digits Zip Code')
    state = st.text_input('State, e.g. "CA"')

# Collect inputs into a dictionary
input_data = {
        'loan_amnt': loan_amnt, #dtype: int (amount of loan applied for)
        'term': term, # dtype: str ('36 months' or '60 months')
        'int_rate': int_rate, # dtype: float, range: 5.31 to 30.99 (interest rate on the loan)
        'installment': installment, # dtype: float (monthly payment owed by the borrower)
        'grade': grade, # dtype: str, range: 'A', 'B', 'C', 'D', 'E', 'F', 'G' (LC assigned loan grade)
        'sub_grade': sub_grade, # dtype: str, range: 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5' (LC assigned loan subgrade)
        'emp_title': emp_title, # dtype: str (borrower's job title)
        'emp_length': emp_length, # dtype: str, range: '<1 years', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years' (employment length in years)
        'home_ownership': home_ownership, # dtype: str, range: 'RENT', 'OWN', 'MORTGAGE', 'OTHER' (home ownership status provided by the borrower during registration)
        'annual_inc': annual_inc, # dtype: float (the self-reported annual income provided by the borrower during registration)
        'verification_status': verification_status, # dtype: str, range: 'Verified', 'Source Verified', 'Not Verified' (indicates if income was verified by LC, not verified, or if the income source was verified)
        'issue_d': issue_d, # dtype: str, format: 'Mon-YYYY' (the month which the loan was funded)
        'purpose': purpose, # dtype: str, range: 'debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'small_business', 'car', 'moving', 'vacation', 'medical', 'other' (a category provided by the borrower for the loan request)
        'title': title, # dtype: str (the loan title provided by the borrower)
        'dti': dti, # dtype: float (a ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income)
        'earliest_cr_line': earliest_cr_line, # dtype: str, format: 'Mon-YYYY' (the month the borrower's earliest reported credit line was opened)
        'open_acc': open_acc, # dtype: int (the number of open credit lines in the borrower's credit file)
        'pub_rec': pub_rec, # dtype: int (number of derogatory public records)
        'revol_bal': revol_bal, # dtype: int (total credit revolving balance)
        'revol_util': revol_util, # dtype: float (revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit)
        'total_acc': total_acc, # dtype: int (the total number of credit lines currently in the borrower's credit file)
        'initial_list_status': initial_list_status, # dtype: str, range: 'w', 'f' (the initial listing status of the loan)
        'application_type': application_type, # dtype: str, range: 'Individual', 'Joint' (indicates whether the loan is an individual application or a joint application with two co-borrowers)
        'mort_acc': mort_acc, # dtype: float (number of mortgage accounts)
        'pub_rec_bankruptcies': pub_rec_bankruptcies, # dtype: int (number of public record bankruptcies)
        'address': address, # dtype: str (borrower's address including postal code)
        'state': state # dtype: str (borrower's state
        }


# Button to make prediction
if st.button('Predict Probability of Default'):
    data = clean_data(input_data)

    # Transform the test set with the Vectorizer Preprocessor
    data_vec = vec_preprocessor.transform(data) # Transform only
    df = util.vec_to_df(data, data_vec, vec_preprocessor)

    predictions, display_proba = make_prediction(df)
    st.subheader(f'Probability of Default: {display_proba * 100:.0f} %')

    if predictions == 1:
        st.subheader('This loan application has HIGH RISK of Default!') 
        st.write('REJECT the Loan Application.')
    else:
        st.subheader('This loan application has LOW RISK of Default')
        st.write('Proceed with the Loan Application.')


