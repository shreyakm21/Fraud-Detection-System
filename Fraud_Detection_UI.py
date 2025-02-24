

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
Format='%(asctime)s:%(name)s:%(levelname)s:%(message)s'
logging.basicConfig(filename="FraudDetection.log",level=logging.DEBUG, filemode='w',format=Format)
from ML_model import main1, main2


# Title
st.title("**Intelligent Fraud Detection system**")


with st.sidebar:
    # Title
    st.subheader("Enter Transaction Data for single Record")
    
    # Payment Details
    step = st.number_input("Enter Step#", value=0)
    #string1 = st.text_input("E")
    type = st.selectbox("Select payment type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    amount = st.number_input("Enter Payment amount", value=0.0)
    nameOrig = st.text_input("Enter Originators name")
    oldbalanceOrg = st.number_input("Enter Originators old balance", value=0.0)
    newbalanceOrig = st.number_input("Enter Originators new balance", value=0.0)
    oldbalanceDest = st.number_input("Enter Destinators old balance", value=0.0)
    newbalanceDest = st.number_input("Enter Destinators new balance", value=0.0)
    nameDest = st.text_input("Enter Destinators name")

    st.write("### OR Upload file for multiple records:")

    # File Upload
    uploaded_file = st.file_uploader("Upload an Excel File", type=["xlsx", "xls", "csv"])
    submit_button1 = st.button("Submit")
# Submit Button

if submit_button1:    
    st.write("### Entered Data:")
    
    sample_data = {
        'step': step,  # Time step
        #'type': 'TRANSFER',  # Transaction type
        'amount': amount,  # Transaction amount
        'type': type,  # Transaction type
        'nameOrig': nameOrig,  # Origin account
        'oldbalanceOrg': oldbalanceOrg,  # Origin balance before transaction
        'newbalanceOrig': newbalanceOrig,  # Placeholder for origin balance after transaction
        'nameDest': nameDest,  # Destination account
        'oldbalanceDest': oldbalanceDest,  # Destination balance before transaction
        'newbalanceDest': newbalanceDest,  # Placeholder for destination balance after transaction
        'isFraud': '',  # Fraud label (0 or 1)

    } 
    # Convert to DataFrame (single row)
    sample_data = pd.DataFrame([sample_data])
    
    # Transpose the DataFrame to show it horizontally
    st.write("### Transaction Details")


    # Process uploaded file
    if uploaded_file is not None:
        #df = pd.read_excel(uploaded_file)
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data File Preview:")
        st.dataframe(df)
        uploaded_file_name = uploaded_file.name
        sample_data = sample_data.head(0)
    else: 
        uploaded_file_name = ''
        st.dataframe(sample_data, use_container_width=True)

    @st.cache_data
    def expensive_function():
        st.write("Model Training completed!")
        #X_train, X_test, y_train, y_test, features, y_pred, model = main1( uploaded_file.name, amount, nameOrig,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest)
        X_train, X_test, y_train, y_test, features, y_pred, model = main1()
        return X_train, X_test, y_train, y_test, features, y_pred, model

    def plot_payment_vs_fraud(result_fraud):
        fraud_rates = result_fraud.groupby("type")["fraudPredict"].mean().sort_values(ascending=False)
    
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=fraud_rates.index, y=fraud_rates.values, ax=ax, palette="coolwarm")
        ax.set_title("Payment Type vs. Fraud Probability")
        ax.set_xlabel("Payment Type")
        ax.set_ylabel("Fraud Probability")
        
        st.pyplot(fig)


    def plot_fraud_distribution(result_fraud):
        """Function to plot Fraud vs. Non-Fraud Transactions."""
    
        # Ensure fraudPredict column is numeric
        result_fraud["fraudPredict"] = pd.to_numeric(result_fraud["fraudPredict"], errors="coerce")
    
        # Count occurrences of fraud (1) and non-fraud (0)
        fraud_counts = result_fraud["fraudPredict"].value_counts()
    
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=fraud_counts.index, y=fraud_counts.values, ax=ax, palette=["green", "red"])
    
        # Labels and title
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
        ax.set_title("Fraud vs. Non-Fraud Transactions")
        ax.set_xlabel("Transaction Type")
        ax.set_ylabel("Count")
    
        # Display the plot in Streamlit
        st.pyplot(fig)
         

    X_train, X_test, y_train, y_test, features, y_pred, model = expensive_function()    

    fraudPredict, result_fraud = main2(X_train, X_test, y_train, y_test, features, y_pred, model, uploaded_file_name, amount,type, nameOrig,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest)

    st.subheader(f"Fraud prediction: {fraudPredict}")
    if uploaded_file_name != '':
        st.write("### Uploaded File Fraud Prediction:")
        st.dataframe(result_fraud)
        st.subheader("Visualization: Payment type vs. Fraud Probability")
        plot_payment_vs_fraud(result_fraud)
        
    
        if 'result_fraud' in locals():
            # Assuming `result_fraud` contains fraud detection results
            st.subheader("Visualization: Fraud vs. Non-Fraud Transactions")
    
            plot_fraud_distribution(result_fraud)
            
        else:
            st.warning("No fraud detection results available to plot.")