
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.basicConfig(
    filename='Fraud_Detection.log',  # Specify a log file
    level=logging.ERROR,  # Set the log level to ERROR or higher
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Load dataset
def load_dataset():
    try:        
        #dataset = pd.read_csv('synthetic_fraud_data.csv')
        #dataset = pd.read_csv('Synthetic_Financial_datasets_log.csv')
        dataset = pd.read_csv('FraudTrainData.csv')
        # Define features and target variable
        features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        X = dataset[features]
        y = dataset['isFraud']
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, features
    except Exception as e:
        logging.error(f"An exception occurred: {str(e)}")
        

def train_Model(X_train, X_test, y_train, y_test):
    try:    
        # Train a RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
    
        # Predict on test data
        y_pred = model.predict(X_test)
        
        return y_pred, model
    except Exception as e:
        logging.error(f"An exception occurred: {str(e)}")        

def evaluate_Model(y_test, y_pred):
    try:    
        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        # Confusion Matrix
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    except Exception as e:
        logging.error(f"An exception occurred: {str(e)}")   

         
def testing_Model_record(y_pred, model, amount,type, nameOrig,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest):
    try:      
        #sample_data.to_csv('sample_data.csv', index=False)
        
        # Load dataset

        sample_data = {
            'step': 1,  # Time step
            'amount': amount,  # Transaction amount
            'type': type,  # Transaction type
            'nameOrig': nameOrig,  # Origin account
            'oldbalanceOrg': oldbalanceOrg,  # Origin balance before transaction
            'newbalanceOrig': newbalanceOrig,  # Placeholder for origin balance after transaction
            'nameDest': '',  # Destination account
            'oldbalanceDest': oldbalanceDest,  # Destination balance before transaction
            'newbalanceDest': newbalanceDest,  # Placeholder for destination balance after transaction
            'isFraud': '',  # Fraud label (0 or 1)
            #'isFlaggedFraud': ''  # Flagged by system (0 or 1)
        }        
        sample_data = pd.DataFrame([sample_data])
        print("sample_data recordL", sample_data)
        # Predict fraud status for sample data
        sample_predictions = model.predict(sample_data.drop(columns=['isFraud','nameDest','nameOrig','type']))
        print("Sample Predictions:", sample_predictions)
        predictedFraud = pd.DataFrame(sample_predictions, columns=['fraudPredict'])
        result_fraud = predictedFraud
        #result_fraud = pd.concat([sample_data, predictedFraud], axis=1).reset_index(drop=True)
        #print("result_fraud :", result_fraud)
        if sample_predictions[0] == 1:
            fraudPredict = "This record looks fraud"
        else:
            fraudPredict = "This record does not looks fraud"
        print("Fraud Predict : ", fraudPredict)
        return fraudPredict, result_fraud
    except Exception as e:
        logging.error(f"An exception occurred: {str(e)}")   

    
def testing_Model_File(y_pred, model, uploaded_file):
    try:      
        #sample_data.to_csv('sample_data.csv', index=False)
        
        # Load dataset

        uploaded_data = pd.read_csv(uploaded_file)

        # Predict fraud status for sample data
        sample_predictions = model.predict(uploaded_data.drop(columns=['isFraud','nameDest','nameOrig','type']))
        print("Sample Predictions:", sample_predictions)
        predictedFraud = pd.DataFrame(sample_predictions, columns=['fraudPredict'])
        result_fraud = pd.concat([uploaded_data, predictedFraud], axis=1).reset_index(drop=True)
        result_file = uploaded_file[:-4]
        result_file = result_file + "_result.csv"
        #result_fraud.to_csv('result_fraud.csv', index=False)
        result_fraud.to_csv(result_file, index=False)
        fraudPredict = f"Fraud predict file ' {result_file} ' is created, please check.."

        return fraudPredict, result_fraud
    except Exception as e:
        logging.error(f"An exception occurred: {str(e)}")   

def main1():
    X_train, X_test, y_train, y_test, features = load_dataset()
    y_pred, model = train_Model(X_train, X_test, y_train, y_test)
    evaluate_Model(y_test, y_pred)
    #testing_Model(y_pred, model, test_file)
    #train_Model_Flag = 1
    return  X_train, X_test, y_train, y_test, features, y_pred, model

def main2(X_train, X_test, y_train, y_test, features, y_pred, model, uploaded_file, amount,type, nameOrig,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest):
    #print(f"file name: {train_Model_Flag}")
    print(f"uploaded_file: {uploaded_file}")
    print(f"amount: {amount}")
    print(f"type: {type}")
    print(f"features: {features}")
    print(f"nameOrig: {nameOrig}")
    print(f"oldbalanceOrg: {oldbalanceOrg}")

    fraudPredict = ''
    if uploaded_file != '': 
        print("uploaded_file :", 'FileYes')
        #sample_data = get_test_data(uploaded_file)
        fraudPredict, result_fraud = testing_Model_File(y_pred, model, uploaded_file)
    else:
        print("uploaded_file :", 'FileNo')
        fraudPredict, result_fraud = testing_Model_record(y_pred, model, amount,type, nameOrig,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest)
    return fraudPredict, result_fraud
    
# MAIN MODULE
# __name__
if __name__=="__main__":
    main1()
    if len(sys.argv) == 1:
        X_train, X_test, y_train, y_test, features, y_pred, model = main1()
    else:
        fraudPredict, result_fraud = main2(sys.argv[1], sys.argv[2], sys.argv[3],sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10],sys.argv[11], sys.argv[12], sys.argv[13], sys.argv[14], sys.argv[15], sys.argv[16])


