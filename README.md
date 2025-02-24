# Fraud-Detection-System
A Machine Learning-powered fraud detection system built using Random Forest Classification to classify fraudulent transactions. The UI is developed with Streamlit, making it interactive and user-friendly.

## 📌 Features
✔ Streamlit-based UI for easy interaction  
✔ Random Forest Classifier for fraud detection  
✔ Trained on a structured dataset of transactions  
✔ Real-time predictions for new transaction data  

## 🛠 Tech Stack
- **Python 3.11**  
- **Machine Learning:** scikit-learn, pandas, numpy  
- **Web UI:** Streamlit  
- **Dataset:** Includes transaction details with fraud labels  

## 📂 Project Structure
│── fraud_model.py            # ML model (Random Forest Classifier)
│── streamlit_ui.py           # Streamlit-based UI
│── dataset.csv               # Training dataset
│── requirements.txt          # Required Python libraries
│── README.md                 # Project documentation

## How to Run the Project
- **Train the ML Model (If not pre-trained)**
python ML_model.py

- **Run the Streamlit UI**
streamlit run Fraud_Detection_UI.py

Dataset Details
The dataset (FraudTrainData.csv) consists of:

Transaction Amount
Merchant Details
User History
Fraudulent Label (0 = Legit, 1 = Fraudulent)
🧠 ML Model Details
✔ Algorithm: Random Forest Classifier
✔ Target Variable: Fraud Label (0/1)
✔ Performance Metrics: Accuracy, Precision, Recall

🎯 Future Improvements
Enhance model performance with hyperparameter tuning
Integrate deep learning techniques for better fraud detection
Deploy using AWS/GCP for scalability

