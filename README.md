# 🩺 Disease Prediction App
This is an AI-powered web application built with Streamlit that predicts the most likely disease based on user-selected symptoms. It also provides severity interpretation and recommended precautions based on the prediction.

# 🚀 Features
✅ Select symptoms from a list (auto-generated from real data)

🧠 Predicts the most likely disease using a trained Random Forest model

🛡️ Offers personalized precautionary advice

🔍 Interprets severity based on symptom weights

📋 Displays top 3 likely diseases without relying on confidence scores

⚠️ Clean and user-friendly UI

# 📁 Files in This Project
File	Description
app.py	Main Streamlit application script
dataset.csv	Main symptom-disease dataset
Symptom-severity.csv	Symptom severity scores
symptom_precaution.csv	Precautions mapped to diseases
disease_prediction_model.pkl	Trained machine learning model
README.md	Project overview and setup guide

# 🧠 Model Info
The model is a Random Forest Classifier trained on a dataset of symptoms and disease labels. It includes:

One-hot encoded symptoms

A computed severity score

Noise handling for improved generalization

# 🛠️ How to Run the App
1. Clone the repository:
bash
Copy
Edit
git clone https://github.com/your-username/disease-prediction-app.git
cd disease-prediction-app
2. Install dependencies:
Make sure you have Python 3.7+ installed.

bash
Copy
Edit
pip install -r requirements.txt
If you don't have a requirements.txt file, here's a basic list:

bash
Copy
Edit
pip install streamlit pandas numpy scikit-learn joblib
3. Run the app:
bash
Copy
Edit
streamlit run app.py
# 📊 How It Works
You select symptoms from a dropdown.

The app generates a binary feature vector for those symptoms.

It sums up the severity of selected symptoms.

The trained model predicts the disease.

The app interprets:

Disease name

Severity level (mild/moderate/severe)

Precautions

Top 3 likely diseases

# ⚠️ Disclaimer
This app is intended for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a doctor for medical concerns.
