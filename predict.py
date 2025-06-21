import pandas as pd
import joblib

# Load model and feature columns
model, feature_columns = joblib.load("disease_prediction_model.pkl")

# Map lowercase symptom names to actual feature names
symptom_lookup = {s.lower(): s for s in feature_columns}

def predict_disease(user_symptoms):
    """
    Predict disease from a list of user-entered symptoms.
    """
    # Initialize all features to 0
    input_data = {symptom: 0 for symptom in feature_columns}

    for symptom in user_symptoms:
        symptom = symptom.strip().lower()
        if symptom in symptom_lookup:
            feature_name = symptom_lookup[symptom]
            input_data[feature_name] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not recognized.")

    # Predict
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return prediction

if __name__ == "__main__":
    print("Disease Predictor")
    print("Enter your symptoms (comma-separated):")
    user_input = input("Symptoms: ")

    # Parse and predict
    user_symptoms = user_input.split(",")
    predicted_disease = predict_disease(user_symptoms)
    print(f"\nPredicted disease: {predicted_disease}")
