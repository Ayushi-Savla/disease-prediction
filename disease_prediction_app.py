import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set up the Streamlit page
st.title("Disease Prediction App")
st.write("Select your symptoms and click 'Predict' to find out the possible disease. Tip: Select at least 3 symptoms for better prediction confidence.")

# Load the dataset to get all possible symptoms
try:
    df = pd.read_csv('dataset.csv')
except FileNotFoundError:
    st.error("Dataset file 'dataset.csv' not found. Please ensure it is in the same folder.")
    st.stop()
df.fillna("None", inplace=True)

# Find all symptom columns
symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]

# Get all unique symptoms (excluding "None")
all_symptoms = pd.unique(df[symptom_cols].values.ravel())
all_symptoms = [s for s in all_symptoms if s != "None"]
all_symptoms.sort()

# Load the saved model and feature columns
try:
    model, feature_columns = joblib.load('disease_prediction_model.pkl')
except FileNotFoundError:
    st.error("Model file 'disease_prediction_model.pkl' not found. Please run the training script.")
    st.stop()

# Create a multi-select dropdown for symptoms
selected_symptoms = st.multiselect(
    "Choose your symptoms:",
    options=all_symptoms,
    help="Select all symptoms you are experiencing. At least 3 symptoms are recommended."
)

# Create a button to trigger prediction
if st.button("Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    elif len(selected_symptoms) < 3:
        st.warning("Selecting fewer than 3 symptoms may lead to low confidence. Please add more if possible.")
    else:
        # Create an input DataFrame with 0s
        input_data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
        
        # Set 1 for selected symptoms
        for symptom in selected_symptoms:
            if symptom in feature_columns:
                input_data[symptom] = 1
            else:
                st.warning(f"Symptom '{symptom}' not recognized by the model.")
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = probabilities[model.classes_.tolist().index(prediction)]
        
        # Get top 3 diseases and their probabilities
        prob_df = pd.DataFrame({
            'Disease': model.classes_,
            'Probability': probabilities
        }).sort_values(by='Probability', ascending=False).head(3)
        
        # Display results
        st.success(f"Predicted Disease: **{prediction}**")
        st.write(f"Confidence Score: **{confidence:.2f}** (how sure the model is)")
        
        st.subheader("Top 3 Possible Diseases:")
        for _, row in prob_df.iterrows():
            st.write(f"- {row['Disease']}: {row['Probability']:.2f}")
        
        st.info("Note: This model uses 2% noise for robustness. It is not a substitute for medical advice. Consult a doctor.")

# Add a footer
st.markdown("---")
st.write("Built with Streamlit. Model trained with 2% noise for robustness.")