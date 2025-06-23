import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set up the Streamlit page
st.title("🩺 Disease Prediction App")
st.write("Select your symptoms and click 'Predict' to find out the possible disease. Tip: Select at least 5 symptoms for better prediction confidence.")

# Load the dataset to get all possible symptoms
try:
    df = pd.read_csv('dataset.csv')
except FileNotFoundError:
    st.error("Dataset file 'dataset.csv' not found.")
    st.stop()

df.fillna("None", inplace=True)
symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]
all_symptoms = pd.unique(df[symptom_cols].values.ravel())
all_symptoms = [s for s in all_symptoms if s != "None"]
all_symptoms.sort()

# Load the symptom severity map
try:
    severity_df = pd.read_csv('Symptom-severity.csv')
    severity_df['Symptom'] = severity_df['Symptom'].str.strip().str.lower()
    severity_map = dict(zip(severity_df['Symptom'], severity_df['weight']))
except FileNotFoundError:
    st.error("File 'Symptom-severity.csv' not found.")
    st.stop()

# Load the precaution map
try:
    precaution_df = pd.read_csv('symptom_precaution.csv')
    precaution_map = precaution_df.set_index("Disease")[["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]].to_dict("index")
except FileNotFoundError:
    st.error("File 'symptom_precaution (1).csv' not found.")
    st.stop()

# Load the saved model and feature columns
try:
    model, feature_columns = joblib.load('disease_prediction_model.pkl')
except FileNotFoundError:
    st.error("Model file 'disease_prediction_model.pkl' not found.")
    st.stop()

# Create a multi-select dropdown for symptoms
selected_symptoms = st.multiselect(
    "Choose your symptoms:",
    options=all_symptoms,
    help="Select all symptoms you are experiencing. At least 5 symptoms are recommended."
)

# Predict button
if st.button("Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    elif len(selected_symptoms) < 3:
        st.warning("Selecting fewer than 5 symptoms may lead to low confidence.")
    else:
        # Create input vector
        input_data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
        
        severity_score = 0
        for symptom in selected_symptoms:
            symptom_clean = symptom.strip()
            if symptom_clean in feature_columns:
                input_data[symptom_clean] = 1
            if symptom_clean.lower() in severity_map:
                severity_score += severity_map[symptom_clean.lower()]
        
        # Set severity score
        if "Severity_Score" in input_data.columns:
            input_data["Severity_Score"] = severity_score

        # Predict
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = probabilities[model.classes_.tolist().index(prediction)]

        # Top 3 predictions
        prob_df = pd.DataFrame({
            'Disease': model.classes_,
            'Probability': probabilities
        }).sort_values(by='Probability', ascending=False).head(3)

        # Display results
        st.success(f"**Predicted Disease:** {prediction}")
        st.write(f"🔍 **Model Confidence Score:** `{confidence:.2f}`")
        st.write(f"🔥 **Your Severity Score:** `{severity_score}`")

        # Interpret severity
        if severity_score >= 18:
            st.error("⚠️ High severity detected. Please consider seeking medical attention.")
        elif severity_score >= 10:
            st.warning("⚠️ Moderate severity. Monitor symptoms closely.")
        else:
            st.info("✅ Mild symptoms detected. Stay hydrated and rest.")

        # Show precautions
        st.subheader("🛡️ Recommended Precautions:")
        precautions = precaution_map.get(prediction)
        if precautions:
            for p in precautions.values():
                if isinstance(p, str) and p.strip():
                    st.write(f"- {p}")
        else:
            st.write("No precautions found for this disease.")

        # Show top 3 diseases
        st.subheader("📋 Top 3 Possible Diseases:")
        for _, row in prob_df.iterrows():
            st.write(f"- {row['Disease']}: `{row['Probability']:.2f}`")

        st.info("Note: This is an AI-powered tool, not a replacement for professional medical advice.")

# Footer
st.markdown("---")
st.caption("🧠 Model trained using symptom severity and robust noise handling.")
