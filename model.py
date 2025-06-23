# Full pipeline including training, evaluation, and prediction enhancements

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

# Load datasets
df = pd.read_csv('dataset.csv')
df.fillna("None", inplace=True)
severity_df = pd.read_csv('Symptom-severity.csv')
precaution_df = pd.read_csv('symptom_precaution.csv')

# Normalize severity mapping
severity_df['Symptom'] = severity_df['Symptom'].str.strip().str.lower()
severity_map = dict(zip(severity_df['Symptom'], severity_df['weight']))

# Identify symptom columns
symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]

# Compute severity score
def compute_severity_score(row):
    total = 0
    for col in symptom_cols:
        symptom = row[col].strip().lower()
        if symptom != "none":
            total += severity_map.get(symptom, 0)
    return total

df['Severity_Score'] = df.apply(compute_severity_score, axis=1)

# One-hot encode symptoms
all_symptoms = pd.unique(df[symptom_cols].values.ravel())
all_symptoms = [s for s in all_symptoms if s != "None"]
symptom_df = pd.DataFrame(0, index=df.index, columns=all_symptoms)
for col in symptom_cols:
    dummies = pd.get_dummies(df[col])
    dummies = dummies.reindex(columns=all_symptoms, fill_value=0)
    symptom_df = symptom_df | dummies

# Final dataset
df_final = pd.concat([df["Disease"], symptom_df], axis=1)
df_final["Severity_Score"] = df["Severity_Score"]
X = df_final.drop(columns=["Disease"])
y = df_final["Disease"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Save feature names
feature_columns = X.columns.tolist()

# Add symptom noise to test set
def add_symptom_noise(X, noise_level=0.08):
    X_noisy = X.copy().astype(int)
    np.random.seed(42)
    n_cells = int(noise_level * X.shape[0] * (X.shape[1] - 1))  # Exclude severity
    row_indices = np.random.randint(0, X.shape[0], n_cells)
    col_indices = np.random.randint(0, X.shape[1] - 1, n_cells)
    for row, col in zip(row_indices, col_indices):
        X_noisy.iat[row, col] = 1 - X_noisy.iat[row, col]
    return X_noisy

X_test_noisy = add_symptom_noise(X_test)

# Add label noise to test labels
def add_label_noise(y, noise_level=0.08):
    y_noisy = y.copy()
    np.random.seed(42)
    unique_classes = y.unique()
    n_noisy = int(noise_level * len(y))
    noisy_indices = np.random.choice(y.index, size=n_noisy, replace=False)
    for idx in noisy_indices:
        current = y_noisy.loc[idx]
        choices = [cls for cls in unique_classes if cls != current]
        y_noisy.loc[idx] = np.random.choice(choices)
    return y_noisy

y_test_noisy = add_label_noise(y_test)

# Train model
base_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model = CalibratedClassifierCV(base_model, cv=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test_noisy)
accuracy = accuracy_score(y_test_noisy, y_pred)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

# Save model
joblib.dump((model, feature_columns), 'disease_prediction_model.pkl')

# Load precaution mapping
precaution_map = precaution_df.set_index("Disease")[["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]].to_dict("index")

# Create a prediction function
def predict_disease(symptoms: list):
    input_vector = pd.Series(0, index=feature_columns)
    input_vector["Severity_Score"] = sum([severity_map.get(s.lower(), 0) for s in symptoms])
    for s in symptoms:
        s = s.strip()
        if s in input_vector.index:
            input_vector[s] = 1
    input_df = pd.DataFrame([input_vector])
    prediction = model.predict(input_df)[0]
    precautions = precaution_map.get(prediction, {})
    return {
        "predicted_disease": prediction,
        "precautions": list(precautions.values())
    }

# Example prediction
example_symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]
example_result = predict_disease(example_symptoms)

# Output summary
{
    "accuracy_on_noisy_test": round(accuracy, 2),
    "cv_mean_accuracy": round(cv_scores.mean(), 2),
    "cv_std": round(cv_scores.std(), 2),
    "model_saved_as": "disease_prediction_model.pkl",
    "example_prediction": example_result
}
