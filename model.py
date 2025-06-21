import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('dataset.csv')
df.fillna("None", inplace=True)

# Identify symptom columns
symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]

# Get all unique symptoms (excluding "None")
all_symptoms = pd.unique(df[symptom_cols].values.ravel())
all_symptoms = [s for s in all_symptoms if s != "None"]

# Build one-hot encoded symptom feature DataFrame
symptom_df = pd.DataFrame(0, index=df.index, columns=all_symptoms)

for col in symptom_cols:
    dummies = pd.get_dummies(df[col])
    dummies = dummies.reindex(columns=all_symptoms, fill_value=0)
    symptom_df = symptom_df | dummies

# Combine symptoms and disease into final DataFrame
df_final = pd.concat([df["Disease"], symptom_df], axis=1)

# Split features and target
X = df_final.drop(columns=["Disease"])
y = df_final["Disease"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save feature names
feature_columns = X.columns.tolist()

# Add symptom noise to X_test (2%)
def add_symptom_noise(X, noise_level=0.02):
    X_noisy = X.copy().astype(int)
    np.random.seed(42)
    n_cells = int(noise_level * X.shape[0] * X.shape[1])
    row_indices = np.random.randint(0, X.shape[0], n_cells)
    col_indices = np.random.randint(0, X.shape[1], n_cells)
    
    for row, col in zip(row_indices, col_indices):
        X_noisy.iat[row, col] = 1 - X_noisy.iat[row, col]  # Flip bit

    return X_noisy

X_test_noisy = add_symptom_noise(X_test, noise_level=0.02)

# Add label noise to y_test (2%)
def add_label_noise(y, noise_level=0.02):
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

y_test_noisy = add_label_noise(y_test, noise_level=0.02)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate on noisy test set
y_pred = model.predict(X_test_noisy)
accuracy = accuracy_score(y_test_noisy, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save model
joblib.dump((model, feature_columns), 'disease_prediction_model.pkl')
print("Model saved as disease_prediction_model.pkl")
