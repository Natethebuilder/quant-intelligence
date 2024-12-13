import pandas as pd
import joblib
import tensorflow as tf
from sklearn.impute import SimpleImputer

# Load the correct preprocessor
preprocessor_path = 'C:/Users/natha/PersonalisedCancerTreatmentProject/models/preprocessor.joblib'
preprocessor = joblib.load(preprocessor_path)

# Load the model
model_path = 'C:/Users/natha/PersonalisedCancerTreatmentProject/models/final_model'
model = tf.keras.models.load_model(model_path)

# Load the new patient data
test_data_path = "C:/Users/natha/PersonalisedCancerTreatmentProject/data/final_cleaned_test_data_confirmed.csv"
new_patient_data = pd.read_csv(test_data_path)

# Define the expected columns
expected_columns = [
    'age_at_index', 'gender', 'race', 'cigarettes_per_day', 'pack_years_smoked',
    'ajcc_pathologic_stage', 'sample_type', 'specimen_type', 'tissue_type',
    'preservation_method', 'portion_weight', 'concentration', 'aliquot_volume',
    'aliquot_quantity', 'treatment_type', 'treatment_therapy',
    'is_ffpe', 'analyte_type', 'primary_diagnosis', 'tissue_or_organ_of_origin',
    'spectrophotometer_method'
]

# Keep only the expected columns and handle data types
new_patient_data = new_patient_data[expected_columns]
numeric_cols = ['age_at_index', 'cigarettes_per_day', 'pack_years_smoked', 
                'portion_weight', 'concentration', 'aliquot_volume', 'aliquot_quantity']
for col in numeric_cols:
    new_patient_data[col] = pd.to_numeric(new_patient_data[col], errors='coerce')

categorical_cols = ['gender', 'race', 'ajcc_pathologic_stage', 'sample_type', 
                    'specimen_type', 'tissue_type', 'preservation_method', 
                    'treatment_type', 'treatment_therapy', 'is_ffpe', 'analyte_type', 
                    'primary_diagnosis', 'tissue_or_organ_of_origin', 'spectrophotometer_method']
for col in categorical_cols:
    new_patient_data[col] = new_patient_data[col].astype(str)

# Impute missing values
numeric_imputer = SimpleImputer(strategy="mean")
categorical_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
new_patient_data[numeric_cols] = numeric_imputer.fit_transform(new_patient_data[numeric_cols])
new_patient_data[categorical_cols] = categorical_imputer.fit_transform(new_patient_data[categorical_cols])

# Transform and predict
try:
    new_patient_processed = preprocessor.transform(new_patient_data)

    # Manually create feature names
    numeric_features = numeric_cols  # Numeric features stay the same
    categorical_features = []  # Initialize an empty list for categorical feature names
    
    # Extract categories for each categorical column and generate feature names
    encoder = preprocessor.named_transformers_['cat']  # Assuming 'cat' is the name in ColumnTransformer
    for col_name, categories in zip(categorical_cols, encoder.categories_):
        categorical_features.extend([f"{col_name}_{category}" for category in categories])

    # Combine all feature names
    all_feature_names = numeric_features + categorical_features

    # Save processed data with these feature names
    processed_data_df = pd.DataFrame(new_patient_processed, columns=all_feature_names)
    processed_data_df.to_csv("C:/Users/natha/PersonalisedCancerTreatmentProject/data/processed_new_patient_data_for_shap_with_names.csv", index=False)
    print("Processed data with feature names saved for SHAP analysis.")

    # Verify the processed data shape matches the model's input shape
    if new_patient_processed.shape[1] != model.input_shape[1]:
        raise ValueError(f"Expected input shape: {model.input_shape[1]}, but got: {new_patient_processed.shape[1]}")

    # Make predictions
    predictions = model.predict(new_patient_processed)
    print("Predicted outcomes:", predictions)
except Exception as e:
    print("Error during transformation or prediction:", e)

# Save predictions to CSV
predictions_df = pd.DataFrame(predictions, columns=["PredictedOutcome"])
predictions_df.to_csv("C:/Users/natha/PersonalisedCancerTreatmentProject/data/predictions.csv", index=False)
print("Predictions saved to CSV.")
