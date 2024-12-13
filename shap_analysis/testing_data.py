import pandas as pd
import joblib
import tensorflow as tf
import shap

# Load the correct preprocessor
preprocessor_path = 'C:/Users/natha/PersonalisedCancerTreatmentProject/models/preprocessor.joblib'
preprocessor = joblib.load(preprocessor_path)

# Load the model
model_path = 'C:/Users/natha/PersonalisedCancerTreatmentProject/models/final_model'
model = tf.keras.models.load_model(model_path)

# Load the new patient data
test_data_path = "C:/Users/natha/PersonalisedCancerTreatmentProject/data/test_data.csv"
new_patient_data = pd.read_csv(test_data_path)

# Print the original shape of new patient data
print("Original shape of new patient data:", new_patient_data.shape)

# Define the expected columns including the new ones
expected_columns = [
    'age_at_index', 'gender', 'race', 'cigarettes_per_day', 'pack_years_smoked',
    'ajcc_pathologic_stage', 'sample_type', 'specimen_type', 'tissue_type',
    'preservation_method', 'portion_weight', 'concentration', 'aliquot_volume',
    'aliquot_quantity', 'treatment_type', 'treatment_therapy',
    'is_ffpe', 'analyte_type', 'primary_diagnosis', 'tissue_or_organ_of_origin',
    'spectrophotometer_method'  # Include new features here
]

# Check for any missing or extra columns
missing_cols = set(expected_columns) - set(new_patient_data.columns)
extra_cols = set(new_patient_data.columns) - set(expected_columns)

if missing_cols:
    print("Missing columns in the new data:", missing_cols)

if extra_cols:
    print("Extra columns in the new data:", extra_cols)

# Keep only the expected columns
new_patient_data = new_patient_data[expected_columns]

# Print the shape after filtering columns
print("Shape after filtering columns:", new_patient_data.shape)

# Print the data types of the filtered data
print("Data types after filtering:")
print(new_patient_data.dtypes)

# Convert data types to ensure compatibility
for col in ['age_at_index', 'cigarettes_per_day', 'pack_years_smoked', 
            'portion_weight', 'concentration', 'aliquot_volume', 'aliquot_quantity']:
    new_patient_data[col] = new_patient_data[col].astype(float)

# Ensure categorical features are strings
for col in ['gender', 'race', 'ajcc_pathologic_stage', 'sample_type', 
            'specimen_type', 'tissue_type', 'preservation_method', 
            'treatment_type', 'treatment_therapy', 'is_ffpe', 'analyte_type', 
            'primary_diagnosis', 'tissue_or_organ_of_origin', 'spectrophotometer_method']:
    new_patient_data[col] = new_patient_data[col].astype(str)

# Check for NaN values
if new_patient_data.isnull().values.any():
    print("Warning: NaN values found in new patient data. Filling NaN with 'Not Reported'.")
    new_patient_data.fillna('Not Reported', inplace=True)

# Verify no NaN values are left
if new_patient_data.isnull().values.any():
    print("Error: NaN values are still present in the data!")
else:
    print("No NaN values detected in cleaned data.")

# Print the cleaned data
print("Cleaned new patient data:")
print(new_patient_data)

# Inspect the preprocessor's transformers without causing AttributeError
print("Preprocessor transformers:")
for name, trans, cols in preprocessor.transformers_:
    print(f"Transformer: {name}, Type: {type(trans).__name__}, Columns: {cols}")

# Preprocess the new patient data
try:
    # Transform the new patient data
    new_patient_processed = preprocessor.transform(new_patient_data)

    # Print the shape of the processed data
    print("Transformed shape of new patient data:", new_patient_processed.shape)

    # Check if the processed data shape matches the model's input shape
    if new_patient_processed.shape[1] != model.input_shape[1]:
        raise ValueError(f"Expected input shape: {model.input_shape[1]}, but got: {new_patient_processed.shape[1]}")

    # Make predictions
    predictions = model.predict(new_patient_processed)
    print("Predicted outcomes:", predictions)

    # SHAP Analysis
    # Load the background data for SHAP (use your original dataset for background)
    background_data_path = "C:/Users/natha/PersonalisedCancerTreatmentProject/data/Final_Combined_Clinical_and_Biospecimen_Data_for_Lung_Cancer.csv"
    background_data = pd.read_csv(background_data_path)

    # Preprocess background data similarly
    background_processed = preprocessor.transform(background_data[expected_columns])

    # Initialize SHAP Explainer
    explainer = shap.KernelExplainer(model.predict, background_processed)

    # Get SHAP values for the new patient data
    shap_values = explainer.shap_values(new_patient_processed)

    # Visualize the SHAP values for the new patient
    shap.initjs()  # Initializes the JavaScript visualization in Jupyter Notebook
    shap.force_plot(explainer.expected_value, shap_values, new_patient_processed)

except Exception as e:
    print("Error during transformation or prediction:", e)
