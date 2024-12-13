import pandas as pd
import tensorflow as tf
import shap

# Load the processed data with feature names
processed_data_path = "C:/Users/natha/PersonalisedCancerTreatmentProject/data/processed_new_patient_data_for_shap_with_names.csv"
processed_data = pd.read_csv(processed_data_path)

# Convert processed data to NumPy array if needed for explainer initialization
processed_data_array = processed_data.to_numpy()

# Load the model
model_path = 'C:/Users/natha/PersonalisedCancerTreatmentProject/models/final_model'
model = tf.keras.models.load_model(model_path)

# Use DeepExplainer
explainer = shap.DeepExplainer(model, processed_data_array)

# Calculate SHAP values for the entire dataset
shap_values = explainer.shap_values(processed_data_array)

# Visualize results with a SHAP summary plot using feature names
shap.summary_plot(shap_values, processed_data, feature_names=processed_data.columns)
