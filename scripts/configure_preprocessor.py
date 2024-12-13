import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load your training data
training_data = pd.read_csv("C:/Users/natha/PersonalisedCancerTreatmentProject/data/Final_Combined_Clinical_and_Biospecimen_Data_for_Lung_Cancer.csv")

# Define numerical and categorical columns
numerical_features = ['age_at_index', 'cigarettes_per_day', 'pack_years_smoked', 
                      'portion_weight', 'concentration', 'aliquot_volume', 'aliquot_quantity']
categorical_features = ['gender', 'race', 'ajcc_pathologic_stage', 'sample_type', 
                        'specimen_type', 'tissue_type', 'preservation_method', 
                        'treatment_type', 'treatment_therapy']

# Replace 'Not Reported' or other non-numeric values in numerical columns
for col in numerical_features:
    training_data[col] = pd.to_numeric(training_data[col], errors='coerce')  # Convert to NaN where necessary

# Fill missing values in numerical columns (optional, based on strategy)
training_data[numerical_features] = training_data[numerical_features].fillna(training_data[numerical_features].median())

# Create transformers
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Build preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessor on cleaned training data and save it
preprocessor.fit(training_data)
joblib.dump(preprocessor, "C:/Users/natha/PersonalisedCancerTreatmentProject/models/preprocessor.joblib")
