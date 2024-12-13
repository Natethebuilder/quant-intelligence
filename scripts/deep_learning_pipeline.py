
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf

# Load the dataset
data_path = "C:/Users/natha/PersonalisedCancerTreatmentProject/data/Final_Combined_Clinical_and_Biospecimen_Data_for_Lung_Cancer.csv"
combined_df = pd.read_csv(data_path)

# Replace 'Not Reported' with NaN and convert necessary columns to numeric
cols_to_convert = [
    'age_at_index', 'cigarettes_per_day', 'pack_years_smoked', 
    'portion_weight', 'concentration', 'aliquot_volume', 'aliquot_quantity'
]
for col in cols_to_convert:
    combined_df[col] = pd.to_numeric(combined_df[col].replace('Not Reported', None), errors='coerce')

# Check for NaN values after conversion
print("Checking for NaN values after conversion:")
print(combined_df.isnull().sum())

# Fill NaN values for numerical columns with the mean
for col in cols_to_convert:
    combined_df[col].fillna(combined_df[col].mean(), inplace=True)

# Define the target and features
combined_df['vital_status'] = combined_df['vital_status'].apply(lambda x: 1 if x == 'Alive' else 0)
y = combined_df['vital_status']

# Selecting feature columns, including new features
feature_columns = [
    'age_at_index', 'gender', 'race', 'cigarettes_per_day', 'pack_years_smoked',
    'ajcc_pathologic_stage', 'sample_type', 'specimen_type', 'tissue_type',
    'preservation_method', 'portion_weight', 'concentration', 'aliquot_volume',
    'aliquot_quantity', 'treatment_type', 'treatment_therapy',
    'is_ffpe', 'analyte_type', 'primary_diagnosis', 'tissue_or_organ_of_origin',
    'spectrophotometer_method'  # Include new features here
]
X = combined_df[feature_columns]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
    ]
)

# Transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save the preprocessor
joblib.dump(preprocessor, 'models/preprocessor.joblib')

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_processed, y_train, epochs=20, validation_data=(X_test_processed, y_test), verbose=2)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy}")

# Save the model
model.save('models/final_model')
