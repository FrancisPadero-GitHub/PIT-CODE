from sklearn.preprocessing import LabelEncoder

import pandas as pd
import pickle
import os
# Get the directory of the current script (somehow you need this on vscode but not on jupyter lab)
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)


# Load the trained ensemble model
with open('ensemble_model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)

# Original categorical columns used during training
cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

# Create new input data
new_data3 = pd.DataFrame({
    'Gender': ['Male'],
    'Customer Type': ['Loyal Customer'],
    'Age': [68],
    'Type of Travel': ['Business'],
    'Class': ['Eco Plus'],
    'Flight Distance': [2800],
    'Inflight wifi service': [4],
    'Departure/Arrival time convenient': [5],
    'Ease of Online booking': [4],
    'Gate location': [3],
    'Food and drink': [4],
    'Online boarding': [5],
    'Seat comfort': [4],
    'Inflight entertainment': [4],
    'On-board service': [5],
    'Leg room service': [5],
    'Baggage handling': [5],
    'Checkin service': [5],
    'Inflight service': [5],
    'Cleanliness': [5],
    'Departure Delay in Minutes': [0],
})

# Encode the categorical columns to match training
new_data_encoded = pd.get_dummies(new_data3, columns=cat_cols, drop_first=True)

# Make sure all expected columns exist (from training)
# Load the original training features to get correct columns
df = pd.read_csv('APS.csv')
df.drop(columns=['id', 'Arrival Delay in Minutes'], axis=1, inplace=True)
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
X = df_encoded.drop(columns='satisfaction')
expected_columns = X.columns

# Align columns with training data
new_data_encoded = new_data_encoded.reindex(columns=expected_columns, fill_value=0)

# Make prediction using the ensemble model
prediction = ensemble_model.predict(new_data_encoded)

# Decode label if you still have access to LabelEncoder from training
# (Recreate it if needed using training data)
le = LabelEncoder()
le.fit(df['satisfaction'])  # Refit encoder with original labels
prediction_label = le.inverse_transform(prediction)

print(f"✅ Predicted Satisfaction: {prediction_label[0]}")
