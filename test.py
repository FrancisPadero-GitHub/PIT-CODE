import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model and encoder
try:
    model = joblib.load('random_forest_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('standard_scaler.pkl')
    model_columns = joblib.load('model_columns.pkl') # Load the list of columns the model was trained on
    print("Model, Label Encoder, Scaler, and Model Columns loaded successfully!")
except FileNotFoundError as e:
    st.error(f"Error: Make sure the following files are in the same directory as this app: 'random_forest_model.pkl', 'label_encoder.pkl', 'standard_scaler.pkl', 'model_columns.pkl'.\n{e}")
    st.stop()

def preprocess_data(df):
    categorical_columns_to_convert = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    df[categorical_columns_to_convert] = df[categorical_columns_to_convert].astype('category')

    df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(axis=0), inplace=True)

    numerical_cols = [c for c in df.columns if df[c].dtype.name != 'category']
    categorical_cols = [c for c in df.columns if df[c].dtype.name == 'category']
    df_describe = df.describe(include=['category'])

    binary_cols = [c for c in categorical_cols if df_describe[c]['unique'] <= 2]
    nonbinary_cols = [c for c in categorical_cols if df_describe[c]['unique'] > 2]

    for col in binary_cols:
        unique_vals = df[col].unique()
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            df[col] = df[col].map(mapping)
        elif len(unique_vals) > 2:
            value_counts = df[col].value_counts()
            if len(value_counts) == 2:
                mapping = {value_counts.index[0]: 0, value_counts.index[1]: 1}
                df[col] = df[col].map(mapping)
            else:
                print(f"Warning: Binary column '{col}' has {len(unique_vals)} unique values and cannot be reliably binarized.")
        else:
            print(f"Warning: Binary column '{col}' has fewer than 2 unique values and cannot be binarized.")
            df[col] = np.nan # Or handle as needed

    df_nonbinary = pd.get_dummies(df[nonbinary_cols])

    df_numerical = df[numerical_cols]
    df_numerical_scaled = pd.DataFrame(scaler.transform(df_numerical), columns=numerical_cols)

    processed_df = pd.concat([df_numerical_scaled, df_nonbinary, df[binary_cols]], axis=1)

    # Ensure processed data has the same columns as the training data
    missing_cols = set(model_columns) - set(processed_df.columns)
    for c in missing_cols:
        processed_df[c] = 0
    processed_df = processed_df[model_columns]

    return processed_df.values # Return as a NumPy array

def main():
    st.title("Airline Passenger Satisfaction Prediction")
    st.subheader("Upload a CSV file to get satisfaction predictions directly.")

    uploaded_file = st.file_uploader("Upload your CSV file (in the format of dummy_test_data.csv)", type=["csv"])

    if uploaded_file is not None:
        try:
            user_data_df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data:")
            st.dataframe(user_data_df)

            if st.button("Predict Satisfaction"):
                processed_array = preprocess_data(user_data_df.copy())
                predictions = model.predict(processed_array)
                original_predictions = label_encoder.inverse_transform(predictions)

                st.subheader("Predictions:")
                results_df = pd.DataFrame({'Prediction': original_predictions})
                st.dataframe(results_df)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()