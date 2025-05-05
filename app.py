from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

# Load trained model and preprocessing artifacts
try:
    model = joblib.load("random_forest_model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: random_forest_model.pkl not found.")
    model = None

try:
    model_columns = joblib.load("model_columns.pkl")
    print("Model columns loaded successfully.")
except FileNotFoundError:
    print("Error: model_columns.pkl not found.")
    model_columns = None

try:
    y_encoder = joblib.load("label_encoder.pkl")
    print("Label encoder loaded successfully.")
except FileNotFoundError:
    print("Warning: label_encoder.pkl not found.")
    y_encoder = None

try:
    scaler = joblib.load("scaler.pkl")
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print("Warning: scaler.pkl not found. Skipping normalization.")
    scaler = None

categorical_columns_to_convert = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Airline Satisfaction Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or model_columns is None:
        return jsonify({'error': 'Model or column data not loaded'}), 500

    try:
        data = request.get_json(force=True)
        features = data['features']
        input_df = pd.DataFrame([features])

        # Convert columns to category
        input_df[categorical_columns_to_convert] = input_df[categorical_columns_to_convert].astype('category')

        # Identify column types
        numerical_columns = [c for c in input_df.columns if input_df[c].dtype.name != 'category']
        categorical_cols = [c for c in input_df.columns if input_df[c].dtype.name == 'category']
        input_df_describe = input_df.describe(include=['category'])

        current_binary_cols = [c for c in categorical_cols if input_df_describe[c]['unique'] == 2]
        current_nonbinary_cols = [c for c in categorical_cols if input_df_describe[c]['unique'] > 2]

        # Binarize binary cols
        for col in current_binary_cols:
            unique_vals = input_df[col].unique()
            if len(unique_vals) == 2:
                input_df[col] = input_df[col].map({unique_vals[0]: 0, unique_vals[1]: 1})

        # Create DataFrames for concatenation
        concat_list = []

        # Normalize numerical columns
        if numerical_columns:
            input_numerical = input_df[numerical_columns]
            if scaler:
                input_numerical = pd.DataFrame(
                    scaler.transform(input_numerical),
                    columns=numerical_columns
                )
            concat_list.append(input_numerical)

        # One-hot encode non-binary
        if current_nonbinary_cols:
            input_nonbinary = pd.get_dummies(input_df[current_nonbinary_cols])
            concat_list.append(input_nonbinary)

        # Binarized binary
        if current_binary_cols:
            input_binary = input_df[current_binary_cols]
            concat_list.append(input_binary)

        if not concat_list:
            return jsonify({'error': 'No valid input features.'}), 400

        X_input = pd.concat(concat_list, axis=1)

        # Reindex to match training columns
        X_input = X_input.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(X_input)

        if y_encoder:
            original_prediction = y_encoder.inverse_transform(prediction)[0]
            return jsonify({'prediction': int(prediction[0]), 'label': str(original_prediction)})
        else:
            return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'trace': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
