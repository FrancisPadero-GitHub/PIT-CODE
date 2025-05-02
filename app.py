from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load your trained model and preprocessor (if you saved one)
with open('ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define expected features and order (MUST match model training input!)
FEATURE_ORDER = [
    'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
    'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
    'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes'
]

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Airline Satisfaction Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        raw_input = data['features']

        # Convert to DataFrame
        input_df = pd.DataFrame([raw_input])

        # Manually one-hot encode using same logic as training
        cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

        # Align with training columns
        # Load original X_train column order used during training
        with open('model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)

        # Ensure all expected columns are present
        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        input_encoded = input_encoded[model_columns]  # Reorder columns

        # Predict
        prediction = model.predict(input_encoded)

        # Map to human-readable labels
        label_map = {
            0: "Neutral or Dissatisfied",
            1: "Satisfied"
        }
        label = label_map.get(int(prediction[0]), "Unknown")
        
        return jsonify({
            'prediction': int(prediction[0]),
            'label': label
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'trace': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
