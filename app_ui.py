import streamlit as st
import requests

# API endpoint for Flask backend
API_URL = "http://127.0.0.1:5000/predict"

def get_prediction(features):
    response = requests.post(API_URL, json={'features': features})
    return response.json()

def main():
    st.title("✈️ Airline Passenger Satisfaction Prediction")
    st.markdown("**Legend for service ratings:** 1 = Bad, 5 = Best")

    # Create columns for a spread-out layout
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("🧑‍💼 Gender", ["Male", "Female"])
        customer_type = st.selectbox("👤 Customer Type", ["Loyal Customer", "Disloyal Customer"])
        age = st.number_input("🔢 Age", min_value=0, max_value=120, value=18)
        time_convenient = st.slider("⏰ Departure/Arrival time convenient", 1, 5, 3)
        booking_ease = st.slider("💻 Ease of Online booking", 1, 5, 3)

        
    with col2:
        travel_class = st.selectbox("🎟️ Class", ["Economy", "Economy Plus", "Business"])
        travel_type = st.selectbox("✈️ Type of Travel", ["Personal","Business"])
        flight_distance = st.number_input("📏 Flight Distance", min_value=0, value=999)
        food = st.slider("🍔 Food and drink", 1, 5, 3)
        gate_location = st.slider("🚪 Gate location", 1, 5, 3)

    # Create another row of columns for more inputs
    col3, col4 = st.columns(2)

    with col3:
        boarding = st.slider("📱 Online boarding", 1, 5, 3)
        comfort = st.slider("💺 Seat comfort", 1, 5, 3)
        entertainment = st.slider("🎬 Inflight entertainment", 1, 5, 3)
        onboard_service = st.slider("🧑‍✈️ On-board service", 1, 5, 3)
        legroom = st.slider("🦵 Leg room service", 1, 5, 3)

    with col4:
        baggage = st.slider("🧳 Baggage handling", 1, 5, 3)
        checkin = st.slider("🛂 Checkin service", 1, 5, 3)
        inflight_service = st.slider(" 🛎️ Inflight service", 1, 5, 3)
        cleanliness = st.slider("🧼 Cleanliness", 1, 5, 3)
        delay_minutes = st.number_input("⏳ Departure Delay in Minutes", min_value=0, value=5)



    features = {
        'Gender': gender,
        'Customer Type': customer_type,
        'Age': age,
        'Type of Travel': travel_type,
        'Class': travel_class,
        'Flight Distance': flight_distance,
        'Departure/Arrival time convenient': time_convenient,
        'Ease of Online booking': booking_ease,
        'Gate location': gate_location,
        'Food and drink': food,
        'Online boarding': boarding,
        'Seat comfort': comfort,
        'Inflight entertainment': entertainment,
        'On-board service': onboard_service,
        'Leg room service': legroom,
        'Baggage handling': baggage,
        'Checkin service': checkin,
        'Inflight service': inflight_service,
        'Cleanliness': cleanliness,
        'Departure Delay in Minutes': delay_minutes
    }

    if st.button("Predict Satisfaction"):
        prediction = get_prediction(features)

        # Ensure that the prediction is returned as a dictionary with 'prediction'
        if 'prediction' in prediction:
            pred_class = prediction['prediction']
            label_map = {
                0: "Neutral or Dissatisfied",
                1: "Satisfied"
            }
            label = label_map.get(pred_class, "Unknown")
            if label == "Satisfied":
                st.success(f"Predicted Satisfaction: **{label}** 😊")
            else:
                st.warning(f"Predicted Satisfaction: **{label}** 😞")
        else:
            st.error(f"Error: {prediction.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()