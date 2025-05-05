import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# API endpoint for Flask backend
API_URL = "http://127.0.0.1:5000/predict"

def get_prediction(features):
    response = requests.post(API_URL, json={'features': features})
    return response.json()

def prediction_page():
    st.title("✈️ Airline Passenger Satisfaction Prediction")
    st.markdown("**Legend for service ratings:** 1 = Bad, 5 = Best")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("🧑‍💼 Gender", ["Male", "Female"])
        customer_type = st.selectbox("👤 Customer Type", ["Loyal Customer", "disloyal Customer"])
        age = st.number_input("🔢 Age", min_value=0, max_value=120, value=18)
        time_convenient = st.slider("⏰ Departure/Arrival time convenient", 1, 5, 3)
        booking_ease = st.slider("💻 Ease of Online booking", 1, 5, 3)

    with col2:
        travel_class = st.selectbox("🎟️ Class", ["Eco", "Eco Plus", "Business"])
        travel_type = st.selectbox("✈️ Type of Travel", ["Personal Travel", "Business Travel"])
        flight_distance = st.number_input("📏 Flight Distance", min_value=0, value=999)
        food = st.slider("🍔 Food and drink", 1, 5, 3)
        gate_location = st.slider("🚪 Gate location", 1, 5, 3)

    col3, col4 = st.columns(2)
    with col3:
        boarding = st.slider("📱 Online boarding", 1, 5, 3)
        comfort = st.slider("💺 Seat comfort", 1, 5, 3)
        entertainment = st.slider("🎬 Inflight entertainment", 1, 5, 3)
        onboard_service = st.slider("🧑‍✈️ On-board service", 1, 5, 3)
        legroom = st.slider("🦵 Leg room service", 1, 5, 3)
        wifi = st.slider("📶 Inflight wifi service", 1, 5, 3)


    with col4:
        baggage = st.slider("🧳 Baggage handling", 1, 5, 3)
        checkin = st.slider("🛂 Checkin service", 1, 5, 3)
        inflight_service = st.slider(" 🛎️ Inflight service", 1, 5, 3)
        cleanliness = st.slider("🧼 Cleanliness", 1, 5, 3)
        delay_minutes = st.number_input("⏳ Departure Delay in Minutes", min_value=0, value=5)
        arrival_delay_minutes = st.number_input("⏳ Arrival Delay in Minutes", min_value=0, value=5)

    features = {
        'Gender': gender,
        'Customer Type': customer_type,
        'Age': age,
        'Type of Travel': travel_type,
        'Class': travel_class,
        'Flight Distance': flight_distance,
        'Inflight wifi service': wifi,
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
        'Departure Delay in Minutes': delay_minutes,
        'Arrival Delay in Minutes': arrival_delay_minutes,


    }

    if st.button("Predict Satisfaction"):
        prediction = get_prediction(features)
        if 'prediction' in prediction:
            pred_class = prediction['prediction']
            label = {0: "Neutral or Dissatisfied", 1: "Satisfied"}.get(pred_class, "Unknown")
            if label == "Satisfied":
                st.success(f"Predicted Satisfaction: **{label}** 😊")
            else:
                st.warning(f"Predicted Satisfaction: **{label}** 😞")
        else:
            st.error(f"Error: {prediction.get('message', 'Unknown error')}")

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

def performance_page():
    st.title("📊 Model Performance")

    # Model performance data
    model_data = {
        "Model": ["XGBoost", "Random Forest", "Logistic Regression", "Naive Bayes", "KNN", "Gradient Boosting"],
        "Accuracy": [0.9638, 0.9625, 0.8752, 0.8491, 0.9259, 0.9434],
        "Precision": [0.9639, 0.9627, 0.8750, 0.8488, 0.9264, 0.9434],
        "Recall": [0.9638, 0.9625, 0.8752, 0.8491, 0.9259, 0.9434],
        "F1 Score": [0.9637, 0.9624, 0.8750, 0.8488, 0.9256, 0.9433]
    }
    df = pd.DataFrame(model_data)

    # Show data table
    st.subheader("🔢 Model Evaluation Metrics")
    st.dataframe(df.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1 Score": "{:.4f}"
    }))

    # Bar chart comparison
    st.subheader("📈 Metric Comparison (Bar Chart)")
    metric_to_plot = st.selectbox("Select a metric to compare:", ["Accuracy", "Precision", "Recall", "F1 Score"])
    fig = px.bar(df, x="Model", y=metric_to_plot, color="Model", text=df[metric_to_plot].round(4),
                 title=f"{metric_to_plot} Comparison Across Models", height=400)
    st.plotly_chart(fig)

    # Radar chart
    st.subheader("🌐 Overall Comparison (Radar Chart)")
    radar_df = df.set_index("Model")
    fig_radar = px.line_polar(radar_df.reset_index(), r="Accuracy", theta="Model", line_close=True,
                              title="Radar Chart - Accuracy Overview")
    st.plotly_chart(fig_radar)

    # Confusion Matrix Example
    st.subheader("📉 Example: Confusion Matrix (XGBoost)")
    fig_cm, ax = plt.subplots()
    ax.matshow([[95, 3], [2, 100]], cmap='Blues')  # Example numbers
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{[[95, 3], [2, 100]][i][j]}", va='center', ha='center', color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig_cm)

    
    # Highlight top model
    st.subheader("🏆 Best Performing Model")
    top_model = df.loc[df["F1 Score"].idxmax()]
    st.metric(label="Top Model", value=top_model["Model"], delta=f"{top_model['F1 Score']:.4f} F1 Score")

    # Add explanation section
    st.subheader("🤔 Why Might Random Forest Be a Better Choice Than XGBoost?")
    with st.expander("View Explanation"):
        st.markdown("""
        Although XGBoost slightly outperforms Random Forest in terms of accuracy and F1 score, there are **practical reasons** why you might still prefer Random Forest:

        - ✅ **Simplicity & Interpretability**: Random Forest is easier to understand and interpret compared to the more complex gradient boosting mechanism in XGBoost.
        - ⚡ **Faster Training**: For many datasets, Random Forest trains significantly faster, especially without extensive hyperparameter tuning.
        - 🛠️ **Less Sensitive to Hyperparameters**: Random Forest generally works well with default settings, while XGBoost often needs careful tuning.
        - 🧠 **Robust to Overfitting**: Random Forest tends to generalize better with fewer risks of overfitting on small to medium datasets.
        - 📦 **Fewer Dependencies**: Random Forest is built into `scikit-learn`, making it simpler to deploy in standard Python environments.

        So even with slightly lower scores, **Random Forest can be the more practical choice** depending on your project's needs.
        """)





def eda_visualizations():
    return





def main():
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to page:", ["Predictions", "Model Performance", "Dataset Analysis"])

    if page == "Predictions":
        prediction_page()
    elif page == "Model Performance":
        performance_page()
    elif page == "Dataset Analysis":
        eda_visualizations

    st.markdown("---")
    st.caption("Developed by Pransisss")

if __name__ == "__main__":
    main()