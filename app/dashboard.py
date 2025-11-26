import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Electricity Demand Forecasting", layout="wide")

st.title("⚡ Electricity Demand Forecasting Dashboard")

# Load model
model = joblib.load("model_demand.pkl")

# Load features data
df = pd.read_csv("data/features.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Demand Plot
st.subheader("📉 Actual Demand Trend")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['timestamp'], df['demand'], label="Actual Demand")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Demand")
ax.legend()
st.pyplot(fig)

st.subheader("🔮 Predict Next Hour Demand")

hour = st.number_input("Hour (0–23)", min_value=0, max_value=23)
day = st.number_input("Day (1–31)", min_value=1, max_value=31)
month = st.number_input("Month (1–12)", min_value=1, max_value=12)
weekday = st.number_input("Weekday (0=Mon … 6=Sun)", min_value=0, max_value=6)
lag_1 = st.number_input("Yesterday same hour demand", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0)

if st.button("Predict Demand"):
    data = {
        "hour": hour,
        "day": day,
        "month": month,
        "weekday": weekday,
        "lag_1": lag_1,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
    }

    df_input = pd.DataFrame([data])
    prediction = model.predict(df_input)[0]

    st.success(f"⚡ Predicted Demand: {prediction:.2f}")
