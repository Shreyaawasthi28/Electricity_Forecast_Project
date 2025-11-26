import pandas as pd
import joblib

# Load trained model
model = joblib.load("model_demand.pkl")

def predict_next(data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    sample = {
        "hour": 10,
        "day": 1,
        "month": 1,
        "weekday": 1,
        "lag_1": 250,
        "temperature": 26,
        "humidity": 55,
        "wind_speed": 8
    }

    result = predict_next(sample)
    print("Predicted Demand:", result)
