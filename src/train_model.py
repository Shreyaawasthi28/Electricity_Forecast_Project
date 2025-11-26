import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib

# -------------------------
# TRAIN MODEL
# -------------------------

def train_model():
    print("🔄 Loading features.csv...")
    df = pd.read_csv("data/features.csv")

    # Target and features
    y = df['demand']
    X = df.drop(columns=['demand', 'timestamp'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("✨ Training LightGBM model...")
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"📊 MAE: {mae}")
    print(f"📉 RMSE: {rmse}")

    # Save model
    joblib.dump(model, "model_demand.pkl")
    print("💾 Model saved as model_demand.pkl")


if __name__ == "__main__":
    train_model()
