import pandas as pd

def load_data():
    demand = pd.read_csv("data/demand.csv", parse_dates=['timestamp'])
    weather = pd.read_csv("data/weather.csv", parse_dates=['timestamp'])
    price = pd.read_csv("data/price.csv", parse_dates=['timestamp'])
    return demand, weather, price

def create_features(demand, weather):
    df = demand.copy()

    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekday'] = df['timestamp'].dt.weekday

    df['lag_1'] = df['demand'].shift(1)

    df = df.merge(weather, on="timestamp", how="left")
    df = df.dropna()

    return df

def save_features(df):
    df.to_csv("data/features.csv", index=False)
    print("✔ features.csv created!")

if __name__ == "__main__":
    demand, weather, price = load_data()
    features = create_features(demand, weather)
    save_features(features)
