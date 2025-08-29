import pickle
import pandas as pd

# Load the trained Kaggle model
with open("model_kaggle.pkl", "rb") as f:
    model = pickle.load(f)

# Test multiple scenarios
test_samples = pd.DataFrame([
    {"meantemp": 42, "humidity": 30, "wind_speed": 2.0, "meanpressure": 1005},  # should be heatwave
    {"meantemp": 28, "humidity": 90, "wind_speed": 4.0, "meanpressure": 1000},  # should be flood
    {"meantemp": 25, "humidity": 60, "wind_speed": 1.5, "meanpressure": 1012},  # should be normal
])

predictions = model.predict(test_samples)

# Print results nicely
for i, row in test_samples.iterrows():
    print(f"Input → Temp: {row['meantemp']}°C, Humidity: {row['humidity']}%, Wind: {row['wind_speed']} m/s, Pressure: {row['meanpressure']}")
    print(f"Prediction → {predictions[i]}")
    print("----")
