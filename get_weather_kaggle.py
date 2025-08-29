import requests
import pickle
import pandas as pd

# Load trained Kaggle model
with open("model_kaggle.pkl", "rb") as f:
    model = pickle.load(f)

# OpenWeather API setup
API_KEY = "170ad135794af4df80fe182a7ded7fcd"  # your key
CITY = "Delhi"  # change to any city you want

URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

# Fetch live weather
response = requests.get(URL)
data = response.json()

if response.status_code == 200:
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    pressure = data["main"]["pressure"]
    wind_speed = data["wind"]["speed"]

    # Rainfall (OpenWeather free tier often doesn't give rainfall data → default 0 if missing)
    rainfall = data.get("rain", {}).get("1h", 0)

    # Input sample must match training features exactly
    sample = pd.DataFrame([{
        "meantemp": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "meanpressure": pressure,
        "rainfall": rainfall
    }])

    # Predict
    prediction = model.predict(sample)[0]

    print(f"City: {CITY}")
    print(f"Temperature: {temperature} °C")
    print(f"Humidity: {humidity} %")
    print(f"Wind Speed: {wind_speed} m/s")
    print(f"Pressure: {pressure}")
    print(f"Rainfall: {rainfall} mm")
    print(f"Disaster Prediction: {prediction}")

else:
    print("Error fetching weather:", data)
