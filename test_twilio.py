import pickle
import requests
import pandas as pd
import config
from twilio_utils import send_alert

# Define the city for weather data
CITY = "Delhi"

def get_prediction_and_send_sms():
    """
    Fetches live weather data, predicts a disaster label, and sends an SMS.
    """
    try:
        # Load the trained model
        with open("model_kaggle.pkl", "rb") as f:
            model = pickle.load(f)

        # OpenWeather API setup
        URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={config.OPENWEATHER_API_KEY}&units=metric"

        # Fetch live weather
        response = requests.get(URL)
        data = response.json()

        if response.status_code == 200:
            temperature = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            pressure = data["main"]["pressure"]
            wind_speed = data["wind"]["speed"]

            # Rainfall (OpenWeather free tier often doesn't give rainfall data -> default 0 if missing)
            # The 'rain' key is not present on most days and in many weather conditions, so we safely get it.
            # We assume it's hourly rainfall since that's what's typically provided.
            rainfall = data.get("rain", {}).get("1h", 0)

            # Create a DataFrame to match the model's expected input
            sample = pd.DataFrame([{
                "meantemp": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "meanpressure": pressure,
                "rainfall": rainfall
            }])

            # Make a prediction
            prediction = model.predict(sample)[0]

            # Send the AI-powered alert using the new send_alert function
            send_alert(prediction, data)
            
        else:
            print("Error fetching weather:", data)

    except FileNotFoundError:
        print("Error: The 'model_kaggle.pkl' file was not found. Please run train_model_kaggle.py first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    get_prediction_and_send_sms()