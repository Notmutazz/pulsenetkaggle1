# 1. Imports
from fastapi import FastAPI
from pydantic import BaseModel
import pickle # Changed from joblib to pickle
import pandas as pd
import requests
from fastapi import UploadFile, File
import io
from fastapi.middleware.cors import CORSMiddleware
import config # NEW: Import the config file

# 2. Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "PulseNet backend is running!"}

# 3. Load the model
try:
    # UPDATED: Load the correct model file name with pickle
    with open("model_kaggle.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully:", type(model))
except Exception as e:
    print("Failed to load model:", e)
    model = None

# 4. Define input schema
class InputData(BaseModel):
    # UPDATED: Added all features that the model was trained on
    meantemp: float
    humidity: float
    wind_speed: float
    meanpressure: float
    rainfall: float

# 5. Manual prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"error": "Model not loaded"}

    # UPDATED: The DataFrame now includes all 5 features with correct names
    input_df = pd.DataFrame([{
        "meantemp": data.meantemp,
        "humidity": data.humidity,
        "wind_speed": data.wind_speed,
        "meanpressure": data.meanpressure,
        "rainfall": data.rainfall
    }])

    prediction = model.predict(input_df)
    return {"prediction": prediction[0]}

# 6. Live weather fetcher
# UPDATED: Uses config.py and fetches all required features
def get_live_weather(city: str = "Delhi") -> dict:
    API_KEY = config.OPENWEATHER_API_KEY
    URL = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(URL)
    data = response.json()

    if response.status_code == 200:
        # UPDATED: Fetch all five features required by the model
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        pressure = data["main"]["pressure"]
        rainfall = data.get("rain", {}).get("1h", 0)
        
        # Return a dictionary with the correct feature names
        return {
            "meantemp": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "meanpressure": pressure,
            "rainfall": rainfall
        }
    else:
        raise ValueError(f"Failed to fetch weather: {data}")

# 7. Live prediction endpoint
@app.get("/predict-live")
def predict_live(city: str = "Delhi"):
    if model is None:
        return {"error": "Model not loaded"}
    try:
        weather = get_live_weather(city)
        input_df = pd.DataFrame([weather])
        prediction = model.predict(input_df)
        return {
            "city": city,
            "weather": weather,
            "prediction": prediction[0]
        }
    except Exception as e:
        return {"error": str(e)}
