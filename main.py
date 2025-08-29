# 1. Imports
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
from fastapi import UploadFile, File
import io
from fastapi.middleware.cors import CORSMiddleware
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
    model = joblib.load("model.pkl")
    print("Model loaded successfully:", type(model))
except Exception as e:
    print("Failed to load model:", e)
    model = None

# 4. Define input schema
class InputData(BaseModel):
    temperature: float
    rainfall: float
    humidity: float

# 5. Manual prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"error": "Model not loaded"}

    input_df = pd.DataFrame([{
        "temperature": data.temperature,
        "rainfall": data.rainfall,
        "humidity": data.humidity
    }])

    prediction = model.predict(input_df)
    return {"prediction": prediction[0]}

# 6. Live weather fetcher
def get_live_weather(city: str = "Bengaluru") -> dict:
    API_KEY = "170ad135794af4df80fe182a7ded7fcd"
    URL = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(URL)
    data = response.json()

    if response.status_code == 200:
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rainfall = data.get("rain", {}).get("1h", 0)
        return {"temperature": temp, "rainfall": rainfall, "humidity": humidity}
    else:
        raise ValueError(f"Failed to fetch weather: {data}")

# 7. Live prediction endpoint
@app.get("/predict-live")
def predict_live(city: str = "Bengaluru"):
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