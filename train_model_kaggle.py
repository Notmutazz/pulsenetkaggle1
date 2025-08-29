import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

# Load dataset
df = pd.read_csv("DailyDelhiClimateTrain.csv")

# Add synthetic rainfall (0–300 mm range)
np.random.seed(42)  # reproducibility
df["rainfall"] = np.random.randint(0, 300, size=len(df))

# Add disaster labels (rules)
def label_disaster(row):
    if row["meantemp"] > 35:
        return "heatwave"
    elif row["rainfall"] > 200:
        return "flood"
    else:
        return "normal"

df["label"] = df.apply(label_disaster, axis=1)

# Features & target
X = df[["meantemp", "humidity", "wind_speed", "meanpressure", "rainfall"]]
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model trained on Kaggle dataset (with rainfall)")
print("Accuracy:", accuracy)

# Save model
with open("model_kaggle.pkl", "wb") as f:
    pickle.dump(model, f)
print(" Model saved as model_kaggle.pkl")
