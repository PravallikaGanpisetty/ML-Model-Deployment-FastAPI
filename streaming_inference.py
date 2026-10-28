import pandas as pd
import joblib
import time

model = joblib.load("model.pkl")
data = pd.read_csv("unseen_passengers.csv")

for i in range(10):
    passenger = data.iloc[[i]]
    prediction = model.predict(passenger)[0]

    result = "Survived" if prediction == 1 else "Did not survive"
    print(f"Passenger {i + 1}: {result}")

    time.sleep(1)
