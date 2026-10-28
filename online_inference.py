import pandas as pd
import joblib

model = joblib.load("model.pkl")
data = pd.read_csv("unseen_passengers.csv")

passenger = data.iloc[[0]]
prediction = model.predict(passenger)[0]

print("Passenger details:")
print(passenger)
print("Prediction:",
      "Survived" if prediction == 1 else "Did not survive")
