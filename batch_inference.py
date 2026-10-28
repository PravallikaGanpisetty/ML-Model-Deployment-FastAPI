import pandas as pd
import joblib

model = joblib.load("model.pkl")
data = pd.read_csv("unseen_passengers.csv")

batch = data.iloc[:20].copy()
batch["Prediction"] = model.predict(batch)

batch.to_csv("batch_predictions.csv", index=False)

print(batch)
print("Saved batch_predictions.csv")
