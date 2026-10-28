import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI()

model = joblib.load("model.pkl")

class Passenger(BaseModel):
    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: float | None = Field(default=None, ge=0, le=100)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    Embarked: Literal["C", "Q", "S"] | None = None

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.model_dump()])
    result = int(model.predict(data)[0])

    return {
        "prediction": result,
        "result": "Survived" if result == 1
        else "Did not survive"
    }
