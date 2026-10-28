# ML Model Deployment Strategies and Model Serving using FastAPI

## Project Description
This project demonstrates training a machine learning model using the Titanic dataset and deploying it through different inference strategies: online, batch, and streaming inference. FastAPI is used to serve predictions through API endpoints.

## Dataset
Kaggle Titanic: Machine Learning from Disaster  
https://www.kaggle.com/competitions/titanic/data

Use `train.csv` for training and evaluation.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Joblib
- FastAPI
- Pydantic
- Requests

## Tasks Completed
1. Trained and evaluated a classification model.
2. Saved the trained pipeline as `model.pkl`.
3. Performed online inference for one passenger.
4. Performed batch inference for 20 records and saved `batch_predictions.csv`.
5. Performed streaming inference for 10 records with a one-second delay.
6. Created FastAPI endpoints for health checking and predictions.
7. Tested valid and invalid API requests.

## Results
- Model accuracy: 80.45%
- Model file: `model.pkl`
- Batch output: `batch_predictions.csv`
- API endpoints: `/health` and `/predict`
- Invalid input is rejected through request validation.

## Installation
Install the required packages:

```bash
pip install -r requirements.txt
```

## Execution
Run the Python scripts from the project directory:

```bash
python train_model.py
python online_inference.py
python batch_inference.py
python streaming_inference.py
```

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the API.

Run the client in a separate terminal while the API server is running:

```bash
python client.py
```

## Project Files
- `train_model.py` — Model training and saving
- `online_inference.py` — Single-record prediction
- `batch_inference.py` — Batch predictions
- `streaming_inference.py` — Sequential predictions
- `app.py` — FastAPI application
- `client.py` — API client
- `model.pkl` — Saved model pipeline
- `batch_predictions.csv` — Batch prediction output
- `unseen_passengers.csv` — Unseen passenger records
- `requirements.txt` — Required Python packages
- `Observation.pdf` — Assignment observation report
- `screenshots/` — Assignment output screenshots
