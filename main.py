from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
import warnings
import os
import pandas as pd
import uvicorn

# Define the FastAPI instance
app = FastAPI()

# Create a Pydantic model to validate the input data
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Define the endpoint to accept the input data
@app.post("/predict")
async def predict(data: PatientData):
    # Setting AWS environment variables
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = '12345678'
    os.environ['AWS_REGION'] = 'us-east-1'
    os.environ['AWS_BUCKET_NAME'] = 'mlflow'

    # Setting MySQL environment variables
    os.environ['MYSQL_DATABASE'] = 'mlflow'
    os.environ['MYSQL_USER'] = 'mlflow_user'
    os.environ['MYSQL_PASSWORD'] = '12345678'
    os.environ['MYSQL_ROOT_PASSWORD'] = 'toor'

    # Setting MLflow environment variables
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://119.59.103.209:9000'
    os.environ['MLFLOW_TRACKING_URI'] = 'http://119.59.103.209:5000'


    # Set the MLflow tracking server URI
    mlflow.set_tracking_uri("http://119.59.103.209:5000")  # Replace with your actual MLflow server URL
    # Suppress the warning
    with warnings.catch_warnings():
         warnings.simplefilter("ignore", category=FutureWarning)
    
    client = MlflowClient()
    model_name = "heart-model"
    model_deployment="Staging"
    # Load the MLflow model
    latest_versions = client.get_latest_versions(model_name)
    print(latest_versions)
    model_uri = f"models:/{model_name}/{model_deployment}"

    model = mlflow.pyfunc.load_model(model_uri)

    # Prepare input data for prediction
    input_data = pd.DataFrame([[
        data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
        data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
        data.ca, data.thal
    ]], columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"])

    # Make a prediction
    prediction = model.predict(input_data)
    print("Prediction",prediction)

    return {"message": "Prediction successful", "prediction": prediction.tolist()}



if __name__ == "__main__":
    # Run the application using uvicorn programmatically
    uvicorn.run(app, host="0.0.0.0", port=8000)