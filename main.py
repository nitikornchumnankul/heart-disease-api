from fastapi import FastAPI
from pydantic import BaseModel

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
    # You can add the logic here to process the input and make predictions
    return {"message": "Data received successfully", "data": data}

