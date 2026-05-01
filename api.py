from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from src.logger import logger
import time

app = FastAPI()

# Load model + preprocessor once
model = joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

class CustomerData(BaseModel):
    Gender: str
    Senior_Citizen: str
    Partner: str
    Dependents: str
    Tenure_Months: int
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    Monthly_Charges: float
    Total_Charges: float
    CLTV: int

@app.get("/")
def home():
    return {"message": "FastAPI is running"}

@app.get("/health")
def health_check():
    return {
        "status":"healthy",
        "model_loaded": True,
        "service": "customer-churn-api"
    }


@app.post("/predict")
def predict(data: CustomerData):
    
    input_dict = {
        "Gender": data.Gender,
        "Senior Citizen": data.Senior_Citizen,
        "Partner": data.Partner,
        "Dependents": data.Dependents,
        "Tenure Months": data.Tenure_Months,
        "Phone Service": data.Phone_Service,
        "Multiple Lines": data.Multiple_Lines,
        "Internet Service": data.Internet_Service,
        "Online Security": data.Online_Security,
        "Online Backup": data.Online_Backup,
        "Device Protection": data.Device_Protection,
        "Tech Support": data.Tech_Support,
        "Streaming TV": data.Streaming_TV,
        "Streaming Movies": data.Streaming_Movies,
        "Contract": data.Contract,
        "Paperless Billing": data.Paperless_Billing,
        "Payment Method": data.Payment_Method,
        "Monthly Charges": data.Monthly_Charges,
        "Total Charges": data.Total_Charges,
        "CLTV": data.CLTV,
        
    }
    input_df = pd.DataFrame([input_dict])
    transformed = preprocessor.transform(input_df)
    start_time = time.time()
    prediction = model.predict(transformed)[0]
    execution_time = time.time() - start_time
    probability = model.predict_proba(transformed)[0][1]
    logger.info("Prediction request received")
    logger.info(f"Input shape: {input_df.shape}")
    
    result = "Customer Will Churn " if prediction == 1 else "Customer Will Not Churn"
    logger.info(f"Prediction: {result}")
    probability = f"{round(float(probability),4) * 100}%"
    return {
        "prediction": result,
        "probability": probability,
        "response_time": round(execution_time,3)

    }
    