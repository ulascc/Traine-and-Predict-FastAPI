from fastapi import FastAPI, Query, Depends
from sqlalchemy.orm import Session
from database_operations import get_db 
from model_operations import load_model
from database import Data
from traine import traine_model
from predict import predict

app = FastAPI()


# Traine
@app.get("/traine/")
def traine_endpoint(db: Session = Depends(get_db)):
    result = traine_model(db)
    return result


# Predict
@app.get("/predict/")
def predict_endpoint(text: str = Query(...)):
    result = predict(text)
    return result
