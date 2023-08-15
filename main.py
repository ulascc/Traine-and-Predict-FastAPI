from fastapi import FastAPI, Query
from traine import traine_model_async
from predict import predict
import datetime

app = FastAPI()


@app.get("/traine/")
def traine_endpoint():
    task_result = traine_model_async.apply_async()
    return {"message": "Model training task started.", "task_id": task_result.id}


@app.get("/traine_result/{task_id}")
def traine_result(task_id: str):
    result = traine_model_async.AsyncResult(task_id)
    
    if result.state == "SUCCESS":
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-1]
        return {"message": "Model trained and saved.", "task_id": task_id, "end_time": end_time, "status" : "complated"}
    elif result.state == "PENDING":
        return {"message": "Task is still pending. Check back later.", "task_id": task_id, "status" : "running" }
    else:
        return {"message": "An error occurred while training the model.", "task_id": task_id, "status" : "error"}


# Predict
@app.get("/predict/")
def predict_endpoint(text: str = Query(...)):
    result = predict(text)
    return result