from fastapi import FastAPI, Query
from traine import traine_model_async
from predict import predict

app = FastAPI()


from traine import traine_model_async

@app.get("/traine/")
def traine_endpoint():
    task_result = traine_model_async.apply_async()
    return {"message": "Model training task started.", "task_id": task_result.id}


@app.get("/traine_result/{task_id}")
def traine_result(task_id: str):
    result = traine_model_async.AsyncResult(task_id)
    
    if result.state == "SUCCESS":
        return {"message": "Model trained and saved."}
    elif result.state == "PENDING":
        return {"message": "Task is still pending. Check back later."}
    else:
        return {"message": "An error occurred while training the model."}


# Predict
@app.get("/predict/")
def predict_endpoint(text: str = Query(...)):
    result = predict(text)
    return result