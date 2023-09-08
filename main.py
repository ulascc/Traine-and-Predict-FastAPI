from fastapi import FastAPI
from traine import traine_model
from rq import Queue
from redis import Redis
from rq.job import Job
from rq_worker import REDIS_HOST, REDIS_PORT
import datetime

app = FastAPI()

redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT)  
queue = Queue('training_queue', connection=redis_conn) 

@app.get("/traine/")
def traine_endpoint():
    try:
        job = queue.enqueue(traine_model)
        return {"message": "Model training task started.", "job_id": job.get_id()}
    except Exception as e:
        error_message = str(e)



@app.get("/traine_result/{job_id}")
def traine_result(job_id: str):
    job = Job.fetch(job_id, redis_conn)

    if job.is_finished:
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-1]
        return {"message": "Model trained and saved.", "task_id": job_id, "end_time": end_time, "status" : "complated"}
    elif job.is_failed:
        return {"message": "An error occurred while training the model.", "task_id": job_id, "status" : "error"}
    else:
        return {"message": "Task is still pending. Check back later.", "task_id": job_id, "status" : "running" }
