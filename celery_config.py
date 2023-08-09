from celery import Celery

app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",  
    backend="redis://localhost:6379/1",  
    include=["traine"]
)

app.conf.broker_connection_retry_on_startup = True

CELERY_ACKS_LATE = True

