from sqlalchemy.orm import Session
from database import Data, SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def fetch_data_for_training(db: Session):
    data_list = db.query(Data).all()
    text_list = [data.text for data in data_list]
    label_list = [data.label for data in data_list]
    return text_list, label_list
