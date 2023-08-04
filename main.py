from fastapi import FastAPI, Query, Depends
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from typing import List

app = FastAPI()

# Database configuration
DATABASE_URL = "postgresql://postgres:1234@localhost/CognitusDB"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLAlchemy model
Base = declarative_base()

class Data(Base):
    __tablename__ = "cognitusApp_data"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    label = Column(String, index=True)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/data/all/")
def read_all_data(skip: int = Query(0, description="Skip records"), limit: int = Query(10, description="Limit records"), db: Session = Depends(get_db)):
    data_list = db.query(Data).offset(skip).limit(limit).all()
    response_list = [{"id": data.id, "text": data.text, "label": data.label} for data in data_list]
    return response_list

