from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DATABASE_URL = "postgresql://postgres:1234@localhost/CognitusDB"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Data(Base):
    __tablename__ = "cognitusApp_data"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    label = Column(String, index=True)
