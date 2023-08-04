from fastapi import FastAPI, Query, Depends
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from typing import List

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split


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


# Veritabanından veri çeken fonksiyon
def fetch_data_for_training(db: Session):
    data_list = db.query(Data).all()
    text_list = [data.text for data in data_list]
    label_list = [data.label for data in data_list]
    return text_list, label_list

def dump_model(model, file_output):
    pickle.dump(model, open(file_output, 'wb'))

def load_model(file_input):
    return pickle.load(open(file_input, 'rb'))

#feature extraction - creating a tf-idf matrix
def tfidf(data, ma = 0.6, mi = 0.0001):
    tfidf_vectorize = TfidfVectorizer()
    tfidf_data = tfidf_vectorize.fit_transform(data)
    return tfidf_data, tfidf_vectorize

#SVM classifier
def test_SVM(x_train, x_test, y_train, y_test):
    SVM = SVC(kernel = 'linear', probability=True)
    SVMClassifier = SVM.fit(x_train, y_train)
    predictions = SVMClassifier.predict(x_test)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    return SVMClassifier, a, p, r


# Veri çekme ve eğitim
@app.get("/traine")
def train_model(db: Session = Depends(get_db)):
    text_list, label_list = fetch_data_for_training(db)
    
    # Veriyi TF-IDF matrisine dönüştürme
    training, vectorizer = tfidf(text_list)
    
    # Veriyi eğitim ve test kümelerine ayırma
    x_train, x_test, y_train, y_test = train_test_split(training, label_list, test_size=0.25, random_state=0)
    
    # SVM modelini eğitme
    model, accuracy, precision, recall = test_SVM(x_train, x_test, y_train, y_test)
    
    # Eğitilmiş modeli ve vektörleme aracını kaydetme
    dump_model(model, 'model.pickle')
    dump_model(vectorizer, 'vectorizer.pickle')
    
    return {"message": "Model trained and saved."}


# PREDICTION Kısmı

@app.get("/predict")
def predict(text: str = Query(...)):
    model = load_model('model.pickle')
    vectorizer = load_model('vectorizer.pickle')
    tfidf_vector = vectorizer.transform([text])
    result = model.predict_proba(tfidf_vector)
    
    prediction = "Confirmation_Yes" if result[0][0] > result[0][1] else "Confirmation_No"
    probability = {"Confirmation_Yes": result[0][0], "Confirmation_No": result[0][1]}

    response_data = {
        "prediction": prediction,
        "probability": probability
    }

    return response_data
