from fastapi import FastAPI, Query, Depends
from sqlalchemy.orm import Session
from database_operations import get_db, fetch_data_for_training
from model_operations import tfidf, test_SVM, dump_model, load_model
from sklearn.model_selection import train_test_split
from database import Data, SessionLocal

app = FastAPI()

# Veritabanından tüm verileri çeken endpoint
@app.get("/data/all/")
def read_all_data(skip: int = Query(0, description="Skip records"), limit: int = Query(10, description="Limit records"), db: Session = Depends(get_db)):
    data_list = db.query(Data).offset(skip).limit(limit).all()
    response_list = [{"id": data.id, "text": data.text, "label": data.label} for data in data_list]
    return response_list


# Traine
@app.get("/traine/")
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


# Predict
@app.get("/predict/")
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
