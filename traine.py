from database_operations import fetch_data_for_training
from model_operations import tfidf, test_SVM, dump_model
from sklearn.model_selection import train_test_split
from database import SessionLocal
from celery_config import app as celery_app


# Traine
@celery_app.task()
def traine_model_async():
    db = SessionLocal()
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