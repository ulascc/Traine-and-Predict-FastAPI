from fastapi import Query
from model_operations import load_model


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