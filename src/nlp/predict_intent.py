import joblib

model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("intent_vectorizer.pkl")


def predict_intent(text):

    X = vectorizer.transform([text])

    intent = model.predict(X)[0]

    confidence = max(model.predict_proba(X)[0])

    if confidence < 0.50:
        return "unknown", confidence

    return intent, confidence