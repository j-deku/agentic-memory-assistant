from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

from nlp.training_data import training_examples

texts = [x[0] for x in training_examples]
labels = [x[1] for x in training_examples]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(texts)

model = LogisticRegression()

model.fit(X, labels)

joblib.dump(model, "intent_model.pkl")
joblib.dump(vectorizer, "intent_vectorizer.pkl")

print("Intent model trained successfully ✔")