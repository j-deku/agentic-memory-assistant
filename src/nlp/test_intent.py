# test_intent.py

from nlp.predict_intent import predict_intent

while True:

    text = input("You: ")

    intent, confidence = predict_intent(text)

    print(
        f"Intent: {intent} | "
        f"Confidence: {confidence:.2f}"
    )