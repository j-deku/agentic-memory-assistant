from nlp.predict_torch_intent import predict_intent

while True:
    text = input("You: ")

    intent, conf = predict_intent(text)

    print(f"Intent: {intent} | Confidence: {conf:.2f}")