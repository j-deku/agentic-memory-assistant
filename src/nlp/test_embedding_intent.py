from nlp.predict_intent_embedding import predict_intent

while True:
    text = input("You: ")

    intent, score = predict_intent(text)

    print(f"Intent: {intent}")
    print(f"Similarity: {score:.2f}")
    print("-" * 30)