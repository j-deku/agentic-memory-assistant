class IntentEngine:
    def __init__(self, patterns):
        self.patterns = patterns

    def detect(self, text: str):
        text = text.lower()
        for intent, pattern in self.patterns:
            if pattern.search(text):
                return intent
        return None