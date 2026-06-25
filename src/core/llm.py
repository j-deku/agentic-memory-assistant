# core/llm.py

from openai import OpenAI
class LLM:
    def __init__(self, api_key = "", model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a strict JSON planner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content