import openai
import os
from dotenv import load_dotenv

load_dotenv()

class LLMWrapper:
    def __init__(self, backend):
        self.backend = backend

    def call(self, prompt):
        return self.backend.call(prompt)


class OpenAIBackend:
    def __init__(self, model="gpt-4"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def call(self, prompt):
        print(f"Calling OpenAI with prompt:\n{prompt[:500]}...\n")
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message['content']
