from abc import ABC, abstractmethod
from typing import Any
from openai import OpenAI
from anthropic import Anthropic

class LLMWrapper(ABC):
    """Abstract base class for LLM interfaces"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM given a prompt
        Args:
            prompt: The input prompt string
        Returns:
            The generated response string
        """
        pass

class OpenAIBackend(LLMWrapper):
    """Implementation using OpenAI's API"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

class AnthropicBackend(LLMWrapper):
    """Implementation using Anthropic's API"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text

class MockLLM(LLMWrapper):
    """Mock implementation for testing"""
    
    def generate(self, prompt: str) -> str:
        # Return a mock response for testing
        return """
        {
            "agents": [
                {
                    "id": "refiner1",
                    "identity": "Midwest oil refinery owner"
                },
                {
                    "id": "canadian_oil",
                    "identity": "Canadian oil producer"
                }
            ],
            "environment": {
                "facts": [
                    "Trump announces 25% tariff on Canadian oil",
                    "Canadian oil production at 4.5M barrels/day",
                    "US refineries process 18M barrels/day"
                ]
            },
            "connectivity": {
                "refiner1": {
                    "visible_facts": [0, 2],
                    "neighbors": ["canadian_oil"]
                },
                "canadian_oil": {
                    "visible_facts": [0, 1],
                    "neighbors": ["refiner1"]
                }
            }
        }
        """
