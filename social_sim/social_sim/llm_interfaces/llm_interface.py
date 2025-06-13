from abc import ABC, abstractmethod
from typing import Any, List
from openai import OpenAI
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import time

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
    
    @abstractmethod
    def batch(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts in a batch
        Args:
            prompts: List of input prompt strings
        Returns:
            List of generated response strings in the same order
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
    
    def batch(self, prompts: List[str]) -> List[str]:
        """OpenAI doesn't have native batching, so raise an error"""
        raise NotImplementedError("OpenAI doesn't have native batching, so raise an error")

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
    
    def batch(self, prompts: List[str]) -> List[str]:
        """Use Anthropic's native batching API"""
        if not prompts:
            return []
        
        # Create batch requests
        requests = []
        for i, prompt in enumerate(prompts):
            new_request = Request(
                custom_id=f"agent-request-{i}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0  # Add temperature for stochastic responses
                )
            )
            requests.append(new_request)
        
        # Submit batch
        message_batch = self.client.messages.batches.create(requests=requests)
        batch_id = message_batch.id
        
        # Poll until complete
        while True:
            message_batch = self.client.messages.batches.retrieve(batch_id)
            if message_batch.processing_status == "ended":
                break
            time.sleep(1)
        
        # Retrieve and order results
        results = self.client.messages.batches.results(batch_id)
        ordered_results = [None] * len(prompts)
        
        for item in results:
            # Extract index from custom_id
            index = int(item.custom_id.split('-')[-1])
            
            # Extract text content from the batch response structure
            if hasattr(item, 'result') and hasattr(item.result, 'message'):
                # Get the text content from the message
                message_content = item.result.message.content
                if message_content and len(message_content) > 0:
                    content = message_content[0].text
                else:
                    content = "<empty>"
            else:
                # Fallback for unexpected response format
                content = str(item)
            
            ordered_results[index] = content
        
        return ordered_results

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
    
    def batch(self, prompts: List[str]) -> List[str]:
        """Mock batch implementation"""
        return [f"Mock response to prompt {i+1}" for i in range(len(prompts))]
