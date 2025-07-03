from abc import ABC, abstractmethod
from typing import Any, List, Optional
from openai import OpenAI
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import time
import threading

class LLMWrapper(ABC):
    """Abstract base class for LLM interfaces"""
    
    def __init__(self):
        # Global character usage tracking
        self.input_characters = 0
        self.output_characters = 0
        self.call_count = 0
        
        # Per-simulation tracking
        self.simulation_usage = {}  # simulation_id -> usage_stats
        self._current_simulation = threading.local()  # Thread-local current simulation
    
    def set_current_simulation(self, simulation_id: str):
        """Set the current simulation context for tracking"""
        self._current_simulation.id = simulation_id
        if simulation_id not in self.simulation_usage:
            self.simulation_usage[simulation_id] = {
                "input_characters": 0,
                "output_characters": 0,
                "call_count": 0
            }
    
    def clear_current_simulation(self):
        """Clear the current simulation context"""
        if hasattr(self._current_simulation, 'id'):
            delattr(self._current_simulation, 'id')
    
    def track_usage(self, input_text: str, output_text: str):
        """Track character usage for input and output"""
        input_chars = len(input_text) if input_text else 0
        output_chars = len(output_text) if output_text else 0
        
        # Track globally
        self.input_characters += input_chars
        self.output_characters += output_chars
        self.call_count += 1
        
        # Track per simulation if context is set
        if hasattr(self._current_simulation, 'id'):
            sim_id = self._current_simulation.id
            if sim_id in self.simulation_usage:
                self.simulation_usage[sim_id]["input_characters"] += input_chars
                self.simulation_usage[sim_id]["output_characters"] += output_chars
                self.simulation_usage[sim_id]["call_count"] += 1
    
    def get_usage_stats(self) -> dict:
        """Get current global usage statistics"""
        return {
            "input_characters": self.input_characters,
            "output_characters": self.output_characters,
            "total_characters": self.input_characters + self.output_characters,
            "call_count": self.call_count
        }
    
    def get_simulation_usage(self, simulation_id: str) -> dict:
        """Get usage statistics for a specific simulation"""
        if simulation_id in self.simulation_usage:
            stats = self.simulation_usage[simulation_id]
            return {
                "input_characters": stats["input_characters"],
                "output_characters": stats["output_characters"],
                "total_characters": stats["input_characters"] + stats["output_characters"],
                "call_count": stats["call_count"]
            }
        return {"input_characters": 0, "output_characters": 0, "total_characters": 0, "call_count": 0}
    
    def get_all_simulation_usage(self) -> dict:
        """Get usage statistics for all simulations"""
        result = {}
        for sim_id, stats in self.simulation_usage.items():
            result[sim_id] = {
                "input_characters": stats["input_characters"],
                "output_characters": stats["output_characters"],
                "total_characters": stats["input_characters"] + stats["output_characters"],
                "call_count": stats["call_count"]
            }
        return result
    
    def reset_usage_stats(self):
        """Reset usage statistics to zero"""
        self.input_characters = 0
        self.output_characters = 0
        self.call_count = 0
        self.simulation_usage.clear()
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate a response from the LLM given a prompt
        Args:
            prompt: The input prompt string
            system_prompt: Optional system prompt to provide context/role
        Returns:
            The generated response string
        """
        pass
    
    @abstractmethod
    def batch(self, prompts: List[str], system_prompts: List[str] = None) -> List[str]:
        """
        Generate responses for multiple prompts in a batch
        Args:
            prompts: List of input prompt strings
            system_prompts: Optional list of system prompts (must match length of prompts if provided)
        Returns:
            List of generated response strings in the same order
        """
        pass

class OpenAIBackend(LLMWrapper):
    """Implementation using OpenAI's API"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate text from a prompt with optional system prompt"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        self.track_usage(prompt, response.choices[0].message.content)
        return response.choices[0].message.content
    
    def batch(self, prompts: List[str], system_prompts: List[str] = None) -> List[str]:
        """Generate text from multiple prompts in batch with optional system prompts"""
        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        
        results = []
        for prompt, system_prompt in zip(prompts, system_prompts):
            result = self.generate(prompt, system_prompt)
            results.append(result)
            # Usage is already tracked in generate() method
        return results

class AnthropicBackend(LLMWrapper):
    """Implementation using Anthropic's API with prompt caching support"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", max_output_tokens: int = 1000, debug: bool = False):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        
        super().__init__()
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.debug = debug
        self.active_batches = set()  # Track active batch IDs
        
    def check_pending_batches(self):
        """Check for and report pending batches"""
        try:
            # List recent batches
            batches = self.client.messages.batches.list(limit=10)
            pending_batches = []
            
            for batch in batches.data:
                if batch.processing_status in ["validating", "in_progress"]:
                    pending_batches.append({
                        "id": batch.id,
                        "status": batch.processing_status,
                        "created_at": batch.created_at,
                        "request_counts": batch.request_counts
                    })
            
            if pending_batches:
                print(f"Found {len(pending_batches)} pending batches:")
                for batch in pending_batches:
                    print(f"  - Batch {batch['id']}: {batch['status']} (created: {batch['created_at']})")
                    if hasattr(batch['request_counts'], 'total'):
                        print(f"    Requests: {batch['request_counts'].total} total")
                        
            return pending_batches
            
        except Exception as e:
            if self.debug:
                print(f"Error checking pending batches: {e}")
            return []
    
    def cancel_pending_batches(self, max_age_hours=1):
        """Cancel batches that have been pending for too long"""
        try:
            from datetime import datetime, timedelta
            
            batches = self.client.messages.batches.list(limit=20)
            cancelled_count = 0
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for batch in batches.data:
                if batch.processing_status in ["validating", "in_progress"]:
                    # Parse the created_at timestamp
                    batch_created = datetime.fromisoformat(batch.created_at.replace('Z', '+00:00'))
                    
                    if batch_created < cutoff_time:
                        try:
                            self.client.messages.batches.cancel(batch.id)
                            cancelled_count += 1
                            if self.debug:
                                print(f"Cancelled old batch {batch.id} (created: {batch.created_at})")
                        except Exception as cancel_error:
                            if self.debug:
                                print(f"Failed to cancel batch {batch.id}: {cancel_error}")
            
            if cancelled_count > 0:
                print(f"Cancelled {cancelled_count} old pending batches")
                
            return cancelled_count
            
        except Exception as e:
            if self.debug:
                print(f"Error cancelling pending batches: {e}")
            return 0
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate text from a prompt with optional system prompt"""
        # Build the messages list
        messages = [{"role": "user", "content": prompt}]
        
        # Create the request parameters
        params = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": messages
        }
        
        # Add system prompt if provided
        if system_prompt:
            params["system"] = system_prompt
        
        response = self.client.messages.create(**params)
        self.track_usage(prompt, response.content[0].text)
        return response.content[0].text
    
    def batch(self, prompts: List[str], system_prompts: List[str] = None) -> List[str]:
        """Use Anthropic's native batching API with system prompt caching for efficiency"""
        if not prompts:
            return []
        
        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        
        # Check if all system prompts are the same for caching optimization
        unique_system_prompts = set(system_prompts)
        use_cache = len(unique_system_prompts) == 1 and list(unique_system_prompts)[0] is not None
        
        if self.debug and use_cache:
            print(f"Using prompt caching for batch of {len(prompts)} requests with consistent system prompt")
        elif self.debug:
            print(f"Processing batch of {len(prompts)} requests without caching (mixed or no system prompts)")
        
        # Create batch requests using simple dictionaries instead of complex types
        requests = []
        for i, (prompt, system_prompt) in enumerate(zip(prompts, system_prompts)):
            prompt_with_id = prompt + str(i)  # Add unique identifier for batch processing
            
            # Build message params as simple dictionary
            message_params = {
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "messages": [{"role": "user", "content": prompt_with_id}],
                "temperature": 1.0  # Add temperature for stochastic responses
            }
            
            # Add system prompt with caching if provided and consistent
            if system_prompt:
                if use_cache:
                    # Use prompt caching for consistent system prompts - cache reused across timesteps
                    message_params["system"] = [
                        {
                            "type": "text", 
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                else:
                    # No caching for mixed system prompts
                    message_params["system"] = system_prompt
            
            # Create request as simple dictionary
            request = {
                "custom_id": f"agent-request-{i}",
                "params": message_params
            }
            requests.append(request)
        
        try:
            # Submit batch using the client's batch creation method
            message_batch = self.client.messages.batches.create(requests=requests)
            batch_id = message_batch.id
            self.active_batches.add(batch_id)
            
            if self.debug:
                print(f"Submitted batch {batch_id} with {len(requests)} requests")
            
            # Poll until complete with a timeout to prevent infinite waiting
            start_time = time.time()
            max_wait_seconds = 120  # 2-minute timeout (adjust as needed)
            while True:
                message_batch = self.client.messages.batches.retrieve(batch_id)
                if message_batch.processing_status == "ended":
                    break

                if (time.time() - start_time) > max_wait_seconds:
                    # Give up waiting – mark entire batch as failed
                    if self.debug:
                        print(f"Batch {batch_id} did not finish within {max_wait_seconds}s – aborting and returning failures")
                    self.active_batches.discard(batch_id)
                    return [None] * len(prompts)

                time.sleep(1)
            
            if self.debug:
                print(f"Batch {batch_id} completed")
            
            # Remove from active batches
            self.active_batches.discard(batch_id)
            
            # Retrieve and order results
            results = self.client.messages.batches.results(batch_id)
            ordered_results = [None] * len(prompts)
            
            # Track batch usage
            total_input = '\n'.join(prompts)
            if system_prompts and any(sp for sp in system_prompts):
                # Add system prompts to input calculation
                system_input = '\n'.join(sp for sp in system_prompts if sp)
                total_input += '\n' + system_input
            
            total_output = ""
            
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
                total_output += content + '\n'
            
            # Track the batch usage
            self.track_usage(total_input, total_output)
            
            return ordered_results
            
        except Exception as e:
            if self.debug:
                print(f"Batch processing failed: {str(e)}")
            # Fallback to individual requests
            if self.debug:
                print("Falling back to individual requests...")
            
            results = []
            for prompt, system_prompt in zip(prompts, system_prompts):
                try:
                    result = self.generate(prompt, system_prompt)
                    results.append(result)
                except Exception as individual_error:
                    if self.debug:
                        print(f"Individual request also failed: {individual_error}")
                    results.append(None)
            
            return results

class MockLLM(LLMWrapper):
    """Mock implementation for testing"""
    
    def __init__(self):
        super().__init__()
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        # Return a mock response for testing
        response = """
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
        self.track_usage(prompt, response)
        return response
    
    def batch(self, prompts: List[str], system_prompts: List[str] = None) -> List[str]:
        """Mock batch implementation"""
        results = [f"Mock response to prompt {i+1}" for i in range(len(prompts))]
        
        # Track batch usage
        total_input = '\n'.join(prompts)
        if system_prompts:
            system_input = '\n'.join(sp for sp in system_prompts if sp)
            total_input += '\n' + system_input
        total_output = '\n'.join(results)
        
        self.track_usage(total_input, total_output)
        return results
