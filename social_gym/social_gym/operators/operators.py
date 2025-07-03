import copy
from typing import List
from genetic_algorithm.individual import Individual
from genetic_algorithm.operators import CrossoverOperator, MutationOperator

import random
import time  # Add time import for delays

class LLMTextCrossover(CrossoverOperator):
    """
    Blend the textual DNA of two parent prompts by asking an LLM
    to synthesise a new prompt that inherits constraints from both.
    """

    def __init__(
        self,
        llm_wrapper,
        use_scheduling: bool = False,
        temperature: float = 0.8,
        api_delay: float = 3.0  # Add delay parameter
    ):
        self._use_scheduling = use_scheduling
        self.llm = llm_wrapper
        self.temperature = temperature
        self.api_delay = api_delay  # Store delay setting

    # ----- API helpers -----------------------------------------------------

    def _call_llm(self, user_prompt, system_prompt=None) -> str:
        result = self.llm.generate(user_prompt, system_prompt)
        # Add delay after API call to prevent rate limiting
        time.sleep(self.api_delay)
        return result

    # ----- Crossover core logic -------------------------------------------

    def apply(self, parents: List[Individual]) -> List[Individual]:
        parent1, parent2 = parents[0], parents[1]
        
        # Create copies of parents for offspring
        child1 = parent1.__class__(parent1._evaluator)
        child2 = parent2.__class__(parent2._evaluator)
        
        child1.dofs = copy.deepcopy(parent1.dofs)
        child2.dofs = copy.deepcopy(parent2.dofs)

        # Handle both string and dictionary DOFs
        if isinstance(parent1.dofs, dict) and isinstance(parent2.dofs, dict):
            # Dictionary DOFs - crossover only the prompt part
            prompt1 = parent1.get_prompt() if hasattr(parent1, 'get_prompt') else parent1.dofs.get("prompt", "")
            prompt2 = parent2.get_prompt() if hasattr(parent2, 'get_prompt') else parent2.dofs.get("prompt", "")
            
            system_prompt = (
                "You are an expert prompt engineer. You will be given two source "
                "prompts. Produce a NEW prompt that:\n"
                " • Preserves any hard constraints mentioned in EITHER parent.\n"
                " • Combines their useful wording and stylistic hints.\n"
                " • Remains concise (≤ 200 tokens).\n"
                "Return ONLY the new prompt."
            )
            
            user_prompt = (
                f"--- Parent A ---\n{prompt1}\n"
                f"--- Parent B ---\n{prompt2}\n"
                "--- End ---"
            )

            children = [child1, child2]
            for child in children:
                new_text = self._call_llm(user_prompt, system_prompt)
                # Update only the prompt part, keep connectivity unchanged
                if hasattr(child, 'set_prompt'):
                    child.set_prompt(new_text)
                else:
                    child.dofs["prompt"] = new_text
        else:
            # String DOFs - original behavior
            system_prompt = (
                "You are an expert prompt engineer. You will be given two source "
                "prompts. Produce a NEW prompt that:\n"
                " • Preserves any hard constraints mentioned in EITHER parent.\n"
                " • Combines their useful wording and stylistic hints.\n"
                " • Remains concise (≤ 200 tokens).\n"
                "Return ONLY the new prompt."
            )
            
            user_prompt = (
                f"--- Parent A ---\n{parent1.dofs}\n"
                f"--- Parent B ---\n{parent2.dofs}\n"
                "--- End ---"
            )

            children = [child1, child2]
            for child in children:
                new_text = self._call_llm(user_prompt, system_prompt)
                child.dofs = new_text

        return children

    # ----- Scheduling ------------------------------------------------------

    def schedule(self, gen: int, n_generations: int):
        pass


class LLMTextMutation(MutationOperator):
    """
    Produce a single mutated prompt by instructing an LLM to
    rewrite / augment the existing one. Optionally also mutate
    connectivity patterns when working with dictionary DOFs.
    """

    def __init__(
        self,
        llm_wrapper,
        use_scheduling: bool = False,
        temperature: float = 0.9,
        mutate_connectivity: bool = False,
        connectivity_mutation_rate: float = 0.1,
        num_agents: int = None,
        mutation_context: str = None,
        api_delay: float = 1.0  # Add delay parameter
    ):
        self._use_scheduling = use_scheduling
        self.llm = llm_wrapper
        self.temperature = temperature
        
        # Connectivity mutation parameters
        self.mutate_connectivity = mutate_connectivity
        self.connectivity_mutation_rate = connectivity_mutation_rate
        self._initial_connectivity_mutation_rate = connectivity_mutation_rate
        self.num_agents = num_agents
        self.mutation_context = mutation_context
        self.api_delay = api_delay  # Store delay setting

    @staticmethod
    def _mutate_via_llm(llm_wrapper, prompt_text: str) -> str:
        """Static version for use by evaluators during initialization"""
        system_prompt = (
            "You are the mutation operator inside of a genetic algorithm in which a group of LLM agents are coordinating to have some desired emergent behavior.\n"
            "Each agent has the same prompt (which is what you are trying to mutate). The agents can send messages to each other and act. An agent can only communicate with the agents to which it is connected. \n"
            "Given a prompt, produce a new prompt that is a mutation of the prompt.\n"
            "Return ONLY the mutated prompt."
        )
        
        user_prompt = f"Here is the original prompt:\n{prompt_text}"
        
        result = llm_wrapper.generate(user_prompt, system_prompt)
        # Add delay after static API call as well
        time.sleep(1.0)  # Default 1 second delay for static calls
        return result

    def _mutate_via_llm_instance(self, prompt_text: str) -> str:
        """Instance method with mutation context for use by mutation operator"""
        system_prompt = (
            "You are the mutation operator inside of a genetic algorithm in which a group of LLM agents are coordinating to have some desired emergent behavior.\n"
            "Each agent has the same prompt (which is what you are trying to mutate). The agents can send messages to each other and act. An agent can only communicate with the agents to which it is connected. \n"
            "Given a prompt, produce a new prompt that is a mutation of the prompt.\n"
        )
        
        # Add mutation context if provided
        if self.mutation_context:
            system_prompt += f"\nAdditional context: {self.mutation_context}\n"
        
        system_prompt += "Return ONLY the mutated prompt."
        
        user_prompt = f"Here is the original prompt:\n{prompt_text} \n Your response should only be the mutated prompt. Do not include any descriptions, explanations, or additional text."
        
        result = self.llm.generate(user_prompt, system_prompt)
        # Add delay after API call to prevent rate limiting
        time.sleep(self.api_delay)
        return result

    def _generate_valid_connectivity_pairs(self) -> List[List[int]]:
        """Generate a valid set of connectivity pairs for the number of agents."""
        # Start with adjacent connectivity as base
        pairs = []
        for i in range(self.num_agents - 1):
            pairs.append([i, i + 1])
        return pairs

    def _add_random_connection(self, connectivity_pairs: List[List[int]]) -> List[List[int]]:
        """Add a random connection between two unconnected agents."""
        # Find all possible pairs
        all_possible_pairs = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                all_possible_pairs.append([i, j])
        
        # Find existing pairs (normalized to [min, max] order)
        existing_pairs = set()
        for pair in connectivity_pairs:
            if len(pair) == 2:
                normalized_pair = tuple(sorted(pair))
                existing_pairs.add(normalized_pair)
        
        # Find available pairs to add
        available_pairs = []
        for pair in all_possible_pairs:
            normalized_pair = tuple(sorted(pair))
            if normalized_pair not in existing_pairs:
                available_pairs.append(pair)
        
        if available_pairs:
            new_pair = random.choice(available_pairs)
            connectivity_pairs.append(new_pair)
        
        return connectivity_pairs

    def _remove_random_connection(self, connectivity_pairs: List[List[int]]) -> List[List[int]]:
        """Remove a random connection while ensuring graph remains connected."""
        if len(connectivity_pairs) <= 1:
            return connectivity_pairs  # Don't remove if too few connections
        
        # Remove a random connection
        if connectivity_pairs:
            random_index = random.randint(0, len(connectivity_pairs) - 1)
            connectivity_pairs.pop(random_index)
        
        return connectivity_pairs

    def _modify_random_connection(self, connectivity_pairs: List[List[int]]) -> List[List[int]]:
        """Modify a random connection by changing one of its endpoints."""
        if not connectivity_pairs:
            return connectivity_pairs
        
        # Pick a random connection to modify
        random_index = random.randint(0, len(connectivity_pairs) - 1)
        pair = connectivity_pairs[random_index]
        
        if len(pair) == 2:
            # Randomly choose which endpoint to change
            endpoint_to_change = random.randint(0, 1)
            # Choose a new random agent
            new_agent = random.randint(0, self.num_agents - 1)
            # Make sure it's different from the other endpoint
            while new_agent == pair[1 - endpoint_to_change]:
                new_agent = random.randint(0, self.num_agents - 1)
            
            pair[endpoint_to_change] = new_agent
        
        return connectivity_pairs

    def _remove_duplicate_connections(self, connectivity_pairs: List[List[int]]) -> List[List[int]]:
        """Remove duplicate connections, treating [i,j] and [j,i] as the same connection."""
        if not connectivity_pairs:
            return connectivity_pairs
        
        # Normalize connections to canonical form [min, max] and track unique ones
        seen_connections = set()
        unique_pairs = []
        
        for pair in connectivity_pairs:
            if len(pair) == 2 and pair[0] != pair[1]:  # Valid pair, not self-connection
                # Normalize to [min, max] form
                normalized_pair = tuple(sorted([pair[0], pair[1]]))
                
                if normalized_pair not in seen_connections:
                    seen_connections.add(normalized_pair)
                    # Keep original order but ensure it's a valid connection
                    unique_pairs.append([normalized_pair[0], normalized_pair[1]])
        
        if len(unique_pairs) != len(connectivity_pairs):
            print(f"Removed {len(connectivity_pairs) - len(unique_pairs)} duplicate connections")
        
        return unique_pairs

    def _mutate_connectivity(self, connectivity_pairs: List[List[int]]) -> List[List[int]]:
        """Apply connectivity mutation to connectivity pairs."""
        if not self.mutate_connectivity or self.num_agents is None:
            return connectivity_pairs
            
        # First, remove any duplicate connections
        current_connectivity = self._remove_duplicate_connections(copy.deepcopy(connectivity_pairs))
        
        # Apply mutations based on mutation rate
        if random.random() < self.connectivity_mutation_rate:
            mutation_type = random.choice(['add', 'remove', 'modify'])
            
            if mutation_type == 'add':
                current_connectivity = self._add_random_connection(current_connectivity)
            elif mutation_type == 'remove':
                current_connectivity = self._remove_random_connection(current_connectivity)
            elif mutation_type == 'modify':
                current_connectivity = self._modify_random_connection(current_connectivity)
            
            # Remove duplicates again after mutation in case any were introduced
            current_connectivity = self._remove_duplicate_connections(current_connectivity)
            
            print(f"Connectivity mutated ({mutation_type}): {current_connectivity}")
        
        return current_connectivity

    # ----------------------------------------------------------------------

    def apply(self, individual: Individual) -> None:
        # Handle both string and dictionary DOFs
        if isinstance(individual.dofs, dict):
            # Dictionary DOFs - mutate prompt and optionally connectivity
            prompt = individual.get_prompt() if hasattr(individual, 'get_prompt') else individual.dofs.get("prompt", "")
            connectivity = individual.dofs.get("connectivity", [])
            
            # Mutate the prompt text
            new_text = self._mutate_via_llm_instance(prompt)
            if hasattr(individual, 'set_prompt'):
                individual.set_prompt(new_text)
            else:
                individual.dofs["prompt"] = new_text
            
            # Optionally mutate connectivity
            if self.mutate_connectivity:
                new_connectivity = self._mutate_connectivity(connectivity)
                if hasattr(individual, 'set_connectivity'):
                    individual.set_connectivity(new_connectivity)
                else:
                    individual.dofs["connectivity"] = new_connectivity
        else:
            # String DOFs - original behavior
            new_text = self._mutate_via_llm_instance(individual.dofs)
            individual.dofs = new_text

    def schedule(self, gen: int, n_generations: int):
        """Update mutation rates based on generation if scheduling is enabled."""
        if self._use_scheduling:
            # Decrease connectivity mutation rate over time
            self.connectivity_mutation_rate = max(
                0.01, 
                self._initial_connectivity_mutation_rate * (1 - gen / n_generations)
            )

