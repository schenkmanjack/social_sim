import copy
from typing import List
from genetic_algorithm.individual import Individual
from genetic_algorithm.operators import CrossoverOperator, MutationOperator

import random

class LLMTextCrossover(CrossoverOperator):
    """
    Blend the textual DNA of two parent prompts by asking an LLM
    to synthesise a new prompt that inherits constraints from both.
    """

    def __init__(
        self,
        llm_wrapper,
        use_scheduling: bool = False,
        temperature: float = 0.8
    ):
        self._use_scheduling = use_scheduling
        self.llm = llm_wrapper
        self.temperature = temperature

    # ----- API helpers -----------------------------------------------------

    def _call_llm(self, prompt) -> str:
        return self.llm.generate(prompt)

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
            
            prompt = (
                "You are an expert prompt engineer. You will be given two source "
                "prompts. Produce a NEW prompt that:\n"
                " • Preserves any hard constraints mentioned in EITHER parent.\n"
                " • Combines their useful wording and stylistic hints.\n"
                " • Remains concise (≤ 200 tokens).\n"
                "Return ONLY the new prompt.\n\n"
                f"--- Parent A ---\n{prompt1}\n"
                f"--- Parent B ---\n{prompt2}\n"
                "--- End ---"
            )

            children = [child1, child2]
            for child in children:
                new_text = self._call_llm(prompt)
                # Update only the prompt part, keep connectivity unchanged
                if hasattr(child, 'set_prompt'):
                    child.set_prompt(new_text)
                else:
                    child.dofs["prompt"] = new_text
        else:
            # String DOFs - original behavior
            prompt = (
                "You are an expert prompt engineer. You will be given two source "
                "prompts. Produce a NEW prompt that:\n"
                " • Preserves any hard constraints mentioned in EITHER parent.\n"
                " • Combines their useful wording and stylistic hints.\n"
                " • Remains concise (≤ 200 tokens).\n"
                "Return ONLY the new prompt.\n\n"
                f"--- Parent A ---\n{parent1.dofs}\n"
                f"--- Parent B ---\n{parent2.dofs}\n"
                "--- End ---"
            )

            children = [child1, child2]
            for child in children:
                new_text = self._call_llm(prompt)
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
        num_agents: int = None
    ):
        self._use_scheduling = use_scheduling
        self.llm = llm_wrapper
        self.temperature = temperature
        
        # Connectivity mutation parameters
        self.mutate_connectivity = mutate_connectivity
        self.connectivity_mutation_rate = connectivity_mutation_rate
        self._initial_connectivity_mutation_rate = connectivity_mutation_rate
        self.num_agents = num_agents
        

    @classmethod
    def _mutate_via_llm(cls, llm_wrapper, prompt_text: str) -> str:
        prompt = (
            "You are a creative but precise prompt-rewriter.\n"
            "Given a prompt, produce a *variation* that:\n"
            " • Keeps the same overall task and constraints.\n"
            " • Uses different wording, synonyms, or sentence order.\n"
            " • Optionally adds ONE helpful hint to improve coordination.\n"
            "Return ONLY the mutated prompt.\n\n"
            f"Here is the original prompt:\n{prompt_text}"
        )
        return llm_wrapper.generate(prompt)

    @classmethod
    def _mutate_via_llm_batch(cls, llm_wrapper, prompt_texts: List[str]) -> List[str]:
        """Mutate multiple prompts in a single LLM call for efficiency."""
        if not prompt_texts:
            return []
        
        if len(prompt_texts) == 1:
            # Single prompt - use regular method
            return [cls._mutate_via_llm(llm_wrapper, prompt_texts[0])]
        
        # Create batch prompt
        batch_prompt = (
            "You are a creative but precise prompt-rewriter.\n"
            "You will be given multiple prompts to mutate. For each prompt, produce a *variation* that:\n"
            " • Keeps the same overall task and constraints.\n"
            " • Uses different wording, synonyms, or sentence order.\n"
            " • Optionally adds ONE helpful hint to improve coordination.\n"
            "Format your response as follows:\n"
            "MUTATION_1: [first mutated prompt]\n"
            "MUTATION_2: [second mutated prompt]\n"
            "... and so on.\n\n"
            "Here are the original prompts:\n"
        )
        
        for i, prompt_text in enumerate(prompt_texts, 1):
            batch_prompt += f"PROMPT_{i}: {prompt_text}\n\n"
        
        batch_prompt += "Now provide the mutations:"
        
        # Get batch response
        batch_response = llm_wrapper.generate(batch_prompt)
        
        # Parse the response to extract individual mutations
        mutations = []
        lines = batch_response.split('\n')
        
        for i in range(1, len(prompt_texts) + 1):
            mutation_found = False
            mutation_prefix = f"MUTATION_{i}:"
            
            for line in lines:
                if line.strip().startswith(mutation_prefix):
                    mutation = line.strip()[len(mutation_prefix):].strip()
                    mutations.append(mutation)
                    mutation_found = True
                    break
            
            if not mutation_found:
                # Fallback: use original method for this prompt
                mutations.append(cls._mutate_via_llm(llm_wrapper, prompt_texts[i-1]))
        
        # Ensure we have the right number of mutations
        while len(mutations) < len(prompt_texts):
            # Fallback for any missing mutations
            missing_idx = len(mutations)
            mutations.append(cls._mutate_via_llm(llm_wrapper, prompt_texts[missing_idx]))
        
        return mutations[:len(prompt_texts)]  # Trim to exact length needed

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
            new_text = LLMTextMutation._mutate_via_llm(self.llm, prompt)
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
            new_text = LLMTextMutation._mutate_via_llm(self.llm, individual.dofs)
            individual.dofs = new_text

    def apply_batch(self, individuals: List[Individual]) -> None:
        """Apply mutation to multiple individuals simultaneously using batched LLM calls."""
        if not individuals:
            return
        
        # Separate individuals by DOF type (string vs dict)
        dict_individuals = []
        string_individuals = []
        
        for individual in individuals:
            if isinstance(individual.dofs, dict):
                dict_individuals.append(individual)
            else:
                string_individuals.append(individual)
        
        # Handle dictionary DOF individuals
        if dict_individuals:
            # Extract prompts for batching
            prompts = []
            for individual in dict_individuals:
                prompt = individual.get_prompt() if hasattr(individual, 'get_prompt') else individual.dofs.get("prompt", "")
                prompts.append(prompt)
            
            # Batch mutate prompts
            mutated_prompts = LLMTextMutation._mutate_via_llm_batch(self.llm, prompts)
            
            # Apply mutations to individuals
            for individual, new_prompt in zip(dict_individuals, mutated_prompts):
                # Update prompt
                if hasattr(individual, 'set_prompt'):
                    individual.set_prompt(new_prompt)
                else:
                    individual.dofs["prompt"] = new_prompt
                
                # Optionally mutate connectivity
                if self.mutate_connectivity:
                    connectivity = individual.dofs.get("connectivity", [])
                    new_connectivity = self._mutate_connectivity(connectivity)
                    if hasattr(individual, 'set_connectivity'):
                        individual.set_connectivity(new_connectivity)
                    else:
                        individual.dofs["connectivity"] = new_connectivity
        
        # Handle string DOF individuals
        if string_individuals:
            # Extract prompt strings for batching
            prompt_strings = [str(individual.dofs) for individual in string_individuals]
            
            # Batch mutate prompts
            mutated_strings = LLMTextMutation._mutate_via_llm_batch(self.llm, prompt_strings)
            
            # Apply mutations to individuals
            for individual, new_string in zip(string_individuals, mutated_strings):
                individual.dofs = new_string

    def schedule(self, gen: int, n_generations: int):
        """Update mutation rates based on generation if scheduling is enabled."""
        if self._use_scheduling:
            # Decrease connectivity mutation rate over time
            self.connectivity_mutation_rate = max(
                0.01, 
                self._initial_connectivity_mutation_rate * (1 - gen / n_generations)
            )

