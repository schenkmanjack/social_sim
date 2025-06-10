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
    rewrite / augment the existing one.
    """

    def __init__(
        self,
        llm_wrapper,
        use_scheduling: bool = False,
        temperature: float = 0.9
    ):
        self._use_scheduling = use_scheduling
        self.llm = llm_wrapper
        self.temperature = temperature
        

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

    # ----------------------------------------------------------------------

    def apply(self, individual: Individual) -> None:
        new_text          = LLMTextMutation._mutate_via_llm(self.llm, individual.dofs)
        individual.dofs = new_text

    def schedule(self, gen: int, n_generations: int):
        pass

