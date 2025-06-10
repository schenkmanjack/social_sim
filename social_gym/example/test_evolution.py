import numpy as np
import matplotlib.pyplot as plt
import time
from genetic_algorithm.ga.genetic_algorithm_base import GeneticAlgorithmBase
from genetic_algorithm.operators.selection import TournamentSelection, NSGA2Selection, QDNeighborTournamentSelection
from social_gym.operators import LLMTextCrossover, LLMTextMutation
from genetic_algorithm.individual import IndividualString
from social_gym.evaluators import RedBlueEvaluator
from social_sim.simulation import Simulation
from social_sim.llm_interfaces import OpenAIBackend, AnthropicBackend
import os

"""Here is a program for training a genetic algorithm to optimize a neural network to classify MNIST digits."""

class MutationRateScheduler:
    def __init__(self, initial_mutation_rate):
        self.initial_mutation_rate = initial_mutation_rate
    
    def __call__(self, gen, n_generations):
        return max(0.1, self.initial_mutation_rate * (1 - gen/n_generations))

class CrossoverRateScheduler:
    def __init__(self, initial_crossover_rate):
        self.initial_crossover_rate = initial_crossover_rate
    
    def __call__(self, gen, n_generations):
        return max(0.1, self.initial_crossover_rate * (1 - gen/n_generations))

def main():
    # Set up LLM backend
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the ANTHROPIC_API_KEY environment variable first")
    llm = AnthropicBackend(api_key=api_key, model="claude-3-haiku-20240307")

    # RedBlue evaluator configuration
    evaluator_config = dict(
        llm_wrapper=llm,
        num_agents=20,
        steps=3,
        connectivity=None, #[[0,1], [2,3], [4,5]],
        connectivity_pattern='adjacent',
        use_batched_evaluation=True,
        initial_prompt="You are an agent who identifies as red or blue. Each step represents a simultaneous decision round. You communicate with your neighbors. The goal is for half of the agents to be red and half to be blue at the end of the simulation. At the end of the simulation you must say what color you are. You must identify as one color and state the color. Provide any answer of either red or blue.",
        results_file="evaluation_results.json"
    )
    
    # Create evaluator with LLM wrapper
    evaluator = RedBlueEvaluator(
        evaluator_config=evaluator_config,
        use_scheduling=True
    )
    
    # Create individual template
    individual_template = IndividualString(evaluator)
    
    # Genetic algorithm parameters
    population_size = 4
    n_generations = 30
    initial_mutation_rate = 0.84
    initial_crossover_rate = 0.2
    initial_tournament_size = 2
    log_freq = 1
    save_freq = 1#100
    plot_freq = 1
    reseed_freq = 50
    
    # Create genetic algorithm operators with LLM wrapper
    selection = TournamentSelection(tournament_size=initial_tournament_size, use_scheduling=True)
    crossover = LLMTextCrossover(llm_wrapper=llm, use_scheduling=True)
    mutation = LLMTextMutation(llm_wrapper=llm, use_scheduling=True)
    
    # MongoDB configuration
    mongo_uri = "mongodb+srv://schenkmanjack:$occeRZeus1999@mongodb-base.9zfdshi.mongodb.net/?retryWrites=true&w=majority&appName=mongodb-base"
    experiment_id = "test_redblue"  # Changed experiment ID
    
    # Create genetic algorithm with MongoDB support
    ga = GeneticAlgorithmBase(
        population_size=population_size,
        individual_template=individual_template,
        mutation_rate=initial_mutation_rate,
        crossover_rate=initial_crossover_rate,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        use_multiprocessing=False,
        initialize_dofs=False,
        mongo_uri=mongo_uri,
        experiment_id=experiment_id,
        load_experiment=False,
        log_mongodb=True,
        load_experiment_id="test_redblue",  # Changed to match experiment_id
        mutation_rate_scheduler=MutationRateScheduler,
        crossover_rate_scheduler=CrossoverRateScheduler
    )
    
    # Run optimization
    start_time = time.time()
    ga.optimize(
        n_generations, 
        log_freq=log_freq, 
        save_freq=save_freq,
        plot_freq=plot_freq,
        reseed_freq=reseed_freq,
        chunk_size=25,
        objective_labels=["Split Deviation", "Neither Fraction"],  # Updated labels
        metric_labels=["Split Deviation", "Neither Fraction"],     # Updated labels
        plot_name="redblue_optimization_results"  # Updated plot name
    )
    end_time = time.time()
    print(f"\nTotal optimization time: {end_time - start_time:.2f} seconds")
    
    # Plot final results
    # ga._plot_optimization_results(n_generations - 1, log_freq)
    # ga._plot_timing()

if __name__ == "__main__":
    main() 