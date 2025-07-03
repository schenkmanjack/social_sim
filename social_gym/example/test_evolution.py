import numpy as np
import matplotlib.pyplot as plt
import time
from genetic_algorithm.ga.genetic_algorithm_base import GeneticAlgorithmBase
from genetic_algorithm.operators.selection import TournamentSelection, NSGA2Selection, QDNeighborTournamentSelection
from social_gym.operators import LLMTextCrossover, LLMTextMutation
from genetic_algorithm.individual import IndividualObject
from social_gym.evaluators import RedBlueEvaluator
from social_gym.evaluators.state_saver import RedBlueStateSaver
from social_sim.simulation import Simulation
from social_sim.llm_interfaces import OpenAIBackend, AnthropicBackend
import os
from pymongo import MongoClient

"""Here is a program for training a genetic algorithm to optimize prompts and connectivity for the RedBlue coordination task."""

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
    llm = AnthropicBackend(api_key=api_key, model="claude-3-5-haiku-20241022", max_output_tokens=12) #model="claude-3-haiku-20240307")

    # Configure which objectives to use
    # Objectives: [split_deviation, neither_fraction, prompt_length, connection_count, failure_count, llm_usage]
    use_objectives = [True, True, True, True, True, False]  # Use all 6 objectives
    
    # Create corresponding labels for selected objectives
    all_objective_names = ["Split Deviation", "Neither Fraction", "Prompt Length", "Connection Count", "Failure Count", "LLM Usage"]
    selected_objective_labels = [name for i, name in enumerate(all_objective_names) if use_objectives[i]]
    
    print(f"Using objectives: {selected_objective_labels}")

    # RedBlue evaluator configuration
    evaluator_config = dict(
        llm_wrapper=llm,
        num_agents=6,
        steps=16,
        connectivity=None, #[[0,1], [2,3], [4,5]],
        connectivity_pattern='random',
        connectivity_density=0.3,
        use_batched_evaluation=True,
        n_eval_duplicates=3,
        disable_summary=True,
        disable_batch_summaries=True,  # Temporarily disable this to debug
        use_objectives=use_objectives,  # New parameter to control which objectives to use
        use_full_agent_memory=False,  # Control whether agents use full memory in prompts (default: True)
        debug=False,  # Set to True to enable debug printing for agent outcome cleaning
        initial_prompt="Choose a color red or blue to achieve a population of half red and half blue. Your response should only be the color.",#"You are an agent who identifies as red or blue. You communicate with your neighbors. The goal is for half of the agents to be red and half to be blue at the end of the simulation. At the end of the simulation you must say what color you are. You must identify as one color and state the color. Provide any answer of either red or blue.",
        results_file="evaluation_results.json"
    )
    
    # Create evaluator with LLM wrapper
    evaluator = RedBlueEvaluator(
        evaluator_config=evaluator_config,
        use_scheduling=True
    )
    
    # Create individual template - now using IndividualObject for dictionary DOFs
    individual_template = IndividualObject(evaluator)
    
    # Genetic algorithm parameters
    population_size = 32
    n_generations = 300 #
    initial_mutation_rate = 1.0 #0.96
    initial_crossover_rate = 0.0
    initial_tournament_size = 3
    epsilon = 0.0
    log_freq = 1
    save_freq = 1#100
    plot_freq = 1
    reseed_freq = 50

    
    # Create genetic algorithm operators with LLM wrapper
    selection = TournamentSelection(tournament_size=initial_tournament_size, use_scheduling=True)
    llm_operator = AnthropicBackend(api_key=api_key, model="claude-3-5-haiku-20241022", max_output_tokens=80)
    crossover = LLMTextCrossover(llm_wrapper=llm_operator, use_scheduling=True)
    mutation = LLMTextMutation(
        llm_wrapper=llm_operator, 
        use_scheduling=True,
        mutate_connectivity=True,
        connectivity_mutation_rate=0.9,
        num_agents=evaluator_config['num_agents'],
        mutation_context=(
            "The goal is for half of the agents to be red and half of the agents to be blue.\n"
            "At the end of the simulation an LLM will inspect the agent to see which color it chose.\n"
            "The agent must make its choice clear.\n"
            "This is a multi-objective optimization problem.\n"
            "The goal is to minimize the following objectives:\n"
            "split_deviation: the difference from an ideal 50-50 split of half of the agents being red and half being blue \n"
            "neither_fraction: the fraction of agents that chose neither color (including agents that failed to choose a color) \n"
            "prompt_length: the length of the prompt \n"
            "connection_count: the number of connections the agent has to other agents \n"
            "failure_count: the number of times that the simulation which the agent is part of fails \n"
            "llm_usage: the total number of characters used in LLM calls during the simulation \n"
        )
    )
    
    # MongoDB configuration - try different connection options
    mongo_uri = "mongodb+srv://schenkmanjack:$occeRZeus1999@mongodb-base.9zfdshi.mongodb.net/?retryWrites=true&w=majority&appName=mongodb-base"
    experiment_id = "test_redblue_connectivity"
    
    # Test MongoDB connection with proper error handling
    print("Testing MongoDB connection...")
    try:
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=10000,  # 10 second timeout
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
            retryWrites=True,
            tlsAllowInvalidCertificates=True
        )
        # Try to ping the database
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        use_mongodb = True
        client.close()
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("Proceeding without MongoDB logging...")
        use_mongodb = False
    
    # Create state saver for JSON logging
    state_saver = RedBlueStateSaver(
        results_file=f"{experiment_id}_generation_state.json",
        num_agents=evaluator_config['num_agents'],
        steps=evaluator_config['steps'],
        connectivity_pattern=evaluator_config['connectivity_pattern'],
        objective_labels=selected_objective_labels,  # Use dynamically selected labels
        metric_labels=selected_objective_labels      # Use dynamically selected labels
    )
    
    # Create genetic algorithm with MongoDB support and state saver
    ga = GeneticAlgorithmBase(
        population_size=population_size,
        individual_template=individual_template,
        mutation_rate=initial_mutation_rate,
        crossover_rate=initial_crossover_rate,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        epsilon=epsilon,
        use_multiprocessing=False,
        initialize_dofs=False,
        mongo_uri=mongo_uri if use_mongodb else None,
        experiment_id=experiment_id,
        load_experiment=False,
        log_mongodb=use_mongodb,
        load_experiment_id="test_redblue_connectivity" if use_mongodb else None,
        mutation_rate_scheduler=MutationRateScheduler,
        crossover_rate_scheduler=CrossoverRateScheduler,
        save_all_elite_dofs=True,
        state_save_class=state_saver
    )
    
    # Run optimization
    start_time = time.time()
    try:
        ga.optimize(
            n_generations, 
            log_freq=log_freq, 
            save_freq=save_freq,
            plot_freq=plot_freq,
            reseed_freq=reseed_freq,
            chunk_size=25,
            objective_labels=selected_objective_labels,  # Use dynamically selected labels
            metric_labels=selected_objective_labels,     # Use dynamically selected labels
            plot_name="redblue_connectivity_optimization_results"  # Updated plot name
        )
    except Exception as e:
        print(f"GA optimization error: {e}")
    
    end_time = time.time()
    print(f"\nTotal optimization time: {end_time - start_time:.2f} seconds")
    
    # Print token usage statistics
    # print("\n" + "="*50)
    # print("TOKEN USAGE SUMMARY")
    # print("="*50)
    # llm.print_token_usage()
    
    # # Also print usage for operators that use LLM
    # print("\nCrossover operator token usage:")
    # if hasattr(crossover, 'llm_wrapper'):
    #     crossover.llm_wrapper.print_token_usage()
    
    # print("\nMutation operator token usage:")
    # if hasattr(mutation, 'llm_wrapper'):
    #     mutation.llm_wrapper.print_token_usage()
    
    # Plot final results
    # ga._plot_optimization_results(n_generations - 1, log_freq)
    # ga._plot_timing()

if __name__ == "__main__":
    main() 