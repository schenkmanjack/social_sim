import json
import os
import argparse
from datetime import datetime
from social_sim.experiment import Experiment
from social_sim.simulation import Simulation
from social_sim.llm_interfaces import OpenAIBackend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def load_config(config_path: str) -> dict:
    """Load experiment configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_path}")
    
    # Set default values if not specified
    defaults = {
        "steps": 5,
        "num_simulations": 3,
        "results_folder": "experiment_results",
        "agent_type": "regular",
        "chunk_size": 1200,
        "plot_results": True
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a social simulation experiment')
    parser.add_argument('config_path', type=str, help='Path to experiment configuration JSON file')
    args = parser.parse_args()

    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Load configuration
    try:
        config = load_config(args.config_path)
    except ValueError as e:
        print(f"Error loading configuration: {e}")
        return

    # Validate required fields
    required_fields = ["query", "name"]
    for field in required_fields:
        if field not in config:
            print(f"Error: Required field '{field}' missing from configuration")
            return

    # Get configuration parameters
    agent_type = config.get("agent_type", "regular")
    chunk_size = config.get("chunk_size", 1200)
    num_simulations = config.get("num_simulations", 3)
    experiment_name = config.get("name", "experiment_default_name")
    results_folder = config.get("results_folder", 
                              f"results_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Create multiple simulations
    simulations = [
        Simulation(
            OpenAIBackend(api_key=api_key), 
            agent_type=agent_type, 
            chunk_size=chunk_size
        )
        for _ in range(num_simulations)
    ]

    # Create experiment with provided name
    experiment = Experiment(simulations, name=experiment_name)

    # Define outcomes to track
    if "outcomes" in config:
        for outcome in config["outcomes"]:
            experiment.define_outcome(
                name=outcome["name"],
                condition=outcome["condition"],
                description=outcome["description"]
            )
    else:
        raise ValueError("No outcomes defined in configuration. Please specify at least one outcome in the 'outcomes' field.")

    # Run experiment
    print(f"Running experiment '{experiment_name}' with query: '{config['query']}'")
    print(f"Number of simulations: {num_simulations}")
    print(f"Steps per simulation: {config['steps']}")
    
    try:
        # Run experiment with steps parameter
        results = []

        for result in experiment.run(
            query=config["query"],
            steps=config["steps"]
        ):
            if isinstance(result, tuple) and len(result) == 2:
                progress, data = result
                
                # Check if this is the final result with statistics
                if progress.get('percentage') == 100 and 'runs' in data:
                    results = data['runs']
                    break

        if not results:
            print("Warning: No results were generated from the experiment")
            return

        # Calculate statistics
        statistics = {}
        for outcome in config["outcomes"]:
            outcome_name = outcome["name"]
            count = sum(1 for sim in results if sim['outcome_analysis'].get(outcome_name, False))
            statistics[outcome_name] = {
                'count': count,
                'percentage': (count / len(results)) * 100,
                'description': outcome.get('description', '')
            }

        # Add statistics to each simulation result
        for result in results:
            result['statistics'] = statistics

        print("Experiment completed successfully.")
    except Exception as e:
        print(f"\n--- Experiment Error ---")
        print(f"An error occurred during experiment: {e}")
        print("Please check the query and simulation parameters.")
        return

    # Save results
    try:
        os.makedirs(results_folder, exist_ok=True)
        if results:
            # Save individual simulation traces
            for i, result in enumerate(results):
                sim_dir = os.path.join(results_folder, f"run_{i+1}")
                os.makedirs(sim_dir, exist_ok=True)
                with open(os.path.join(sim_dir, "trace.json"), 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved trace for run {i+1}")

            # Save overall experiment results
            print("Saving experiment results...")
            experiment.save_results(
                output_dir=results_folder,
                plot_results=config.get("plot_results", True),
                results=results
            )
            print(f"Results saved to {os.path.abspath(results_folder)}")
            if config.get("plot_results", True):
                print("Plot generated successfully")
            else:
                print("Plotting disabled")
    except Exception as e:
        print(f"Error saving results: {e}")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()

    # Print statistics
    print("\nExperiment Statistics:")
    if results and "statistics" in results[-1]:
        for outcome_name, stats in results[-1]["statistics"].items():
            print(f"\n{outcome_name}:")
            print(f"  Count: {stats['count']}/{num_simulations}")
            print(f"  Percentage: {stats['percentage']:.1f}%")
            print(f"  Description: {stats['description']}")

if __name__ == "__main__":
    main()