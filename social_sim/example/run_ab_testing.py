import json
import os
import argparse
from datetime import datetime
from social_sim.llm_interfaces import OpenAIBackend
from social_sim.simulation.simulation import Simulation
from social_sim.experiment import Experiment

def load_config(config_path: str) -> dict:
    """Load A/B testing configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_path}")
    
    # Set default values if not specified
    defaults = {
        "num_agents": 5,
        "results_folder": "ab_testing_results",
        "steps": 1,
        "chunk_size": 1200
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run an A/B testing simulation')
    parser.add_argument('config_path', type=str, help='Path to A/B testing configuration JSON file')
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
    required_fields = ["query", "name", "agent_outcome_definitions"]
    for field in required_fields:
        if field not in config:
            print(f"Error: Required field '{field}' missing from configuration")
            return

    # Initialize LLM backend
    llm = OpenAIBackend(api_key=api_key)

    # Initialize Simulation
    simulation = Simulation(
        llm_wrapper=llm,
        agent_type=config.get("agent_type", "regular"),
        chunk_size=config.get("chunk_size", 1200),
        agent_outcome_definitions=config.get("agent_outcome_definitions", {})
    )

    # Create Experiment
    experiment = Experiment([simulation], name=config["name"])

    # Define outcomes
    for outcome in config["agent_outcome_definitions"]:
        if isinstance(outcome, dict):
            experiment.define_outcome(
                name=outcome["name"],
                condition=outcome["condition"],
                description=outcome["description"]
            )
        else:
            print(f"Warning: Outcome definition is not a dictionary: {outcome}")

    # Run experiment
    print(f"Running A/B testing '{config['name']}'")
    try:
        results = []
        for result in experiment.run(query=config["query"], steps=config.get("steps", 1)):
            if isinstance(result, tuple) and len(result) == 2:
                progress, data = result
                if progress.get('percentage') == 100 and 'runs' in data:
                    results = data['runs']
                    break

        if not results:
            print("Warning: No results were generated from the experiment")
            return

        # Save results
        results_folder = config.get("results_folder", 
                                    f"results_{config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(results_folder, exist_ok=True)
        
        experiment.save_results(
            output_dir=results_folder,
            plot_results=config.get("plot_results", True),
            results=results
        )
        
        print(f"Results saved to {os.path.abspath(results_folder)}")

    except Exception as e:
        print(f"\n--- Experiment Error ---")
        print(f"An error occurred during experiment: {e}")
        print("Please check the configuration and simulation parameters.")
        return

if __name__ == "__main__":
    main()
