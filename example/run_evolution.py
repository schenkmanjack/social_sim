import json
import os
import argparse
from datetime import datetime
from social_sim.agents.agent import Agent
from social_sim.interactions.connectivity import ConnectivityGraph
from social_sim.llm_interfaces import OpenAIBackend
from social_sim.experiment import Experiment
from social_sim.simulation import Simulation

def load_config(config_path: str) -> dict:
    """Load evolution configuration from JSON file"""
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
        "steps": 10,
        "results_folder": "evolution_results",
        "chunk_size": 1200
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config

def create_connectivity_graph(num_agents, connectivity_graph=None):
    """Create a connectivity graph based on custom configuration or default to full connectivity
    
    Args:
        num_agents: Number of agents in the system
        connectivity_graph: Optional custom connectivity configuration
    """
    graph = {}
    
    if connectivity_graph is None:
        # Default to full connectivity
        for i in range(num_agents):
            graph[f"agent_{i}"] = {
                "visible_facts": list(range(100)),  # Assuming max 100 facts
                "neighbors": [f"agent_{j}" for j in range(num_agents) if j != i]
            }
    else:
        # Use custom connectivity configuration
        for agent_id, connections in connectivity_graph.items():
            if not isinstance(connections, dict) or "visible_facts" not in connections or "neighbors" not in connections:
                raise ValueError(f"Invalid connectivity configuration for agent {agent_id}")
            
            graph[agent_id] = {
                "visible_facts": connections["visible_facts"],
                "neighbors": connections["neighbors"]
            }
    
    return ConnectivityGraph(graph)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a social evolution simulation')
    parser.add_argument('config_path', type=str, help='Path to evolution configuration JSON file')
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
    required_fields = ["agent_prompt", "name"]
    for field in required_fields:
        if field not in config:
            print(f"Error: Required field '{field}' missing from configuration")
            return

    # Initialize LLM backend
    llm = OpenAIBackend(api_key=api_key)

    # Create simulation
    simulation = Simulation(
        llm,
        agent_type=config.get("agent_type", "regular"),
        chunk_size=config.get("chunk_size", 1200)
    )

    # Create experiment
    experiment = Experiment([simulation], name=config["name"])

    # Define outcomes if specified in config
    if "outcomes" in config:
        for outcome in config["outcomes"]:
            experiment.define_outcome(
                name=outcome["name"],
                condition=outcome["condition"],
                description=outcome["description"]
            )

    # Run experiment
    print(f"Running evolution '{config['name']}'")
    print(f"Steps: {config['steps']}")

    try:
        results = []
        for result in experiment.run(
            query=config["agent_prompt"],
            steps=config["steps"]
        ):
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
        
        # Save experiment results
        experiment.save_results(
            output_dir=results_folder,
            plot_results=config.get("plot_results", True),
            results=results
        )
        
        print(f"Results saved to {os.path.abspath(results_folder)}")

        # Print statistics
        print("\nExperiment Statistics:")
        if results and "statistics" in results[-1]:
            for outcome_name, stats in results[-1]["statistics"].items():
                print(f"\n{outcome_name}:")
                print(f"  Count: {stats['count']}/1")  # Since we're running one simulation
                print(f"  Percentage: {stats['percentage']:.1f}%")
                print(f"  Description: {stats['description']}")

    except Exception as e:
        print(f"\n--- Experiment Error ---")
        print(f"An error occurred during experiment: {e}")
        print("Please check the configuration and simulation parameters.")
        return

if __name__ == "__main__":
    main()
