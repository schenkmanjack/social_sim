"""
Run a simulation in which each agent (id + prompt) is supplied
explicitly in the configuration file – i.e. the Orchestrator is
NOT asked to generate the setup.

Config JSON *must* contain:
{
  "name": "...",
  "steps": 10,
  "agents": [
      { "id": "agent_0", "prompt": "..." },
      { "id": "agent_1", "prompt": "..." },
      ...
  ],

  // optional
  "connectivity": {
      "agent_0": {
          "visible_facts": [0, 1, 2],
          "neighbors": ["agent_1"]
      },
      ...
  }
}

Everything else (chunk_size, plot_results, outcomes …) follows the
same conventions as example/run_evolution.py
"""
import argparse
import json
import os
from datetime import datetime

from social_sim.agents.agent import Agent
from social_sim.interactions.connectivity import ConnectivityGraph
from social_sim.llm_interfaces import OpenAIBackend, AnthropicBackend
from social_sim.simulation import Simulation
from social_sim.experiment import Experiment

# --------------------------------------------------------------------------- #
# helper functions
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    """Load & *strictly* validate a manual-setup configuration file"""
    with open(path) as f:
        cfg = json.load(f)

    required = ["name", "steps", "agents", "connectivity"]
    for field in required:
        if field not in cfg:
            raise ValueError(f"Missing required field '{field}' in config")

    if not isinstance(cfg["agents"], list) or not cfg["agents"]:
        raise ValueError("'agents' must be a non-empty list")
    for ag in cfg["agents"]:
        if "id" not in ag or "prompt" not in ag:
            raise ValueError("Every agent entry needs 'id' and 'prompt'")

    # quick structure check for connectivity
    if not isinstance(cfg["connectivity"], dict):
        raise ValueError("'connectivity' must be an object mapping agent_id → {visible_facts, neighbors}")
    for aid, entry in cfg["connectivity"].items():
        if not {"visible_facts", "neighbors"} <= entry.keys():
            raise ValueError(f"Connectivity for '{aid}' must include visible_facts and neighbors lists")

    cfg.setdefault("chunk_size", 1200)
    cfg.setdefault("results_folder", "manual_results")
    cfg.setdefault("plot_results", True)

    return cfg


def build_connectivity(cfg: dict) -> ConnectivityGraph:
    """
    Simply wrap the user-supplied connectivity definition.
    (No auto-generation fallback any more.)
    """
    return ConnectivityGraph(cfg["connectivity"])


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual social-sim experiment")
    parser.add_argument("config_path", type=str, help="Path to JSON config")
    args = parser.parse_args()

    # LLM backend
    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     raise EnvironmentError("Set the OPENAI_API_KEY environment variable first")
    # llm = OpenAIBackend(api_key=api_key)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the ANTHROPIC_API_KEY environment variable first")
    llm = AnthropicBackend(api_key=api_key)

    # Create simulation and set it up from config
    sim = Simulation(
        llm_wrapper=llm,
        chunk_size=1200,  # Default, can be overridden by config
    )
    
    # Set up from config file - this does all the parsing and validation
    sim.setup_from_config(args.config_path)
    
    # Get config values for experiment setup
    config = sim.config
    
    # Create experiment
    experiment = Experiment([sim], name=config["name"], debug=False)
    
    # # Add outcomes if specified
    # for outcome in config.get("outcomes", []):
    #     experiment.define_outcome(
    #         name=outcome["name"],
    #         condition=outcome["condition"],
    #         description=outcome["description"],
    #     )
    
    # Run experiment
    print(f"Running manual experiment '{config['name']}' for {config['steps']} steps…")
    runs = []
    
    for progress, data in experiment.run_manual(
        steps=config["steps"], 
        time_scale=config.get("time_scale"),
        use_batched=True
    ):
    # for progress, data in experiment.run_manual_batch(
    #     steps=config["steps"], 
    #     time_scale=config.get("time_scale")
    # ):
        if progress.get("percentage") == 100 and "runs" in data:
            runs = data["runs"]
            break

    # Save results
    if runs:
        out_dir = config.get("results_folder", f"manual_results_{config['name']}_{datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(out_dir, exist_ok=True)
        experiment.save_results(
            output_dir=out_dir, 
            plot_results=config.get("plot_results", True), 
            results=runs
        )
        print("Results saved to", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
