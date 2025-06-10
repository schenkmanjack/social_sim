import os
import json
import tempfile
from datetime import datetime
import numpy as np

from social_sim.agents.agent import Agent
from social_sim.interactions.connectivity import ConnectivityGraph
from social_sim.simulation import Simulation
from social_sim.experiment import Experiment
from social_gym.operators import LLMTextMutation

class RedBlueEvaluator:
    def __init__(self, evaluator_config=dict(), use_scheduling=False):
        self.evaluator_config = evaluator_config
        self.llm = evaluator_config.get('llm_wrapper')
        self._initial_evaluator_config = evaluator_config.copy()
        self._use_scheduling = use_scheduling
        self.use_batched_evaluation = evaluator_config.get('use_batched_evaluation', False)
        
        # Get required config parameters
        self.num_agents = evaluator_config.get('num_agents', 7)
        self.steps = evaluator_config.get('steps', 5)
        self.connectivity_pairs = evaluator_config.get('connectivity', None)
        self.connectivity_pattern = evaluator_config.get('connectivity_pattern', 'adjacent')
        if self.connectivity_pairs is None:
            if self.connectivity_pattern == 'adjacent':
                self.connectivity_pairs = [[i, i+1] for i in range(self.num_agents-1)]
            elif self.connectivity_pattern == 'all':
                self.connectivity_pairs = [[i, j] for i in range(self.num_agents) for j in range(self.num_agents) if i != j]
            else:
                raise ValueError(f"Invalid evaluator pattern: {self.evaluator_pattern}")
            
            print(f"Connectivity pairs: {self.connectivity_pairs}")
            print(f"RedBlueEvaluator initialized with {self.num_agents} agents, {self.steps} steps")
        
        # Initialize JSON file for saving results
        self.results_file = evaluator_config.get('results_file', 'evaluation_results.json')
        self._ensure_results_file()

    def _ensure_results_file(self):
        """Ensure the results file exists and is properly initialized"""
        if not os.path.exists(self.results_file):
            initial_data = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "num_agents": self.num_agents,
                    "steps": self.steps,
                    "connectivity_pattern": self.connectivity_pattern
                },
                "evaluations": []
            }
            with open(self.results_file, 'w') as f:
                json.dump(initial_data, f, indent=2)

    def save_evaluation_results(self, prompt, objectives, constraints=None, metrics=None, additional_info=None):
        """Save evaluation results to JSON file"""
        try:
            # Load existing data
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            # Create new evaluation entry
            evaluation_entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "objectives": objectives,
                "constraints": constraints or [],
                "metrics": metrics or [],
                "additional_info": additional_info or {}
            }
            
            # Add to evaluations list
            data["evaluations"].append(evaluation_entry)
            
            # Save back to file
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Saved evaluation result to {self.results_file}")
            
        except Exception as e:
            print(f"Error saving evaluation results: {str(e)}")

    def save_batch_evaluation_results(self, prompts, all_objectives, all_constraints=None, all_metrics=None, additional_info=None):
        """Save batch evaluation results to JSON file"""
        try:
            # Load existing data
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            # Create batch evaluation entries
            batch_timestamp = datetime.now().isoformat()
            for i, (prompt, objectives) in enumerate(zip(prompts, all_objectives)):
                constraints = all_constraints[i] if all_constraints else []
                metrics = all_metrics[i] if all_metrics else []
                
                evaluation_entry = {
                    "timestamp": batch_timestamp,
                    "batch_index": i,
                    "prompt": prompt,
                    "objectives": objectives,
                    "constraints": constraints,
                    "metrics": metrics,
                    "additional_info": additional_info or {}
                }
                
                data["evaluations"].append(evaluation_entry)
            
            # Save back to file
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Saved {len(prompts)} batch evaluation results to {self.results_file}")
            
        except Exception as e:
            print(f"Error saving batch evaluation results: {str(e)}")

    def get_bounds(self):
        # Return bounds for string DOF - not really applicable but needed for interface
        pass
    
    def get_dofs(self):
        # Return number of DOFs - for string this is more conceptual
        return 100  # Placeholder
    
    def initialize_dofs(self):
        """Initialize with a default prompt from evaluator_config"""
        initial_prompt = self.evaluator_config.get('initial_prompt', 
            "You are an agent who identifies as red or blue. Each step represents a simultaneous decision round. You communicate with your neighbors. The goal is for half of the agents to be red and half to be blue at the end of the simulation. At the end of the simulation you must say what color you are. You must identify as one color and state the color. Provide any answer of either red or blue.")
        initial_prompt = LLMTextMutation._mutate_via_llm(self.llm, initial_prompt)
        return initial_prompt

    def create_config_from_dofs(self, prompt_dof, config_name="eval_config"):
        """Create simulation config from DOF (prompt string)"""
        
        # Create agents list - all agents get the same prompt
        agents = []
        for i in range(self.num_agents):
            agents.append({
                "id": f"agent_{i}",
                "prompt": prompt_dof
            })
        
        # Create connectivity from pairs
        connectivity = {}
        for i in range(self.num_agents):
            connectivity[f"agent_{i}"] = {
                "visible_facts": [],
                "neighbors": []
            }
        
        # Add bidirectional connections
        for pair in self.connectivity_pairs:
            if len(pair) == 2 and 0 <= pair[0] < self.num_agents and 0 <= pair[1] < self.num_agents:
                agent_0 = f"agent_{pair[0]}"
                agent_1 = f"agent_{pair[1]}"
                connectivity[agent_0]["neighbors"].append(agent_1)
                connectivity[agent_1]["neighbors"].append(agent_0)
        
        # Create full config
        config = {
            "name": config_name,
            "steps": self.steps,
            "agent_type": "timescale_aware",
            "chunk_size": 1200,
            "results_folder": f"temp_results_{config_name}",
            "plot_results": False,
            "agents": agents,
            "connectivity": connectivity,
            "outcomes": [],
            "agent_outcome_definitions": {
                "red": "Only consider the final piece of the agents memory. The agent has to make a choice. It cannot be two colors. Does the agent choose red? For the agent to be red, it must state it unambiguously.",
                "blue": "Only consider the final piece of the agents memory. The agent has to make a choice. It cannot be two colors. Does the agent choose blue? For the agent to be blue, it must state it unambiguously."
            }
        }
        
        return config

    def run_simulation(self, config):
        """Run a single simulation with the given config"""
        
        # Create simulation
        sim = Simulation(
            llm_wrapper=self.llm,
            chunk_size=config.get("chunk_size", 1200),
        )
        
        # Set up from config dict directly - no temp file needed
        sim.setup_from_config(config)
        
        # Get config values for experiment setup
        config = sim.config
        
        # Create experiment
        experiment = Experiment([sim], name=config["name"])
        
        # Run experiment
        runs = []
        for progress, data in experiment.run_manual(
            steps=config["steps"], 
            time_scale=config.get("time_scale")
        ):
            if progress.get("percentage") == 100 and "runs" in data:
                runs = data["runs"]
                break
        
        self.experiment  = experiment #return runs[0] if runs else None

    def evaluate_simulation_outcomes(self, simulation=None):
        """Evaluate simulation outcomes and return objectives"""
        if simulation is None:
            raise ValueError("Simulation is required for evaluation")
        agent_outcomes = simulation.agent_outcomes
        red_agents = agent_outcomes.get('red', [])
        blue_agents = agent_outcomes.get('blue', [])
        
        num_red = len(red_agents)
        num_blue = len(blue_agents)
        total_agents = self.num_agents
        
        # Objective 1: Deviation from 50-50 red/blue split
        target_split = total_agents / 2.0
        red_deviation = abs(num_red - target_split) / target_split
        blue_deviation = abs(num_blue - target_split) / target_split
        split_deviation = (red_deviation + blue_deviation) / 2.0
        
        # Objective 2: Fraction of agents that are neither red nor blue
        num_classified = num_red + num_blue
        num_neither = total_agents - num_classified
        neither_fraction = num_neither / total_agents
        
        # If neither_fraction is negative, return infinite objectives (invalid solution)
        if neither_fraction < 0:
            print(f"Warning: Invalid classification - neither_fraction = {neither_fraction}. Setting objectives to infinity.")
            return [float('inf'), float('inf')]
        
        return [split_deviation, neither_fraction]

    def evaluate(self, dofs):
        """Evaluate a single individual with DOF as prompt string"""
        try:
            # DOF should be a string (the prompt)
            if not isinstance(dofs, str):
                raise ValueError("DOF must be a string")
            
            # Create config from DOF
            config_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            config = self.create_config_from_dofs(dofs, config_name)
            
            # Run simulation
            self.run_simulation(config)
            # Evaluate outcomes
            objectives = self.evaluate_simulation_outcomes(simulation=self.experiment.simulations[0])
            
            # For this evaluator, objectives and metrics are the same
            constraints = []
            metrics = objectives.copy()
            print(f"Objectives: {objectives}")
            
            # Save results to JSON
            self.save_evaluation_results(
                prompt=dofs,
                objectives=objectives,
                constraints=constraints,
                metrics=metrics,
                additional_info={"config_name": config_name}
            )
            
            return objectives, constraints, metrics
            
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
            # Save failed evaluation
            self.save_evaluation_results(
                prompt=dofs if isinstance(dofs, str) else str(dofs),
                objectives=[1.0, 1.0],
                constraints=[],
                metrics=[1.0, 1.0],
                additional_info={"error": str(e), "failed": True}
            )
            return [1.0, 1.0], [], [1.0, 1.0]  # Return penalty for errors
    
    def evaluate_batch(self, individuals=None):
        """Evaluate a batch of DOFs"""
        if individuals is None or len(individuals) == 0:
            return [], [], []

        try:
            # Create a simulation for each individual
            simulations = []
            for individual in individuals:
                config_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                config = self.create_config_from_dofs(individual.dofs, config_name)
                
                sim = Simulation(
                    llm_wrapper=self.llm,
                    chunk_size=config.get("chunk_size", 1200),
                )
                sim.setup_from_config(config)
                simulations.append(sim)

            # Create experiment with all simulations
            print(f"Creating experiment with {len(simulations)} simulations...")
            experiment = Experiment(simulations, name=f"batch_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Run experiment and collect results
            print(f"Starting experiment execution (batched={self.use_batched_evaluation})...")
            runs = []
            if self.use_batched_evaluation:
                print("Using batched evaluation...")
                for progress, data in experiment.run_manual_batch(steps=self.steps):
                    print(f"Batch progress: {progress}")
                    if progress.get("percentage") == 100 and "runs" in data:
                        runs = data["runs"]
                        break
            else:
                print("Using non-batched evaluation...")
                for progress, data in experiment.run_manual(steps=self.steps, use_batched=False):
                    print(f"Manual progress: {progress}")
                    if progress.get("percentage") == 100 and "runs" in data:
                        runs = data["runs"]
                        break

            print(f"Experiment completed with {len(runs)} runs")
            self.experiment = experiment

            # Process each run's outcomes
            all_objectives = []
            all_constraints = []
            all_metrics = []

            print(f"Runs: {len(runs)}")
            for idx, individual in enumerate(individuals):
                # Check if simulation failed or has no outcomes
                run = runs[idx]
                if (run.get("failed", False) or 
                    not run.get("agent_outcomes")):
                    # Assign infinite objectives for failed simulations
                    objectives = [float('inf')] * 2
                    constraints = []
                    metrics = objectives.copy()
                    print(f"Run {idx}: FAILED - {run.get('summary', 'No summary')}")
                else:
                    # Use the batched agent outcomes
                    agent_outcomes = run["agent_outcomes"]
                    red_agents = agent_outcomes.get('red', [])
                    blue_agents = agent_outcomes.get('blue', [])
                    
                    print(f"\nRun {idx} Agent Colors:")
                    print(f"  Red agents: {red_agents} (count: {len(red_agents)})")
                    print(f"  Blue agents: {blue_agents} (count: {len(blue_agents)})")
                    print(f"  Total classified: {len(red_agents) + len(blue_agents)}/{self.num_agents}")
                    
                    num_red = len(red_agents)
                    num_blue = len(blue_agents)
                    total_agents = self.num_agents
                    
                    # Calculate objectives using the batched outcomes
                    target_split = total_agents / 2.0
                    red_deviation = abs(num_red - target_split) / target_split
                    blue_deviation = abs(num_blue - target_split) / target_split
                    split_deviation = (red_deviation + blue_deviation) / 2.0
                    
                    num_classified = num_red + num_blue
                    num_neither = total_agents - num_classified
                    neither_fraction = num_neither / total_agents
                    
                    print(f"  Target split: {target_split} each")
                    print(f"  Red deviation: {red_deviation:.3f}, Blue deviation: {blue_deviation:.3f}")
                    print(f"  Split deviation: {split_deviation:.3f}")
                    print(f"  Neither count: {num_neither}, Neither fraction: {neither_fraction:.3f}")
                    
                    objectives = [split_deviation, neither_fraction]
                    constraints = []
                    metrics = objectives.copy()
                    
                    print(f"  Final objectives: {objectives}")

                individual._objectives = objectives
                individual._constraints = constraints
                individual._metrics = metrics
                
                all_objectives.append(objectives)
                all_constraints.append(constraints)
                all_metrics.append(metrics)
            print(f"All objectives: {all_objectives}")
            print(f"All constraints: {all_constraints}")
            print(f"All metrics: {all_metrics}")

            # Save batch results to JSON
            prompts = [individual.dofs for individual in individuals]
            self.save_batch_evaluation_results(
                prompts=prompts,
                all_objectives=all_objectives,
                all_constraints=all_constraints,
                all_metrics=all_metrics,
                additional_info={"batch_size": len(individuals), "use_batched_evaluation": self.use_batched_evaluation}
            )

            return all_objectives, all_constraints, all_metrics

        except Exception as e:
            print(f"Error in evaluate_batch: {str(e)}")
            num_individuals = len(individuals)
            
            # Save failed batch evaluation
            if individuals:
                prompts = [individual.dofs for individual in individuals]
                failed_objectives = [[float('inf')] * 2 for _ in range(num_individuals)]
                failed_constraints = [[] for _ in range(num_individuals)]
                failed_metrics = [[float('inf')] * 2 for _ in range(num_individuals)]
                
                self.save_batch_evaluation_results(
                    prompts=prompts,
                    all_objectives=failed_objectives,
                    all_constraints=failed_constraints,
                    all_metrics=failed_metrics,
                    additional_info={"error": str(e), "failed": True}
                )
            
            return (
                [[float('inf')] * 2 for _ in range(num_individuals)],
                [[] for _ in range(num_individuals)],
                [[float('inf')] * 2 for _ in range(num_individuals)]
            )

    def schedule(self, gen, n_generations):
        """Update evaluator parameters based on generation"""
        if self._use_scheduling:
            # Could implement scheduling logic here if needed
            pass
    
    def visualize(self, gen):
        """Visualize evaluation results"""
        if gen % 10 == 0:
            print(f"Generation {gen}: RedBlue evaluation visualization")
            # Could implement visualization logic here if needed
            pass