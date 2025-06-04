import os
import json
import tempfile
from datetime import datetime
import numpy as np

from social_sim.agents.agent import Agent
from social_sim.interactions.connectivity import ConnectivityGraph
from social_sim.simulation import Simulation
from social_sim.experiment import Experiment

class RedBlueEvaluator:
    def __init__(self, evaluator_config=dict(), use_scheduling=False):
        self.evaluator_config = evaluator_config
        self.llm = evaluator_config.get('llm_wrapper')
        self._initial_evaluator_config = evaluator_config.copy()
        self._use_scheduling = use_scheduling
        self.use_batched_evaluation = False
        
        # Get required config parameters
        self.num_agents = evaluator_config.get('num_agents', 7)
        self.steps = evaluator_config.get('steps', 5)
        self.connectivity_pairs = evaluator_config.get('connectivity', [[0,1], [2,3], [4,5]])
        
        print(f"RedBlueEvaluator initialized with {self.num_agents} agents, {self.steps} steps")

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
        return initial_prompt

    def create_config_from_dof(self, prompt_dof, config_name="eval_config"):
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

    def evaluate_simulation_outcomes(self):
        """Evaluate simulation outcomes and return objectives"""
        
        agent_outcomes = self.experiment.simulations[0].agent_outcomes
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
            config = self.create_config_from_dof(dofs, config_name)
            
            # Run simulation
            self.run_simulation(config)
            # Evaluate outcomes
            objectives = self.evaluate_simulation_outcomes()
            
            # For this evaluator, objectives and metrics are the same
            constraints = []
            metrics = objectives.copy()
            print(f"Objectives: {objectives}")
            
            return objectives, constraints, metrics
            
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
            return [1.0, 1.0], [], [1.0, 1.0]  # Return penalty for errors

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