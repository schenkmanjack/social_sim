import os
import json
import tempfile
from datetime import datetime
import numpy as np
import random
import time

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
        self.n_eval_duplicates = evaluator_config.get('n_eval_duplicates', 1)  # Number of duplicate evaluations per individual
        self.debug = evaluator_config.get('debug', False)  # Debug flag for print statements
        self.disable_batch_summaries = evaluator_config.get('disable_batch_summaries', False)  # NEW: Control batch summary processing
        
        # Agent memory configuration
        self.use_full_agent_memory = evaluator_config.get('use_full_agent_memory', True)
        
        # Objective selection configuration
        self.use_objectives = evaluator_config.get('use_objectives', [True, True, True, True, True, True])  # Default: use all 6 objectives
        if len(self.use_objectives) != 6:
            raise ValueError("use_objectives must be a boolean array of length 6")
        self.objective_names = ["split_deviation", "neither_fraction", "prompt_length", "connection_count", "failure_count", "llm_usage"]
        self.selected_objectives = [name for i, name in enumerate(self.objective_names) if self.use_objectives[i]]
        print(f"Using objectives: {self.selected_objectives}")
        
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
            elif self.connectivity_pattern == 'random':
                self.connectivity_pairs = self._generate_random_connectivity_pairs()
            else:
                raise ValueError(f"Invalid evaluator pattern: {self.evaluator_pattern}")
            
            print(f"Connectivity pairs: {self.connectivity_pairs}")
            print(f"RedBlueEvaluator initialized with {self.num_agents} agents, {self.steps} steps, {self.n_eval_duplicates} duplicates per individual")
            print(f"use_full_agent_memory: {self.use_full_agent_memory}")
        
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
        """Initialize with a default prompt and connectivity from evaluator_config"""
        initial_prompt = self.evaluator_config.get('initial_prompt', 
            "You are an agent who identifies as red or blue. Each step represents a simultaneous decision round. You communicate with your neighbors. The goal is for half of the agents to be red and half to be blue at the end of the simulation. At the end of the simulation you must say what color you are. You must identify as one color and state the color. Provide any answer of either red or blue.")
        # initial_prompt = LLMTextMutation._mutate_via_llm(self.llm, initial_prompt)
        
        # Create connectivity_pairs if it doesn't exist by checking config
        if not hasattr(self, 'connectivity_pairs') or self.connectivity_pairs is None:
            connectivity_pattern = self.evaluator_config.get('connectivity_pattern', 'adjacent')
            
            if connectivity_pattern == 'adjacent':
                self.connectivity_pairs = [[i, i+1] for i in range(self.num_agents-1)]
            elif connectivity_pattern == 'all':
                self.connectivity_pairs = [[i, j] for i in range(self.num_agents) for j in range(i+1, self.num_agents)]
            elif connectivity_pattern == 'random':
                self.connectivity_pairs = self._generate_random_connectivity_pairs()
            else:
                print(f"Warning: Unknown connectivity pattern '{connectivity_pattern}', using 'adjacent'")
                self.connectivity_pairs = [[i, i+1] for i in range(self.num_agents-1)]
            
            print(f"Created connectivity_pairs with pattern '{connectivity_pattern}': {self.connectivity_pairs}")
        
        # Get initial connectivity from evaluator configuration
        initial_connectivity = self.connectivity_pairs.copy()
        
        # Return dictionary DOF with both prompt and connectivity
        return {
            "prompt": initial_prompt,
            "connectivity": initial_connectivity
        }

    def create_config_from_dofs(self, dofs, config_name="eval_config"):
        """Create simulation config from DOF (prompt string or dictionary)"""
        
        # Handle both string and dictionary DOFs
        if isinstance(dofs, dict):
            prompt_dof = dofs.get("prompt", "")
            connectivity_pairs = dofs.get("connectivity", self.connectivity_pairs)
        else:
            # Backward compatibility - dofs is a string
            prompt_dof = dofs
            connectivity_pairs = self.connectivity_pairs
        
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
        for pair in connectivity_pairs:
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
                "red": "Does the agent choose red? For the agent to be red, it must state it unambiguously.",
                "blue": "Does the agent choose blue? For the agent to be blue, it must state it unambiguously."
            },
            "use_full_agent_memory": self.use_full_agent_memory
        }
        
        return config

    def run_simulation(self, config):
        """Run a single simulation with the given config"""
        
        # Create simulation
        sim = Simulation(
            llm_wrapper=self.llm,
            chunk_size=config.get("chunk_size", 1200),
            disable_summary=self.evaluator_config.get('disable_summary', False)
        )
        
        # Set up from config dict directly - no temp file needed
        sim.setup_from_config(config)
        
        # Get config values for experiment setup
        config = sim.config
        
        # Create experiment
        experiment = Experiment([sim], name=config["name"], disable_batch_summaries=self.disable_batch_summaries)
        
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

    def _clean_agent_outcomes(self, agent_outcomes):
        """
        Clean agent outcomes by moving agents that appear in multiple color categories to undefined.
        
        Args:
            agent_outcomes: Dict with 'red', 'blue', and possibly 'undefined' keys
            
        Returns:
            Dict with cleaned agent outcomes where each agent appears in only one category
        """
        if not agent_outcomes or not isinstance(agent_outcomes, dict):
            return {"red": [], "blue": [], "undefined": []}
        
        # Get lists for each category
        red_agents = agent_outcomes.get('red', [])
        blue_agents = agent_outcomes.get('blue', [])
        undefined_agents = agent_outcomes.get('undefined', [])
        
        # Convert to sets for easier operations
        red_set = set(red_agents)
        blue_set = set(blue_agents)
        undefined_set = set(undefined_agents)
        
        # Find all unique agents
        all_agents = red_set.union(blue_set).union(undefined_set)
        
        # Create cleaned categories
        cleaned_red = []
        cleaned_blue = []
        cleaned_undefined = []
        
        conflicts_found = False
        conflicted_agents = []
        
        for agent in all_agents:
            categories = []
            if agent in red_set:
                categories.append('red')
            if agent in blue_set:
                categories.append('blue')
            if agent in undefined_set:
                categories.append('undefined')
            
            # If agent appears in multiple categories, move to undefined
            if len(categories) > 1:
                cleaned_undefined.append(agent)
                conflicts_found = True
                conflicted_agents.append((agent, categories))
                if self.debug:
                    print(f"Debug: Agent {agent} found in multiple categories {categories}, moved to undefined")
            else:
                # Agent appears in only one category, keep it there
                if agent in red_set:
                    cleaned_red.append(agent)
                elif agent in blue_set:
                    cleaned_blue.append(agent)
                elif agent in undefined_set:
                    cleaned_undefined.append(agent)
        
        if conflicts_found and self.debug:
            print(f"Debug: Data integrity cleaning moved {len(conflicted_agents)} agents to undefined due to conflicts: {conflicted_agents}")
        
        return {
            'red': cleaned_red,
            'blue': cleaned_blue,
            'undefined': cleaned_undefined
        }

    def evaluate_simulation_outcomes(self, simulation=None):
        """Evaluate simulation outcomes and return objectives"""
        if simulation is None:
            raise ValueError("Simulation is required for evaluation")
        
        # Clean agent outcomes before computing objectives
        raw_agent_outcomes = simulation.agent_outcomes
        agent_outcomes = self._clean_agent_outcomes(raw_agent_outcomes)
        
        # Store cleaned outcomes back on simulation for potential use by state saver
        simulation.agent_outcomes = agent_outcomes
        
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
            return [float('inf'), float('inf'), float('inf')]
        
        return [split_deviation, neither_fraction]

    def calculate_prompt_length_objective(self, dofs):
        """Calculate prompt length as normalized objective"""
        # Handle both string and dictionary DOFs
        if isinstance(dofs, dict):
            prompt = dofs.get("prompt", "")
        else:
            prompt = str(dofs)
        
        # Objective 3: Character length of the prompt (normalized)
        prompt_length = len(prompt)
        # Normalize by a reasonable maximum length (e.g., 1000 characters)
        # This makes it a value between 0 and 1+ where shorter prompts are better
        max_reasonable_length = 1000
        normalized_length = prompt_length / max_reasonable_length
        
        return normalized_length

    def calculate_connectivity_objective(self, dofs):
        """Calculate number of connections as objective (raw count for minimization)"""
        # Handle both string and dictionary DOFs
        if isinstance(dofs, dict):
            connectivity = dofs.get("connectivity", [])
        else:
            # For string DOFs, use evaluator's default connectivity
            connectivity = self.connectivity_pairs
        
        # Count number of connections
        connection_count = len(connectivity) if connectivity else 0
        
        return connection_count

    def calculate_llm_usage_objective(self, simulation_or_run):
        """Calculate LLM character usage as normalized objective"""
        # Handle both Simulation objects and run dictionaries for backward compatibility
        if hasattr(simulation_or_run, 'get_llm_usage_stats'):
            # It's a Simulation object - use the proper method
            usage_stats = simulation_or_run.get_llm_usage_stats()
        elif isinstance(simulation_or_run, dict) and 'llm_usage' in simulation_or_run:
            # It's a run dictionary - use the llm_usage key
            usage_stats = simulation_or_run['llm_usage']
        else:
            # No usage tracking available
            return 0
        
        if usage_stats and 'total_characters' in usage_stats:
            total_chars = usage_stats['total_characters']
            # Normalize by reasonable maximum (e.g., 50,000 characters for a typical simulation)
            max_reasonable_chars = 50000
            normalized_usage = total_chars / max_reasonable_chars
            return normalized_usage
        else:
            return 0  # No usage tracking available

    def evaluate(self, dofs):
        """Evaluate a single individual with DOF as prompt string or dictionary"""
        try:
            # Handle both string and dictionary DOFs
            if isinstance(dofs, dict):
                prompt = dofs.get("prompt", "")
            else:
                prompt = str(dofs)
                
            if not prompt:
                raise ValueError("DOF must contain a valid prompt")
            
            # Create config from DOF
            config_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            config = self.create_config_from_dofs(dofs, config_name)
            
            # Run simulation
            self.run_simulation(config)
            # Evaluate outcomes (returns 2 objectives: split_deviation, neither_fraction)
            sim_objectives = self.evaluate_simulation_outcomes(simulation=self.experiment.simulations[0])
            
            # Store agent outcomes for potential use by state saver
            agent_outcomes = self.experiment.simulations[0].agent_outcomes
            
            # Add prompt length as third objective
            prompt_length_objective = self.calculate_prompt_length_objective(dofs)
            
            # Add connection count as fourth objective
            connectivity_objective = self.calculate_connectivity_objective(dofs)
            
            # Add failure count as fifth objective (always 0 for single evaluation)
            failure_count_objective = 0  # Single evaluation: either completely fails or succeeds
            
            # Add LLM usage as sixth objective
            llm_usage_objective = self.calculate_llm_usage_objective(self.experiment.simulations[0])
            
            # Combine all objectives
            all_objectives = sim_objectives + [prompt_length_objective, connectivity_objective, failure_count_objective, llm_usage_objective]
            
            # Filter objectives based on use_objectives configuration
            objectives = self._filter_objectives(all_objectives)
            
            # For this evaluator, objectives and metrics are the same
            constraints = np.array([])
            metrics = objectives.copy()
            print(f"All objectives: {all_objectives}")
            print(f"Filtered objectives: {objectives}")
            
            return np.array(objectives), constraints, np.array(metrics), agent_outcomes
            
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
            # Return penalty for errors - filter the penalty objectives too
            penalty_all_objectives = [1e4, 1e4, 1e4, 1e4, 1e4, 1e4]
            penalty_objectives = self._filter_objectives(penalty_all_objectives)
            return np.array(penalty_objectives), np.array([]), np.array(penalty_objectives), {"red": [], "blue": []}  # Return penalty for errors
    
    def evaluate_batch(self, individuals=None):
        """Evaluate a batch of DOFs with multiple duplicates per individual"""
        if individuals is None or len(individuals) == 0:
            return [], [], [], []

        try:
            # Add delay to space out API calls between evaluations
            time.sleep(1.0)  # 1 second delay to help with rate limiting
            
            # Create multiple simulations for each individual (duplicates)
            simulations = []
            individual_indices = []  # Track which individual each simulation belongs to
            
            print(f"Creating {self.n_eval_duplicates} duplicate simulations for each of {len(individuals)} individuals...")
            
            for ind_idx, individual in enumerate(individuals):
                for dup_idx in range(self.n_eval_duplicates):
                    config_name = f"eval_{ind_idx}_{dup_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                    config = self.create_config_from_dofs(individual.dofs, config_name)
                    
                    sim = Simulation(
                        llm_wrapper=self.llm,
                        chunk_size=config.get("chunk_size", 1200),
                        disable_summary=self.evaluator_config.get('disable_summary', False)
                    )
                    sim.setup_from_config(config)
                    simulations.append(sim)
                    individual_indices.append(ind_idx)  # Track which individual this simulation belongs to

            print(f"Created {len(simulations)} total simulations ({len(individuals)} individuals × {self.n_eval_duplicates} duplicates)")

            # Create experiment with all simulations (including duplicates)
            experiment = Experiment(simulations, name=f"batch_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}", disable_batch_summaries=self.disable_batch_summaries)
            
            # Run experiment and collect results with error handling
            print(f"Starting experiment execution (batched={self.use_batched_evaluation})...")
            runs = []
            
            try:
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
            except Exception as e:
                error_str = str(e).lower()
                print(f"Experiment execution failed: {str(e)}")
                
                # Check if this is an API overload error
                if any(keyword in error_str for keyword in ['overload', 'rate limit', '429', '529', 'too many requests']):
                    print("API overload detected, returning penalty objectives for all individuals")
                    # Return penalty objectives for all individuals
                    full_penalty_objectives = [1e4] * 6
                    filtered_penalty_objectives = self._filter_objectives(full_penalty_objectives)
                    return (
                        np.array([filtered_penalty_objectives for _ in range(len(individuals))]),
                        np.array([[] for _ in range(len(individuals))]),
                        np.array([filtered_penalty_objectives for _ in range(len(individuals))]),
                        [{"red": [], "blue": []} for _ in range(len(individuals))]
                    )
                else:
                    # Re-raise non-API errors
                    raise

            print(f"Experiment completed with {len(runs)} runs")
            self.experiment = experiment

            # Group runs by individual and average results
            all_objectives = []
            all_constraints = []
            all_metrics = []
            all_agent_outcomes = []

            print(f"Processing {len(runs)} runs for {len(individuals)} individuals...")
            
            for ind_idx, individual in enumerate(individuals):
                # Collect all duplicate runs for this individual
                individual_objectives = []
                individual_constraints = []
                individual_metrics = []
                individual_agent_outcomes = []
                
                # Find all runs belonging to this individual
                individual_runs = []
                for run_idx, run in enumerate(runs):
                    if individual_indices[run_idx] == ind_idx:
                        individual_runs.append(run)
                
                print(f"\nIndividual {ind_idx}: Processing {len(individual_runs)} duplicate runs")
                
                # Process each duplicate run
                for dup_idx, run in enumerate(individual_runs):
                    # Debug: Print run structure
                    print(f"  Duplicate {dup_idx} DEBUG:")
                    print(f"    Run keys: {list(run.keys()) if isinstance(run, dict) else 'Not a dict'}")
                    print(f"    Failed flag: {run.get('failed', 'No failed key')}")
                    print(f"    Agent outcomes: {run.get('agent_outcomes', 'No agent_outcomes key')}")
                    if 'agent_outcomes' in run:
                        print(f"    Agent outcomes type: {type(run['agent_outcomes'])}")
                        print(f"    Agent outcomes content: {run['agent_outcomes']}")
                    
                    # Check if simulation actually failed
                    simulation_failed = run.get("failed", False)
                    agent_outcomes = run.get("agent_outcomes", {})
                    
                    # Clean agent outcomes before processing
                    if agent_outcomes:
                        agent_outcomes = self._clean_agent_outcomes(agent_outcomes)
                    
                    # More robust check for valid agent outcomes
                    has_valid_outcomes = (
                        agent_outcomes and 
                        isinstance(agent_outcomes, dict) and 
                        ('red' in agent_outcomes or 'blue' in agent_outcomes)
                    )
                    
                    if simulation_failed:
                        # Actually failed simulation
                        objectives = [1e4] * 5
                        constraints = []
                        metrics = objectives.copy()
                        agent_outcomes = {"red": [], "blue": []}
                        print(f"  Duplicate {dup_idx}: SIMULATION FAILED - {run.get('summary', 'No summary')}")
                    elif not has_valid_outcomes:
                        # No valid agent outcomes found - try manual extraction as fallback
                        print(f"  Duplicate {dup_idx}: Attempting manual agent outcome extraction...")
                        
                        # Try to manually extract red/blue from agent memory
                        manual_outcomes = self._extract_agent_outcomes_manually(run)
                        
                        if manual_outcomes and (manual_outcomes.get('red') or manual_outcomes.get('blue')):
                            # Manual extraction succeeded
                            agent_outcomes = manual_outcomes
                            
                            # Clean manually extracted outcomes as well
                            agent_outcomes = self._clean_agent_outcomes(agent_outcomes)
                            
                            red_agents = agent_outcomes.get('red', [])
                            blue_agents = agent_outcomes.get('blue', [])
                            
                            print(f"  Duplicate {dup_idx}: MANUAL EXTRACTION SUCCESS - Red={len(red_agents)}, Blue={len(blue_agents)}")
                            
                            num_red = len(red_agents)
                            num_blue = len(blue_agents)
                            total_agents = self.num_agents
                            
                            # Calculate all 6 objectives
                            red_fraction = num_red / total_agents if total_agents > 0 else 0
                            target_split = total_agents / 2.0
                            red_deviation = abs(num_red - target_split) / target_split if target_split > 0 else 0
                            blue_deviation = abs(num_blue - target_split) / target_split if target_split > 0 else 0
                            split_deviation = (red_deviation + blue_deviation) / 2.0
                            
                            num_classified = num_red + num_blue
                            neither_count = total_agents - num_classified
                            neither_fraction = neither_count / total_agents if total_agents > 0 else 0
                            prompt_length_objective = len(individual.dofs["prompt"]) / 1000.0  # Normalize by 1000 chars
                            connectivity_objective = len(individual.dofs["connectivity"]) / 10.0  # Normalize by 10 connections
                            failure_count_objective = 0  # No failures for successful runs
                            
                            # NEW: LLM usage objective
                            llm_usage_objective = self.calculate_llm_usage_objective(self.experiment.simulations[0])
                            
                            # Create full objectives array (6 objectives)
                            all_objectives = [split_deviation, neither_fraction, prompt_length_objective, 
                                            connectivity_objective, failure_count_objective, llm_usage_objective]
                            
                            # Filter objectives based on use_objectives configuration
                            objectives = self._filter_objectives(all_objectives)
                            constraints = []
                            metrics = objectives.copy()
                            
                            print(f"    Split deviation: {split_deviation:.3f}, Neither fraction: {neither_fraction:.3f}, Prompt length: {len(str(individual.dofs))} chars (normalized: {prompt_length_objective:.3f}), Connection count: {connectivity_objective}")
                        else:
                            # Manual extraction also failed
                            penalty_all_objectives = [1e4] * 6
                            objectives = self._filter_objectives(penalty_all_objectives)
                            constraints = []
                            metrics = objectives.copy()
                            agent_outcomes = {"red": [], "blue": []}
                            print(f"  Duplicate {dup_idx}: MANUAL EXTRACTION FAILED - {run.get('summary', 'No summary')}")
                    else:
                        # Successful simulation - calculate objectives
                        red_agents = agent_outcomes.get('red', [])
                        blue_agents = agent_outcomes.get('blue', [])
                        
                        print(f"  Duplicate {dup_idx}: SUCCESS - Red={len(red_agents)}, Blue={len(blue_agents)}")
                        
                        num_red = len(red_agents)
                        num_blue = len(blue_agents)
                        total_agents = self.num_agents
                        
                        # Calculate all 6 objectives
                        red_fraction = num_red / total_agents if total_agents > 0 else 0
                        target_split = total_agents / 2.0
                        red_deviation = abs(num_red - target_split) / target_split if target_split > 0 else 0
                        blue_deviation = abs(num_blue - target_split) / target_split if target_split > 0 else 0
                        split_deviation = (red_deviation + blue_deviation) / 2.0
                        
                        num_classified = num_red + num_blue
                        neither_count = total_agents - num_classified
                        neither_fraction = neither_count / total_agents if total_agents > 0 else 0
                        prompt_length_objective = len(individual.dofs["prompt"]) / 1000.0  # Normalize by 1000 chars
                        connectivity_objective = len(individual.dofs["connectivity"]) / 10.0  # Normalize by 10 connections
                        failure_count_objective = 0  # No failures for successful runs
                        
                        # NEW: LLM usage objective
                        llm_usage_objective = self.calculate_llm_usage_objective(self.experiment.simulations[0])
                        
                        # Create full objectives array (6 objectives)
                        all_objectives = [split_deviation, neither_fraction, prompt_length_objective, 
                                        connectivity_objective, failure_count_objective, llm_usage_objective]
                        
                        # Filter objectives based on use_objectives configuration
                        objectives = self._filter_objectives(all_objectives)
                        constraints = []
                        metrics = objectives.copy()
                        
                        print(f"    Split deviation: {split_deviation:.3f}, Neither fraction: {neither_fraction:.3f}, Prompt length: {len(str(individual.dofs))} chars (normalized: {prompt_length_objective:.3f}), Connection count: {connectivity_objective}")
                    
                    individual_objectives.append(objectives)
                    individual_constraints.append(constraints)
                    individual_metrics.append(metrics)
                    individual_agent_outcomes.append(agent_outcomes)
                
                # Count failed duplicates for this individual
                failed_duplicates = sum(1 for obj in individual_objectives if any(np.array(obj) >= 1e4))
                total_duplicates = len(individual_objectives)
                
                # Average the results across duplicates (excluding failed runs if any succeeded)
                valid_objectives = [obj for obj in individual_objectives if not any(np.array(obj) >= 1e4)]
                valid_agent_outcomes = [ao for i, ao in enumerate(individual_agent_outcomes) if not any(np.array(individual_objectives[i]) >= 1e4)]
                
                if valid_objectives:
                    # Reconstruct full 6-objective arrays from filtered objectives
                    expanded_objectives = []
                    for filtered_obj in valid_objectives:
                        # Create full 6-objective array and fill in the filtered values
                        full_obj = [0] * 6  # Initialize with zeros
                        filtered_idx = 0
                        for i in range(6):
                            if self.use_objectives[i]:
                                if i == 4:  # failure_count index - always 0 for valid runs before aggregation
                                    full_obj[i] = 0
                                elif i == 5:  # llm_usage index - get from simulation
                                    # For valid runs, calculate LLM usage from the simulation
                                    full_obj[i] = self.calculate_llm_usage_objective(self.experiment.simulations[0])
                                else:
                                    full_obj[i] = filtered_obj[filtered_idx]
                                    filtered_idx += 1
                            else:
                                full_obj[i] = 0  # Unused objectives set to 0
                        expanded_objectives.append(full_obj)
                    
                    # Handle each objective type appropriately
                    if expanded_objectives:
                        # Average simulation outcomes that vary between duplicates
                        avg_split_deviation = np.mean([obj[0] for obj in expanded_objectives])  # split_deviation
                        avg_neither_fraction = np.mean([obj[1] for obj in expanded_objectives])  # neither_fraction
                        
                        # Take deterministic values from first duplicate (same for all duplicates)
                        prompt_length_objective = expanded_objectives[0][2]  # prompt_length
                        connection_count_objective = expanded_objectives[0][3]  # connection_count
                        
                        # Count failures properly (not averaged)
                        failure_count_objective = failed_duplicates
                        
                        # Average LLM usage across valid duplicates
                        avg_llm_usage = np.mean([obj[5] for obj in expanded_objectives])  # llm_usage
                        
                        # Combine all objectives
                        full_avg_objectives = [avg_split_deviation, avg_neither_fraction, prompt_length_objective, 
                                             connection_count_objective, failure_count_objective, avg_llm_usage]
                    else:
                        # Fallback in case of no expanded objectives
                        full_avg_objectives = [1e4, 1e4, 1e4, 1e4, failed_duplicates, 1e4]
                    
                    # Now filter the full averaged objectives
                    avg_objectives = np.array(self._filter_objectives(full_avg_objectives))
                    avg_constraints = np.array([])  # No constraints for this problem
                    avg_metrics = avg_objectives.copy()
                    
                    # Store ALL agent outcomes from valid duplicates (not just first one)
                    all_duplicate_outcomes = valid_agent_outcomes
                    
                    print(f"  Individual {ind_idx} final averaged objectives: {avg_objectives} (from {len(valid_objectives)}/{len(individual_objectives)} valid runs, {failed_duplicates} failures)")
                else:
                    # All runs failed - create full penalty objectives then filter
                    full_penalty_objectives = [1e4] * 4 + [total_duplicates, 1e4]  # First 4 objectives get penalty, failure count = total duplicates, max LLM usage penalty
                    avg_objectives = np.array(self._filter_objectives(full_penalty_objectives))
                    avg_constraints = np.array([])
                    avg_metrics = avg_objectives.copy()
                    all_duplicate_outcomes = []
                    
                    print(f"  Individual {ind_idx}: ALL RUNS FAILED - using infinite objectives, {total_duplicates} failures")

                # Store results on individual
                individual._objectives = avg_objectives
                individual._constraints = avg_constraints
                individual._metrics = avg_metrics
                
                # Store agent outcomes from ALL duplicates for state saving
                individual._agent_outcomes_all_duplicates = all_duplicate_outcomes
                
                all_objectives.append(avg_objectives)
                all_constraints.append(avg_constraints)
                all_metrics.append(avg_metrics)
                all_agent_outcomes.append(all_duplicate_outcomes)

            print(f"Final averaged objectives for all individuals: {all_objectives}")

            # Safety check: Replace any None values with penalty objectives
            safe_objectives = []
            safe_constraints = []
            safe_metrics = []
            
            full_penalty_objectives = [1e4] * 6
            filtered_penalty_objectives = self._filter_objectives(full_penalty_objectives)
            
            for i, obj in enumerate(all_objectives):
                if obj is None or np.any(np.isnan(obj)) or np.any(np.isinf(obj)):
                    print(f"WARNING: Individual {i} has invalid objectives {obj}, replacing with penalty objectives")
                    safe_objectives.append(filtered_penalty_objectives)
                    safe_constraints.append(np.array([]))
                    safe_metrics.append(filtered_penalty_objectives)
                else:
                    safe_objectives.append(obj)
                    safe_constraints.append(all_constraints[i] if i < len(all_constraints) else np.array([]))
                    safe_metrics.append(all_metrics[i] if i < len(all_metrics) else obj)

            return (
                np.array(safe_objectives) if safe_objectives else np.array([]),
                np.array(safe_constraints) if safe_constraints else np.array([]),
                np.array(safe_metrics) if safe_metrics else np.array([]),
                all_agent_outcomes
            )

        except Exception as e:
            print(f"Error in evaluate_batch: {str(e)}")
            import traceback
            traceback.print_exc()
            num_individuals = len(individuals)
            
            # Create penalty objectives and filter them
            full_penalty_objectives = [1e4] * 6
            filtered_penalty_objectives = self._filter_objectives(full_penalty_objectives)
            
            return (
                np.array([filtered_penalty_objectives for _ in range(num_individuals)]),  # Updated to use filtered penalty objectives
                np.array([[] for _ in range(num_individuals)]),
                np.array([filtered_penalty_objectives for _ in range(num_individuals)]),  # Updated to use filtered penalty objectives
                [{"red": [], "blue": []} for _ in range(num_individuals)]
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

    def _generate_random_connectivity_pairs(self):
        """Generate random connectivity pairs for agents based on density from config."""
        
        # Get density from evaluator config (default 0.6 = 60% of possible connections)
        density = self.evaluator_config.get('connectivity_density', 0.6)
        
        # Generate all possible pairs (excluding self-connections)
        all_possible_pairs = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):  # Only upper triangle to avoid duplicates
                all_possible_pairs.append([i, j])
        
        # Calculate target number of connections based on density
        max_connections = len(all_possible_pairs)
        target_connections = int(density * max_connections)
        
        # Randomly select connections (risk disconnected components)
        if target_connections >= max_connections:
            connectivity_pairs = all_possible_pairs.copy()
        elif target_connections <= 0:
            connectivity_pairs = []
        else:
            connectivity_pairs = random.sample(all_possible_pairs, target_connections)
        
        # Remove any duplicate connections
        connectivity_pairs = self._remove_duplicate_connections(connectivity_pairs)
        
        return connectivity_pairs
    
    def _remove_duplicate_connections(self, connectivity_pairs):
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
                    # Keep in canonical [min, max] form
                    unique_pairs.append([normalized_pair[0], normalized_pair[1]])
        
        return unique_pairs

    def _extract_agent_outcomes_manually(self, run):
        """
        Manually extract red/blue agent outcomes from run data when LLM analysis fails.
        
        Args:
            run: Dictionary containing simulation run data
            
        Returns:
            Dict with 'red' and 'blue' keys containing lists of agent IDs
        """
        red_agents = []
        blue_agents = []
        
        try:
            # Method 1: Check agent_states for final color choices
            agent_states = run.get('agent_states', {})
            if agent_states:
                for agent_id, state in agent_states.items():
                    state_str = str(state).lower()
                    if 'red' in state_str and 'blue' not in state_str:
                        red_agents.append(agent_id)
                    elif 'blue' in state_str and 'red' not in state_str:
                        blue_agents.append(agent_id)
            
            # Method 2: Check summary for color mentions
            summary = run.get('summary', '')
            if summary and not red_agents and not blue_agents:
                summary_lower = summary.lower()
                
                # Look for patterns like "Agent 0: Red", "agent_1 chose blue", etc.
                import re
                
                # Pattern for "Agent X: Color" or "agent_X chose color"
                red_patterns = [
                    r'agent[_\s]*(\d+)[:\s]*red',
                    r'agent[_\s]*(\d+)[^a-z]*chose[^a-z]*red',
                    r'agent[_\s]*(\d+)[^a-z]*identifies?[^a-z]*as[^a-z]*red',
                    r'agent[_\s]*(\d+)[^a-z]*self[^a-z]*identified[^a-z]*as[^a-z]*red'
                ]
                
                blue_patterns = [
                    r'agent[_\s]*(\d+)[:\s]*blue',
                    r'agent[_\s]*(\d+)[^a-z]*chose[^a-z]*blue',
                    r'agent[_\s]*(\d+)[^a-z]*identifies?[^a-z]*as[^a-z]*blue',
                    r'agent[_\s]*(\d+)[^a-z]*self[^a-z]*identified[^a-z]*as[^a-z]*blue'
                ]
                
                for pattern in red_patterns:
                    matches = re.findall(pattern, summary_lower)
                    for match in matches:
                        agent_id = f"agent_{match}"
                        if agent_id not in red_agents and agent_id not in blue_agents:
                            red_agents.append(agent_id)
                
                for pattern in blue_patterns:
                    matches = re.findall(pattern, summary_lower)
                    for match in matches:
                        agent_id = f"agent_{match}"
                        if agent_id not in blue_agents and agent_id not in red_agents:
                            blue_agents.append(agent_id)
            
            # Method 3: Check actions for color choices
            actions = run.get('actions', [])
            if actions and not red_agents and not blue_agents:
                for step_actions in actions:
                    if isinstance(step_actions, list):
                        for action in step_actions:
                            action_str = str(action).lower()
                            # Look for agent color declarations in actions
                            if 'red' in action_str and 'blue' not in action_str:
                                # Try to extract agent ID from action
                                import re
                                agent_match = re.search(r'agent[_\s]*(\d+)', action_str)
                                if agent_match:
                                    agent_id = f"agent_{agent_match.group(1)}"
                                    if agent_id not in red_agents:
                                        red_agents.append(agent_id)
                            elif 'blue' in action_str and 'red' not in action_str:
                                import re
                                agent_match = re.search(r'agent[_\s]*(\d+)', action_str)
                                if agent_match:
                                    agent_id = f"agent_{agent_match.group(1)}"
                                    if agent_id not in blue_agents:
                                        blue_agents.append(agent_id)
            
            print(f"    Manual extraction found: Red={red_agents}, Blue={blue_agents}")
            
            return {
                'red': red_agents,
                'blue': blue_agents
            }
            
        except Exception as e:
            print(f"    Error in manual extraction: {e}")
            return {'red': [], 'blue': []}

    def _filter_objectives(self, all_objectives):
        """Filter objectives based on use_objectives boolean array"""
        if len(all_objectives) != 6:
            raise ValueError(f"Expected 6 objectives, got {len(all_objectives)}")
        
        filtered_objectives = [obj for i, obj in enumerate(all_objectives) if self.use_objectives[i]]
        return filtered_objectives