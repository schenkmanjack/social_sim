import os
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np


class StateSaver(ABC):
    """Abstract base class for saving genetic algorithm state"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self._ensure_results_file()
    
    @abstractmethod
    def _ensure_results_file(self):
        """Initialize the results file with appropriate structure"""
        pass
    
    @abstractmethod
    def save_generation_state(self, generation: int, population: List, elites: List, **kwargs):
        """Save the complete state for a generation"""
        pass


class RedBlueStateSaver(StateSaver):
    """State saver specific to RedBlue evaluator format"""
    
    def __init__(self, results_file: str, num_agents: int, steps: int, connectivity_pattern: str,
                 objective_labels: List[str] = None, metric_labels: List[str] = None):
        self.num_agents = num_agents
        self.steps = steps
        self.connectivity_pattern = connectivity_pattern
        self.objective_labels = objective_labels or ["Split Deviation", "Neither Fraction"]
        self.metric_labels = metric_labels or self.objective_labels
        self._seen_solutions = set()  # Track previously seen solutions by their signature
        super().__init__(results_file)
    
    def _ensure_results_file(self):
        """Initialize results file with metadata if it doesn't exist"""
        if not os.path.exists(self.results_file):
            initial_data = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "num_agents": self.num_agents,
                    "steps": self.steps,
                    "connectivity_pattern": self.connectivity_pattern,
                    "objective_labels": self.objective_labels,
                    "metric_labels": self.metric_labels
                },
                "generations": []
            }
            
            try:
                with open(self.results_file, 'w') as f:
                    json.dump(initial_data, f, indent=2)
                print(f"[DEBUG] Created new results file: {self.results_file}")
            except Exception as e:
                print(f"[DEBUG] Error creating results file: {e}")
                raise
    
    def _get_solution_signature(self, individual):
        """Create a unique signature for a solution to detect duplicates"""
        try:
            # Extract prompt and connectivity for signature
            if hasattr(individual, 'get_prompt'):
                prompt = individual.get_prompt()
                connectivity = individual.dofs.get("connectivity", []) if isinstance(individual.dofs, dict) else []
            elif isinstance(individual.dofs, dict):
                prompt = individual.dofs.get("prompt", "")
                connectivity = individual.dofs.get("connectivity", [])
            else:
                prompt = str(individual.dofs)
                connectivity = []
            
            # Create signature from prompt + connectivity (sorted for consistency)
            connectivity_str = str(sorted(connectivity)) if connectivity else "[]"
            signature = f"{prompt}|{connectivity_str}"
            return signature
        except Exception as e:
            print(f"[DEBUG] Error creating solution signature: {e}")
            return str(hash(str(individual.dofs)))  # Fallback to hash
    
    def _filter_new_elites(self, elites):
        """Filter elites to only include new solutions (not seen before)"""
        print(f"[DEBUG] Filtering {len(elites)} total elites for new solutions")
        print(f"[DEBUG] Previously seen solutions: {len(self._seen_solutions)}")
        
        new_elites = []
        new_signatures = []
        
        for individual in elites:
            signature = self._get_solution_signature(individual)
            
            if signature not in self._seen_solutions:
                new_elites.append(individual)
                new_signatures.append(signature)
        
        # Add new signatures to seen set
        self._seen_solutions.update(new_signatures)
        
        print(f"[DEBUG] Found {len(new_elites)} new elite solutions (should be ≤ population size)")
        print(f"[DEBUG] Total seen solutions now: {len(self._seen_solutions)}")
        
        return new_elites
    
    def save_generation_state(self, generation: int, population: List, elites: List, **kwargs):
        """Save complete generation state including elite status"""
        print(f"[DEBUG] Starting save_generation_state for generation {generation}")
        print(f"[DEBUG] Population size: {len(population)}, Elites count: {len(elites)}")
        print(f"[DEBUG] Results file: {self.results_file}")
        
        try:
            # Load existing data with fallback if file is corrupted
            print(f"[DEBUG] Attempting to load existing JSON file...")
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                print(f"[DEBUG] Successfully loaded existing JSON with {len(data.get('generations', []))} generations")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"[DEBUG] Failed to load JSON file: {e}")
                print(f"[DEBUG] Reinitializing JSON structure...")
                # Reinitialize the file if it's corrupted
                data = {
                    "metadata": {
                        "created": datetime.now().isoformat(),
                        "num_agents": self.num_agents,
                        "steps": self.steps,
                        "connectivity_pattern": self.connectivity_pattern
                    },
                    "generations": []
                }
                print(f"[DEBUG] JSON structure reinitialized")
            
            # Filter elites to get only new ones BEFORE creating generation entry
            new_elites = self._filter_new_elites(elites)
            print(f"[DEBUG] Processing {len(new_elites)} new elite individuals out of {len(population)} total...")
            
            # Create generation entry
            generation_timestamp = datetime.now().isoformat()
            print(f"[DEBUG] Creating generation entry for generation {generation} at {generation_timestamp}")
            generation_entry = {
                "generation": generation,
                "timestamp": generation_timestamp,
                "population_size": len(population),
                "total_elites_found": len(elites),  # Total elites found by GA
                "new_elites_saved": len(new_elites),  # New elites saved to file
                "num_elites": len(new_elites),  # For backward compatibility
                "individuals": []
            }
            
            # Save only elite individuals in population
            for i, individual in enumerate(new_elites):
                print(f"[DEBUG] Processing elite individual {i}")
                print(f"[DEBUG] Individual has attributes: {list(vars(individual).keys())}")
                
                # Check individual's objectives
                if hasattr(individual, 'objectives'):
                    print(f"[DEBUG] Elite individual {i} objectives: {individual.objectives} (type: {type(individual.objectives)})")
                else:
                    print(f"[DEBUG] Elite individual {i} has no objectives attribute!")
                
                # Check elite status (should always be True since we filtered)
                is_elite = getattr(individual, 'is_elite', False)
                print(f"[DEBUG] Elite individual {i} is_elite: {is_elite} (type: {type(is_elite)})")
                
                # Safely convert objectives, handling inf values
                def safe_convert_to_list(arr):
                    print(f"[DEBUG] Converting array to list: {arr} (type: {type(arr)})")
                    if hasattr(arr, 'tolist'):
                        result = arr.tolist()
                    else:
                        result = list(arr)
                    
                    print(f"[DEBUG] Converted to list: {result}")
                    
                    # Replace inf/-inf with string representations for JSON safety
                    def replace_inf(obj):
                        if isinstance(obj, list):
                            return [replace_inf(item) for item in obj]
                        elif obj == float('inf'):
                            print(f"[DEBUG] Replacing float('inf') with 'Infinity'")
                            return "Infinity"
                        elif obj == float('-inf'):
                            print(f"[DEBUG] Replacing float('-inf') with '-Infinity'")
                            return "-Infinity"
                        elif isinstance(obj, float) and np.isnan(obj):
                            print(f"[DEBUG] Replacing NaN with 'NaN'")
                            return "NaN"
                        else:
                            return obj
                    
                    result_safe = replace_inf(result)
                    print(f"[DEBUG] After inf replacement: {result_safe}")
                    return result_safe
                
                # Extract prompt and connectivity safely
                if hasattr(individual, 'get_prompt'):
                    # Use getter method for prompt, direct access for connectivity
                    prompt = individual.get_prompt()
                    connectivity = individual.dofs.get("connectivity", []) if isinstance(individual.dofs, dict) else []
                elif isinstance(individual.dofs, dict):
                    # Fallback to direct access for dict DOFs
                    prompt = individual.dofs.get("prompt", "")
                    connectivity = individual.dofs.get("connectivity", [])
                else:
                    # String DOFs
                    prompt = individual.dofs if isinstance(individual.dofs, str) else str(individual.dofs)
                    connectivity = []
                
                # Extract agent outcomes (colors) from all duplicates
                all_duplicate_outcomes = getattr(individual, '_agent_outcomes_all_duplicates', [])
                
                if all_duplicate_outcomes:
                    # Create agent colors for each duplicate
                    agent_colors_per_duplicate = []
                    for dup_idx, agent_outcomes in enumerate(all_duplicate_outcomes):
                        red_agents = agent_outcomes.get('red', [])
                        blue_agents = agent_outcomes.get('blue', [])
                        
                        # Determine undefined agents (those not classified as red or blue)
                        all_agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
                        classified_agents = set(red_agents + blue_agents)
                        undefined_agents = [agent_id for agent_id in all_agent_ids if agent_id not in classified_agents]
                        
                        agent_colors_per_duplicate.append({
                            "duplicate": dup_idx,
                            "red": red_agents,
                            "blue": blue_agents,
                            "undefined": undefined_agents
                        })
                    
                    # Also create a summary showing color consistency across duplicates
                    color_consistency = self._analyze_color_consistency(all_duplicate_outcomes)
                else:
                    # No valid duplicates
                    agent_colors_per_duplicate = []
                    color_consistency = {}
                
                individual_data = {
                    "individual_id": i,
                    "prompt": prompt,
                    "connectivity": connectivity,
                    "agent_colors_per_duplicate": agent_colors_per_duplicate,
                    "color_consistency": color_consistency,
                    "objectives": safe_convert_to_list(individual.objectives),
                    "constraints": (safe_convert_to_list(individual.constraints) 
                                  if hasattr(individual, 'constraints') else []),
                    "metrics": (safe_convert_to_list(individual.metrics) 
                              if hasattr(individual, 'metrics') else []),
                    "is_elite": True,  # Always True since we filtered for elites
                    "additional_info": kwargs.get('additional_info', {})
                }
                print(f"[DEBUG] Elite individual {i} data created successfully")
                generation_entry["individuals"].append(individual_data)
            
            print(f"[DEBUG] All individuals processed, adding generation entry to data")
            # Add generation entry to data
            data["generations"].append(generation_entry)
            
            print(f"[DEBUG] Total generations in data: {len(data['generations'])}")
            
            # Atomic write: write to temp file first, then rename
            temp_file = self.results_file + '.tmp'
            print(f"[DEBUG] Writing to temporary file: {temp_file}")
            
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, allow_nan=False)
            
            print(f"[DEBUG] Successfully wrote to temporary file")
            
            # Atomic move
            print(f"[DEBUG] Moving temporary file to final location...")
            os.rename(temp_file, self.results_file)
            print(f"[DEBUG] File move completed")
            
            print(f"Saved generation {generation} state with {len(population)} individuals ({len(elites)} elites) to {self.results_file}")
            
        except Exception as e:
            print(f"[DEBUG] Exception occurred during save: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            import traceback
            print(f"[DEBUG] Full traceback:")
            traceback.print_exc()
            
            print(f"Error saving generation state: {str(e)}")
            # Clean up temp file if it exists
            temp_file = self.results_file + '.tmp'
            if os.path.exists(temp_file):
                print(f"[DEBUG] Cleaning up temporary file: {temp_file}")
                os.remove(temp_file)
    
    def get_generation_data(self, generation: Optional[int] = None) -> Dict[str, Any]:
        """Retrieve data for a specific generation or all generations"""
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            if generation is None:
                return data
            
            for gen_data in data["generations"]:
                if gen_data["generation"] == generation:
                    return gen_data
            
            return {}
            
        except Exception as e:
            print(f"Error retrieving generation data: {str(e)}")
            return {}
    
    def get_pareto_front_history(self) -> List[Dict[str, Any]]:
        """Extract Pareto front evolution across generations"""
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            pareto_history = []
            for gen_data in data["generations"]:
                elites = [ind for ind in gen_data["individuals"] if ind["is_elite"]]
                pareto_history.append({
                    "generation": gen_data["generation"],
                    "timestamp": gen_data["timestamp"],
                    "pareto_front": elites
                })
            
            return pareto_history
            
        except Exception as e:
            print(f"Error extracting Pareto front history: {str(e)}")
            return []
    
    def _analyze_color_consistency(self, all_duplicate_outcomes):
        """Analyze how consistently each agent chooses the same color across duplicates"""
        if not all_duplicate_outcomes:
            return {}
        
        agent_color_choices = {}
        
        # Collect color choices for each agent across all duplicates
        for agent_id in [f"agent_{i}" for i in range(self.num_agents)]:
            choices = []
            for outcomes in all_duplicate_outcomes:
                if agent_id in outcomes.get('red', []):
                    choices.append('red')
                elif agent_id in outcomes.get('blue', []):
                    choices.append('blue')
                else:
                    choices.append('undefined')
            
            # Calculate consistency metrics
            total_duplicates = len(choices)
            red_count = choices.count('red')
            blue_count = choices.count('blue')
            undefined_count = choices.count('undefined')
            
            # Determine most frequent choice
            choice_counts = {'red': red_count, 'blue': blue_count, 'undefined': undefined_count}
            most_frequent_choice = max(choice_counts, key=choice_counts.get)
            consistency_percentage = choice_counts[most_frequent_choice] / total_duplicates
            
            agent_color_choices[agent_id] = {
                "choices": choices,
                "most_frequent": most_frequent_choice,
                "consistency": consistency_percentage,
                "red_count": red_count,
                "blue_count": blue_count,
                "undefined_count": undefined_count
            }
        
        return agent_color_choices 