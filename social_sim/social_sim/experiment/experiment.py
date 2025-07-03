from typing import List, Dict, Any, Generator, Tuple
import json
import os
from datetime import datetime
import re
import matplotlib.pyplot as plt

# Import Simulation to fix NameError in _generate_experiment_results
from social_sim.simulation.simulation import Simulation


class Experiment:
    def __init__(self, simulations: List['Simulation'], name: str = None, debug: bool = False, disable_batch_summaries: bool = False):
        """
        Initialize an experiment with multiple simulations.
        
        Args:
            simulations: List of Simulation objects to run
            name: Optional name for the experiment
            debug: Whether to print debug information during execution
            disable_batch_summaries: Whether to disable expensive batch summary processing
        """
        self.simulations = simulations
        self.name = name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = None
        self.outcome_definitions = {}
        self.debug = debug
        self.disable_batch_summaries = disable_batch_summaries
        
    def define_outcome(self, name: str, condition: str, description: str):
        """
        Define an outcome to track across all simulations.
        
        Args:
            name: Unique identifier for the outcome
            condition: Natural language description of when this outcome occurs
            description: Human-readable description of the outcome
        """
        self.outcome_definitions[name] = {
            "condition": condition,
            "description": description,
            "count": 0
        }
        
    def analyze_outcome(self, history: List[Dict]) -> Dict[str, bool]:
        """
        Analyze a simulation run to determine which defined outcomes occurred.
        Uses the outcome definitions from the configuration to determine results.
        """
        if not self.outcome_definitions:
            if self.debug:
                print("Warning: No outcomes defined for analysis")
            return {}
            
        # Extract the final state from the history
        final_state = {}
        for step in reversed(history):  # Look at the most recent steps first
            if "actions" in step:
                final_state["actions"] = step["actions"]
                if self.debug:
                    print(f"\nFound actions in step: {step['actions']}")
            if "environment_state" in step:
                final_state["environment"] = step["environment_state"]
                if self.debug:
                    print(f"Found environment state: {step['environment_state']}")
            if "agent_states" in step:
                final_state["agent_states"] = step["agent_states"]
                if self.debug:
                    print(f"Found agent states: {step['agent_states']}")
            if final_state:  # If we found any state information
                break
        
        if not final_state:
            if self.debug:
                print("Warning: No final state found in history")
            return {name: False for name in self.outcome_definitions}
        
        if self.debug:
            print(f"\nFinal state for analysis: {json.dumps(final_state, indent=2)}")
        
        # Use the first simulation's orchestrator for analysis
        orchestrator = self.simulations[0].orchestrator
        
        # Create a more detailed prompt for the LLM
        prompt = f"""
        Analyze this simulation state to determine which outcomes occurred.
        
        Final state:
        {json.dumps(final_state, indent=2)}
        
        Outcomes to check:
        {json.dumps(self.outcome_definitions, indent=2)}
        
        For each outcome, determine if its condition was met based on the final state.
        Pay special attention to the actions taken by the agents in the final state.
        
        Return a JSON object with the outcome names as keys and boolean values indicating if they occurred.
        Example format:
        {{
            "outcome1": true,
            "outcome2": false,
            "outcome3": false,
            "outcome4": false
        }}
        
        Important: 
        1. Carefully analyze the actions and environment state to determine which outcome occurred.
        2. Apply each outcome's condition exactly as specified in the configuration.
        3. Return exactly one true outcome based on the conditions.
        4. If you're unsure, make your best judgment based on the final state.
        """
        
        try:
            # Get the LLM's analysis using _call_llm_with_retry instead of analyze
            analysis = orchestrator._call_llm_with_retry(prompt)
            if self.debug:
                print(f"\nRaw analysis from LLM: {analysis}")
            
            try:
                outcome = json.loads(analysis)
            except json.JSONDecodeError:
                if self.debug:
                    print(f"Warning: Failed to parse LLM response as JSON: {analysis}")
                    # Try to extract JSON from the response
                    match = re.search(r'\{.*\}', analysis, re.DOTALL)
                    if match:
                        try:
                            outcome = json.loads(match.group(0))
                        except json.JSONDecodeError:
                            if self.debug:
                                print("Warning: Could not extract valid JSON from response")
                                return {name: False for name in self.outcome_definitions}
                        else:
                            if self.debug:
                                print("Warning: No JSON found in response")
                            return {name: False for name in self.outcome_definitions}
            
            # Validate that all defined outcomes are present in the result
            for outcome_name in self.outcome_definitions:
                if outcome_name not in outcome:
                    if self.debug:
                        print(f"Warning: Outcome '{outcome_name}' not found in analysis")
                    outcome[outcome_name] = False
            
            # Ensure exactly one outcome is true
            true_outcomes = [name for name, value in outcome.items() if value]
            if len(true_outcomes) != 1:
                if self.debug:
                    print(f"Warning: Expected exactly one true outcome, but found {len(true_outcomes)}")
                    print(f"True outcomes: {true_outcomes}")
                    print(f"Full outcome analysis: {outcome}")
                    # If we have multiple true outcomes, keep the first one
                    if true_outcomes:
                        outcome = {name: (name == true_outcomes[0]) for name in self.outcome_definitions}
                    else:
                        # If no outcomes are true, make a best guess based on the final state
                        if self.debug:
                            print("No outcomes were true, making best guess based on final state")
                            print(f"Final state: {json.dumps(final_state, indent=2)}")
                            # Default to the first outcome if we can't determine
                            outcome = {name: (name == list(self.outcome_definitions.keys())[0]) 
                                     for name in self.outcome_definitions}
            
            if self.debug:
                print(f"Final determined outcome: {outcome}")
            return outcome
            
        except Exception as e:
            if self.debug:
                print(f"Error analyzing outcome: {e}")
                print(f"Error details: {str(e)}")
                import traceback
                traceback.print_exc()
            return {name: False for name in self.outcome_definitions}
            
    def run(self, query: str, steps: int = 5) -> Generator[Tuple[Dict, Dict], None, None]:
        """
        Run the experiment with multiple simulations.
        
        Args:
            query: The initial query or scenario
            steps: Number of steps per simulation
            
        Yields:
            Tuple of (progress, data) for each step
        """
        total_simulations = len(self.simulations)
        simulation_histories = []  # Store complete histories for each simulation
        
        for sim_idx, simulation in enumerate(self.simulations, 1):
            if self.debug:
                print(f"\nRunning simulation {sim_idx}/{total_simulations}")
            sim_results = []
            sim_history = []  # Store history for this simulation
            
            # Run the simulation
            for step_idx, result in enumerate(simulation.run(query, steps), 1):
                progress = {
                    'current_sim': sim_idx,
                    'total_sims': total_simulations,
                    'current_sim_step': step_idx,
                    'total_steps': steps,
                    'percentage': ((sim_idx - 1) * steps + step_idx) / (total_simulations * steps) * 100
                }
                
                # Store the result and history
                sim_results.append(result)
                sim_history.append(result)
                
                # Print the current step's actions for debugging
                if "actions" in result:
                    if self.debug:
                        print(f"Step {step_idx} actions: {result['actions']}")
                
                # Yield progress and current data
                yield progress, result
            
            # Store simulation history
            simulation_histories.append(sim_history)
        
        # Generate final results using extracted method
        final_result = self._generate_experiment_results(simulation_histories, expected_steps=steps)
        if final_result:
            self.experiment_results = final_result
            yield {'percentage': 100}, final_result

    def _process_results(self, results):
        """Process and analyze the results"""
        # Group results by simulation
        simulation_results = []
        current_sim_results = []
        
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                progress, data = result
                if progress.get('current_sim_step', 0) == 1:  # Start of new simulation
                    if current_sim_results:
                        simulation_results.append(current_sim_results)
                    current_sim_results = []
                current_sim_results.append(data)
        
        if current_sim_results:
            simulation_results.append(current_sim_results)
            
        # Analyze outcomes for each simulation
        outcome_results = []
        for sim_results in simulation_results:
            # Extract the history from the simulation results
            history = []
            for step in sim_results:
                if isinstance(step, dict) and "history" in step:
                    history.extend(step["history"])
                elif isinstance(step, dict):
                    history.append(step)
            
            # Analyze the outcome based on the complete history
            outcome_analysis = self.analyze_outcome(history)
            outcome_results.append(outcome_analysis)
            
            # Print debug info for each run
            if self.debug:
                print(f"\nRun outcomes: {outcome_analysis}")
            
        # Calculate statistics
        statistics = self._calculate_statistics(outcome_results)
        
        # Store results for later use
        self.experiment_results = {
            "runs": simulation_results,
            "statistics": statistics
        }
        
        return self.experiment_results
        
    def _calculate_statistics(self, outcome_results: List[Dict[str, bool]]) -> Dict[str, Dict]:
        """
        Calculate statistics for each outcome across all simulations.
        """
        statistics = {}
        num_simulations = len(outcome_results)
        
        if num_simulations == 0:
            if self.debug:
                print("Warning: No simulation results to analyze")
            return {}
            
        # Initialize statistics for each outcome
        for outcome_name in self.outcome_definitions:
            statistics[outcome_name] = {
                "count": 0,
                "percentage": 0.0,
                "description": self.outcome_definitions[outcome_name]["description"]
            }
        
        # Count occurrences of each outcome
        for result in outcome_results:
            if not result:  # Skip empty results
                continue
            if self.debug:
                print(f"\nAnalyzing outcome result: {result}")  # Debug print
            for outcome_name, occurred in result.items():
                if occurred:
                    statistics[outcome_name]["count"] += 1
        
        # Calculate percentages
        for outcome_name in statistics:
            count = statistics[outcome_name]["count"]
            statistics[outcome_name]["percentage"] = (count / num_simulations) * 100 if num_simulations > 0 else 0.0
            
        # Print debug information
        if self.debug:
            print("\nOutcome counts:")
            for outcome_name, stats in statistics.items():
                print(f"{outcome_name}: {stats['count']}/{num_simulations}")
            
        return statistics
        
    def _generate_plot(self, output_dir: str):
        """
        Generate a plot of the experiment results.
        
        Args:
            output_dir: Directory to save the plot in
        """
        if not hasattr(self, 'experiment_results') or not self.experiment_results:
            if self.debug:
                print("Warning: No results to plot")
            return
            
        # Create a bar plot of outcome frequencies
        outcomes = list(self.outcome_definitions.keys())
        counts = [self.experiment_results['statistics'][outcome]['count'] 
                 for outcome in outcomes]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(outcomes, counts)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title(f'Outcome Distribution: {self.name}')
        plt.xlabel('Outcome')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'outcome_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        
        if self.debug:
            print(f"Plot saved to {plot_path}")

    def save_results(self,
                     output_dir: str,
                     plot_results: bool = True,
                     results: List[Dict] = None):
        """
        Save experiment results to the specified directory, including a new
        sub-folder that stores per-simulation agent_outcomes.
        """
        if results is None:
            results = []

        # Ensure root results directory exists
        os.makedirs(output_dir, exist_ok=True)
        if self.debug:
            print(f"Created results directory at: {os.path.abspath(output_dir)}")

        # ------------------------------------------------------------------ #
        # 1.  Save the detailed trace for every run (existing behaviour + AO)
        # ------------------------------------------------------------------ #
        for i, (result, history) in enumerate(
            zip(results, self.experiment_results.get("histories", []))
        ):
            sim_dir = os.path.join(output_dir, f"run_{i+1}")
            os.makedirs(sim_dir, exist_ok=True)

            trace = {
                "steps": history,
                "outcome_analysis": result.get("outcome_analysis", {}),
                "statistics": self.experiment_results["statistics"],
                # include agent outcomes in the trace for completeness
                "agent_outcomes": result.get("agent_outcomes", {})
            }

            with open(os.path.join(sim_dir, "trace.json"), "w") as f:
                json.dump(trace, f, indent=2)
            if self.debug:
                print(f"Saved trace for run {i+1}")

        # ------------------------------------------------------------------ #
        # 2.  NEW: write agent_outcomes for each run into a dedicated folder
        # ------------------------------------------------------------------ #
        agent_outcomes_dir = os.path.join(output_dir, "agent_outcomes")
        os.makedirs(agent_outcomes_dir, exist_ok=True)

        for i, result in enumerate(results):
            ao_path = os.path.join(
                agent_outcomes_dir, f"run_{i+1}_agent_outcomes.json"
            )
            with open(ao_path, "w") as f:
                json.dump(result.get("agent_outcomes", {}), f, indent=2)
            if self.debug:
                print(f"Saved agent outcomes for run {i+1} to {ao_path}")

        # ------------------------------------------------------------------ #
        # 3.  Summary & optional plot (unchanged)
        # ------------------------------------------------------------------ #
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Experiment: {self.name}\n")
            f.write(f"Number of simulations: {len(results)}\n\n")
            
            # Add LLM usage statistics
            if hasattr(self, 'experiment_results') and 'llm_usage' in self.experiment_results:
                llm_stats = self.experiment_results['llm_usage']
                f.write("LLM Usage Statistics:\n")
                f.write(f"  Total Input Characters: {llm_stats['total_input_characters']:,}\n")
                f.write(f"  Total Output Characters: {llm_stats['total_output_characters']:,}\n")
                f.write(f"  Total Characters: {llm_stats['total_characters']:,}\n")
                f.write(f"  Total LLM Calls: {llm_stats['total_call_count']}\n")
                f.write(f"  Average Input per Simulation: {llm_stats['avg_input_per_simulation']:.1f}\n")
                f.write(f"  Average Output per Simulation: {llm_stats['avg_output_per_simulation']:.1f}\n")
                f.write(f"  Average Calls per Simulation: {llm_stats['avg_calls_per_simulation']:.1f}\n\n")
            
            f.write("Outcome Statistics:\n")
            for outcome_name, stats in self.experiment_results["statistics"].items():
                f.write(f"\n{outcome_name}:\n")
                f.write(f"  Count: {stats['count']}/{len(results)}\n")
                f.write(f"  Percentage: {stats['percentage']:.1f}%\n")
                f.write(f"  Description: {stats['description']}\n")
        if self.debug:
            print(f"Summary saved to {summary_path}")

        if plot_results:
            self._generate_plot(output_dir)

    def _calculate_outcome_probability(self, outcome_name: str, history: List[Dict]) -> float:
        """
        Calculate the probability of a specific outcome based on the simulation history.
        
        Args:
            outcome_name: Name of the outcome to calculate probability for
            history: List of simulation steps
            
        Returns:
            float: Probability of the outcome (0.0 to 1.0)
        """
        # Count occurrences of key actions or states that indicate this outcome
        outcome_indicators = 0
        total_steps = len(history)
        
        if total_steps == 0:
            return 0.0
            
        # Look for indicators in each step
        for step in history:
            if isinstance(step, dict):
                # Check agent actions
                if "actions" in step:
                    for action in step["actions"]:
                        if isinstance(action, str) and outcome_name.lower() in action.lower():
                            outcome_indicators += 1
                
                # Check agent states
                if "agent_states" in step:
                    for agent_state in step["agent_states"].values():
                        if isinstance(agent_state, str) and outcome_name.lower() in agent_state.lower():
                            outcome_indicators += 1
                        elif isinstance(agent_state, dict):
                            # Handle nested dictionaries
                            for value in agent_state.values():
                                if isinstance(value, str) and outcome_name.lower() in value.lower():
                                    outcome_indicators += 1
                
                # Check environment state
                if "environment_state" in step:
                    env_state = step["environment_state"]
                    if isinstance(env_state, str) and outcome_name.lower() in env_state.lower():
                        outcome_indicators += 1
                    elif isinstance(env_state, list):
                        for item in env_state:
                            if isinstance(item, str) and outcome_name.lower() in item.lower():
                                outcome_indicators += 1
        
        # Calculate probability based on indicators
        probability = outcome_indicators / total_steps if total_steps > 0 else 0.0
        return probability 

    def run_manual(self, steps: int = 5, time_scale: str = None, use_batched: bool = False) -> Generator[Tuple[Dict, Dict], None, None]:
        """
        Run the experiment manually with optional batching.
        
        Args:
            steps: Number of steps per simulation
            time_scale: Optional time scale for TimescaleAwareAgents
            use_batched: Whether to use batched processing within each simulation
            
        Yields:
            Tuple of (progress, data) for each step
        """
        total_simulations = len(self.simulations)
        simulation_histories = []
        
        for sim_idx, simulation in enumerate(self.simulations, 1):
            if self.debug:
                print(f"\nRunning simulation {sim_idx}/{total_simulations}")
            
            sim_results = []
            sim_history = []
            
            # Choose the appropriate run method
            run_method = simulation.run_manual_batching if use_batched else simulation.run_manual
            
            # Run the simulation
            for step_idx, result in enumerate(run_method(steps, time_scale), 1):
                progress = {
                    'current_sim': sim_idx,
                    'total_sims': total_simulations,
                    'current_sim_step': step_idx,
                    'total_steps': steps,
                    'percentage': ((sim_idx - 1) * steps + step_idx) / (total_simulations * steps) * 100
                }
                
                # Store the result and history
                sim_results.append(result)
                sim_history.append(result)
                
                # Yield progress and current data
                yield progress, result
            
            # Store simulation history
            simulation_histories.append(sim_history)
        
        # Generate final results using extracted method
        final_result = self._generate_experiment_results(simulation_histories, expected_steps=steps)
        if final_result:
            self.experiment_results = final_result
            yield {'percentage': 100}, final_result

    def _build_simulation_result(self, simulation, sim_history, sim_index, batched_summary=None, batched_agent_outcomes=None, expected_steps=None):
        """Build result dict for a single simulation."""
        # Helper: decide if a single step indicates a failed simulation (placeholders or explicit failures)
        def _is_failed_step(step):
            """Return True if the step is a placeholder (None) or a dict with failed_agents."""
            return (step is None) or (isinstance(step, dict) and step.get("failed_agents"))

        # ------------------------------------------------------------
        # Determine whether the simulation history represents a valid
        # (i.e. non-failed) simulation.  Treat placeholder `None` steps
        # inserted by `run_manual_batch` as failures so that they do not
        # trigger AttributeError further below.
        # ------------------------------------------------------------
        if expected_steps is not None:
            is_valid = (
                sim_history and
                len(sim_history) == expected_steps and
                not any(_is_failed_step(step) for step in sim_history)
            )
        else:
            is_valid = (
                sim_history and
                not any(_is_failed_step(step) for step in sim_history)
            )
        
        if not is_valid:
            if expected_steps is not None:
                failure_reason = (
                    "no history" if not sim_history else
                    f"incomplete steps ({len(sim_history)}/{expected_steps})" if len(sim_history) != expected_steps else
                    "had failed agents"
                )
            else:
                failure_reason = (
                    "no history" if not sim_history else
                    "had failed agents"
                )
            return {
                "steps": [],
                "outcome_analysis": {},
                "environment": [],
                "agent_states": {},
                "actions": [],
                "agent_outcomes": {},
                "summary": f"Simulation failed - {failure_reason}",
                "failed": True
            }

        # Process successful simulation
        if batched_summary and batched_summary.get("success"):
            # Use batched agent outcomes if available, otherwise fall back to simulation's outcomes
            agent_outcomes = batched_agent_outcomes if batched_agent_outcomes is not None else simulation.agent_outcomes
            final_result = {
                "summary": batched_summary["summary"],
                "environment_state": simulation.env.get_state(),
                "agent_states": {aid: agent.state for aid, agent in simulation.agents.items()},
                "agent_outcomes": agent_outcomes
            }
        else:
            final_result = simulation._generate_final_summary(sim_history)
            # If we have batched agent outcomes, use them instead of the ones from final summary
            if batched_agent_outcomes is not None:
                final_result["agent_outcomes"] = batched_agent_outcomes

        outcome = self.analyze_outcome(sim_history)
        return {
            "steps": sim_history,
            "outcome_analysis": outcome,
            "environment": final_result.get("environment_state", []),
            "agent_states": final_result.get("agent_states", {}),
            "actions": [
                step.get("actions", []) if isinstance(step, dict) else []
                for step in sim_history
            ],
            "agent_outcomes": final_result.get("agent_outcomes", {}),
            "summary": final_result.get("summary", ""),
            "failed": False
        }

    def _generate_experiment_results(self, simulation_histories, expected_steps=None):
        """
        Generate final results for all simulations using batched summary processing.
        
        Args:
            simulation_histories: List of simulation histories
            expected_steps: Expected number of steps per simulation
        """
        # Use Simulation's batch summary processing only if not disabled
        summaries = []
        if not self.disable_batch_summaries:
            summaries = Simulation.batch_process_summaries(
                simulation_histories,
                llm=self.simulations[0].orchestrator.llm if self.simulations else None,
                debug=self.debug
            )
            if self.debug:
                print(f"Generated {len(summaries)} batch summaries")
        else:
            if self.debug:
                print("Batch summary processing disabled - skipping expensive LLM calls")
            # Create dummy summaries to maintain compatibility
            summaries = [{"success": False, "summary": "Batch summaries disabled", "error": None} 
                        for _ in simulation_histories]
        
        # Batch analyze agent outcomes for all simulations
        outcome_results = []
        if self.simulations:
            try:
                outcome_results = Simulation.batch_analyze_agent_outcomes(
                    self.simulations,
                    simulation_histories,
                    llm=self.simulations[0].orchestrator.llm,
                    debug=self.debug
                )
                if self.debug:
                    print(f"Batch analyzed agent outcomes: {outcome_results}")
            except Exception as e:
                if self.debug:
                    print(f"Failed to batch analyze agent outcomes: {e}")
                # Create empty results for failed analyses
                outcome_results = [None] * len(self.simulations)
        else:
            outcome_results = []
        
        # Build results for each simulation
        all_results = []
        for sim_index in range(len(self.simulations)):
            simulation = self.simulations[sim_index]
            sim_history = simulation_histories[sim_index]
            batched_summary = summaries[sim_index] if summaries else None
            batched_agent_outcomes = outcome_results[sim_index] if sim_index < len(outcome_results) else None
            
            sim_result = self._build_simulation_result(
                simulation=simulation,
                sim_history=sim_history,
                sim_index=sim_index,
                batched_summary=batched_summary,
                batched_agent_outcomes=batched_agent_outcomes,
                expected_steps=expected_steps
            )
            all_results.append(sim_result)
        
        # Calculate final statistics and assemble results
        if all_results:
            statistics = self._calculate_statistics([r['outcome_analysis'] for r in all_results])
            
            # Aggregate LLM usage from all simulations
            llm_usage_stats = self.aggregate_llm_usage()
            
            final_result = {
                "runs": all_results,
                "statistics": statistics,
                "histories": simulation_histories,
                "llm_usage": llm_usage_stats
            }
            return final_result
        else:
            if self.debug:
                print("Warning: No results were generated from the experiment")
            return None

    def aggregate_llm_usage(self) -> dict:
        """
        Aggregate LLM usage statistics from all simulations.
        
        Returns:
            Dict with total usage statistics across all simulations
        """
        total_input = 0
        total_output = 0
        total_calls = 0
        individual_stats = []
        
        for i, simulation in enumerate(self.simulations):
            if hasattr(simulation, 'get_llm_usage_stats'):
                stats = simulation.get_llm_usage_stats()
                total_input += stats['input_characters']
                total_output += stats['output_characters']
                total_calls += stats['call_count']
                
                individual_stats.append({
                    "simulation_index": i,
                    "simulation_id": getattr(simulation, 'simulation_id', f'sim_{i}'),
                    "input_characters": stats['input_characters'],
                    "output_characters": stats['output_characters'],
                    "total_characters": stats['total_characters'],
                    "call_count": stats['call_count']
                })
        
        return {
            "total_input_characters": total_input,
            "total_output_characters": total_output,
            "total_characters": total_input + total_output,
            "total_call_count": total_calls,
            "simulations_count": len(self.simulations),
            "avg_input_per_simulation": total_input / len(self.simulations) if self.simulations else 0,
            "avg_output_per_simulation": total_output / len(self.simulations) if self.simulations else 0,
            "avg_calls_per_simulation": total_calls / len(self.simulations) if self.simulations else 0,
            "individual_simulations": individual_stats
        }

    def run_manual_batch(self, steps: int = 5, time_scale: str = None) -> Generator[Tuple[Dict, Dict], None, None]:
        """
        Run the experiment with cross-simulation batching for better performance.
        All simulations run simultaneously with agents batched together.
        """
        total_simulations = len(self.simulations)
        if total_simulations == 0:
            return
        
        # Initialize simulation histories
        simulation_histories = [[] for _ in range(total_simulations)]
        active_simulations = set(range(total_simulations))
        
        # Run all steps across all simulations simultaneously
        for step in range(steps):
            if self.debug:
                print(f"\nRunning step {step + 1}/{steps} across {len(active_simulations)} active simulations...")
            
            # Create step placeholders for all simulations first
            for sim_index in range(total_simulations):
                if sim_index not in active_simulations:
                    # Failed simulation gets None placeholder
                    simulation_histories[sim_index].append(None)
            
            if not active_simulations:
                # All simulations failed, break out early to avoid unnecessary looping
                if self.debug:
                    print("All simulations have failed – ending run_manual_batch early.")
                break
            
            # Collect all agents from all active simulations for batching
            agents_data = []
            result_mapping = {}  # Maps result index to (sim_index, agent_id, agent_context)
            
            for sim_index in active_simulations:
                simulation = self.simulations[sim_index]
                
                # Get failed agents from previous steps
                failed_agents = set()
                if simulation_histories[sim_index]:
                    last_step = simulation_histories[sim_index][-1]
                    failed_agents = set(last_step.get("failed_agents", []))
                
                # Get active agents
                active_agents = {aid: agent for aid, agent in simulation.agents.items() 
                               if aid not in failed_agents}
                
                if not active_agents:
                    active_simulations.discard(sim_index)
                    continue
                
                # Prepare agent data using existing simulation logic
                for agent_id in active_agents.keys():
                    # Reuse existing methods from simulation
                    visible_state = simulation.env.snapshot_for_agent(agent_id, simulation.graph)
                    messages_with_senders = simulation._get_messages_for_agent(agent_id)
                    
                    agent_data = {
                        "agent": simulation.agents[agent_id],
                        "agent_id": agent_id,
                        "simulation_index": sim_index,
                        "visible_state": visible_state,
                        "messages_with_senders": messages_with_senders,
                        "step": step + 1,
                        "total_steps": steps
                    }
                    
                    if time_scale:
                        agent_data["time_scale"] = time_scale
                    
                    agents_data.append(agent_data)
                    
                    # Store context for _process_batch_responses
                    agent_context = {
                        "agent_id": agent_id,
                        "visible_state": visible_state,
                        "messages_with_senders": messages_with_senders
                    }
                    
                    result_mapping[len(agents_data) - 1] = (sim_index, agent_id, agent_context)
            
            if not agents_data:
                # No active agents this step, continue to next step (failed sims will get None placeholders)
                continue
            
            # Use the class method to batch process all agents
            batch_results = Simulation.batch_process_agents(agents_data, debug=self.debug)
            
            # Group results by simulation and use existing _process_batch_responses
            simulation_results = {}
            for result_index, result in enumerate(batch_results):
                sim_index, agent_id, agent_context = result_mapping[result_index]
                
                if sim_index not in simulation_results:
                    simulation_results[sim_index] = {
                        'agent_ids': [],
                        'batch_responses': [],
                        'agent_contexts': []
                    }
                
                simulation_results[sim_index]['agent_ids'].append(agent_id)
                simulation_results[sim_index]['batch_responses'].append(result)
                simulation_results[sim_index]['agent_contexts'].append(agent_context)
            
            # Process each simulation's results using existing method
            step_data = {}
            for sim_index in simulation_results:
                simulation = self.simulations[sim_index]
                sim_data = simulation_results[sim_index]
                
                # Leverage existing _process_batch_responses method
                step_actions, step_failed_agents = simulation._process_batch_responses(
                    sim_data['agent_ids'],
                    sim_data['batch_responses'], 
                    sim_data['agent_contexts'],
                    step
                )
                
                # If any agents failed, remove this simulation
                if step_failed_agents:
                    active_simulations.discard(sim_index)
                    if self.debug:
                        print(f"Simulation {sim_index + 1} removed due to failed agents: {step_failed_agents}")
                    # Don't add step result for failed simulation - it will get None placeholder next iteration
                    continue
                
                # Create step result using same format as existing methods
                step_result = {
                    "step": step + 1,
                    "actions": step_actions,
                    "environment": simulation.env.get_state(),
                    "agent_states": {aid: agent.state for aid, agent in simulation.agents.items()},
                    "failed_agents": step_failed_agents
                }
                
                simulation_histories[sim_index].append(step_result)
                step_data[sim_index] = step_result
            
            # Yield progress
            progress = {
                'current_step': step + 1,
                'total_steps': steps,
                'active_simulations': len(active_simulations),
                'total_simulations': total_simulations,
                'percentage': ((step + 1) / steps) * 100
            }
            
            yield progress, {'step': step + 1, 'simulation_results': step_data}
        
        # Generate final results using extracted method
        final_result = self._generate_experiment_results(simulation_histories, expected_steps=steps)
        if final_result:
            self.experiment_results = final_result
            yield {'percentage': 100}, final_result
        else:
            if self.debug:
                print("Warning: No results were generated from the experiment") 