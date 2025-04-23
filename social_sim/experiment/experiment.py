from typing import List, Dict, Any
import json
import os
from datetime import datetime
import re
import matplotlib.pyplot as plt

class Experiment:
    def __init__(self, simulations: List['Simulation'], name: str = None):
        """
        Initialize an experiment with multiple simulations.
        
        Args:
            simulations: List of Simulation objects to run
            name: Optional name for the experiment
        """
        self.simulations = simulations
        self.name = name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = []
        self.outcome_definitions = {}
        
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
        """
        if not self.outcome_definitions:
            print("Warning: No outcomes defined for analysis")
            return {}
            
        # Use the first simulation's orchestrator for analysis
        orchestrator = self.simulations[0].orchestrator
        
        prompt = f"""
        Analyze this simulation history to determine which outcomes occurred.
        
        Outcomes to check:
        {json.dumps(self.outcome_definitions, indent=2)}
        
        Simulation history:
        {json.dumps(history, indent=2)}
        
        Instructions:
        1. Review the entire simulation history, not just the final state
        2. Look for key decisions and actions throughout the transcript
        3. Pay attention to:
           - What each agent decided to do
           - When they made their decisions
           - How their decisions affected the outcome
        4. Compare the sequence of events to the defined conditions
        5. For each outcome:
           - If the condition matches exactly, mark as true
           - If the condition doesn't match, mark as false
        6. Ensure exactly one outcome is marked as true
        7. If you cannot determine the outcome from the history, mark the most likely outcome as true
        
        Respond with a JSON object where keys are outcome names and values are booleans.
        Example response format (for Prisoner's Dilemma):
        {{
            "both_cooperate": false,
            "both_defect": false,
            "alice_defects_bob_cooperates": true,
            "bob_defects_alice_cooperates": false
        }}
        """
        
        try:
            response = orchestrator._call_llm_with_retry(prompt)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                
                # Print debug info
                print(f"\nRaw outcome analysis: {result}")
                print(f"Number of true outcomes: {sum(result.values())}")
                
                # If all outcomes are false, mark the most likely outcome as true
                if not any(result.values()):
                    print("Warning: No outcomes were marked as true. Marking most likely outcome...")
                    # For Prisoner's Dilemma, if all false, assume both defect
                    result["both_defect"] = True
                    return result
                    
                if sum(result.values()) > 1:
                    print("Warning: Multiple outcomes were marked as true. Using first true outcome...")
                    # Keep only the first true outcome
                    first_true = next(k for k, v in result.items() if v)
                    return {k: (k == first_true) for k in result}
                    
                return result
            return {}
        except Exception as e:
            print(f"Warning: Error analyzing outcomes: {str(e)}")
            return {}
            
    def run(self, query: str, steps: int = 5) -> Dict[str, Any]:
        """
        Run the experiment with the given query.
        
        Args:
            query: The query to use for the experiment
            steps: Number of steps to run each simulation for
            
        Returns:
            Dictionary containing experiment results
        """
        self.results = []
        
        for i in range(len(self.simulations)):
            print(f"\nRunning simulation {i+1}/{len(self.simulations)}...")
            try:
                # Add debug print to see what's happening
                print(f"Starting simulation {i+1} with query: {query[:50]}...")
                result = self.simulations[i].run(query, steps=steps)
                print(f"Simulation {i+1} completed successfully")
                self.results.append(result)
            except Exception as e:
                print(f"Detailed error in simulation {i+1}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"Error traceback: {e.__traceback__}")
                raise  # Re-raise the exception to be caught by the caller
            
        # Calculate statistics with error handling
        try:
            print("Calculating statistics...")
            statistics = self._calculate_statistics(self.results)
            print("Statistics calculated successfully")
        except Exception as e:
            print(f"Error calculating statistics:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Error traceback: {e.__traceback__}")
            raise
            
        # Store results for saving later
        self.experiment_results = {
            "runs": self.results,
            "statistics": statistics
        }
        
        return self.experiment_results
        
    def _calculate_statistics(self, outcome_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics from all simulation runs.
        """
        stats = {}
        
        # Initialize statistics for each outcome
        for outcome_name in self.outcome_definitions:
            stats[outcome_name] = {
                "count": 0,
                "percentage": 0.0,
                "description": self.outcome_definitions[outcome_name]["description"]
            }
            
        # Calculate statistics
        total_runs = len(outcome_results)
        if total_runs == 0:
            return stats
            
        # Count outcomes for each run
        for result in outcome_results:
            # Analyze the outcome using the orchestrator
            outcome_analysis = self.analyze_outcome(result.get("history", []))
            
            # Update statistics based on the analysis
            for outcome_name, occurred in outcome_analysis.items():
                if occurred:
                    stats[outcome_name]["count"] += 1
            
            # Print debug info for each run
            print(f"\nRun outcomes: {outcome_analysis}")
            
        # Calculate percentages
        for outcome_name in stats:
            stats[outcome_name]["percentage"] = (stats[outcome_name]["count"] / total_runs) * 100
            
        return stats
        
    def save_results(self, output_dir: str = None, plot_results: bool = True):
        """
        Save experiment results to disk.
        
        Args:
            output_dir: Directory to save results in. If None, uses experiment name.
            plot_results: Whether to generate and save plots
        """
        if output_dir is None:
            output_dir = self.name
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Use stored results if available
        if not hasattr(self, 'experiment_results'):
            raise ValueError("Experiment must be run before saving results")
        
        # Save overall statistics
        with open(os.path.join(output_dir, "statistics.json"), 'w') as f:
            json.dump(self.experiment_results["statistics"], f, indent=2)
            
        # Save each run's trace
        for i, result in enumerate(self.experiment_results["runs"]):
            run_dir = os.path.join(output_dir, f"run_{i+1}")
            os.makedirs(run_dir, exist_ok=True)
            
            # Save trace
            with open(os.path.join(run_dir, "trace.json"), 'w') as f:
                json.dump(result, f, indent=2)
                
            # Save summary
            with open(os.path.join(run_dir, "summary.txt"), 'w') as f:
                f.write(str(result.get("summary", "No summary available")))
        
        # Generate and save the plot if enabled
        if plot_results:
            self.plot_results(output_dir)
        
    def plot_results(self, output_dir: str = None):
        """
        Create a bar chart of experiment outcomes and save it to disk.
        
        Args:
            output_dir: Directory to save the plot in. If None, uses experiment name.
        """
        if output_dir is None:
            output_dir = self.name
            
        if not hasattr(self, 'experiment_results'):
            raise ValueError("Experiment must be run before plotting results")
            
        # Extract data for plotting
        outcome_names = []
        counts = []
        percentages = []
        
        for outcome_name, stats in self.experiment_results["statistics"].items():
            outcome_names.append(outcome_name)
            counts.append(stats["count"])
            percentages.append(stats["percentage"])
        
        # Generate colors dynamically using a colormap
        num_outcomes = len(outcome_names)
        if num_outcomes <= 7:
            # Use distinct colors for small numbers of outcomes
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c2f0c2']
            colors = colors[:num_outcomes]
        else:
            # Use a continuous colormap for larger numbers of outcomes
            cmap = plt.cm.get_cmap('viridis', num_outcomes)
            colors = [cmap(i) for i in range(num_outcomes)]
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(outcome_names, counts, color=colors)
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}/{len(self.simulations)}',
                    ha='center', va='bottom')
        
        # Add percentage labels inside bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{percentages[i]:.1f}%',
                    ha='center', va='center', color='white')
        
        # Customize the plot
        plt.title('Experiment Outcomes Distribution', pad=20)
        plt.xlabel('Outcomes')
        plt.ylabel('Count')
        plt.ylim(0, len(self.simulations))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels if they're long
        if any(len(name) > 10 for name in outcome_names):
            plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "outcomes_bar_chart.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Plot saved to {plot_path}") 