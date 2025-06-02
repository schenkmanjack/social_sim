import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from social_sim.llm_interfaces import OpenAIBackend
from social_sim.simulation import Simulation
from social_sim.orchestrator import Orchestrator
from wordcloud import WordCloud
import re
import time
from openai import OpenAI

def save_trace(trace_data, trace_file="full_trace.txt"):
    """Save detailed simulation trace to file"""
    if not trace_data or not isinstance(trace_data, dict):
        print(f"Warning: Invalid trace_data provided to save_trace. Skipping.")
        return

    setup_data = trace_data.get("setup", {})
    history_data = trace_data.get("history", [])

    with open(trace_file, 'w') as f:
        f.write("=== Initial Setup ===\n")

        agents = setup_data.get("agents", [])
        if agents:
            f.write("\nAgents:\n")
            for agent in agents:
                f.write(f"- {agent.get('id', 'N/A')}: {agent.get('identity', 'N/A')}\n")
        else:
            f.write("\nAgents: Not available\n")

        environment = setup_data.get("environment", {})
        facts = environment.get("facts", [])
        if facts:
            f.write("\nInitial Environment:\n")
            for fact in facts:
                f.write(f"- {fact}\n")
        else:
            f.write("\nInitial Environment: Not available\n")

        f.write("\n=== Simulation History ===\n")
        if not history_data:
             f.write("No history recorded.\n")

        for step_idx, step in enumerate(history_data):
            f.write(f"\nStep {step.get('step', step_idx)}:\n")
            actions = step.get("actions", [])
            if not actions:
                f.write("  No actions recorded for this step.\n")
                continue

            for action in actions:
                agent_id = action.get('agent', 'Unknown Agent')
                identity = action.get('identity', 'Unknown Identity')
                f.write(f"\n  {agent_id} ({identity}):\n")

                visible_state = action.get("visible_state", [])
                f.write("    Visible state:\n")
                if not visible_state:
                    f.write("      - None\n")
                else:
                    for fact in visible_state:
                        f.write(f"      - {fact}\n")

                received_messages = action.get("received_messages", [])
                if received_messages:
                    f.write("    Received messages:\n")
                    for msg in received_messages:
                        f.write(f"      - {msg if msg is not None else '[No message content]'}\n")

                action_desc = action.get('action', '[No action recorded]')
                f.write(f"    Action: {action_desc}\n")


def save_summary(summary, summary_file="summary.txt"):
    """Save simulation summary to file"""
    with open(summary_file, 'w') as f:
        f.write(str(summary) if summary is not None else "Summary was not generated.")

def determine_category(action_text, metric):
    """
    Determines the category for a given action based on the metric's keywords.
    For pie charts, we need to categorize actions into distinct groups.
    """
    # For pie charts, we need to determine which category this action belongs to
    if metric["visualization"] == "pie":
        # Example: For AI Chip Origin, categorize based on source
        if "AI Chip Origin" in metric["metric_name"]:
            if "China" in action_text:
                return "China"
            elif "America" in action_text:
                return "America"
            else:
                return "Other"
        # Add more category determination logic for other metrics
        return "Unknown"
    return None

# --- Helper function to get total unique agents ---
def get_total_agents(history):
    agent_ids = set()
    if not history:
        return 0
    for step in history:
        if "actions" in step:
            for action in step["actions"]:
                if "agent" in action: # Assuming agent ID is stored under 'agent' key
                    agent_ids.add(action["agent"])
    return len(agent_ids)
# --- End Helper ---

def extract_metric_data(history, metric):
    """
    Extracts relevant data from history based on metric specification.
    Focuses on the requested visualization type and uses keywords generically.
    """
    visualization = metric.get("visualization", "unknown").lower().replace(" chart", "").strip()
    keywords = [k.lower() for k in metric.get("keywords", [])]
    question = metric.get("question", "Metric")
    metric_name = metric.get("metric_name", "Metric")

    if not keywords:
        print(f"Warning: No keywords provided for metric '{metric_name}'. Cannot extract data.")
        return None

    # --- Find all actions relevant to the metric's keywords ---
    relevant_actions = []
    for step in history:
        for action in step.get("actions", []):
            action_text = action.get("action", "").lower()
            # Action is relevant if *any* of the metric's keywords are present
            if any(keyword in action_text for keyword in keywords):
                relevant_actions.append(action) # Store the whole action dict

    if not relevant_actions:
        print(f"Warning: No relevant actions found for metric '{metric_name}' using keywords: {keywords}")
        return None

    print(f"Info: Found {len(relevant_actions)} relevant actions for metric '{metric_name}'.")

    # --- Generate data based on Visualization Type ---

    # --- Bar or Pie Chart: Requires Categorization ---
    if visualization == "bar" or visualization == "pie":
        # Generic Strategy 1: Categorize by which keyword(s) matched
        category_counts = {kw: 0 for kw in keywords}
        unmatched_relevant_count = 0 # Should ideally be 0 if filtering is correct

        for action in relevant_actions:
            action_text = action.get("action", "").lower()
            matched_at_least_one = False
            for kw in keywords:
                if kw in action_text:
                    category_counts[kw] += 1
                    matched_at_least_one = True
                    # Note: An action can contribute to multiple keyword counts if it contains multiple keywords.
            if not matched_at_least_one:
                 unmatched_relevant_count += 1 # Track if any relevant actions didn't match keywords (shouldn't happen)


        # Filter zero counts
        category_counts = {k: v for k, v in category_counts.items() if v > 0}

        # If keyword categorization yields nothing useful (e.g., only one keyword dominates)
        # Generic Strategy 2: Fallback to categorizing by Agent ID
        if not category_counts or len(category_counts) < 2 : # If results are trivial
             print(f"Info: Keyword categorization for '{metric_name}' was trivial. Falling back to agent counts.")
             agent_counts = {}
             for action in relevant_actions:
                 agent_id = action.get("agent", "Unknown Agent") # Make sure 'agent' key exists in your action dict
                 agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1

             if agent_counts and len(agent_counts) >= 1: # Use agent counts if available and meaningful
                 category_counts = agent_counts
             elif not category_counts: # If agent counts also failed, return None
                 print(f"Warning: Could not categorize relevant actions for metric '{metric_name}' by keywords or agents.")
                 return None
             # If agent counts failed but keyword counts had one item, keep the keyword count


        return {
            "values": list(category_counts.values()),
            "labels": list(category_counts.keys()),
            "title": question,
            "xlabel": "Category" if visualization == "bar" else None,
            "ylabel": "Frequency" if visualization == "bar" else None
        }

    # --- Line Chart: Requires Time Series Data ---
    elif visualization == "line":
        # Generic Strategy: Count relevant actions per step
        values_per_step = []
        for i, step in enumerate(history):
            count_in_step = 0
            for action in step.get("actions", []):
                 action_text = action.get("action", "").lower()
                 # Check against the metric's keywords again for this step
                 if any(keyword in action_text for keyword in keywords):
                     count_in_step += 1
            values_per_step.append(count_in_step)

        # Check if all steps have zero counts
        if not any(v > 0 for v in values_per_step):
             print(f"Warning: No relevant actions found across any steps for line chart metric '{metric_name}'.")
             return None

        return {
            "x": list(range(len(history))),
            "y": values_per_step,
            "title": question,
            "xlabel": "Simulation Step",
            "ylabel": "Frequency of Relevant Actions"
        }

    # --- Word Cloud: Requires Text Aggregation ---
    elif visualization == "word cloud":
        # This logic is already fairly generic
        text_for_cloud = []
        stop_words = set(['the', 'and', 'to', 'of', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'a', 'an', 'it', 'that', 'this', 'will', 'be', 'has', 'have', 'its', 'their'])
        # Add metric keywords to stop words for the cloud
        stop_words.update(keywords)

        for action in relevant_actions:
            action_text = action.get("action", "")
            words = re.findall(r'\b\w+\b', action_text.lower())
            # Filter stop words, metric keywords, and short words
            words = [w for w in words if w not in stop_words and len(w) > 3]
            text_for_cloud.extend(words)

        word_freq = {}
        for word in text_for_cloud:
            word_freq[word] = word_freq.get(word, 0) + 1

        if not word_freq:
            print(f"Warning: No suitable words found for word cloud metric '{metric_name}' after filtering.")
            return None

        return { "word_freq": word_freq, "title": question }

    # --- Sentiment Analysis (Heuristic Trigger) ---
    # Keep basic version, triggered by name/question keywords. Less generic but hard to avoid.
    elif ("sentiment" in metric_name.lower() or "sentiment" in question.lower()) and visualization == "pie":
        print(f"Info: Attempting basic sentiment analysis for metric '{metric_name}'.")
        sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
        positive_words = ["positive", "good", "agree", "support", "benefit", "opportunity", "cooperate", "resolve", "success", "achieve", "effective", "helpful"]
        negative_words = ["negative", "bad", "disagree", "oppose", "harm", "risk", "restrict", "concern", "threat", "reject", "fail", "problem", "issue", "challenge", "difficult"]

        for action in relevant_actions:
            action_text = action.get("action", "").lower()
            pos_score = sum(1 for word in positive_words if word in action_text)
            neg_score = sum(1 for word in negative_words if word in action_text)

            if pos_score > neg_score:
                sentiments["Positive"] += 1
            elif neg_score > pos_score:
                sentiments["Negative"] += 1
            else:
                sentiments["Neutral"] += 1

        sentiments = {k: v for k, v in sentiments.items() if v > 0}
        if not sentiments:
             print(f"Warning: No sentiment identified for metric '{metric_name}'.")
             return None

        return {
            "values": list(sentiments.values()),
            "labels": list(sentiments.keys()),
            "title": question
        }

    # --- Fallback for unknown/unhandled visualization types ---
    else:
        print(f"Warning: Visualization type '{visualization}' is not implemented or not suitable for generic handling.")
        return None


def generate_plots(history, plot_keywords, results_folder):
    """
    Generate plots based on the simulation history and keywords
    """
    if not history or not plot_keywords:
        print("No history or plot keywords provided. Skipping plot generation.")
        return

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate plots for each set of keywords
    for keywords in plot_keywords:
        try:
            # Extract data for these keywords
            metric_data = extract_metric_data(history, {
                "keywords": keywords,
                "visualization": "line",  # Default visualization
                "metric_name": " ".join(keywords)
            })

            if not metric_data:
                print(f"No data found for keywords: {keywords}")
                continue

            # Generate plot
            plt.figure(figsize=(10, 6))
            plt.plot(metric_data['x'], metric_data['y'])
            plt.title(f"Trend for {' '.join(keywords)}")
            plt.xlabel("Step")
            plt.ylabel("Count")
            
            # Save plot
            filename = f"{'_'.join(keywords)}.png"
            filepath = os.path.join(plots_dir, filename)
            plt.savefig(filepath)
            plt.close()
            
            print(f"Plot saved to {filepath}")
            
        except Exception as e:
            print(f"Error generating plot for keywords {keywords}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run a social simulation')
    parser.add_argument('--query', type=str, required=True, help='Query to simulate')
    parser.add_argument('--steps', type=int, default=5, help='Number of simulation steps')
    parser.add_argument('--results-folder', type=str, default="results",
                       help='Folder to save trace, summary, and plots (default: results)')
    parser.add_argument('--plot-keywords', type=str, action='append', nargs='+',
                       help='Keywords defining an action category to plot (e.g., --plot-keywords defect betray --plot-keywords cooperate share)')
    parser.add_argument('--agent-type', type=str, default="regular",
                       choices=["regular", "timescale_aware"],
                       help='Type of agent to use in simulation (default: regular)')
    parser.add_argument('--chunk-size', type=int, default=1200,
                       help='Number of steps to include in each summary chunk (default: 1200)')
    parser.add_argument('--plots', action='store_true',
                       help='Enable plot generation (default: False)')
    args = parser.parse_args()

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    llm = OpenAIBackend(api_key=api_key)

    simulation = Simulation(llm, agent_type=args.agent_type, chunk_size=args.chunk_size)
    print(f"Running simulation for query: '{args.query}' with {args.steps} steps...")
    try:
        result = simulation.run(query=args.query, steps=args.steps)
        print("Simulation finished.")
    except Exception as e:
        print(f"\n--- Simulation Error ---")
        print(f"An error occurred during simulation run: {e}")
        print("Please check the query, API key, and simulation logic.")
        return

    try:
        os.makedirs(args.results_folder, exist_ok=True)
        print(f"Created results folder at: {os.path.abspath(args.results_folder)}")
    except OSError as e:
        print(f"Error creating results folder '{args.results_folder}': {e}. Using current directory.")
        args.results_folder = "."

    trace_filepath = os.path.join(args.results_folder, "full_trace.txt")
    summary_filepath = os.path.join(args.results_folder, "summary.txt")

    try:
        save_trace(result, trace_filepath)
        print(f"Detailed trace saved to {trace_filepath}")
    except Exception as e:
        print(f"Error saving trace file to {trace_filepath}: {e}")


    try:
        save_summary(result.get("summary"), summary_filepath)
        print(f"Summary saved to {summary_filepath}")
    except Exception as e:
        print(f"Error saving summary file to {summary_filepath}: {e}")


    # Only generate plots if explicitly enabled
    if args.plots:
        plot_keywords_to_use = None
        if args.plot_keywords:
            print("Using user-provided keywords for plotting.")
            plot_keywords_to_use = args.plot_keywords
        else:
            print("No plot keywords provided by user. Attempting to determine metrics via Orchestrator...")
            if hasattr(simulation, 'orchestrator'):
                keywords_from_orchestrator = simulation.orchestrator.determine_plot_metrics(
                    args.query,
                    result.get('history', []),
                    should_plot=args.plots
                )
                if keywords_from_orchestrator:
                    print(f"Orchestrator suggested keywords: {keywords_from_orchestrator}")
                    plot_keywords_to_use = keywords_from_orchestrator
                else:
                    print("Orchestrator could not determine plot keywords. Skipping plot generation.")
            else:
                print("Error: Simulation object does not have an orchestrator. Cannot determine plot keywords.")

        if plot_keywords_to_use:
            print("Generating plots...")
            try:
                generate_plots(result.get('history', []), plot_keywords_to_use, args.results_folder)
            except Exception as e:
                print(f"Error during plot generation: {e}")
    else:
        print("Plot generation disabled by default. Use --plots to enable.")


if __name__ == "__main__":
    import matplotlib
    try:
        matplotlib.use('Agg')
    except ImportError:
        print("Warning: Matplotlib 'Agg' backend not available. Plots might not save correctly on headless server.")
    main()

class Orchestrator:
    def __init__(self, model="gpt-3.5-turbo", delay_seconds=1):
        self.client = OpenAI()
        self.model = model
        self.delay_seconds = delay_seconds
        self.last_request_time = 0

    def _wait_for_rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.delay_seconds:
            time.sleep(self.delay_seconds - time_since_last_request)
        self.last_request_time = time.time()

    def determine_plot_metrics(self, query, history, should_plot: bool = False, max_tokens_per_chunk: int = 10000) -> list:
        """
        Analyzes the simulation history to identify meaningful metrics and visualization types.
        Only runs if should_plot is True.
        """
        if not should_plot:
            print("Plotting disabled, skipping metrics analysis")
            return []
        
        if not history:
            return []
        
        print("Orchestrator: Analyzing simulation for meaningful metrics...")
        
        # ... rest of the existing determine_plot_metrics code ...
