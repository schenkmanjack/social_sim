import argparse
import os
import json
from social_sim.llm_interfaces import OpenAIBackend
from social_sim.simulation import Simulation

def save_trace(trace_data, trace_file="full_trace.txt"):
    """Save detailed simulation trace to file"""
    with open(trace_file, 'w') as f:
        f.write("=== Initial Setup ===\n")
        f.write("\nAgents:\n")
        for agent in trace_data["setup"]["agents"]:
            f.write(f"- {agent['id']}: {agent['identity']}\n")
        
        f.write("\nInitial Environment:\n")
        for fact in trace_data["setup"]["environment"]["facts"]:
            f.write(f"- {fact}\n")
        
        f.write("\n=== Simulation History ===\n")
        for step in trace_data["history"]:
            f.write(f"\nStep {step['step']}:\n")
            for action in step["actions"]:
                f.write(f"\n{action['agent']} ({action['identity']}):\n")
                f.write("Visible state:\n")
                for fact in action["visible_state"]:
                    f.write(f"- {fact}\n")
                if action["received_messages"]:
                    f.write("Received messages:\n")
                    for msg in action["received_messages"]:
                        f.write(f"- {msg}\n")
                f.write(f"Action: {action['action']}\n")

def save_summary(summary, summary_file="summary.txt"):
    """Save simulation summary to file"""
    with open(summary_file, 'w') as f:
        f.write(summary)

def main():
    parser = argparse.ArgumentParser(description='Run a social simulation')
    parser.add_argument('--query', type=str, required=True, help='Query to simulate')
    parser.add_argument('--steps', type=int, default=5, help='Number of simulation steps')
    parser.add_argument('--trace-file', type=str, default="full_trace.txt", 
                       help='File to save detailed trace (default: full_trace.txt)')
    parser.add_argument('--summary-file', type=str, default="summary.txt",
                       help='File to save summary (default: summary.txt)')
    parser.add_argument('--agent-type', type=str, default="regular",
                       choices=["regular", "timescale_aware"],
                       help='Type of agent to use in simulation (default: regular)')
    parser.add_argument('--chunk-size', type=int, default=1200,
                       help='Number of steps to include in each summary chunk (default: 1200)')
    args = parser.parse_args()

    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Initialize LLM with your API key
    llm = OpenAIBackend(api_key=api_key)
    
    # Create and run simulation
    simulation = Simulation(llm, agent_type=args.agent_type, chunk_size=args.chunk_size)
    result = simulation.run(query=args.query, steps=args.steps)
    
    # Save detailed trace
    save_trace(result, args.trace_file)
    print(f"Detailed trace saved to {args.trace_file}")
    
    # Save summary
    save_summary(result["summary"], args.summary_file)
    print(f"Summary saved to {args.summary_file}")

if __name__ == "__main__":
    main()
