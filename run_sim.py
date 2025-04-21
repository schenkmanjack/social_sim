import argparse
from simulation import Simulation
from orchestrator import Orchestrator
from llm_interface import LLMWrapper, OpenAIBackend

def main():
    parser = argparse.ArgumentParser(description="Run a social simulation scenario.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Simulation query (e.g., 'Trump tariffs on Canadian oil')"
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of simulation steps")
    parser.add_argument("--model", type=str, default="gpt-4", help="OpenAI model to use")

    args = parser.parse_args()

    backend = OpenAIBackend(model=args.model)
    llm = LLMWrapper(backend)
    orchestrator = Orchestrator(llm)
    simulation = Simulation(orchestrator, llm)

    summary_json = simulation.run(query=args.query, steps=args.steps)

    print("\n=== SIMULATION SUMMARY ===\n")
    print(summary_json)

if __name__ == "__main__":
    main()
