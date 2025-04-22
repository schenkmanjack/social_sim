from social_sim.orchestrator import Orchestrator
from social_sim.interactions import Environment, ConnectivityGraph
from social_sim.agents import Agent, TimescaleAwareAgent

class Simulation:
    def __init__(self, llm_wrapper, agent_type="regular", chunk_size=1000):
        """
        Initialize the simulation with an LLM wrapper
        Args:
            llm_wrapper: An instance of LLMWrapper for generating responses
            agent_type: Type of agent to use ("regular" or "timescale_aware")
            chunk_size: Number of steps to include in each summary chunk
        """
        self.orchestrator = Orchestrator(llm_wrapper)
        self.agents = {}
        self.env = None
        self.graph = None
        self.agent_type = agent_type
        self.chunk_size = chunk_size

    def _summarize_chunk(self, chunk):
        """Summarize a chunk of simulation steps"""
        try:
            return self.orchestrator.summarize_outcome(chunk)
        except Exception as e:
            print(f"Warning: Error summarizing chunk: {str(e)}")
            return "Chunk summary generation failed."

    def _summarize_chunks(self, chunk_summaries):
        """Combine and summarize multiple chunk summaries"""
        combined_summary = "\n\n".join(chunk_summaries)
        try:
            return self.orchestrator.summarize_outcome([{
                "summary": combined_summary
            }])
        except Exception as e:
            print(f"Warning: Error combining chunk summaries: {str(e)}")
            return "Final summary generation failed."

    def run(self, query: str, steps: int = 5) -> dict:
        """
        Run the simulation for the given number of steps
        Args:
            query: The initial query to simulate
            steps: Number of simulation steps to run
        Returns:
            Dictionary containing both detailed history and summary
        """
        # Setup the simulation world
        setup = self.orchestrator.setup_simulation(query)
        print("Setup data:", setup)  # Debug print
        
        # Initialize environment
        self.env = Environment(setup["environment"]["facts"])
        
        # Initialize connectivity graph
        self.graph = ConnectivityGraph(setup["connectivity"])
        
        # Initialize agents
        for agent_data in setup["agents"]:
            print("Agent data:", agent_data)  # Debug print
            agent_id = agent_data["id"]
            if self.agent_type == "timescale_aware":
                self.agents[agent_id] = TimescaleAwareAgent(
                    agent_id=agent_id,
                    identity=agent_data["identity"],
                    llm=self.orchestrator.llm
                )
            else:
                self.agents[agent_id] = Agent(
                    agent_id=agent_id,
                    identity=agent_data["identity"],
                    llm=self.orchestrator.llm
                )

        # Run simulation steps
        history = []
        for step in range(steps):
            step_actions = []
            
            # Each agent takes their turn
            for agent_id, agent in self.agents.items():
                # Get visible environment state for this agent
                visible_state = self.env.snapshot_for_agent(agent_id, self.graph)
                
                # Get messages from neighbors
                neighbors = self.graph.get_neighbors(agent_id)
                messages = []
                for neighbor_id in neighbors:
                    if neighbor_id in self.agents:
                        messages.append(self.agents[neighbor_id].get_last_message())
                
                # Agent decides on action
                action = agent.act(visible_state, messages)
                step_actions.append({
                    "agent": agent_id,
                    "identity": self.agents[agent_id].identity,
                    "visible_state": visible_state,
                    "received_messages": messages,
                    "action": action
                })
                
                # Update environment with action
                self.env.update(action)
            
            history.append({
                "step": step,
                "actions": step_actions,
                "environment": self.env.get_state()
            })

        # Generate final summary using chunked approach
        try:
            # Split history into chunks
            chunks = []
            for i in range(0, len(history), self.chunk_size):
                chunks.append(history[i:i + self.chunk_size])
            
            # Summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                print(f"Summarizing chunk {i+1}/{len(chunks)}...")
                chunk_summary = self._summarize_chunk(chunk)
                chunk_summaries.append(chunk_summary)
            
            # Combine and summarize chunk summaries
            if len(chunk_summaries) > 1:
                print("Combining chunk summaries...")
                summary = self._summarize_chunks(chunk_summaries)
            else:
                summary = chunk_summaries[0]
                
        except Exception as e:
            print(f"Warning: Could not generate summary due to error: {str(e)}")
            summary = "Summary generation failed. Please refer to the detailed trace file for the simulation results."
        
        return {
            "setup": setup,  # Initial configuration
            "history": history,  # Detailed step-by-step history
            "summary": summary  # Final summary
        }

    def should_activate(self, agent_id):
        return True

    def get_communications(self, agent_id):
        return []
