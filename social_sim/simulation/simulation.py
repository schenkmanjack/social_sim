from social_sim.orchestrator import Orchestrator
from social_sim.interactions import Environment, ConnectivityGraph
from social_sim.agents import Agent, TimescaleAwareAgent
from typing import Generator, Dict, List
import json

class Simulation:
    def __init__(self, llm_wrapper, agent_type="regular", chunk_size=1000, agent_outcome_definitions=None):
        """
        Initialize the simulation with an LLM wrapper
        Args:
            llm_wrapper: An instance of LLMWrapper for generating responses
            agent_type: Type of agent to use ("regular" or "timescale_aware")
            chunk_size: Number of steps to include in each summary chunk
            agent_outcome_definitions: Definitions for agent-specific outcomes
        """
        self.orchestrator = Orchestrator(llm_wrapper)
        self.agents = {}
        self.env = None
        self.graph = None
        self.agent_type = agent_type
        self.chunk_size = chunk_size
        self.agent_outcome_definitions = agent_outcome_definitions or []

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

    def run(self, query: str, steps: int = 5) -> Generator[Dict, None, None]:
        """
        Run the simulation for a given number of steps.
        """
        # Setup the simulation world
        setup = self.orchestrator.setup_simulation(query)
        print("Setup data:", setup)
        
        # Initialize environment
        self.env = Environment(setup["environment"]["facts"])
        
        # Initialize connectivity graph
        self.graph = ConnectivityGraph(setup["connectivity"])
        
        # Initialize agents
        for agent_data in setup["agents"]:
            print("Agent data:", agent_data)
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
            print(f"\nRunning step {step + 1}/{steps}...")
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
                    "identity": agent.identity,
                    "visible_state": visible_state,
                    "received_messages": messages,
                    "action": action
                })
                
                # Update environment with action
                self.env.update(action)
            
            # Record step results
            step_result = {
                "step": step + 1,
                "actions": step_actions,
                "environment": self.env.get_state(),
                "agent_states": {agent_id: agent.state for agent_id, agent in self.agents.items()}
            }
            history.append(step_result)
            
            # Yield step result
            yield step_result

        # Generate final summary
        try:
            # Split history into chunks
            chunks = []
            for i in range(0, len(history), self.chunk_size):
                chunks.append(history[i:i + self.chunk_size])
            
            # Summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                print(f"Summarizing chunk {i+1}/{len(chunks)} (size: {len(chunk)} steps)...")
                chunk_summary = self._summarize_chunk(chunk)
                chunk_summaries.append(chunk_summary)
            
            # Combine and summarize chunk summaries
            if len(chunk_summaries) > 1:
                print("Combining chunk summaries...")
                summary = self._summarize_chunks(chunk_summaries)
            else:
                summary = chunk_summaries[0]
                
            # Analyze agent outcomes
            agent_outcomes = self.analyze_agent_outcomes()
            print(f"Agent outcomes: {agent_outcomes}")
            
            # Yield final result
            final_result = {
                "summary": summary,
                "history": history,
                "agent_states": {agent_id: agent.state for agent_id, agent in self.agents.items()},
                "environment_state": self.env.get_state(),
                "agent_outcomes": agent_outcomes
            }
            yield final_result
            
        except Exception as e:
            print(f"Warning: Could not generate summary due to error: {str(e)}")
            summary = "Summary generation failed. Please refer to the detailed trace file for the simulation results."
            agent_outcomes = {}
            yield steps, {
                "setup": setup,
                "history": history,
                "summary": summary,
                "metrics": [],
                "agent_outcomes": agent_outcomes
            }

    def should_activate(self, agent_id):
        return True

    def get_communications(self, agent_id):
        return []

    def run_with_progress(self, query: str, steps: int = 5):
        """
        Run the simulation for the given number of steps with progress tracking
        Args:
            query: The initial query to simulate
            steps: Number of simulation steps to run
        Yields:
            Tuple of (current_step, result) for progress tracking
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
            print(f"\nRunning step {step + 1}/{steps}...")  # Added print statement
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
                    "identity": agent.identity,
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
            
            # Yield progress after each step
            yield step + 1, {
                "setup": setup,
                "history": history,
                "current_step": step + 1,
                "total_steps": steps
            }

        # Generate final summary using chunked approach
        try:
            # Split history into chunks using instance variable
            chunks = []
            for i in range(0, len(history), self.chunk_size):
                chunks.append(history[i:i + self.chunk_size])
            
            # Summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                print(f"Summarizing chunk {i+1}/{len(chunks)} (size: {len(chunk)} steps)...")
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
        
        # Convert metrics to the expected format using chunked approach
        try:
            metrics = []
            # Process history in chunks for metric determination
            for i, chunk in enumerate(chunks):
                print(f"Analyzing metrics for chunk {i+1}/{len(chunks)} (size: {len(chunk)} steps)...")
                try:
                    chunk_metrics = self.orchestrator.determine_plot_metrics(query, chunk)
                    for name, keywords in chunk_metrics:
                        # Only add if not already present
                        if not any(m["metric_name"] == name for m in metrics):
                            metrics.append({
                                "question": f"How did {name.lower()} change over time?",
                                "metric_name": name,
                                "keywords": keywords,
                                "visualization": "line",  # Default to line chart for time series
                                "data_type": "trend"
                            })
                except Exception as e:
                    print(f"Warning: Could not determine metrics for chunk {i+1}: {str(e)}")
                    continue
        except Exception as e:
            print(f"Warning: Could not format metrics: {str(e)}")
            metrics = []
        
        # Analyze agent outcomes
        agent_outcomes = self.analyze_agent_outcomes()
        print(f"Agent outcomes: {agent_outcomes}")
        
        # Yield final result
        yield steps, {
            "setup": setup,
            "history": history,
            "summary": summary,
            "metrics": metrics,
            "agent_outcomes": agent_outcomes
        }

    def analyze_agent_outcomes(self) -> Dict[str, List[str]]:
        """
        Analyze which agents match each agent-specific outcome based on their final states.
        Updates each agent's agent_outcomes property with a dict mapping outcome conditions to analysis results.
        
        Returns:
            Dict mapping outcome names to lists of agent IDs that matched
        """
        agent_outcomes = {name: [] for name in self.agent_outcome_definitions}
        
        if not self.agent_outcome_definitions:
            print("No agent outcome definitions provided.")
            return agent_outcomes
        
        # Initialize agent_outcomes dictionary for each agent
        for agent in self.agents.values():
            agent.agent_outcomes = {}
        
        # Process each agent
        for agent_id, agent in self.agents.items():
            # Process each outcome definition for this agent
            for outcome_name, outcome_definition in self.agent_outcome_definitions.items():
                # Create a prompt for the LLM
                prompt = f"""
                Analyze if this agent matches this specific outcome condition.
                
                Agent ID: {agent_id}
                Agent Memory: {agent.memory}
                
                Outcome Name: {outcome_name}
                Outcome Definition: {json.dumps(outcome_definition, indent=2)}
                
                Provide a detailed analysis of whether and why this agent matches this outcome.
                Return a JSON string containing the analysis.
                Example format:
                "Agent matches because they achieved X and Y"
                or
                "Agent does not match because they failed to achieve Z"
                """
                
                try:
                    # Get the LLM's analysis for this agent and outcome
                    analysis = self.orchestrator._call_llm_with_retry(prompt)
                    analysis_result = json.loads(analysis)["analysis"]
                    print(f"Analysis result: {analysis_result}")
                    
                    # Update agent's agent_outcomes with their analysis for this outcome
                    agent.agent_outcomes[outcome_name] = analysis_result
                    
                    # If agent matches this outcome, add them to the overall results
                    if "matches" in analysis_result.lower():
                        agent_outcomes[outcome_name].append(agent_id)
                
                except Exception as e:
                    print(f"Error analyzing outcome {outcome_name} for agent {agent_id}: {e}")
                    agent.agent_outcomes[outcome_name] = "Analysis failed"
                    continue
        
        self.agent_outcomes = agent_outcomes
        return agent_outcomes
