from social_sim.orchestrator import Orchestrator
from social_sim.interactions import Environment, ConnectivityGraph
from social_sim.agents import Agent, TimescaleAwareAgent
from typing import Generator, Dict, List
import json

class Simulation:
    def __init__(self, llm_wrapper, agent_type="regular", chunk_size=1000, agent_outcome_definitions=None, debug=False):
        """
        Initialize the simulation with an LLM wrapper
        Args:
            llm_wrapper: An instance of LLMWrapper for generating responses
            agent_type: Type of agent to use ("regular" or "timescale_aware")
            chunk_size: Number of steps to include in each summary chunk
            agent_outcome_definitions: Definitions for agent-specific outcomes
            debug: Whether to print debug statements (default: False)
        """
        self.orchestrator = Orchestrator(llm_wrapper)
        self.agents = {}
        self.env = None
        self.graph = None
        self.agent_type = agent_type
        self.chunk_size = chunk_size
        self.agent_outcome_definitions = agent_outcome_definitions or []
        self.debug = debug

    def _summarize_chunk(self, chunk):
        """Summarize a chunk of simulation steps"""
        try:
            return self.orchestrator.summarize_outcome(chunk)
        except Exception as e:
            if self.debug:
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
            if self.debug:
                print(f"Warning: Error combining chunk summaries: {str(e)}")
            return "Final summary generation failed."

    def _get_messages_for_agent(self, agent_id):
        """
        Get formatted messages for an agent from their neighbors.
        Returns messages with sender information.
        """
        neighbors = self.graph.get_neighbors(agent_id)
        messages_with_senders = []
        
        for neighbor_id in neighbors:
            if neighbor_id in self.agents:
                message = self.agents[neighbor_id].get_last_message()
                if message:  # Only add non-empty messages
                    messages_with_senders.append({
                        "sender": neighbor_id,
                        "message": message
                    })
        
        return messages_with_senders

    def run(self, query: str, steps: int = 5) -> Generator[Dict, None, None]:
        """
        Run the simulation for a given number of steps.
        """
        # Setup the simulation world
        setup = self.orchestrator.setup_simulation(query)
        if self.debug:
            print("Setup data:", setup)
        
        # Initialize environment
        self.env = Environment(setup["environment"]["facts"])
        
        # Initialize connectivity graph
        self.graph = ConnectivityGraph(setup["connectivity"])
        
        # Initialize agents
        for agent_data in setup["agents"]:
            if self.debug:
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
            if self.debug:
                print(f"\nRunning step {step + 1}/{steps}...")
            step_actions = []
            
            # Each agent takes their turn
            for agent_id, agent in self.agents.items():
                # Get visible environment state for this agent
                visible_state = self.env.snapshot_for_agent(agent_id, self.graph)
                
                # Get messages from neighbors using the dedicated function
                messages_with_senders = self._get_messages_for_agent(agent_id)
                
                # Agent decides on action - pass time information for TimescaleAwareAgents
                if hasattr(agent, 'time_scale'):  # Check if it's a TimescaleAwareAgent
                    action = agent.act(
                        visible_state, 
                        messages_with_senders, 
                        current_step=step + 1, 
                        total_steps=steps, 
                        time_scale=time_scale
                    )
                else:
                    action = agent.act(visible_state, messages_with_senders)
                    
                step_actions.append({
                    "agent": agent_id,
                    "identity": agent.identity,
                    "visible_state": visible_state,
                    "received_messages": messages_with_senders,
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
                if self.debug:
                    print(f"Summarizing chunk {i+1}/{len(chunks)} (size: {len(chunk)} steps)...")
                chunk_summary = self._summarize_chunk(chunk)
                chunk_summaries.append(chunk_summary)
            
            # Combine and summarize chunk summaries
            if len(chunk_summaries) > 1:
                if self.debug:
                    print("Combining chunk summaries...")
                summary = self._summarize_chunks(chunk_summaries)
            else:
                summary = chunk_summaries[0]
                
            # Analyze agent outcomes
            agent_outcomes = self.analyze_agent_outcomes()
            if self.debug:
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
            if self.debug:
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
        if self.debug:
            print("Setup data:", setup)
        
        # Initialize environment
        self.env = Environment(setup["environment"]["facts"])
        
        # Initialize connectivity graph
        self.graph = ConnectivityGraph(setup["connectivity"])
        
        # Initialize agents
        for agent_data in setup["agents"]:
            if self.debug:
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
            if self.debug:
                print(f"\nRunning step {step + 1}/{steps}...")
            step_actions = []
            
            # Each agent takes their turn
            for agent_id, agent in self.agents.items():
                # Get visible environment state for this agent
                visible_state = self.env.snapshot_for_agent(agent_id, self.graph)
                
                # Get messages from neighbors using the dedicated function
                messages_with_senders = self._get_messages_for_agent(agent_id)
                
                # Agent decides on action - pass time information for TimescaleAwareAgents
                if hasattr(agent, 'time_scale'):  # Check if it's a TimescaleAwareAgent
                    action = agent.act(
                        visible_state, 
                        messages_with_senders, 
                        current_step=step + 1, 
                        total_steps=steps, 
                        time_scale=time_scale
                    )
                else:
                    action = agent.act(visible_state, messages_with_senders)
                    
                step_actions.append({
                    "agent": agent_id,
                    "identity": agent.identity,
                    "visible_state": visible_state,
                    "received_messages": messages_with_senders,
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
                if self.debug:
                    print(f"Summarizing chunk {i+1}/{len(chunks)} (size: {len(chunk)} steps)...")
                chunk_summary = self._summarize_chunk(chunk)
                chunk_summaries.append(chunk_summary)
            
            # Combine and summarize chunk summaries
            if len(chunk_summaries) > 1:
                if self.debug:
                    print("Combining chunk summaries...")
                summary = self._summarize_chunks(chunk_summaries)
            else:
                summary = chunk_summaries[0]
                
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not generate summary due to error: {str(e)}")
            summary = "Summary generation failed. Please refer to the detailed trace file for the simulation results."
        
        # Convert metrics to the expected format using chunked approach
        try:
            metrics = []
            # Process history in chunks for metric determination
            for i, chunk in enumerate(chunks):
                if self.debug:
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
                    if self.debug:
                        print(f"Warning: Could not determine metrics for chunk {i+1}: {str(e)}")
                    continue
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not format metrics: {str(e)}")
            metrics = []
        
        # Analyze agent outcomes
        agent_outcomes = self.analyze_agent_outcomes()
        if self.debug:
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
            if self.debug:
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
                    if self.debug:
                        print(f"Agent {agent_id} Analysis result: {analysis_result}")
                    
                    # Update agent's agent_outcomes with their analysis for this outcome
                    agent.agent_outcomes[outcome_name] = analysis_result
                    
                    # If agent matches this outcome, add them to the overall results
                    if "matches" in analysis_result.lower():
                        agent_outcomes[outcome_name].append(agent_id)
                
                except Exception as e:
                    if self.debug:
                        print(f"Error analyzing outcome {outcome_name} for agent {agent_id}: {e}")
                    agent.agent_outcomes[outcome_name] = "Analysis failed"
                    continue
        
        self.agent_outcomes = agent_outcomes
        return agent_outcomes

    def run_manual(self, steps: int = 5, time_scale: str = None) -> Generator[Dict, None, None]:
        """
        Run the simulation for a given number of steps without orchestrator setup.
        Assumes agents, environment, and connectivity are already configured.
        
        Args:
            steps: Number of steps to run
            time_scale: Optional time scale for TimescaleAwareAgents ("years", "months", "weeks", "days", "hours")
        """
        if not self.agents:
            raise ValueError("No agents configured. Use manual setup before calling run_manual.")
        if self.env is None:
            raise ValueError("Environment not configured. Use manual setup before calling run_manual.")
        if self.graph is None:
            raise ValueError("Connectivity graph not configured. Use manual setup before calling run_manual.")

        # Run simulation steps
        history = []
        for step in range(steps):
            if self.debug:
                print(f"\nRunning step {step + 1}/{steps}...")
            step_actions = []
            
            # Each agent takes their turn
            for agent_id, agent in self.agents.items():
                # Get visible environment state for this agent
                visible_state = self.env.snapshot_for_agent(agent_id, self.graph)
                
                # Get messages from neighbors using the dedicated function
                messages_with_senders = self._get_messages_for_agent(agent_id)
                
                # Agent decides on action - pass time information for TimescaleAwareAgents
                if hasattr(agent, 'time_scale'):  # Check if it's a TimescaleAwareAgent
                    action = agent.act(
                        visible_state, 
                        messages_with_senders, 
                        current_step=step + 1, 
                        total_steps=steps, 
                        time_scale=time_scale
                    )
                else:
                    action = agent.act(visible_state, messages_with_senders)
                    
                step_actions.append({
                    "agent": agent_id,
                    "identity": agent.identity,
                    "visible_state": visible_state,
                    "received_messages": messages_with_senders,
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
                if self.debug:
                    print(f"Summarizing chunk {i+1}/{len(chunks)} (size: {len(chunk)} steps)...")
                chunk_summary = self._summarize_chunk(chunk)
                chunk_summaries.append(chunk_summary)
            
            # Combine and summarize chunk summaries
            if len(chunk_summaries) > 1:
                if self.debug:
                    print("Combining chunk summaries...")
                summary = self._summarize_chunks(chunk_summaries)
            else:
                summary = chunk_summaries[0] if chunk_summaries else "No summary available"
                
            # Analyze agent outcomes
            agent_outcomes = self.analyze_agent_outcomes()
            if self.debug:
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
            if self.debug:
                print(f"Warning: Could not generate summary due to error: {str(e)}")
            summary = "Summary generation failed. Please refer to the detailed trace file for the simulation results."
            agent_outcomes = {}
            yield {
                "history": history,
                "summary": summary,
                "agent_outcomes": agent_outcomes,
                "agent_states": {agent_id: agent.state for agent_id, agent in self.agents.items()},
                "environment_state": self.env.get_state()
            }

    def setup_from_config(self, config) -> None:
        """
        Set up the simulation from a JSON configuration object for manual runs.
        
        Args:
            config: Dictionary containing the configuration (can be loaded from JSON)
        """
        # If config is a string path, load it as JSON
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = json.load(f)
        
        # Validate required fields
        required_fields = ["name", "steps", "agents", "connectivity"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in config")
        
        if not isinstance(config["agents"], list) or not config["agents"]:
            raise ValueError("'agents' must be a non-empty list")
        
        for agent_def in config["agents"]:
            if "id" not in agent_def or "prompt" not in agent_def:
                raise ValueError("Every agent entry needs 'id' and 'prompt'")
        
        if not isinstance(config["connectivity"], dict):
            raise ValueError("'connectivity' must be an object mapping agent_id → {visible_facts, neighbors}")
        
        for agent_id, conn_info in config["connectivity"].items():
            if not {"visible_facts", "neighbors"} <= conn_info.keys():
                raise ValueError(f"Connectivity for '{agent_id}' must include visible_facts and neighbors lists")
        
        # Store config for later use
        self.config = config
        
        # Set agent outcome definitions from config
        if "agent_outcome_definitions" in config:
            self.agent_outcome_definitions = config["agent_outcome_definitions"]
            if self.debug:
                print(f"Loaded {len(self.agent_outcome_definitions)} agent outcome definitions: {list(self.agent_outcome_definitions.keys())}")
        else:
            if self.debug:
                print("No agent outcome definitions found in config")
        
        # Set up environment with empty facts (manual setup)
        self.env = Environment([])
        
        # Set up connectivity graph
        self.graph = ConnectivityGraph(config["connectivity"])
        
        # Create and add agents
        for agent_def in config["agents"]:
            agent_id = agent_def["id"]
            
            if config.get("agent_type") == "timescale_aware":
                agent = TimescaleAwareAgent(
                    agent_id=agent_id,
                    identity=agent_def["prompt"],
                    llm=self.orchestrator.llm
                )
            else:
                agent = Agent(
                    agent_id=agent_id,
                    identity=agent_def["prompt"],
                    llm=self.orchestrator.llm
                )
            
            self.agents[agent_id] = agent
        
        if self.debug:
            print(f"Successfully set up simulation '{config['name']}' with {len(self.agents)} agents")
