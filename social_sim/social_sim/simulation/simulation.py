from social_sim.orchestrator import Orchestrator
from social_sim.interactions import Environment, ConnectivityGraph
from social_sim.agents import Agent, TimescaleAwareAgent
from typing import Generator, Dict, List
import json
import time
import random

class Simulation:
    def __init__(self, llm_wrapper, agent_type="regular", chunk_size=1000, agent_outcome_definitions=None, debug=False, disable_summary=False):
        """
        Initialize the simulation with an LLM wrapper
        Args:
            llm_wrapper: An instance of LLMWrapper for generating responses
            agent_type: Type of agent to use ("regular" or "timescale_aware")
            chunk_size: Number of steps to include in each summary chunk
            agent_outcome_definitions: Definitions for agent-specific outcomes
            debug: Whether to print debug statements (default: False)
            disable_summary: Whether to disable summary generation (default: False)
        """
        self.orchestrator = Orchestrator(llm_wrapper)
        self.agents = {}
        self.env = None
        self.graph = None
        self.agent_type = agent_type
        self.chunk_size = chunk_size
        self.agent_outcome_definitions = agent_outcome_definitions or {}
        self.agent_outcomes = {}  # Initialize to prevent AttributeError
        self.debug = debug
        self.disable_summary = disable_summary

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

    def _generate_final_summary(self, history):
        """
        Generate final summary and agent outcomes from simulation history.
        
        Args:
            history: List of simulation step results
            
        Returns:
            Dict containing summary, agent_outcomes, agent_states, and environment_state
        """
        try:
            # Skip summarization if disabled
            if self.disable_summary:
                if self.debug:
                    print("Summary generation disabled - skipping...")
                
                agent_outcomes = self.analyze_agent_outcomes()
                
                # Calculate LLM usage from trace (no summary)
                llm_usage = self.calculate_llm_usage_from_trace(history, None)
                
                return {
                    "summary": "Summary generation disabled for performance",
                    "history": history,
                    "agent_states": {agent_id: agent.state for agent_id, agent in self.agents.items()},
                    "environment_state": self.env.get_state(),
                    "agent_outcomes": agent_outcomes,
                    "llm_usage": llm_usage
                }
            
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
            
            # Calculate LLM usage from trace
            llm_usage = self.calculate_llm_usage_from_trace(history, summary)
            
            final_result = {
                "summary": summary,
                "history": history,
                "agent_states": {agent_id: agent.state for agent_id, agent in self.agents.items()},
                "environment_state": self.env.get_state(),
                "agent_outcomes": agent_outcomes,
                "llm_usage": llm_usage
            }
            
            # Store for later access
            self._last_final_result = final_result
            return final_result
            
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not generate summary due to error: {str(e)}")
            
            # Calculate LLM usage even in error case
            llm_usage = self.calculate_llm_usage_from_trace(history, None)
            
            return {
                "history": history,
                "summary": "Summary generation failed. Please refer to the detailed trace file for the simulation results.",
                "agent_outcomes": {},
                "agent_states": {agent_id: agent.state for agent_id, agent in self.agents.items()},
                "environment_state": self.env.get_state(),
                "llm_usage": llm_usage
            }

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
                    llm=self.orchestrator.llm,
                    use_full_agent_memory=True  # Default to True for backward compatibility
                )
            else:
                self.agents[agent_id] = Agent(
                    agent_id=agent_id,
                    identity=agent_data["identity"],
                    llm=self.orchestrator.llm,
                    use_full_agent_memory=True  # Default to True for backward compatibility
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

        # Generate final summary using the new method
        final_result = self._generate_final_summary(history)
        yield final_result

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
                    llm=self.orchestrator.llm,
                    use_full_agent_memory=True  # Default to True for backward compatibility
                )
            else:
                self.agents[agent_id] = Agent(
                    agent_id=agent_id,
                    identity=agent_data["identity"],
                    llm=self.orchestrator.llm,
                    use_full_agent_memory=True  # Default to True for backward compatibility
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

        # Generate final summary using the new method
        final_result = self._generate_final_summary(history)
        yield final_result

    def analyze_agent_outcomes(self) -> Dict[str, List[str]]:
        """
        Analyze which agents match each agent-specific outcome based on their final actions.
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
                # Use final piece of agent memory instead of just last_message to match outcome definitions
                if hasattr(agent, 'memory') and agent.memory:
                    # Use the last few entries of memory for context
                    recent_memory = agent.memory[-3:] if len(agent.memory) >= 3 else agent.memory
                    final_memory_text = '\n'.join([str(msg) for msg in recent_memory])
                elif agent.last_message:
                    final_memory_text = agent.last_message
                else:
                    final_memory_text = "No action taken"
                
                # Create a prompt for the LLM
                prompt = f"""
                Analyze if this agent matches this specific outcome condition based on their final memory.
                
                Agent ID: {agent_id}
                Agent Final Memory: {final_memory_text}
                
                Outcome Name: {outcome_name}
                Outcome Definition: {json.dumps(outcome_definition, indent=2)}
                
                Does this agent match the outcome condition based on their final memory?
                
                RESPOND WITH ONLY ONE WORD: either "true" or "false"
                
                Your response:"""
                
                try:
                    # Get the LLM's analysis for this agent and outcome
                    analysis = self.orchestrator._call_llm_with_retry(prompt)
                    analysis_clean = analysis.strip().lower()
                    
                    # Check if response contains "true"
                    matches = "true" in analysis_clean
                    
                    if self.debug:
                        print(f"Agent {agent_id} {outcome_name}: response='{analysis.strip()}', matches={matches}")
                    
                    # Update agent's agent_outcomes with the raw response for reference
                    agent.agent_outcomes[outcome_name] = f"Response: {analysis.strip()}, Matches: {matches}"
                    
                    # If agent matches this outcome, add them to the overall results
                    if matches:
                        agent_outcomes[outcome_name].append(agent_id)
                    
                    # Small delay to prevent overwhelming the API
                    time.sleep(0.1)
                
                except Exception as e:
                    if self.debug:
                        print(f"Error analyzing outcome {outcome_name} for agent {agent_id}: {e}")
                    agent.agent_outcomes[outcome_name] = "Analysis failed"
                    continue
        
        self.agent_outcomes = agent_outcomes
        return agent_outcomes

    def _process_batch_responses(self, agent_ids, batch_responses, agent_contexts, step):
        """
        Process batch LLM responses and update agent states.
        
        Args:
            agent_ids: List of agent IDs in the same order as batch_responses
            batch_responses: List of LLM responses from batch call OR list of result dicts from batch_process_agents
            agent_contexts: List of context dictionaries for each agent
            step: Current simulation step (0-indexed)
            
        Returns:
            Tuple of (step_actions, failed_agents)
        """
        step_actions = []
        failed_agents = []
        
        for i, (agent_id, response_data) in enumerate(zip(agent_ids, batch_responses)):
            try:
                agent = self.agents[agent_id]
                context = agent_contexts[i]
                
                # Handle new response format from batch_process_agents
                if isinstance(response_data, dict) and 'success' in response_data:
                    if not response_data['success']:
                        if self.debug:
                            print(f"Agent {agent_id} failed: {response_data.get('error', 'Unknown error')}")
                        failed_agents.append(agent_id)
                        continue
                    response = response_data['response']
                else:
                    # Handle old format (direct response)
                    response = response_data
                    if response is None:
                        if self.debug:
                            print(f"Agent {agent_id} failed: No response received")
                        failed_agents.append(agent_id)
                        continue
                
                # Update agent state with the response
                agent.last_message = response
                agent.memory.append(response)
                
                step_actions.append({
                    "agent": agent_id,
                    "identity": agent.identity,
                    "visible_state": context["visible_state"],
                    "received_messages": context["messages_with_senders"],
                    "action": response
                })
                
                # Update environment with action
                self.env.update(response)
                
            except Exception as e:
                if self.debug:
                    print(f"Error processing response for agent {agent_id}: {str(e)}")
                failed_agents.append(agent_id)
                continue
        
        return step_actions, failed_agents

    def run_manual_batching(self, steps: int = 5, time_scale: str = None) -> Generator[Dict, None, None]:
        """
        Run the simulation for a given number of steps without orchestrator setup using batch LLM calls.
        Assumes agents, environment, and connectivity are already configured.
        
        Args:
            steps: Number of steps to run
            time_scale: Optional time scale for TimescaleAwareAgents ("years", "months", "weeks", "days", "hours")
        """
        if not self.agents:
            raise ValueError("No agents configured. Use manual setup before calling run_manual_batching.")
        if self.env is None:
            raise ValueError("Environment not configured. Use manual setup before calling run_manual_batching.")
        if self.graph is None:
            raise ValueError("Connectivity graph not configured. Use manual setup before calling run_manual_batching.")

        # Track failed agents to exclude from subsequent steps
        failed_agents = set()
        
        # Run simulation steps
        history = []
        for step in range(steps):
            if self.debug:
                print(f"\nRunning step {step + 1}/{steps}...")
            
            # Get active agents (exclude failed ones)
            active_agents = {aid: agent for aid, agent in self.agents.items() if aid not in failed_agents}
            
            if not active_agents:
                if self.debug:
                    print("No active agents remaining. Ending simulation.")
                break
            
            # Prepare data for batch_process_agents
            agents_data = []
            agent_contexts = []
            agent_ids = list(active_agents.keys())
            
            for agent_id in agent_ids:
                agent = active_agents[agent_id]
                
                # Get visible environment state for this agent
                visible_state = self.env.snapshot_for_agent(agent_id, self.graph)
                
                # Get messages from neighbors using the dedicated function
                messages_with_senders = self._get_messages_for_agent(agent_id)
                
                # Prepare data structure for batch_process_agents
                agent_data = {
                    "agent": agent,
                    "agent_id": agent_id,
                    "simulation_index": 0,  # Always 0 for single simulation
                    "visible_state": visible_state,
                    "messages_with_senders": messages_with_senders,
                    "step": step + 1,
                    "total_steps": steps
                }
                
                if time_scale:
                    agent_data["time_scale"] = time_scale
                
                agents_data.append(agent_data)
                
                # Keep context for _process_batch_responses
                agent_contexts.append({
                    "agent_id": agent_id,
                    "visible_state": visible_state,
                    "messages_with_senders": messages_with_senders
                })
            
            # Use batch_process_agents class method properly
            if self.debug:
                print(f"Processing batch for {len(agents_data)} agents...")
            
            batch_results = Simulation.batch_process_agents(agents_data, debug=self.debug)
            
            # Process batch responses using the updated method
            step_actions, step_failed_agents = self._process_batch_responses(agent_ids, batch_results, agent_contexts, step)
            
            # Add newly failed agents to the set
            failed_agents.update(step_failed_agents)
            
            if step_failed_agents and self.debug:
                print(f"Agents failed in this step: {step_failed_agents}")
            
            # Record step results
            step_result = {
                "step": step + 1,
                "actions": step_actions,
                "environment": self.env.get_state(),
                "agent_states": {agent_id: agent.state for agent_id, agent in active_agents.items()},
                "failed_agents": list(failed_agents)
            }
            history.append(step_result)
            
            # Yield step result
            yield step_result

        # Generate final summary using the new method
        final_result = self._generate_final_summary(history)
        final_result["failed_agents"] = list(failed_agents)
        yield final_result

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
        
        # Get use_full_agent_memory parameter from config (default: True)
        use_full_agent_memory = config.get("use_full_agent_memory", True)
        
        # Create and add agents
        for agent_def in config["agents"]:
            agent_id = agent_def["id"]
            
            if config.get("agent_type") == "timescale_aware":
                agent = TimescaleAwareAgent(
                    agent_id=agent_id,
                    identity=agent_def["prompt"],
                    llm=self.orchestrator.llm,
                    use_full_agent_memory=use_full_agent_memory
                )
            else:
                agent = Agent(
                    agent_id=agent_id,
                    identity=agent_def["prompt"],
                    llm=self.orchestrator.llm,
                    use_full_agent_memory=use_full_agent_memory
                )
            
            self.agents[agent_id] = agent
        
        if self.debug:
            print(f"Successfully set up simulation '{config['name']}' with {len(self.agents)} agents")
            print(f"use_full_agent_memory: {use_full_agent_memory}")

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

        # Generate final summary using the new method
        final_result = self._generate_final_summary(history)
        yield final_result

    @classmethod
    def batch_process_agents(cls, agents_data, llm=None, debug=False):
        """
        Class method to process a batch of agents across multiple simulations.
        
        Args:
            agents_data: List of dicts with agent, context, and simulation info
            llm: Orchestrator instance for LLM calls
            debug: Debug flag
        
        Returns:
            List of results in same order as input
        """
        if llm is None:
            # Get LLM from the first agent
            llm = agents_data[0]["agent"].llm
        
        if not agents_data:
            return []
        
        # Extract prompts for batch call
        prompts = []
        for agent_data in agents_data:
            agent = agent_data["agent"]
            
            # Generate prompt based on agent type
            if hasattr(agent, 'time_scale'):  # TimescaleAwareAgent
                prompt = agent.generate_prompt(
                    agent_data["visible_state"], 
                    agent_data["messages_with_senders"], 
                    current_step=agent_data["step"], 
                    total_steps=agent_data["total_steps"], 
                    time_scale=agent_data.get("time_scale")
                )
            else:  # Regular Agent
                prompt = agent.generate_prompt(
                    agent_data["visible_state"], 
                    agent_data["messages_with_senders"]
                )
            
            prompts.append(prompt)
        
        # Make batch LLM call with retry logic and rate limiting
        max_retries = 3
        base_delay = 2.0  # Start with 2 second delay
        
        for attempt in range(max_retries):
            try:
                if debug:
                    print(f"Making batch LLM call for {len(prompts)} agents across simulations (attempt {attempt + 1}/{max_retries})...")
                
                # Add a small delay before the request to help with rate limiting
                if attempt > 0:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    if debug:
                        print(f"Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                
                batch_responses = llm.batch(prompts)
                break  # Success, exit retry loop
                
            except Exception as e:
                error_str = str(e).lower()
                if debug:
                    print(f"Batch LLM call failed (attempt {attempt + 1}): {str(e)}")
                
                # Check if this is a rate limiting / overload error
                is_rate_limit_error = any(keyword in error_str for keyword in [
                    'overload', 'rate limit', '429', '529', 'too many requests', 
                    'service unavailable', 'timeout'
                ])
                
                if is_rate_limit_error and attempt < max_retries - 1:
                    # This is a rate limiting error and we have retries left
                    if debug:
                        print(f"Rate limiting detected, will retry after delay...")
                    continue
                else:
                    # Either not a rate limit error, or we're out of retries
                    if debug:
                        print(f"Batch LLM call failed permanently: {str(e)}")
                    # Return all failures
                    return [
                        {
                            "agent_id": agent_data["agent_id"],
                            "simulation_index": agent_data["simulation_index"],
                            "response": None,
                            "success": False,
                            "error": f"Batch call failed: {str(e)}"
                        }
                        for agent_data in agents_data
                    ]
        
        # Process responses
        results = []
        for i, (agent_data, response) in enumerate(zip(agents_data, batch_responses)):
            try:
                # Check if response indicates an error
                if response is None or (isinstance(response, dict) and response.get('error')):
                    error_msg = response.get('error', 'No response received') if isinstance(response, dict) else 'No response received'
                    results.append({
                        "agent_id": agent_data["agent_id"],
                        "simulation_index": agent_data["simulation_index"],
                        "response": None,
                        "success": False,
                        "error": error_msg
                    })
                    continue
                
                results.append({
                    "agent_id": agent_data["agent_id"],
                    "simulation_index": agent_data["simulation_index"],
                    "response": response,
                    "success": True,
                    "error": None
                })
                
            except Exception as e:
                if debug:
                    print(f"Error processing response for agent {agent_data['agent_id']}: {str(e)}")
                results.append({
                    "agent_id": agent_data["agent_id"],
                    "simulation_index": agent_data["simulation_index"],
                    "response": None,
                    "success": False,
                    "error": str(e)
                })
        
        return results

    @classmethod
    def batch_process_summaries(cls, histories_data, llm=None, debug=False):
        """Process summaries for multiple simulations in batch."""
        if not histories_data:
            return []

        # Infer expected steps from the first history
        expected_steps = len(histories_data[0]) if histories_data[0] else 0
        if debug:
            print(f"Inferred expected_steps = {expected_steps} from simulation_histories structure")

        # Filter valid histories - they should all have the expected length with non-None entries
        valid_histories = []
        history_mapping = {}  # Maps result index back to original index
        
        for idx, history in enumerate(histories_data):
            # Check for valid history - proper length with no failed agents in any step
            is_valid = (
                history and 
                len(history) == expected_steps and 
                all(step is not None for step in history) and  # No None steps
                not any(step.get("failed_agents") for step in history if step)  # No failed agents
            )
            
            if is_valid:
                valid_histories.append(history)
                history_mapping[len(valid_histories)-1] = idx

        # Only process valid histories
        batch_responses = []
        batch_outcomes = []
        if valid_histories:
            try:
                # First get summaries
                batch_responses = llm.batch([
                    f"S:{json.dumps(history, separators=(',', ':'))}"
                    for history in valid_histories
                ])
                
                # Note: batch_outcomes will be empty since we can't analyze outcomes without simulation objects
                batch_outcomes = [None] * len(valid_histories)
                
            except Exception as e:
                if debug:
                    print(f"Batch summary processing failed: {e}")
                return [{"success": False, "error": str(e), "summary": None}] * len(histories_data)

        # Build results array with failure info for invalid/failed histories
        results = [{"success": False, "error": "Invalid or failed simulation", "summary": None}] * len(histories_data)
        for valid_idx, response in enumerate(batch_responses):
            orig_idx = history_mapping[valid_idx]
            if response is None or (isinstance(response, dict) and response.get('error')):
                results[orig_idx] = {
                    "success": False,
                    "error": response.get('error', 'No response received') if isinstance(response, dict) else 'No response received',
                    "summary": None,
                    "agent_outcomes": None
                }
            else:
                results[orig_idx] = {
                    "success": True,
                    "error": None,
                    "summary": response,
                    "agent_outcomes": batch_outcomes[valid_idx] if valid_idx < len(batch_outcomes) else None
                }

        return results

    @classmethod
    def batch_analyze_agent_outcomes(cls, simulations, histories, llm=None, debug=False):
        """
        Batch analyze agent outcomes for multiple simulations.
        
        Args:
            simulations: List of Simulation instances
            histories: List of simulation histories
            llm: LLM interface to use (will use first simulation's if None)
            debug: Whether to print debug info
        
        Returns:
            List of agent outcomes dictionaries
        """
        if not simulations or not histories:
            return []
        
        if llm is None:
            llm = simulations[0].orchestrator.llm
        
        # Prepare batch of prompts for outcome analysis
        prompts = []
        valid_indices = []
        
        for idx, (simulation, history) in enumerate(zip(simulations, histories)):
            if not history or not simulation.agent_outcome_definitions:
                continue
            
            # For each simulation, create one combined prompt for all outcomes using FINAL MEMORY instead of just final actions
            agents_final_memory = {}
            for agent_id, agent in simulation.agents.items():
                if hasattr(agent, 'memory') and agent.memory:
                    # Use the last few entries of memory for context
                    recent_memory = agent.memory[-3:] if len(agent.memory) >= 3 else agent.memory
                    final_memory_text = '\n'.join([str(msg) for msg in recent_memory])
                elif agent.last_message:
                    final_memory_text = agent.last_message
                else:
                    final_memory_text = "No action taken"
                agents_final_memory[agent_id] = final_memory_text
            
            prompt = f"""
            Analyze if these agents match specific outcome conditions based on their final memory.
            
            Agents Final Memory:
            {json.dumps(agents_final_memory, indent=2)}
            
            Outcome Definitions:
            {json.dumps(simulation.agent_outcome_definitions, indent=2)}
            
            For each agent and each outcome, determine if the agent matches the outcome based on their final memory.
            
            RESPOND WITH A SIMPLE JSON FORMAT:
            {{
                "agent_0": {{
                    "outcome_1": "true",
                    "outcome_2": "false"
                }},
                "agent_1": {{
                    "outcome_1": "false", 
                    "outcome_2": "true"
                }}
            }}
            
            IMPORTANT: 
            1. Use exactly "true" or "false" as string values
            2. Include ALL agents and ALL outcomes
            3. Base decisions on the outcome definitions provided
            4. Use the exact outcome names from the definitions
            """
            prompts.append(prompt)
            valid_indices.append(idx)
        
        if not prompts:
            return [None] * len(simulations)
        
        # Make batch LLM call with retry logic
        max_retries = 3
        base_delay = 2.0  # Start with 2 second delay
        
        for attempt in range(max_retries):
            try:
                if debug:
                    print(f"Making batch LLM call for {len(prompts)} outcome analyses (attempt {attempt + 1}/{max_retries})...")
                
                # Add a small delay before the request to help with rate limiting
                if attempt > 0:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    if debug:
                        print(f"Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                
                batch_responses = llm.batch(prompts)
                break  # Success, exit retry loop
                
            except Exception as e:
                error_str = str(e).lower()
                if debug:
                    print(f"Batch outcome analysis failed (attempt {attempt + 1}): {str(e)}")
                
                # Check if this is a rate limiting / overload error
                is_rate_limit_error = any(keyword in error_str for keyword in [
                    'overload', 'rate limit', '429', '529', 'too many requests', 
                    'service unavailable', 'timeout'
                ])
                
                if is_rate_limit_error and attempt < max_retries - 1:
                    # This is a rate limiting error and we have retries left
                    if debug:
                        print(f"Rate limiting detected, will retry after delay...")
                    continue
                else:
                    # Either not a rate limit error, or we're out of retries
                    if debug:
                        print(f"Batch outcome analysis failed permanently: {str(e)}")
                    return [None] * len(simulations)
        
        # Process responses and map back to all simulations
        results = [None] * len(simulations)
        
        for i, response in enumerate(batch_responses):
            orig_idx = valid_indices[i]
            try:
                outcomes = json.loads(response)
                simulation = simulations[orig_idx]
                
                # Process the simplified response format
                agent_outcomes = {name: [] for name in simulation.agent_outcome_definitions}
                
                # Parse responses for each agent
                for agent_id, agent_results in outcomes.items():
                    if agent_id in simulation.agents:
                        agent = simulation.agents[agent_id]
                        if not hasattr(agent, 'agent_outcomes'):
                            agent.agent_outcomes = {}
                        
                        # Check each outcome for this agent
                        for outcome_name in simulation.agent_outcome_definitions.keys():
                            if outcome_name in agent_results:
                                response_value = str(agent_results[outcome_name]).strip().lower()
                                matches = "true" in response_value
                                
                                # Store the response for debugging
                                agent.agent_outcomes[outcome_name] = f"Response: {agent_results[outcome_name]}, Matches: {matches}"
                                
                                # Add to outcome list if matches
                                if matches:
                                    agent_outcomes[outcome_name].append(agent_id)
                            else:
                                # Fallback if outcome not found in response
                                agent.agent_outcomes[outcome_name] = "No response for this outcome"
                
                # Store results in simulation
                simulation.agent_outcomes = agent_outcomes
                results[orig_idx] = agent_outcomes
                
            except Exception as e:
                if debug:
                    print(f"Error processing outcome analysis for simulation {orig_idx}: {e}")
                    print(f"Raw response: {response}")
                results[orig_idx] = None
            
        return results

    def calculate_llm_usage_from_trace(self, history: List[Dict], final_summary: str = None) -> dict:
        """
        Calculate LLM character usage post-hoc from simulation trace.
        Reconstructs all prompts and responses to count characters.
        
        Args:
            history: Complete simulation history from trace
            final_summary: Final summary text if available
            
        Returns:
            Dict with input_characters, output_characters, total_characters, call_count
        """
        total_input_chars = 0
        total_output_chars = 0
        total_calls = 0
        
        if self.debug:
            print("Calculating LLM usage from simulation trace...")
        
        # 1. Calculate agent action LLM calls
        for step_data in history:
            if "actions" not in step_data:
                continue
                
            step_num = step_data.get("step", 0)
            
            for action_data in step_data["actions"]:
                agent_id = action_data["agent"]
                agent = self.agents.get(agent_id)
                
                if not agent:
                    continue
                
                # Reconstruct the exact prompt that was sent to LLM
                visible_state = action_data["visible_state"]
                received_messages = action_data["received_messages"]
                agent_response = action_data["action"]
                
                # Generate the same prompt the agent would have used
                if hasattr(agent, 'time_scale') and hasattr(self, 'config'):
                    # TimescaleAwareAgent with time context
                    total_steps = self.config.get("steps", 5) if hasattr(self, 'config') else 5
                    time_scale = getattr(self, 'time_scale', None)
                    
                    system_prompt, user_prompt = agent.generate_prompts(
                        visible_state, 
                        received_messages,
                        current_step=step_num,
                        total_steps=total_steps,
                        time_scale=time_scale
                    )
                else:
                    # Regular Agent
                    system_prompt, user_prompt = agent.generate_prompts(
                        visible_state, 
                        received_messages
                    )
                
                # Count input characters (system + user prompt)
                input_chars = len(user_prompt)
                if system_prompt:
                    input_chars += len(system_prompt)
                
                # Count output characters (agent response)
                output_chars = len(str(agent_response)) if agent_response else 0
                
                total_input_chars += input_chars
                total_output_chars += output_chars
                total_calls += 1
                
                if self.debug:
                    print(f"  Step {step_num}, Agent {agent_id}: {input_chars} input, {output_chars} output chars")
        
        # 2. Calculate summary generation LLM calls (if not disabled)
        if not self.disable_summary and history:
            # Chunk summaries
            chunks = []
            for i in range(0, len(history), self.chunk_size):
                chunks.append(history[i:i + self.chunk_size])
            
            # Each chunk summary call
            for chunk in chunks:
                chunk_str = json.dumps(chunk, separators=(',', ':'))
                total_input_chars += len(chunk_str)
                total_calls += 1
                
                # Estimate chunk summary output (we don't have exact text, use reasonable estimate)
                estimated_chunk_summary_chars = min(len(chunk_str) // 4, 500)  # Rough estimate
                total_output_chars += estimated_chunk_summary_chars
            
            # Final summary combination (if multiple chunks)
            if len(chunks) > 1:
                # Estimate combined summary input
                estimated_combined_input = len(chunks) * 200  # Rough estimate of chunk summary length
                total_input_chars += estimated_combined_input
                total_calls += 1
                
                # Use actual final summary length if provided
                if final_summary:
                    total_output_chars += len(final_summary)
                else:
                    total_output_chars += 300  # Rough estimate
        
        # 3. Calculate agent outcome analysis LLM calls
        if self.agent_outcome_definitions:
            for agent_id, agent in self.agents.items():
                # Construct final_memory_text for this agent
                if hasattr(agent, 'memory') and agent.memory:
                    # Use the last few entries of memory for context
                    recent_memory = agent.memory[-3:] if len(agent.memory) >= 3 else agent.memory
                    final_memory_text = '\n'.join([str(msg) for msg in recent_memory])
                elif agent.last_message:
                    final_memory_text = agent.last_message
                else:
                    final_memory_text = "No action taken"
                
                for outcome_name, outcome_definition in self.agent_outcome_definitions.items():
                    # Reconstruct outcome analysis prompt
                    prompt = f"""
                    Analyze if this agent matches this specific outcome condition based on their final memory.
                    
                    Agent ID: {agent_id}
                    Agent Final Memory: {final_memory_text}
                    
                    Outcome Name: {outcome_name}
                    Outcome Definition: {json.dumps(outcome_definition, indent=2)}
                    
                    Does this agent match the outcome condition based on their final memory?
                    
                    RESPOND WITH ONLY ONE WORD: either "true" or "false"
                    
                    Your response:"""
                    
                    total_input_chars += len(prompt)
                    total_calls += 1
                    
                    # Estimate outcome analysis response length
                    estimated_analysis_chars = 150  # Typical analysis response length
                    total_output_chars += estimated_analysis_chars
        
        usage_stats = {
            "input_characters": total_input_chars,
            "output_characters": total_output_chars,
            "total_characters": total_input_chars + total_output_chars,
            "call_count": total_calls
        }
        
        if self.debug:
            print(f"Total LLM usage: {usage_stats}")
        
        return usage_stats

    def get_llm_usage_stats(self) -> dict:
        """
        Get LLM usage statistics for this simulation.
        This method looks for usage stats in the last final result.
        """
        # Try to get from last final result if available
        if hasattr(self, '_last_final_result') and 'llm_usage' in self._last_final_result:
            return self._last_final_result['llm_usage']
        
        # Fallback: return empty stats
        return {
            "input_characters": 0,
            "output_characters": 0,
            "total_characters": 0,
            "call_count": 0
        }
