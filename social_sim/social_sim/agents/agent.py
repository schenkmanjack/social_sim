class Agent:
    def __init__(self, agent_id, identity, llm, use_full_agent_memory=True):
        """
        Initialize an agent
        Args:
            agent_id: Unique identifier for the agent
            identity: Description of who the agent is and their role
            llm: LLM wrapper for generating responses
            use_full_agent_memory: Whether to include full memory in prompts (default: True)
        """
        self.agent_id = agent_id
        self.identity = identity
        self.llm = llm
        self.use_full_agent_memory = use_full_agent_memory
        self.last_message = None
        self.memory = []  # Initialize empty memory
        self.state = {}  # Generic state dictionary

    def generate_prompts(self, visible_state, messages):
        """
        Generate separate system and user prompts (lean version to reduce token usage)
        Returns: (system_prompt, user_prompt)
        """
        system_prompt = self.identity  # unchanged

        # --- Build compact user prompt ----------------------------------
        # 1. Messages – already formatted succinctly
        messages_text = self._format_messages(messages)
        messages_block = f"Msgs:\n{messages_text}" if messages_text else ""

        # 2. Optional environment state if any facts are visible
        env_block = f"Env:{visible_state}\n" if visible_state else ""

        # 3. Memory (full or last action)
        if self.use_full_agent_memory:
            mem_block = f"Memory:{self.memory}"
        else:
            mem_block = f"Prev:{self.last_message}" if self.last_message else "Prev:None"

        # Assemble minimal prompt – each section separated by a newline only
        user_prompt = (
            f"{env_block}{messages_block}\n{mem_block}\n"
            "Next action (one short sentence):"
        )

        return system_prompt, user_prompt

    def generate_prompt(self, visible_state, messages):
        """
        Generate the prompt that would be used for acting, without actually calling the LLM
        (Kept for backward compatibility)
        """
        system_prompt, user_prompt = self.generate_prompts(visible_state, messages)
        
        # If no system prompt, return just user prompt
        if not system_prompt:
            return user_prompt
        
        # Otherwise combine them for backward compatibility
        return f"System: {system_prompt}\n\nUser: {user_prompt}"

    def act(self, visible_state, messages):
        """Generate an action given the visible state and messages from neighbors"""
        system_prompt, user_prompt = self.generate_prompts(visible_state, messages)
        
        # Call LLM with system and user prompts
        action = self.llm.generate(user_prompt, system_prompt)
        
        # Update memory and last message
        self.last_message = action
        self.memory.append(action)
        
        return action

    def _format_messages(self, messages):
        """Format messages for display, handling both old and new message formats"""
        if not messages:
            return "No messages received."
        
        formatted_messages = []
        for i, message in enumerate(messages):
            if isinstance(message, dict) and 'sender' in message:
                # New format with sender information
                formatted_messages.append(f"{message['sender']}: {message['message']}")
            else:
                # Old format - just the message text
                formatted_messages.append(f"Message {i+1}: {message}")
        
        return "\n".join(formatted_messages)

    def get_last_message(self):
        """
        Get the last message sent by this agent
        """
        return self.last_message

    def update_state(self, key, value):
        """
        Update the agent's state with a key-value pair
        """
        self.state[key] = value

    def get_state(self):
        """
        Get the agent's current state
        """
        return self.state.copy()
