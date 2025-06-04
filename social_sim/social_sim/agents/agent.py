class Agent:
    def __init__(self, agent_id, identity, llm):
        """
        Initialize an agent
        Args:
            agent_id: Unique identifier for the agent
            identity: Description of who the agent is and their role
            llm: LLM wrapper for generating responses
        """
        self.agent_id = agent_id
        self.identity = identity
        self.llm = llm
        self.last_message = None
        self.memory = []  # Initialize empty memory
        self.state = {}  # Generic state dictionary

    def generate_prompt(self, visible_state, messages):
        """
        Generate the prompt that would be used for acting, without actually calling the LLM
        """
        # Format messages with sender information if available
        messages_text = self._format_messages(messages)
        
        prompt = f"""
        You are {self.identity}
        
        Current environment state:
        {visible_state}
        
        Messages from others:
        {messages_text}
        
        Your previous actions:
        {self.memory}
        
        What action do you take? Respond with a clear description of your action.
        """
        
        return prompt

    def act(self, visible_state, messages):
        """
        Decide on an action based on visible state and messages
        """
        prompt = self.generate_prompt(visible_state, messages)
        action = self.llm.generate(prompt)
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
