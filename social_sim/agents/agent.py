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

    def act(self, visible_state, messages):
        """
        Decide on an action based on visible state and messages
        """
        prompt = f"""
        You are {self.identity}
        
        Current environment state:
        {visible_state}
        
        Messages from others:
        {messages}
        
        Your previous actions:
        {self.memory}
        
        What action do you take? Respond with a clear description of your action.
        """
        
        action = self.llm.generate(prompt)
        self.last_message = action
        self.memory.append(action)
        return action

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
