class Environment:
    def __init__(self, initial_facts):
        self.facts = initial_facts
        self.history = []

    def snapshot_for_agent(self, agent_id, graph):
        """
        Get the subset of environment facts visible to an agent
        """
        return graph.get_visible_state(agent_id, self.facts)

    def update(self, action):
        """
        Update the environment based on an agent's action
        """
        # For now, just append the action as a new fact
        self.facts.append(action)
        self.history.append(action)

    def get_state(self):
        """
        Get current state of the environment
        Returns:
            List of all current facts in the environment
        """
        return self.facts
