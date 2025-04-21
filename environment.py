class Environment:
    def __init__(self, state):
        self.state = state
        self.history = []

    def snapshot_for_agent(self, agent_id, graph):
        return graph.get_visible_state(agent_id, self.state)

    def update(self, agent_actions):
        self.history.append(agent_actions)
        pass
