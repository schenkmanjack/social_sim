class Agent:
    def __init__(self, agent_id, identity, neighbors):
        self.agent_id = agent_id
        self.identity = identity
        self.memory = []
        self.neighbors = neighbors

    def step(self, visible_env, comms, llm):
        context = self._build_context(visible_env, comms)
        response = llm.call(context)
        self.memory.append(response)
        return self._parse_response(response)

    def _build_context(self, env, comms):
        return f"""
        Agent Identity: {self.identity}
        Visible Environment: {env}
        Messages from Neighbors: {comms}
        What action should the agent take next?
        """

    def _parse_response(self, response):
        return response
