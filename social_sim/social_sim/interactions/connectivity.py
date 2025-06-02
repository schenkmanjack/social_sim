class ConnectivityGraph:
    def __init__(self, graph_dict):
        """
        Initialize with a dictionary mapping agent IDs to their connectivity info
        graph_dict should be of form:
        {
            "agent_id": {
                "visible_facts": [fact_indices],
                "neighbors": [neighbor_ids]
            }
        }
        """
        self.graph = graph_dict

    def get_visible_state(self, agent_id, env_state):
        """
        Returns the subset of environment facts visible to the given agent
        """
        if agent_id not in self.graph:
            return []
            
        visible_indices = self.graph[agent_id]["visible_facts"]
        return [env_state[i] for i in visible_indices if i < len(env_state)]

    def get_neighbors(self, agent_id):
        """
        Returns list of agent IDs that this agent can communicate with
        """
        if agent_id not in self.graph:
            return []
        return self.graph[agent_id]["neighbors"]
