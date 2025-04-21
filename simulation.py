from environment import Environment
from agent import Agent

class Simulation:
    def __init__(self, orchestrator, llm):
        self.orchestrator = orchestrator
        self.llm = llm
        self.env = None
        self.agents = {}
        self.graph = None
        self.history = []

    def run(self, query, steps=10):
        agents_data, env_data, graph = self.orchestrator.setup_simulation(query)
        self.env = Environment(env_data)
        self.agents = {
            a['id']: Agent(a['id'], a['identity'], a['neighbors'])
            for a in agents_data
        }
        self.graph = graph

        for t in range(steps):
            step_log = {}
            for agent_id, agent in self.agents.items():
                if self.should_activate(agent_id):
                    visible = self.env.snapshot_for_agent(agent_id, self.graph)
                    comms = self.get_communications(agent_id)
                    action = agent.step(visible, comms, self.llm)
                    step_log[agent_id] = action
            self.env.update(step_log)
            self.history.append(step_log)

        return self.orchestrator.summarize_simulation(query, self.history)

    def should_activate(self, agent_id):
        return True

    def get_communications(self, agent_id):
        return []
