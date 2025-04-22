import json

class Orchestrator:
    def __init__(self, llm_wrapper):
        self.llm = llm_wrapper

    def setup_simulation(self, query):
        """
        Uses LLM to generate the simulation setup based on the query
        Returns agents, environment, and connectivity graph
        """
        prompt = f"""
        Given the query: '{query}'
        Define a list of agents with identities and roles,
        a synthetic environment state,
        and a connectivity graph (who sees what and can talk to whom).

        Output the result as JSON with the following structure:
        {{
            "agents": [
                {{
                    "id": "unique_agent_id",
                    "identity": "detailed description of who they are and their role"
                }},
                ...
            ],
            "environment": {{
                "facts": [
                    "fact 1",
                    "fact 2",
                    ...
                ]
            }},
            "connectivity": {{
                "agent_id": {{
                    "visible_facts": [0, 2, 3],  // indices of facts this agent can see
                    "neighbors": ["agent2", "agent3"]  // IDs of agents they can communicate with
                }},
                ...
            }}
        }}

        For the connectivity graph:
        - visible_facts should be indices into the environment facts array
        - neighbors should be IDs of other agents they can communicate with
        - Make sure the connectivity is realistic based on the agents' roles
        """

        response = self.llm.generate(prompt)
        try:
            setup = json.loads(response)
            
            # Validate the structure
            if not all(key in setup for key in ["agents", "environment", "connectivity"]):
                raise ValueError("Missing required keys in setup")
            
            # Validate connectivity graph
            for agent in setup["agents"]:
                agent_id = agent["id"]
                if agent_id not in setup["connectivity"]:
                    raise ValueError(f"Missing connectivity info for agent {agent_id}")
                
                conn = setup["connectivity"][agent_id]
                if not all(key in conn for key in ["visible_facts", "neighbors"]):
                    raise ValueError(f"Invalid connectivity structure for agent {agent_id}")
                
                # Validate fact indices
                for fact_idx in conn["visible_facts"]:
                    if not isinstance(fact_idx, int) or fact_idx < 0:
                        raise ValueError(f"Invalid fact index for agent {agent_id}")
                    if fact_idx >= len(setup["environment"]["facts"]):
                        raise ValueError(f"Fact index out of bounds for agent {agent_id}")
                
                # Validate neighbor IDs
                for neighbor_id in conn["neighbors"]:
                    if not any(a["id"] == neighbor_id for a in setup["agents"]):
                        raise ValueError(f"Invalid neighbor ID {neighbor_id} for agent {agent_id}")

            return setup

        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise ValueError(f"Error in simulation setup: {str(e)}")

    def summarize_outcome(self, simulation_history):
        """
        Uses LLM to generate a summary of the simulation outcome
        """
        prompt = f"""
        Given the following simulation history:
        {json.dumps(simulation_history, indent=2)}

        Provide a concise summary of the key outcomes and implications.
        """
        return self.llm.generate(prompt)
