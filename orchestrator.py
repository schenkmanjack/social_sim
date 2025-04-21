import json

class Orchestrator:
    def __init__(self, llm):
        self.llm = llm

    def setup_simulation(self, query):
        prompt = f"""
        Given the query: '{query}'
        Define a list of agents with identities and roles,
        a synthetic environment state,
        and a connectivity graph (who sees what and can talk to whom).

        Output the result as JSON with the following structure:
        {{
            "agents": [
                {{"id": ..., "identity": ..., "neighbors": [...]}},
                ...
            ],
            "environment": {{
                "facts": [...]
            }},
            "connectivity": {{
                "agent_id": ["visible_fact_indices", "neighbors"]
            }}
        }}
        """
        result = self.llm.call(prompt)
        return self.parse_result(result)

    def parse_result(self, result):
        parsed = json.loads(result)
        return parsed['agents'], parsed['environment'], parsed['connectivity']

    def summarize_simulation(self, query, history):
        summary_prompt = f"""
        Given the original query: '{query}' and the following simulation trace:

        {history}

        Summarize the key events, outcomes, and trends. Suggest useful graphs or tables to visualize these outcomes.
        Output:
        {{
            "summary": "...",
            "visualizations": ["..."]
        }}
        """
        summary_result = self.llm.call(summary_prompt)
        return summary_result
