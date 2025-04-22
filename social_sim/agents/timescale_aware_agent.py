from .agent import Agent
class TimescaleAwareAgent(Agent):
    def __init__(self, agent_id, identity, llm):
        super().__init__(agent_id, identity, llm)
        self.time_scale = None
        self.current_step = None
        self.total_steps = None

    def act(self, visible_state, messages, current_step=None, total_steps=None, time_scale=None):
        """
        Decide on an action with time scale awareness
        """
        self.time_scale = time_scale
        self.current_step = current_step
        self.total_steps = total_steps

        time_context = self._build_time_context()
        action_guidance = self._get_action_guidance()
        
        prompt = f"""
        You are {self.identity}
        
        {time_context}
        
        {action_guidance}
        
        Current environment state:
        {visible_state}
        
        Messages from others:
        {messages}
        
        What action do you take? Respond with a single sentence describing your action.
        """
        
        action = self.llm.generate(prompt)
        self.last_message = action
        self.memory.append(action)
        return action

    def _build_time_context(self):
        """Build a detailed time context string"""
        if self.current_step is None or self.total_steps is None:
            return "This is an ongoing simulation with no fixed end point."
            
        time_context = []
        
        # Add time scale information
        if self.time_scale:
            time_context.append(f"This simulation is operating on a {self.time_scale} time scale.")
        
        # Add progress information
        progress = (self.current_step / self.total_steps) * 100
        time_context.append(f"Current progress: {progress:.1f}% complete (step {self.current_step} of {self.total_steps})")
        
        # Add phase information
        if progress < 25:
            phase = "early"
        elif progress < 75:
            phase = "middle"
        else:
            phase = "final"
        time_context.append(f"This is the {phase} phase of the simulation.")
        
        return "\n".join(time_context)

    def _get_action_guidance(self):
        """Provide guidance on appropriate actions based on time scale"""
        if not self.time_scale:
            return ""
            
        guidance = []
        
        # Time scale specific guidance
        if self.time_scale == "years":
            guidance.extend([
                "Focus on long-term strategic planning and policy development.",
                "Consider multi-year impacts and consequences.",
                "Think about legacy and lasting effects."
            ])
        elif self.time_scale == "months":
            guidance.extend([
                "Focus on medium-term planning and resource allocation.",
                "Consider quarterly impacts and milestones.",
                "Think about organizational changes and restructuring."
            ])
        elif self.time_scale == "weeks":
            guidance.extend([
                "Focus on operational planning and project management.",
                "Consider weekly milestones and deliverables.",
                "Think about team coordination and resource deployment."
            ])
        elif self.time_scale == "days":
            guidance.extend([
                "Focus on immediate actions and daily operations.",
                "Consider short-term impacts and quick wins.",
                "Think about tactical responses and immediate needs."
            ])
        elif self.time_scale == "hours":
            guidance.extend([
                "Focus on immediate response and crisis management.",
                "Consider minute-by-minute developments.",
                "Think about emergency procedures and rapid decision-making."
            ])
            
        return "\n".join(guidance)