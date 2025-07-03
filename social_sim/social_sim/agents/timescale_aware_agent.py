from .agent import Agent

class TimescaleAwareAgent(Agent):
    def __init__(self, agent_id, identity, llm, use_full_agent_memory=True):
        super().__init__(agent_id, identity, llm, use_full_agent_memory)
        self.time_scale = None
        self.current_step = None
        self.total_steps = None

    def generate_prompts(self, visible_state, messages, current_step=None, total_steps=None, time_scale=None):
        """
        Generate separate system and user prompts for timescale-aware agent (compact version)
        Returns: (system_prompt, user_prompt)
        """
        # Set context
        self.time_scale = time_scale
        self.current_step = current_step
        self.total_steps = total_steps

        # System prompt
        system_prompt = self.identity

        # --- Compact time context ---------------------------------------
        time_context = self._build_time_context()

        # Messages
        messages_text = self._format_messages(messages)
        messages_block = f"Msgs:\n{messages_text}" if messages_text else ""

        # Memory (full or last)
        if self.use_full_agent_memory:
            mem_block = f"Memory:{self.memory}"
        else:
            mem_block = f"Prev:{self.last_message}" if self.last_message else "Prev:None"

        # Build user prompt (no unnecessary headers)
        user_prompt_parts = []
        if time_context:
            user_prompt_parts.append(time_context)
        if messages_block:
            user_prompt_parts.append(messages_block)
        if mem_block:
            user_prompt_parts.append(mem_block)
        user_prompt_parts.append("Next action (one sentence):")

        user_prompt = "\n".join(user_prompt_parts)

        return system_prompt, user_prompt

    def generate_prompt(self, visible_state, messages, current_step=None, total_steps=None, time_scale=None):
        """
        Generate the prompt that would be used for acting, without actually calling the LLM
        (Kept for backward compatibility)
        """
        system_prompt, user_prompt = self.generate_prompts(visible_state, messages, current_step, total_steps, time_scale)
        
        # If no system prompt, return just user prompt
        if not system_prompt:
            return user_prompt
        
        # Otherwise combine them for backward compatibility
        return f"System: {system_prompt}\n\nUser: {user_prompt}"

    def act(self, visible_state, messages, current_step=None, total_steps=None, time_scale=None):
        """Generate an action given the visible state, messages, and time context"""
        system_prompt, user_prompt = self.generate_prompts(visible_state, messages, current_step, total_steps, time_scale)
        
        # Call LLM with system and user prompts
        action = self.llm.generate(user_prompt, system_prompt)
        
        # Update memory and last message
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