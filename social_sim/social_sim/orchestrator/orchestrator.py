import json
import re
import time

class Orchestrator:
    def __init__(self, llm_wrapper, delay_seconds=1):
        self.llm = llm_wrapper
        self.delay_seconds = delay_seconds
        self.last_request_time = 0

    def _wait_for_rate_limit(self):
        """Wait between requests to respect rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.delay_seconds:
            time.sleep(self.delay_seconds - time_since_last_request)
        self.last_request_time = time.time()

    def _call_llm_with_retry(self, prompt, max_retries=3):
        """Wrapper for LLM calls with rate limiting and retry logic"""
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                return self.llm.generate(prompt)
            except Exception as e:
                if "rate_limit_exceeded" in str(e).lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        self.delay_seconds *= 2
                        print(f"Rate limit hit, increasing delay to {self.delay_seconds} seconds")
                        time.sleep(self.delay_seconds)
                        continue
                raise e

    def setup_simulation(self, query):
        """
        Uses LLM to generate the simulation setup based on the query
        Returns agents, environment, and connectivity graph
        """
        prompt = f"""
        Given the query: '{query}'
        Define a list of agents with identities and roles,
        a synthetic environment state,
        and a connectivity graph (who sees what and can talk to whom). The agent ids should be of the form agent_1, agent_2, agent_3, ...

        Output the result as JSON with the following structure:
        {{
            "agents": [
                {{
                    "id": "agent_1",
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
                "agent_1": {{
                    "visible_facts": [0, 2, 3],  // indices of facts this agent can see (can be empty)
                    "neighbors": ["agent_2", "agent_3"]  // IDs of agents they can communicate with (can be empty)
                }},
                ...
            }}
        }}

        For the connectivity graph:
        - visible_facts should be indices into the environment facts array (can be empty)
        - neighbors should be IDs of other agents they can communicate with (can be empty)
        - Make sure the connectivity is realistic based on the agents' roles
        """

        response = self._call_llm_with_retry(prompt)
        print(f"Orchestrator: LLM response: {response}")
        try:
            # Attempt to find JSON within potential markdown/text
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                raise ValueError("No valid JSON object found in LLM response.")
            json_str = match.group(0)
            setup = json.loads(json_str)

            # Validate the structure
            if not all(key in setup for key in ["agents", "environment", "connectivity"]):
                raise ValueError("Missing required keys in setup")
            if not isinstance(setup["agents"], list):
                 raise ValueError("'agents' must be a list")
            if not isinstance(setup["environment"], dict) or "facts" not in setup["environment"] or not isinstance(setup["environment"]["facts"], list):
                 raise ValueError("'environment' must be a dict with a 'facts' list")
            if not isinstance(setup["connectivity"], dict):
                 raise ValueError("'connectivity' must be a dict")

            # Ensure all agents defined in 'agents' list have connectivity info
            for agent in setup["agents"]:
                if agent["id"] not in setup["connectivity"]:
                     raise ValueError(f"Missing connectivity info for agent {agent['id']}")

            return setup

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}. Response:\n{response}")
        except Exception as e:
            # Add more context to the error
            raise ValueError(f"Error processing simulation setup from LLM: {str(e)}. Response:\n{response}")

    def _estimate_total_tokens(self, prompt_template: str, data: any) -> int:
        """
        Estimate total tokens including prompt template and data
        """
        # More conservative token estimation (1.5 chars ≈ 1 token)
        template_tokens = len(prompt_template) // 1.5
        data_tokens = len(json.dumps(data)) // 1.5
        response_buffer = 2000  # Conservative buffer
        
        return template_tokens + data_tokens + response_buffer

    def _chunk_by_tokens(self, data: list, max_tokens_per_chunk: int = 500) -> list:
        """
        Split data into chunks based on token count, accounting for prompt overhead.
        
        Args:
            data: List of items to chunk
            max_tokens_per_chunk: Maximum tokens per chunk (default 500 to stay well under limit)
        Returns:
            List of chunks
        """
        chunks = []
        current_chunk = []
        current_chunk_tokens = 0
        
        # Minimal prompt template
        prompt_template = "S:\n{data}"  # Ultra-minimal prompt
        
        for item in data:
            # Estimate total tokens including prompt overhead
            item_tokens = self._estimate_total_tokens(prompt_template, item)
            
            if current_chunk_tokens + item_tokens > max_tokens_per_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_chunk_tokens = 0
            
            current_chunk.append(item)
            current_chunk_tokens += item_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _combine_summaries(self, chunk_summaries):
        """
        Combines multiple chunk summaries into a final summary
        """
        if not chunk_summaries:
            return "No summaries available"
        
        # If only one chunk, return it directly
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
        
        # Combine summaries in smaller batches
        combined = []
        current_batch = []
        current_chars = 0
        
        for summary in chunk_summaries:
            summary_chars = len(summary)
            
            if current_chars + summary_chars > 8000 and current_batch:
                prompt = "Combine these summaries into one:" + "\n" + "\n".join(current_batch)
                try:
                    batch_summary = self._call_llm_with_retry(prompt)
                    combined.append(batch_summary)
                except Exception as e:
                    print(f"Error combining batch: {str(e)}")
                    combined.extend(current_batch)
                
                current_batch = []
                current_chars = 0
            
            current_batch.append(summary)
            current_chars += summary_chars
        
        # Process final batch
        if current_batch:
            prompt = "Combine these summaries into one:" + "\n" + "\n".join(current_batch)
            try:
                batch_summary = self._call_llm_with_retry(prompt)
                combined.append(batch_summary)
            except Exception as e:
                print(f"Error combining final batch: {str(e)}")
                combined.extend(current_batch)
        
        # If we have multiple combined summaries, combine them one final time
        if len(combined) > 1:
            prompt = "Combine these summaries into one final summary:" + "\n" + "\n".join(combined)
            try:
                return self._call_llm_with_retry(prompt)
            except Exception as e:
                print(f"Error creating final summary: {str(e)}")
                return "\n".join(combined)
        
        return combined[0]

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimate of token count (4 chars ≈ 1 token)
        """
        return len(text) // 4

    def _combine_within_limits(self, summaries: list, max_tokens: int = 15000) -> str:
        """
        Combines summaries while ensuring result stays within token limits
        """
        if not summaries:
            return "No summaries to combine"
        
        # If single summary, return it if within limits
        if len(summaries) == 1:
            if self._estimate_tokens(summaries[0]) <= max_tokens:
                return summaries[0]
            else:
                return "Summary too long to process"
        
        # Try combining all summaries
        combined = "\n".join(summaries)
        if self._estimate_tokens(combined) <= max_tokens:
            prompt = "Combine these summaries into one:\n" + combined
            try:
                return self._call_llm_with_retry(prompt)
            except Exception as e:
                print(f"Error combining summaries: {str(e)}")
                return combined
        
        # If too long, split into smaller groups
        mid = len(summaries) // 2
        first_half = self._combine_within_limits(summaries[:mid], max_tokens)
        second_half = self._combine_within_limits(summaries[mid:], max_tokens)
        
        # Combine the halves
        final_combined = first_half + "\n" + second_half
        if self._estimate_tokens(final_combined) <= max_tokens:
            prompt = "Combine these two summaries into one:\n" + final_combined
            try:
                return self._call_llm_with_retry(prompt)
            except Exception as e:
                print(f"Error combining final summaries: {str(e)}")
                return final_combined
        else:
            return final_combined

    def analyze_metrics(self, simulation_history, metrics_to_analyze: list = None, max_chars: int = 1000):
        """
        Analyzes metrics one at a time to stay within token limits
        """
        if not simulation_history:
            return "No simulation history available"
        
        # Extract all unique numeric metrics
        all_metrics = set()
        for step in simulation_history:
            metrics = step.get('metrics', {})
            all_metrics.update(k for k, v in metrics.items() if isinstance(v, (int, float)))
        
        if not all_metrics:
            return "No numeric metrics found in simulation"
        
        # Process one metric at a time
        metric_summaries = []
        
        for metric in all_metrics:
            try:
                # Extract data for this metric only
                metric_data = []
                for step in simulation_history:
                    value = step.get('metrics', {}).get(metric)
                    if isinstance(value, (int, float)):
                        metric_data.append({
                            's': step.get('step'),
                            'v': value
                        })
                
                if not metric_data:
                    continue
                    
                print(f"Analyzing metric: {metric}")
                
                # Create minimal prompt
                prompt = f"""
                Analyze trends in this metric: {metric}
                Data: {json.dumps(metric_data, separators=(',', ':'))}
                """
                
                # Check token count before making API call
                if self._estimate_tokens(prompt) > 15000:
                    print(f"Warning: Metric {metric} data too large, skipping")
                    continue
                
                try:
                    metric_summary = self._call_llm_with_retry(prompt)
                    if metric_summary:
                        metric_summaries.append(f"{metric}: {metric_summary}")
                except Exception as e:
                    print(f"Warning: Error analyzing metric {metric}: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"Error processing metric {metric}: {str(e)}")
                continue
        
        if not metric_summaries:
            return "No metrics analysis available"
        
        # Combine metric summaries
        try:
            prompt = "Combine these metric analyses into one summary:\n" + "\n".join(metric_summaries)
            if self._estimate_tokens(prompt) <= 15000:
                return self._call_llm_with_retry(prompt)
            else:
                # If too long, just return the individual summaries
                return "\n\n".join(metric_summaries)
        except Exception as e:
            print(f"Error combining metric summaries: {str(e)}")
            return "\n\n".join(metric_summaries)

    def summarize_outcome(self, simulation_history, max_chars: int = 8000):
        """
        Summarizes simulation with minimal data extraction
        """
        chunk_summaries = []
        current_chunk = []
        current_chars = 0
        
        for i, step in enumerate(simulation_history):
            try:
                # Extract only essential information
                minimal_step = {
                    'step': step.get('step'),
                    'actions': [
                        {
                            'agent': action.get('agent'),
                            'action': action.get('action'),
                            'visible_state': action.get('visible_state', [])[:3]  # Only first 3 state items
                        }
                        for action in step.get('actions', [])[:5]  # Only first 5 actions
                    ]
                }
                
                step_json = json.dumps(minimal_step, separators=(',', ':'))
                step_chars = len(step_json)
                
                print(f"Step {i+1}: {step_chars} chars")
                
                if current_chars + step_chars > max_chars and current_chunk:
                    print(f"Processing chunk {len(chunk_summaries) + 1} with {len(current_chunk)} steps...")
                    
                    prompt = f"S:{json.dumps(current_chunk, separators=(',', ':'))}"
                    try:
                        summary = self._call_llm_with_retry(prompt)
                        if summary:
                            chunk_summaries.append(summary)
                            print(f"Successfully summarized chunk {len(chunk_summaries)}")
                        else:
                            print(f"Warning: Empty summary for chunk {len(chunk_summaries) + 1}")
                    except Exception as e:
                        print(f"Error summarizing chunk {len(chunk_summaries) + 1}: {str(e)}")
                        print(f"Chunk size: {len(current_chunk)} steps, {current_chars} chars")
                    
                    current_chunk = []
                    current_chars = 0
                
                current_chunk.append(minimal_step)
                current_chars += step_chars
                
            except Exception as e:
                print(f"Error processing step {i+1}: {str(e)}")
                continue
        
        # Process final chunk
        if current_chunk:
            print(f"Processing final chunk with {len(current_chunk)} steps...")
            try:
                prompt = f"S:{json.dumps(current_chunk, separators=(',', ':'))}"
                summary = self._call_llm_with_retry(prompt)
                if summary:
                    chunk_summaries.append(summary)
                    print(f"Successfully summarized final chunk")
                else:
                    print("Warning: Empty final summary")
            except Exception as e:
                print(f"Error summarizing final chunk: {str(e)}")
        
        if not chunk_summaries:
            print("Error: No chunks were successfully summarized")
            return {
                'summary': "Summary generation failed - no chunks processed successfully",
                'metrics_analysis': None
            }
        
        # Combine summaries
        try:
            final_summary = self._combine_summaries(chunk_summaries)
            if not final_summary:
                print("Error: Combined summary is empty")
                final_summary = "Summary combination failed"
        except Exception as e:
            print(f"Error combining summaries: {str(e)}")
            final_summary = "Summary combination failed"
        
        return {
            'summary': final_summary,
            'metrics_analysis': None
        }

    def determine_plot_metrics(self, query: str, history: list, should_plot: bool = False, max_tokens_per_chunk: int = 10000) -> list:
        """
        Analyzes the simulation history to identify meaningful metrics and visualization types.
        Only runs if should_plot is True.
        """
        if not should_plot:
            print("Plotting disabled, skipping metrics analysis")
            return []
        
        if not history:
            print("No history provided, skipping metrics analysis")
            return []
        
        print("Orchestrator: Analyzing simulation for meaningful metrics...")
        
        # Process history in chunks to avoid token limits
        chunks = self._chunk_by_tokens(history, max_tokens_per_chunk)
        all_metrics = []
        
        for i, chunk in enumerate(chunks):
            print(f"Analyzing metrics for chunk {i+1}/{len(chunks)}...")
            try:
                chunk_summary = self._summarize_chunk_for_metrics(query, chunk)
                if not chunk_summary:
                    print(f"Warning: Empty summary for chunk {i+1}")
                    continue
                    
                chunk_metrics = self._extract_metrics_from_summary(chunk_summary, query)
                if chunk_metrics:
                    for metric in chunk_metrics:
                        if not any(m["metric_name"] == metric["metric_name"] for m in all_metrics):
                            all_metrics.append(metric)
            except Exception as e:
                print(f"Warning: Could not determine metrics for chunk {i+1}: {str(e)}")
                continue
        
        return all_metrics

    def _summarize_chunk_for_metrics(self, query: str, chunk: list) -> str:
        """
        Summarize a chunk with the specific intent of finding meaningful metrics
        """
        prompt = f"""
        You are analyzing a simulation with the following query: "{query}"
        
        Your task is to summarize this chunk of the simulation with a focus on identifying 
        meaningful metrics that could be plotted to visualize important trends or patterns.
        
        Consider:
        1. What measurable quantities changed over time?
        2. What categories of actions or events occurred?
        3. What relationships between agents or events could be quantified?
        
        Simulation chunk:
        {json.dumps(chunk, indent=2)}
        
        Please provide a summary that highlights potential metrics and their significance.
        """
        
        try:
            response = self._call_llm_with_retry(prompt)
            return response
        except Exception as e:
            print(f"Warning: Error summarizing chunk for metrics: {str(e)}")
            return ""

    def _extract_metrics_from_summary(self, summary: str, query: str) -> list:
        """
        Extract specific metrics and their keywords from a summary, guided by the simulation query
        """
        prompt = f"""
        Based on the following simulation summary and original query, identify specific metrics that could be plotted.
        Focus on metrics relevant to the simulation's domain and objectives.

        Original simulation query:
        {query}

        Summary to analyze:
        {summary}

        For each metric, provide:
        1. A clear question the metric answers
        2. The metric name
        3. The data type (e.g., economic, political, technological, social)
        4. Suggested visualization type (line, bar, pie, etc.)
        5. How to extract the data
        6. Relevant keywords for data extraction

        Format the response as a JSON array of objects with these fields:
        - question
        - metric_name
        - data_type
        - visualization
        - extraction_method
        - keywords
        """

        try:
            response = self._call_llm_with_retry(prompt)
            match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                metrics = json.loads(json_str)
                return metrics
            return []
        except Exception as e:
            print(f"Warning: Error extracting metrics from summary: {str(e)}")
            return []
