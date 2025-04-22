from flask import Flask, render_template, request, jsonify
from social_sim.simulation import Simulation
from social_sim.orchestrator import Orchestrator
from social_sim.llm import OpenAIBackend
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        # Get parameters from request
        data = request.json
        query = data.get('query')
        steps = int(data.get('steps', 5))
        agent_type = data.get('agent_type', 'regular')
        chunk_size = int(data.get('chunk_size', 5))

        # Initialize LLM
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return jsonify({'error': 'OPENAI_API_KEY not set'}), 500

        llm = OpenAIBackend(api_key=api_key)
        
        # Run simulation
        simulation = Simulation(llm, agent_type=agent_type, chunk_size=chunk_size)
        result = simulation.run(query=query, steps=steps)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
