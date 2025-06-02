from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import tempfile
from datetime import datetime

print("=== Starting Flask App ===")

try:
    from social_sim.llm_interfaces import OpenAIBackend
    print("✓ OpenAIBackend imported successfully")
except ImportError as e:
    print(f"✗ OpenAIBackend import failed: {e}")

try:
    from social_sim.simulation.simulation import Simulation
    print("✓ Simulation imported successfully")
except ImportError as e:
    print(f"✗ Simulation import failed: {e}")

# Comment out the Experiment import for now since it's failing
# try:
#     from social_sim.experiment import Experiment
#     print("✓ Experiment imported successfully")
# except ImportError as e:
#     print(f"✗ Experiment import failed: {e}")

app = Flask(__name__)
print("✓ Flask app created")

# Temporarily disable CORS restrictions for debugging
CORS(app, origins="*", methods="*", allow_headers="*")
print("✓ CORS configured (permissive)")

def get_hardcoded_config():
    """Return the exact config from ab_testing_config.json for testing"""
    return {
        "name": "makeup_ad_ab_test",
        "query": "Get sentiments for older women and young women on the advertisement below: Tell me if you have a positive or negative sentiment. The agents should have no communication with any other agent and no access to facts of the environment. Each agent should have no visible facts and no neighbors. 'Buy new makeup. It will make you look great.' Create agents representing different demographics and analyze their reactions to this marketing message.",
        "num_agents": 2,
        "steps": 1,
        "results_folder": "results_makeup_ad_ab_test",
        "agent_type": "regular",
        "chunk_size": 1200,
        "plot_results": True,
        "outcomes": [
            {
                "name": "positive_sentiment",
                "condition": "Most agents express positive sentiment towards the advertisement, showing interest or approval",
                "description": "Most agents have a positive reaction to the makeup advertisement"
            },
            {
                "name": "negative_sentiment", 
                "condition": "Most agents express negative sentiment towards the advertisement, showing disinterest or disapproval",
                "description": "Most agents have a negative reaction to the makeup advertisement"
            },
            {
                "name": "neutral_sentiment",
                "condition": "Most agents express neutral sentiment towards the advertisement, showing indifference",
                "description": "Most agents have a neutral reaction to the makeup advertisement"
            }
        ],
        "agent_outcome_definitions": {
            "positive_sentiment": "Agent has a positive reaction to the makeup advertisement",
            "negative_sentiment": "Agent has a negative reaction to the makeup advertisement", 
            "neutral_sentiment": "Agent has a neutral reaction to the makeup advertisement"
        }
    }

@app.route('/api/generate-personas', methods=['POST'])
def generate_personas():
    print("=== Generate Personas Endpoint Called ===")
    try:
        print("About to return hardcoded personas...")
        
        # Ignore user input for now, return hardcoded personas based on the config
        dummy_personas = [
            {
                "id": "persona_1",
                "name": "Older Woman",
                "description": "Mature woman with established beauty routines and preferences",
                "avatar": "👩‍🦳",
                "evaluation_focus": "Practicality and value"
            },
            {
                "id": "persona_2",
                "name": "Young Woman",
                "description": "Young woman interested in beauty trends and new products",
                "avatar": "👩",
                "evaluation_focus": "Trends and innovation"
            }
        ]
        
        print(f"Returning {len(dummy_personas)} hardcoded personas")
        
        return jsonify({
            "success": True,
            "personas": dummy_personas
        })
        
    except Exception as e:
        print(f"Error generating personas: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        # Ignore user input, use hardcoded config
        config = get_hardcoded_config()
        
        # Check for OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return jsonify({
                "success": False,
                "error": "OPENAI_API_KEY environment variable is not set"
            }), 500
        
        # Initialize LLM backend (exactly like run_ab_testing.py)
        llm = OpenAIBackend(api_key=api_key)
        
        # Initialize Simulation (exactly like run_ab_testing.py)
        simulation = Simulation(
            llm_wrapper=llm,
            agent_type=config.get("agent_type", "regular"),
            chunk_size=config.get("chunk_size", 1200),
            agent_outcome_definitions=config.get("agent_outcome_definitions", {})
        )
        
        # Create Experiment (exactly like run_ab_testing.py)
        experiment = Experiment([simulation], name=config["name"])
        
        # Define outcomes (exactly like run_ab_testing.py)
        # Note: ab_testing_config.json has both "outcomes" and "agent_outcome_definitions"
        # run_ab_testing.py uses "agent_outcome_definitions" for the loop
        for outcome_name, outcome_condition in config["agent_outcome_definitions"].items():
            experiment.define_outcome(
                name=outcome_name,
                condition=outcome_condition,
                description=f"Outcome: {outcome_name}"
            )
        
        # Run experiment (exactly like run_ab_testing.py)
        print(f"Running evaluation '{config['name']}'")
        results = []
        for result in experiment.run(query=config["query"], steps=config.get("steps", 1)):
            if isinstance(result, tuple) and len(result) == 2:
                progress, data = result
                if progress.get('percentage') == 100 and 'runs' in data:
                    results = data['runs']
                    break
        
        if not results:
            return jsonify({
                "success": False,
                "error": "No results were generated from the experiment"
            }), 500
        
        # Format results for frontend
        formatted_results = {
            'experiment_name': config.get('name', 'Unknown'),
            'query_used': config.get('query', ''),
            'agents_generated': config.get('num_agents', 0),
            'outcomes': config.get('agent_outcome_definitions', {}),
            'evaluations': [],
            'summary': {
                'total_evaluations': len(results) if results else 0,
                'outcome_counts': {}
            }
        }
        
        # Process results
        if results:
            for result in results:
                if 'agent_responses' in result:
                    formatted_results['evaluations'].extend(result.get('agent_responses', []))
                
                # Count outcomes
                for outcome in result.get('outcomes', []):
                    outcome_name = outcome.get('name', 'unknown')
                    if outcome_name not in formatted_results['summary']['outcome_counts']:
                        formatted_results['summary']['outcome_counts'][outcome_name] = 0
                    formatted_results['summary']['outcome_counts'][outcome_name] += 1
        
        return jsonify({
            "success": True,
            "results": formatted_results
        })
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/test', methods=['GET'])
def test():
    print("=== TEST ROUTE CALLED ===")
    try:
        response = jsonify({"message": "Flask server is working on port 5000!", "timestamp": datetime.now().isoformat()})
        print(f"=== TEST ROUTE RESPONSE: {response} ===")
        return response
    except Exception as e:
        print(f"=== TEST ROUTE ERROR: {e} ===")
        return str(e), 500

@app.route('/')
def home():
    print("=== HOME ROUTE CALLED ===")
    try:
        response = jsonify({
            "message": "Flask API Server is running!",
            "endpoints": [
                "/api/test - Test endpoint",
                "/api/generate-personas - Generate personas from query", 
                "/api/evaluate - Evaluate content with personas"
            ],
            "timestamp": datetime.now().isoformat()
        })
        print(f"=== HOME ROUTE RESPONSE: {response} ===")
        return response
    except Exception as e:
        print(f"=== HOME ROUTE ERROR: {e} ===")
        return str(e), 500

if __name__ == '__main__':
    print("=== Starting Flask Server ===")
    print("Available routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.endpoint}: {rule.rule} [{', '.join(rule.methods)}]")
    print("Starting server on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)