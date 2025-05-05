from flask import Flask, render_template, request, jsonify, send_from_directory
from social_sim.simulation import Simulation
from social_sim.orchestrator import Orchestrator
from social_sim.llm_interfaces import OpenAIBackend
from social_sim.experiment import Experiment
import os
import json
from datetime import datetime

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

# Add this route to serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        # Get parameters from request
        data = request.json
        query = data.get('query')
        steps = data.get('steps')
        
        # Validate required parameters
        if not query:
            return jsonify({'error': 'query parameter is required'}), 400
        if not steps:
            return jsonify({'error': 'steps parameter is required'}), 400
            
        steps = int(steps)
        agent_type = data.get('agent_type', 'regular')
        chunk_size = int(data.get('chunk_size', 1200))
        plots = data.get('plots', False)
        plot_keywords = data.get('plot_keywords', [])
        results_folder = data.get('results_folder', 'results')

        # Initialize LLM
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return jsonify({'error': 'OPENAI_API_KEY not set'}), 500

        llm = OpenAIBackend(api_key=api_key)
        
        def clean_text(text):
            if isinstance(text, str):
                # Replace newlines with \n escape sequence
                text = text.replace('\n', '\\n')
                # Escape other special characters
                text = text.replace('"', '\\"')
                text = text.replace('\r', '\\r')
                text = text.replace('\t', '\\t')
                # Ensure no unescaped quotes remain
                text = text.replace('"', '\\"')
            return text

        def clean_dict(d):
            if not isinstance(d, dict):
                return d
            cleaned = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    cleaned[k] = clean_dict(v)
                elif isinstance(v, list):
                    cleaned[k] = [clean_dict(item) if isinstance(item, dict) else clean_text(item) for item in v]
                else:
                    cleaned[k] = clean_text(v)
            return cleaned
        
        def progress_generator():
            try:
                simulation = Simulation(llm, agent_type=agent_type, chunk_size=chunk_size)
                for step, result in simulation.run_with_progress(query=query, steps=steps):
                    try:
                        if step == steps:  # Final result
                            print(f"Step {step}: Processing final result")
                            
                            # Clean and prepare the summary
                            summary_text = result.get('summary', {}).get('summary', '')
                            if summary_text:
                                summary_text = clean_text(summary_text)
                            
                            # Clean history items
                            cleaned_history = []
                            for hist_item in result.get('history', []):
                                if isinstance(hist_item, dict):
                                    cleaned_item = {k: clean_text(v) for k, v in hist_item.items()}
                                elif isinstance(hist_item, list):
                                    cleaned_item = [clean_text(v) for v in hist_item]
                                else:
                                    cleaned_item = clean_text(hist_item)
                                cleaned_history.append(cleaned_item)
                            
                            response = {
                                'final_result': {
                                    'summary': {'summary': summary_text},
                                    'history': cleaned_history,
                                    'metrics': result.get('metrics', [])
                                },
                                'progress': {
                                    'current_step': step,
                                    'total_steps': steps,
                                    'percentage': 100
                                }
                            }
                            
                            # Debug: Print and save the response
                            print("\n=== DEBUG: Final Response ===")
                            print(json.dumps(response, indent=2, ensure_ascii=False))
                            
                            # Save to file for debugging
                            with open('debug_response.json', 'w', encoding='utf-8') as f:
                                json.dump(response, f, indent=2, ensure_ascii=False)
                            
                            # Ensure proper JSON encoding
                            json_str = json.dumps(response, ensure_ascii=False)
                            print("JSON string length:", len(json_str))
                            yield f"data: {json_str}\n\n"
                            print("Final response sent")
                        else:
                            # Progress update
                            progress_data = {
                                'progress': {
                                    'current_step': step,
                                    'total_steps': steps,
                                    'percentage': (step / steps) * 100
                                }
                            }
                            json_str = json.dumps(progress_data, ensure_ascii=False)
                            yield f"data: {json_str}\n\n"
                            
                    except Exception as e:
                        print(f"Step {step} error: {str(e)}")
                        error_response = {'error': f'Step processing error: {str(e)}'}
                        print("\n=== DEBUG: Error Response ===")
                        print(json.dumps(error_response, indent=2))
                        yield f"data: {json.dumps(error_response)}\n\n"
                        
            except Exception as e:
                print(f"Simulation error: {str(e)}")
                error_response = {'error': f'Simulation error: {str(e)}'}
                print("\n=== DEBUG: Error Response ===")
                print(json.dumps(error_response, indent=2))
                yield f"data: {json.dumps(error_response)}\n\n"
        
        return app.response_class(progress_generator(), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/run_experiment', methods=['GET'])
def run_experiment():
    try:
        # Get parameters from URL
        data = json.loads(request.args.get('data', '{}'))
        
        print("DEBUG: run_experiment route hit")  # Debug log
        print("DEBUG: Received data:", data)  # Debug log
        
        if not data:
            print("DEBUG: No data received")  # Debug log
            return jsonify({'error': 'No data received'}), 400
            
        query = data.get('query')
        steps = data.get('steps')
        num_simulations = data.get('num_simulations')
        agent_type = data.get('agent_type', 'regular')
        chunk_size = int(data.get('chunk_size', 1200))
        outcomes = data.get('outcomes', [])
        experiment_name = data.get('name', f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        results_folder = data.get('results_folder', f'results_{experiment_name}')
        
        print("DEBUG: Parsed parameters:", {  # Debug log
            'query': query,
            'steps': steps,
            'num_simulations': num_simulations,
            'outcomes': outcomes
        })
        
        # Validate required parameters
        if not all([query, steps, num_simulations]):
            print("DEBUG: Missing required parameters")  # Debug log
            return jsonify({'error': 'Missing required parameters: query, steps, or num_simulations'}), 400
        if not outcomes:
            print("DEBUG: No outcomes defined")  # Debug log
            return jsonify({'error': 'No outcomes defined. Please add at least one outcome.'}), 400
            
        # Initialize LLM
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return jsonify({'error': 'OPENAI_API_KEY not set'}), 500

        llm = OpenAIBackend(api_key=api_key)
        
        def progress_generator():
            try:
                # Create multiple simulations
                simulations = [
                    Simulation(llm, agent_type=agent_type, chunk_size=chunk_size)
                    for _ in range(num_simulations)
                ]
                
                # Create experiment with name
                experiment = Experiment(simulations, name=experiment_name)
                
                # Define outcomes and log them
                for outcome in outcomes:
                    experiment.define_outcome(
                        name=outcome['name'],
                        condition=outcome['condition'],
                        description=outcome['description']
                    )
                    # Log outcome definition
                    log_message = f"Defined outcome: {outcome['name']}"
                    yield f'data: {json.dumps({"log": log_message})}\n\n'
                
                # Run experiment with progress tracking
                for progress, result in experiment.run(query, steps):
                    # Send progress update
                    yield f'data: {json.dumps({"progress": progress})}\n\n'
                    
                    # If this is the final result, save it
                    if progress['percentage'] == 100:
                        # Save results
                        os.makedirs(results_folder, exist_ok=True)
                        experiment.save_results(results_folder, plot_results=True)
                        
                        # Send final result
                        yield f'data: {json.dumps({"final_result": result})}\n\n'
                
            except Exception as e:
                error_response = {'error': f'Experiment error: {str(e)}'}
                yield f'data: {json.dumps(error_response)}\n\n'
        
        return app.response_class(progress_generator(), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_experiment_route', methods=['GET'])
def test_experiment_route():
    return jsonify({'status': 'experiment route is accessible'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5023, debug=True, use_reloader=False)
