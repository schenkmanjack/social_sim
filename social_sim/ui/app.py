from flask import Flask, render_template, request, jsonify
from social_sim.simulation import Simulation
from social_sim.orchestrator import Orchestrator
from social_sim.llm_interfaces import OpenAIBackend
import os
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', version="v2")

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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
