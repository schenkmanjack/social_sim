import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from typing import List, Dict, Any, Optional
import json
from pymongo import MongoClient
from datetime import datetime

# Configure the Streamlit page
st.set_page_config(
    page_title="Interactive Pareto Front Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InteractiveParetoApp:
    def __init__(self):
        self.data = None
        self.selected_point = None
        self.objective_labels = []
        self.metric_labels = []
        self.mongo_client = None
        self.db = None
    
    def connect_mongodb(self, mongo_uri: str, db_name: str = "genetic_algorithm"):
        """Connect to MongoDB"""
        try:
            self.mongo_client = MongoClient(mongo_uri)
            self.db = self.mongo_client[db_name]
            # Test connection - use the client's admin database to ping
            self.mongo_client.admin.command('ping')
            return True
        except Exception as e:
            st.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
    
    def list_experiments(self) -> List[Dict]:
        """List available experiments from MongoDB"""
        if self.db is None:
            return []
        
        try:
            # Get all experiments from the experiments collection (history type)
            experiments = list(self.db.experiments.find(
                {"type": "history", "experiment_id": {"$exists": True}},
                {"experiment_id": 1, "current_generation": 1, "_id": 0}
            ))
            
            # Remove duplicates and get latest generation for each experiment
            experiment_dict = {}
            for exp in experiments:
                exp_id = exp["experiment_id"]
                if exp_id not in experiment_dict or exp["current_generation"] > experiment_dict[exp_id]["current_generation"]:
                    experiment_dict[exp_id] = exp
            
            return list(experiment_dict.values())
        except Exception as e:
            st.error(f"Error listing experiments: {str(e)}")
            return []
    
    def load_mongodb_data(self, experiment_id: str) -> bool:
        """Load GA data from MongoDB using the exact structure from genetic_algorithm_base.py"""
        if self.db is None:
            st.error("Not connected to MongoDB")
            return False
        
        try:
            # Load history document (matches GA's _save_state structure)
            history_doc = self.db.experiments.find_one({
                "experiment_id": experiment_id,
                "type": "history"
            })
            
            if not history_doc:
                st.error(f"No history data found for experiment: {experiment_id}")
                return False
            
            # Debug info
            st.info(f"Found history document for experiment: {experiment_id}")
            current_generation = history_doc.get('current_generation', 0)
            st.info(f"Current generation: {current_generation}")
            
            # Check what keys are in the history document
            history_keys = list(history_doc.keys())
            st.info(f"History document keys: {history_keys}")
            
            # Load elite chunks from CURRENT generation only (for current Pareto front)
            current_gen_chunks = list(self.db.experiments.find({
                "experiment_id": experiment_id,
                "type": "chunk",
                "generation": current_generation - 1  # GA saves generation-1 as current
            }).sort("chunk_index", 1))
            
            st.info(f"Found {len(current_gen_chunks)} chunks for current generation ({current_generation - 1})")
            
            # Reconstruct current elite individuals from chunks
            elite_data = []
            elite_id = 0
            
            for chunk_idx, chunk in enumerate(current_gen_chunks):
                if 'elites' in chunk:
                    st.info(f"Chunk {chunk_idx} has {len(chunk['elites'])} elites")
                    for elite_dict in chunk['elites']:
                        # DOFs are strings (LLM prompts), keep them as strings
                        dofs = elite_dict['dofs']
                        if isinstance(dofs, list) and len(dofs) == 1:
                            dofs = dofs[0]  # Handle case where DOF is wrapped in array
                        
                        elite_data.append({
                            'id': elite_id,
                            'is_elite': True,
                            'objectives': np.array(elite_dict['objectives']),
                            'dofs': dofs,  # Keep as string
                            'metrics': np.array(elite_dict.get('metrics', elite_dict['objectives'])),
                            'constraints': elite_dict.get('constraints', []),
                            'generation': chunk.get('generation', current_generation - 1)
                        })
                        elite_id += 1
                else:
                    st.info(f"Chunk {chunk_idx} has no elites")
            
            st.info(f"Reconstructed {len(elite_data)} elite individuals")
            
            # Load ALL generation chunks for evolution data
            all_chunks = list(self.db.experiments.find({
                "experiment_id": experiment_id,
                "type": "chunk"
            }).sort([("generation", 1), ("chunk_index", 1)]))
            
            st.info(f"Found {len(all_chunks)} total chunks across all generations")
            
            # Organize chunks by generation
            chunks_by_generation = {}
            for chunk in all_chunks:
                gen = chunk.get('generation', 0)
                if gen not in chunks_by_generation:
                    chunks_by_generation[gen] = []
                chunks_by_generation[gen].append(chunk)
            
            st.info(f"Chunks organized by generation: {list(chunks_by_generation.keys())}")
            
            # Extract evolution data with actual DOFs from chunks
            self.evolution_data = []
            if 'elite_history' in history_doc and history_doc['elite_history']:
                st.info(f"Elite history has {len(history_doc['elite_history'])} generations")
                for gen_idx, elite_objectives in enumerate(history_doc['elite_history']):
                    generation_elites = []
                    
                    # Get chunks for this generation
                    gen_chunks = chunks_by_generation.get(gen_idx, [])
                    st.info(f"Generation {gen_idx}: {len(elite_objectives)} elites in history, {len(gen_chunks)} chunks")
                    
                    # Build a mapping from objectives to DOFs for this generation
                    obj_to_dof = {}
                    for chunk in gen_chunks:
                        if 'elites' in chunk:
                            for elite_dict in chunk['elites']:
                                obj_key = tuple(elite_dict['objectives'])
                                dofs = elite_dict['dofs']
                                if isinstance(dofs, list) and len(dofs) == 1:
                                    dofs = dofs[0]
                                obj_to_dof[obj_key] = dofs
                    
                    # Match objectives from history with DOFs from chunks
                    for obj_idx, objectives in enumerate(elite_objectives):
                        obj_key = tuple(objectives)
                        actual_dofs = obj_to_dof.get(obj_key, f"DOF not found for gen {gen_idx}")
                        
                        generation_elites.append({
                            'id': f"gen{gen_idx}_elite{obj_idx}",
                            'is_elite': True,
                            'objectives': np.array(objectives),
                            'dofs': actual_dofs,  # Actual DOF from chunks
                            'metrics': np.array(objectives),  # Use objectives as metrics
                            'constraints': [],
                            'generation': gen_idx
                        })
                    
                    self.evolution_data.append(generation_elites)
            else:
                st.warning("No elite_history found in history document")
            
            # For non-elites, use latest population objectives from history
            # Note: DOFs are only stored for elites, so non-elites won't have actual DOFs
            non_elite_data = []
            if 'objectives_history' in history_doc and history_doc['objectives_history']:
                latest_objectives = history_doc['objectives_history'][-1]
                st.info(f"Found {len(latest_objectives)} objectives in latest population")
                
                # Create non-elite entries (approximations since full population DOFs aren't stored)
                for i, obj in enumerate(latest_objectives):
                    # Skip if this might be an elite (rough approximation)
                    is_likely_elite = any(
                        np.allclose(obj, elite['objectives'], rtol=1e-5) 
                        for elite in elite_data
                    )
                    
                    if not is_likely_elite:
                        non_elite_data.append({
                            'id': len(elite_data) + len(non_elite_data),
                            'is_elite': False,
                            'objectives': np.array(obj),
                            'dofs': f"DOF not saved (non-elite from generation {current_generation-1})",
                            'metrics': np.array(obj),
                            'constraints': [],
                            'generation': current_generation - 1
                        })
            else:
                st.warning("No objectives_history found in history document")
            
            # Store data
            self.data = {
                'elites': elite_data,
                'non_elites': non_elite_data,
                'all_individuals': elite_data + non_elite_data
            }
            
            # Store labels (use defaults if not present in history)
            if elite_data:
                n_objectives = len(elite_data[0]['objectives'])
                default_obj_labels = [f'Objective {i+1}' for i in range(n_objectives)]
            else:
                default_obj_labels = ['Objective 1', 'Objective 2']
                
            self.objective_labels = history_doc.get('objective_labels', default_obj_labels)
            self.metric_labels = history_doc.get('metric_labels', self.objective_labels)
            
            st.success(f"✅ Loaded {len(elite_data)} elites and {len(non_elite_data)} non-elites from experiment '{experiment_id}'")
            st.info(f"Objective labels: {self.objective_labels}")
            
            return True
            
        except Exception as e:
            st.error(f"Error loading MongoDB data: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_ga_data(self, ga_instance):
        """Load data from a GeneticAlgorithmBase instance"""
        try:
            # Get current elites (Pareto front points)
            current_elites = [ind for ind in ga_instance.population if ind.is_elite]
            non_elites = [ind for ind in ga_instance.population if not ind.is_elite]
            
            # Extract data from individuals
            elite_data = []
            for i, ind in enumerate(current_elites):
                objectives = ind.objectives
                if torch.is_tensor(objectives):
                    objectives = objectives.cpu().numpy()
                
                dofs = ind.dofs
                if torch.is_tensor(dofs):
                    dofs = dofs.cpu().numpy()
                
                metrics = getattr(ind, 'metrics', objectives)
                if torch.is_tensor(metrics):
                    metrics = metrics.cpu().numpy()
                
                constraints = getattr(ind, 'constraints', [])
                
                elite_data.append({
                    'id': i,
                    'is_elite': True,
                    'objectives': objectives,
                    'dofs': dofs,
                    'metrics': metrics,
                    'constraints': constraints,
                    'individual': ind
                })
            
            # Extract data from non-elite population
            non_elite_data = []
            for i, ind in enumerate(non_elites):
                objectives = ind.objectives
                if torch.is_tensor(objectives):
                    objectives = objectives.cpu().numpy()
                
                dofs = ind.dofs
                if torch.is_tensor(dofs):
                    dofs = dofs.cpu().numpy()
                
                metrics = getattr(ind, 'metrics', objectives)
                if torch.is_tensor(metrics):
                    metrics = metrics.cpu().numpy()
                
                constraints = getattr(ind, 'constraints', [])
                
                non_elite_data.append({
                    'id': len(elite_data) + i,
                    'is_elite': False,
                    'objectives': objectives,
                    'dofs': dofs,
                    'metrics': metrics,
                    'constraints': constraints,
                    'individual': ind
                })
            
            self.data = {
                'elites': elite_data,
                'non_elites': non_elite_data,
                'all_individuals': elite_data + non_elite_data
            }
            
            # Store labels
            self.objective_labels = getattr(ga_instance, 'objective_labels', [f'Objective {i+1}' for i in range(len(elite_data[0]['objectives']) if elite_data else 2)])
            self.metric_labels = getattr(ga_instance, 'metric_labels', self.objective_labels)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading GA data: {str(e)}")
            return False
    
    def load_sample_data(self):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        n_points = 50
        n_objectives = 3
        n_dofs = 10
        
        # Generate Pareto-like front
        elite_data = []
        for i in range(20):
            # Generate objectives that form a rough Pareto front
            obj1 = np.random.uniform(0.1, 1.0)
            obj2 = np.random.uniform(0.1, 1.0 - obj1 + 0.3)
            obj3 = np.random.uniform(0.1, 2.0 - obj1 - obj2 + 0.5)
            
            objectives = np.array([obj1, obj2, obj3])
            dofs = np.random.uniform(-1, 1, n_dofs)
            metrics = objectives.copy()
            
            elite_data.append({
                'id': i,
                'is_elite': True,
                'objectives': objectives,
                'dofs': dofs,
                'metrics': metrics,
                'constraints': [],
            })
        
        # Generate non-elite population
        non_elite_data = []
        for i in range(30):
            objectives = np.random.uniform(0.5, 2.0, n_objectives)
            dofs = np.random.uniform(-1, 1, n_dofs)
            metrics = objectives.copy()
            
            non_elite_data.append({
                'id': 20 + i,
                'is_elite': False,
                'objectives': objectives,
                'dofs': dofs,
                'metrics': metrics,
                'constraints': [],
            })
        
        self.data = {
            'elites': elite_data,
            'non_elites': non_elite_data,
            'all_individuals': elite_data + non_elite_data
        }
        
        self.objective_labels = ['Objective 1', 'Objective 2', 'Objective 3']
        self.metric_labels = ['Metric 1', 'Metric 2', 'Metric 3']
        
        return True
    
    def create_pareto_plot(self, x_axis_idx: int, y_axis_idx: int, plot_type: str = 'objectives'):
        """Create interactive Pareto front evolution plot"""
        if not self.data:
            return None
        
        # Select data based on plot type  
        if plot_type == 'objectives':
            data_key = 'objectives'
            labels = self.objective_labels
        else:
            data_key = 'metrics'
            labels = self.metric_labels
        
        # Check if we have evolution data from MongoDB
        has_evolution_data = hasattr(self, 'evolution_data') and self.evolution_data
        
        # Create the plot
        fig = go.Figure()
        
        if has_evolution_data:
            # Plot evolution of Pareto fronts across generations (like GA's second subplot)
            max_generation = len(self.evolution_data) - 1
            
            for gen_idx, generation_elites in enumerate(self.evolution_data):
                if len(generation_elites) > 0:
                    # Get data for this generation
                    gen_x = [ind[data_key][x_axis_idx] for ind in generation_elites]
                    gen_y = [ind[data_key][y_axis_idx] for ind in generation_elites]
                    gen_ids = [ind['id'] for ind in generation_elites]
                    
                    # Color based on generation (similar to GA's viridis colormap)
                    color_intensity = gen_idx / max(max_generation, 1)
                    
                    # Add trace for this generation
                    fig.add_trace(go.Scatter(
                        x=gen_x,
                        y=gen_y,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=color_intensity,
                            colorscale='viridis',
                            opacity=0.7,
                            line=dict(width=1, color='black')
                        ),
                        name=f'Generation {gen_idx}',
                        customdata=[f"Gen{gen_idx}_ID{id}" for id in gen_ids],
                        hovertemplate=f'<b>Generation {gen_idx}</b><br>Individual %{{customdata}}<br>{labels[x_axis_idx]}: %{{x:.4f}}<br>{labels[y_axis_idx]}: %{{y:.4f}}<extra></extra>',
                        showlegend=False  # Too many generations for legend
                    ))
            
            title = f'Evolution of Pareto Front: {labels[x_axis_idx]} vs {labels[y_axis_idx]}'
            
        else:
            # Fallback: Current population view (original behavior)
            # Prepare data for plotting
            elite_x = [ind[data_key][x_axis_idx] for ind in self.data['elites']]
            elite_y = [ind[data_key][y_axis_idx] for ind in self.data['elites']]
            elite_ids = [ind['id'] for ind in self.data['elites']]
            
            non_elite_x = [ind[data_key][x_axis_idx] for ind in self.data['non_elites']]
            non_elite_y = [ind[data_key][y_axis_idx] for ind in self.data['non_elites']]
            non_elite_ids = [ind['id'] for ind in self.data['non_elites']]
            
            # Add non-elite points
            fig.add_trace(go.Scatter(
                x=non_elite_x,
                y=non_elite_y,
                mode='markers',
                marker=dict(
                    size=8,
                    color='lightgray',
                    opacity=0.6,
                    line=dict(width=1, color='gray')
                ),
                name=f'Population (n={len(non_elite_x)})',
                customdata=non_elite_ids,
                hovertemplate=f'<b>Individual %{{customdata}}</b><br>{labels[x_axis_idx]}: %{{x:.4f}}<br>{labels[y_axis_idx]}: %{{y:.4f}}<extra></extra>'
            ))
            
            # Add elite points (Pareto front)
            fig.add_trace(go.Scatter(
                x=elite_x,
                y=elite_y,
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    opacity=0.8,
                    line=dict(width=2, color='darkred')
                ),
                name=f'Pareto Front (n={len(elite_x)})',
                customdata=elite_ids,
                hovertemplate=f'<b>Individual %{{customdata}}</b><br>{labels[x_axis_idx]}: %{{x:.4f}}<br>{labels[y_axis_idx]}: %{{y:.4f}}<extra></extra>'
            ))
            
            title = f'{plot_type.title()} Space: {labels[x_axis_idx]} vs {labels[y_axis_idx]}'
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=labels[x_axis_idx],
            yaxis_title=labels[y_axis_idx],
            hovermode='closest',
            height=500,
            showlegend=True if not has_evolution_data else False,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add colorbar for evolution plot
        if has_evolution_data:
            fig.update_layout(
                coloraxis=dict(
                    colorscale='viridis',
                    colorbar=dict(
                        title=dict(text="Generation", side="right")
                    )
                )
            )
        
        return fig
    
    def display_individual_details(self, individual_id: int):
        """Display detailed information about a selected individual"""
        if not self.data or individual_id is None:
            return
        
        # Find the individual
        individual = None
        for ind in self.data['all_individuals']:
            if ind['id'] == individual_id:
                individual = ind
                break
        
        if not individual:
            st.error("Individual not found")
            return
        
        st.subheader(f"Individual {individual_id} Details")
        
        # Elite status and generation info
        status = "🏆 Pareto Front Member" if individual['is_elite'] else "👥 Population Member"
        generation = individual.get('generation', 'Unknown')
        st.markdown(f"**Status:** {status}")
        st.markdown(f"**Generation:** {generation}")
        
        # MAIN FOCUS: LLM Prompt (DOF)
        st.subheader("📝 Optimized Prompt")
        dofs = individual['dofs']
        
        if isinstance(dofs, str) and not dofs.startswith("DOF not") and not dofs.startswith("No prompt"):
            # This is an actual LLM prompt - show it prominently
            st.text_area(
                "Prompt Text:",
                value=dofs,
                height=300,  # Larger height for main focus
                disabled=True,
                help="This is the LLM prompt that was optimized by the genetic algorithm"
            )
            
            # Prompt statistics in a compact format
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", len(dofs))
            with col2:
                st.metric("Words", len(dofs.split()))
            with col3:
                st.metric("Lines", len(dofs.split('\n')))
        
        elif isinstance(dofs, str):
            # Placeholder message
            st.info(dofs)
        
        else:
            # Legacy numeric DOFs - show compact version
            st.info("Numeric DOF data available (legacy format)")
        
        # SECONDARY INFO: Objectives - shown compactly
        st.subheader("📊 Performance Metrics")
        
        # Show objectives in a compact table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Objectives:**")
            obj_df = pd.DataFrame({
                'Metric': self.objective_labels,
                'Value': [f"{val:.4f}" if not np.isinf(val) and not np.isnan(val) else str(val) 
                         for val in individual['objectives']]
            })
            st.dataframe(obj_df, hide_index=True, height=150)
        
        with col2:
            # Show metrics only if different from objectives
            if not np.array_equal(individual['metrics'], individual['objectives']):
                st.markdown("**Additional Metrics:**")
                metrics_df = pd.DataFrame({
                    'Metric': self.metric_labels,
                    'Value': [f"{val:.4f}" if not np.isinf(val) and not np.isnan(val) else str(val) 
                             for val in individual['metrics']]
                })
                st.dataframe(metrics_df, hide_index=True, height=150)
            else:
                st.markdown("**Performance Summary:**")
                # Show a simple performance indicator
                obj_vals = individual['objectives']
                if len(obj_vals) >= 2:
                    # For RedBlue problem: obj[0] is split deviation, obj[1] is neither fraction
                    if not np.isinf(obj_vals[0]) and not np.isnan(obj_vals[0]):
                        if obj_vals[0] == 0.0 and obj_vals[1] == 0.0:
                            st.success("🎯 Perfect solution! (50-50 split, all classified)")
                        elif obj_vals[0] <= 0.5:
                            st.info("✅ Good solution (close to 50-50 split)")
                        else:
                            st.warning("⚠️ Suboptimal solution (far from 50-50 split)")
                    else:
                        st.error("❌ Failed simulation")
        
        # Constraints (if any) - very compact
        if individual['constraints']:
            with st.expander("⚠️ Constraints", expanded=False):
                const_df = pd.DataFrame({
                    'Constraint': [f'Constraint_{i+1}' for i in range(len(individual['constraints']))],
                    'Value': individual['constraints']
                })
                st.dataframe(const_df, hide_index=True)

    def load_json_state_data(self, json_file) -> bool:
        """Load data from JSON state saver format"""
        try:
            # Read the JSON file
            if hasattr(json_file, 'read'):
                # It's a file-like object (from streamlit file uploader)
                data = json.load(json_file)
            else:
                # It's a file path
                with open(json_file, 'r') as f:
                    data = json.load(f)
            
            # Validate the JSON structure
            if 'generations' not in data:
                st.error("Invalid JSON format: missing 'generations' key")
                return False
            
            generations = data['generations']
            if not generations:
                st.error("No generation data found in JSON file")
                return False
            
            st.info(f"Found {len(generations)} generations in JSON file")
            
            # Extract all elite individuals from all generations
            all_elites = []
            elite_id = 0
            
            for gen_data in generations:
                generation = gen_data['generation']
                individuals = gen_data.get('individuals', [])
                
                # Filter only elite individuals
                gen_elites = [ind for ind in individuals if ind.get('is_elite', False)]
                st.info(f"Generation {generation}: {len(gen_elites)} elites out of {len(individuals)} individuals")
                
                for individual in gen_elites:
                    # Handle objectives with special values
                    objectives = individual.get('objectives', [0, 0])
                    
                    # Convert string representations back to float values for plotting
                    def convert_special_values(obj_list):
                        converted = []
                        for val in obj_list:
                            if val == "Infinity":
                                converted.append(float('inf'))
                            elif val == "-Infinity":
                                converted.append(float('-inf'))
                            elif val == "NaN":
                                converted.append(float('nan'))
                            else:
                                converted.append(float(val))
                        return converted
                    
                    objectives = convert_special_values(objectives)
                    metrics = convert_special_values(individual.get('metrics', objectives))
                    constraints = individual.get('constraints', [])
                    
                    elite_data = {
                        'id': elite_id,
                        'is_elite': True,
                        'objectives': np.array(objectives),
                        'dofs': individual.get('prompt', 'No prompt available'),  # Use prompt as DOF
                        'metrics': np.array(metrics),
                        'constraints': constraints,
                        'generation': generation,
                        'individual_id': individual.get('individual_id', elite_id)
                    }
                    all_elites.append(elite_data)
                    elite_id += 1
            
            # Store the data (only elites, no non-elites for JSON format)
            self.data = {
                'elites': all_elites,
                'non_elites': [],  # Empty for JSON format since we only show elites
                'all_individuals': all_elites
            }
            
            # Set up evolution data for plotting
            self.evolution_data = []
            for gen_data in generations:
                generation = gen_data['generation']
                individuals = gen_data.get('individuals', [])
                gen_elites = [ind for ind in individuals if ind.get('is_elite', False)]
                
                generation_elites = []
                for individual in gen_elites:
                    objectives = convert_special_values(individual.get('objectives', [0, 0]))
                    metrics = convert_special_values(individual.get('metrics', objectives))
                    
                    generation_elites.append({
                        'id': f"gen{generation}_elite{individual.get('individual_id', 0)}",
                        'is_elite': True,
                        'objectives': np.array(objectives),
                        'dofs': individual.get('prompt', 'No prompt available'),
                        'metrics': np.array(metrics),
                        'constraints': individual.get('constraints', []),
                        'generation': generation
                    })
                
                self.evolution_data.append(generation_elites)
            
            # Extract metadata for labels
            metadata = data.get('metadata', {})
            
            # Determine number of objectives from first elite
            if all_elites:
                n_objectives = len(all_elites[0]['objectives'])
                # Use default labels for now (could be enhanced later)
                default_labels = [f'Objective {i+1}' for i in range(n_objectives)]
            else:
                default_labels = ['Objective 1', 'Objective 2']
            
            self.objective_labels = default_labels
            self.metric_labels = default_labels
            
            st.success(f"✅ Loaded {len(all_elites)} elite individuals from {len(generations)} generations")
            st.info(f"Metadata: {metadata}")
            
            return True
            
        except Exception as e:
            st.error(f"Error loading JSON data: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return False

    def plot_pareto_front(self):
        """Create interactive Pareto front visualization showing all elites across generations"""
        if not self.data:
            st.warning("No data available. Please load data first.")
            return None
        
        all_individuals = self.data['all_individuals']
        
        if not all_individuals:
            st.warning("No elite individuals found in the dataset.")
            return None
        
        # For JSON data, we only have elites from all generations
        # Filter to only elites (though for JSON format, all should be elites)
        elites = [ind for ind in all_individuals if ind.get('is_elite', False)]
        
        if not elites:
            st.warning("No elite individuals found.")
            return None
        
        # Prepare data for plotting
        plot_data = []
        for individual in elites:
            objectives = individual['objectives']
            generation = individual.get('generation', 0)
            
            # Skip infinite or NaN values for plotting
            if np.any(np.isinf(objectives)) or np.any(np.isnan(objectives)):
                continue
                
            plot_data.append({
                'id': individual['id'],
                'obj1': objectives[0],
                'obj2': objectives[1] if len(objectives) > 1 else 0,
                'generation': generation,
                'prompt': individual['dofs'][:100] + "..." if len(str(individual['dofs'])) > 100 else str(individual['dofs'])
            })
        
        if not plot_data:
            st.warning("No valid elite individuals to plot (all have infinite/NaN objectives).")
            return None
        
        df = pd.DataFrame(plot_data)
        
        # Create generation color mapping
        unique_generations = sorted(df['generation'].unique())
        n_generations = len(unique_generations)
        
        # Use a color palette that works well for multiple generations
        if n_generations <= 10:
            colors = px.colors.qualitative.Set3[:n_generations]
        else:
            # For many generations, use a continuous color scale
            colors = px.colors.sample_colorscale('viridis', n_generations)
        
        generation_colors = {gen: colors[i] for i, gen in enumerate(unique_generations)}
        
        # Add color column to dataframe
        df['color'] = df['generation'].map(generation_colors)
        
        # Create the scatter plot
        fig = px.scatter(
            df,
            x='obj1',
            y='obj2',
            color='generation',
            hover_data=['id', 'prompt'],
            title="Pareto Front Evolution - All Elite Solutions",
            labels={
                'obj1': self.objective_labels[0] if len(self.objective_labels) > 0 else 'Objective 1',
                'obj2': self.objective_labels[1] if len(self.objective_labels) > 1 else 'Objective 2',
                'generation': 'Generation'
            },
            color_continuous_scale='viridis' if n_generations > 10 else None
        )
        
        # Update layout for better appearance
        fig.update_traces(
            marker=dict(
                size=12,
                opacity=0.8,
                line=dict(width=1, color='white')
            )
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(
                title="Generation",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            margin=dict(r=150)  # Make room for legend
        )
        
        # Add custom hover template
        fig.update_traces(
            hovertemplate="<b>Individual %{customdata[0]}</b><br>" +
                         f"{self.objective_labels[0] if len(self.objective_labels) > 0 else 'Objective 1'}: %{{x:.4f}}<br>" +
                         f"{self.objective_labels[1] if len(self.objective_labels) > 1 else 'Objective 2'}: %{{y:.4f}}<br>" +
                         "Generation: %{marker.color}<br>" +
                         "Prompt: %{customdata[1]}<extra></extra>",
            customdata=df[['id', 'prompt']].values
        )
        
        return fig

def main():
    st.title("🎯 Interactive Pareto Front Explorer")
    st.markdown("Explore Pareto optimal solutions interactively. Click on points to see detailed information.")
    
    app = InteractiveParetoApp()
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Controls")
    
    # Data loading options
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Load from JSON", "Load from MongoDB", "Sample Data", "Load from GA Instance"],
        help="Choose data source for the Pareto front visualization"
    )
    
    if data_source == "Load from JSON":
        st.sidebar.subheader("📁 JSON File Upload")
        json_file = st.sidebar.file_uploader(
            "Upload JSON state file",
            type="json",
            help="Upload a JSON file generated by the RedBlueStateSaver (e.g., *_generation_state.json)"
        )
        
        if json_file:
            if app.load_json_state_data(json_file):
                st.sidebar.success("✅ Data loaded successfully from JSON file!")
            else:
                st.sidebar.error("❌ Failed to load data from JSON file")
        else:
            st.sidebar.info("Please upload a JSON file to visualize the Pareto front evolution")
    
    elif data_source == "Sample Data":
        if st.sidebar.button("Load Sample Data"):
            app.load_sample_data()
            st.sidebar.success("Sample data loaded!")
    
    elif data_source == "Load from MongoDB":
        st.sidebar.subheader("🗄️ MongoDB Connection")
        
        # MongoDB connection inputs
        mongo_uri = st.sidebar.text_input(
            "MongoDB URI",
            value="mongodb://localhost:27017/",
            help="MongoDB connection string"
        )
        
        db_name = st.sidebar.text_input(
            "Database Name",
            value="genetic_algorithm",
            help="Database name containing GA data"
        )
        
        # Auto-connect on first load
        if 'mongodb_connected' not in st.session_state:
            if app.connect_mongodb(mongo_uri, db_name):
                st.sidebar.success("Connected to MongoDB!")
                st.session_state.mongodb_connected = True
            else:
                st.sidebar.error("Failed to connect to MongoDB. Using sample data instead.")
                app.load_sample_data()
                st.session_state.mongodb_connected = False
        
        if st.sidebar.button("Connect to MongoDB"):
            if app.connect_mongodb(mongo_uri, db_name):
                st.sidebar.success("Connected to MongoDB!")
                st.session_state.mongodb_connected = True
            else:
                st.sidebar.error("Failed to connect to MongoDB")
                st.session_state.mongodb_connected = False
        
        # List experiments if connected
        if app.db is not None:
            experiments = app.list_experiments()
            
            # Always show manual experiment input option
            st.sidebar.subheader("🔍 Manual Experiment Load")
            manual_exp_id = st.sidebar.text_input(
                "Enter Experiment ID:",
                value="",
                help="Enter the exact experiment ID to load manually"
            )
            
            if st.sidebar.button("Load Manual Experiment"):
                if manual_exp_id.strip():
                    if app.load_mongodb_data(manual_exp_id.strip()):
                        st.sidebar.success(f"Loaded data for experiment: {manual_exp_id}")
                        st.session_state.auto_loaded = True
                    else:
                        st.sidebar.error(f"Failed to load experiment: {manual_exp_id}")
                else:
                    st.sidebar.error("Please enter an experiment ID")
            
            if experiments:
                st.sidebar.subheader("📊 Available Experiments")
                st.sidebar.info(f"Found {len(experiments)} experiments in database")
                
                experiment_options = [
                    f"{exp['experiment_id']} (Gen: {exp['current_generation']})"
                    for exp in experiments
                ]
                
                selected_exp = st.sidebar.selectbox(
                    "Select Experiment",
                    experiment_options,
                    help="Choose an experiment to load"
                )
                
                # Auto-load the first experiment if none loaded yet
                if not app.data and 'auto_loaded' not in st.session_state:
                    exp_id = selected_exp.split(" (Gen:")[0]
                    if app.load_mongodb_data(exp_id):
                        st.sidebar.success(f"Auto-loaded data for experiment: {exp_id}")
                        st.session_state.auto_loaded = True
                    else:
                        st.sidebar.error("Failed to auto-load experiment data")
                
                if st.sidebar.button("Load Selected Experiment"):
                    # Extract experiment ID from selection
                    exp_id = selected_exp.split(" (Gen:")[0]
                    if app.load_mongodb_data(exp_id):
                        st.sidebar.success(f"Loaded data for experiment: {exp_id}")
                        st.session_state.auto_loaded = True
                    else:
                        st.sidebar.error("Failed to load experiment data")
            else:
                st.sidebar.info("No experiments found in database")
                # Always load sample data when no experiments found
                if not app.data:
                    st.sidebar.info("Loading sample data instead...")
                    if app.load_sample_data():
                        st.sidebar.success("Sample data loaded successfully!")
                    else:
                        st.sidebar.error("Failed to load sample data")
        else:
            # MongoDB connection failed, load sample data
            if not app.data:
                st.sidebar.info("MongoDB not connected. Loading sample data...")
                if app.load_sample_data():
                    st.sidebar.success("Sample data loaded successfully!")
                else:
                    st.sidebar.error("Failed to load sample data")
    
    else:  # Load from GA Instance
        st.sidebar.markdown("**Note:** To load from GA instance, use the `load_ga_data(ga_instance)` method")
        if st.sidebar.button("Load Sample Data for Demo"):
            app.load_sample_data()
            st.sidebar.success("Sample data loaded!")
    
    # Fallback to sample data only if no other data source worked
    if not app.data and data_source != "Load from MongoDB":
        app.load_sample_data()
    
    if app.data:
        # Plot type selection
        plot_type = st.sidebar.selectbox(
            "Plot Type",
            ["objectives", "metrics"],
            help="Choose between objectives or metrics for plotting"
        )
        
        # Axis selection
        labels = app.objective_labels if plot_type == 'objectives' else app.metric_labels
        
        x_axis = st.sidebar.selectbox(
            "X-Axis",
            labels,
            index=0,
            help="Select objective/metric for X-axis"
        )
        
        y_axis = st.sidebar.selectbox(
            "Y-Axis", 
            labels,
            index=1 if len(labels) > 1 else 0,
            help="Select objective/metric for Y-axis"
        )
        
        x_axis_idx = labels.index(x_axis)
        y_axis_idx = labels.index(y_axis)
        
        # Statistics
        st.sidebar.subheader("📊 Statistics")
        st.sidebar.metric("Pareto Front Size", len(app.data['elites']))
        st.sidebar.metric("Population Size", len(app.data['non_elites']))
        st.sidebar.metric("Total Individuals", len(app.data['all_individuals']))
        
        # Main layout
        col1, col2 = st.columns([1.2, 0.8])
        
        with col1:
            # Check if we have evolution data to show
            has_evolution_data = hasattr(app, 'evolution_data') and app.evolution_data
            
            if has_evolution_data:
                st.subheader("🎯 Pareto Front Evolution")
                st.info(f"Showing evolution across {len(app.evolution_data)} generations. Color indicates generation (dark=early, bright=recent).")
            else:
                st.subheader("🎯 Pareto Front Visualization")
            
            # Create and display the plot
            fig = app.create_pareto_plot(x_axis_idx, y_axis_idx, plot_type)
            
            if fig:
                # Display the plot and capture click events
                selected_points = st.plotly_chart(
                    fig, 
                    use_container_width=True,
                    key="pareto_plot",
                    on_select="rerun",
                    selection_mode="points"
                )
                
                # Handle point selection
                if selected_points and selected_points.selection and selected_points.selection.points:
                    # Get the first selected point
                    point = selected_points.selection.points[0]
                    if hasattr(point, 'customdata'):
                        individual_id = point.customdata
                        app.selected_point = individual_id
        
        with col2:
            st.subheader("🔍 Individual Details")
            
            # Manual individual selection
            individual_ids = [ind['id'] for ind in app.data['all_individuals']]
            selected_id = st.selectbox(
                "Select Individual:",
                individual_ids,
                index=0,
                help="Select an individual to view details"
            )
            
            # Display details for selected individual
            if selected_id is not None:
                app.display_individual_details(selected_id)
            elif app.selected_point is not None:
                app.display_individual_details(app.selected_point)
            else:
                st.info("Select an individual from the dropdown or click on a point in the plot.")
    
    else:
        st.warning("No data loaded. Please check MongoDB connection or load sample data.")

# Function to integrate with existing GA code
def launch_interactive_pareto_explorer(ga_instance=None):
    """
    Launch the interactive Pareto front explorer.
    
    Args:
        ga_instance: GeneticAlgorithmBase instance (optional)
    """
    app = InteractiveParetoApp()
    
    if ga_instance:
        if app.load_ga_data(ga_instance):
            st.success("GA data loaded successfully!")
        else:
            st.error("Failed to load GA data. Using sample data instead.")
            app.load_sample_data()
    else:
        app.load_sample_data()
    
    return app

if __name__ == "__main__":
    import sys
    import subprocess
    import os
    
    # Check if running in Streamlit context by looking for Streamlit-specific environment
    # This avoids calling Streamlit functions that cause warnings
    is_streamlit_context = (
        'streamlit' in sys.modules or 
        os.environ.get('STREAMLIT_SERVER_PORT') is not None or
        any('streamlit' in arg.lower() for arg in sys.argv)
    )
    
    if is_streamlit_context:
        # We're in Streamlit context, run the main app
        main()
    else:
        # We're not in Streamlit context, so launch with streamlit run
        print("🎯 Interactive Pareto Front Explorer")
        print("=" * 50)
        print("This is a Streamlit application that needs to be run with:")
        print(f"  streamlit run {sys.argv[0]}")
        print("\nStarting Streamlit server automatically...")
        print("=" * 50)
        
        try:
            # Try to run with streamlit
            subprocess.run([sys.executable, "-m", "streamlit", "run"] + sys.argv)
        except KeyboardInterrupt:
            print("\nStreamlit server stopped.")
        except Exception as e:
            print(f"\nError starting Streamlit: {e}")
            print(f"Please run manually: streamlit run {sys.argv[0]}") 