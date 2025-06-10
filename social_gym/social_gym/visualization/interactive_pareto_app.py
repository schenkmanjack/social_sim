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
            # Test connection
            self.db.admin.command('ping')
            return True
        except Exception as e:
            st.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
    
    def list_experiments(self) -> List[Dict]:
        """List available experiments from MongoDB"""
        if not self.db:
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
        if not self.db:
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
            
            # Load elite chunks (matches GA's _save_state structure)
            elite_chunks = list(self.db.experiments.find({
                "experiment_id": experiment_id,
                "type": "chunk"
            }).sort("chunk_index", 1))
            
            if not elite_chunks:
                st.error(f"No elite chunk data found for experiment: {experiment_id}")
                return False
            
            # Reconstruct elite individuals from chunks (current generation)
            elite_data = []
            elite_id = 0
            
            for chunk in elite_chunks:
                if 'elites' in chunk:
                    for elite_dict in chunk['elites']:
                        elite_data.append({
                            'id': elite_id,
                            'is_elite': True,
                            'objectives': np.array(elite_dict['objectives']),
                            'dofs': np.array(elite_dict['dofs']),
                            'metrics': np.array(elite_dict.get('metrics', elite_dict['objectives'])),
                            'constraints': elite_dict.get('constraints', []),
                        })
                        elite_id += 1
            
            # Extract evolution data from elite_history
            self.evolution_data = []
            if 'elite_history' in history_doc and history_doc['elite_history']:
                for gen_idx, elite_objectives in enumerate(history_doc['elite_history']):
                    generation_elites = []
                    for obj_idx, objectives in enumerate(elite_objectives):
                        generation_elites.append({
                            'id': f"gen{gen_idx}_elite{obj_idx}",
                            'is_elite': True,
                            'objectives': np.array(objectives),
                            'dofs': np.zeros(len(elite_data[0]['dofs']) if elite_data else 10),  # Placeholder
                            'metrics': np.array(objectives),  # Use objectives as metrics
                            'constraints': [],
                        })
                    self.evolution_data.append(generation_elites)
            
            # For non-elites, use the latest population objectives from history
            non_elite_data = []
            if 'objectives_history' in history_doc and history_doc['objectives_history']:
                latest_objectives = history_doc['objectives_history'][-1]
                
                # Create non-elite entries (approximations since full population DOFs aren't stored)
                for i, obj in enumerate(latest_objectives):
                    # Skip if this might be an elite (rough approximation)
                    is_likely_elite = any(
                        np.allclose(obj, elite['objectives'], rtol=1e-5) 
                        for elite in elite_data
                    )
                    
                    if not is_likely_elite:
                        # Create placeholder DOFs for non-elites (since GA doesn't save full population DOFs)
                        placeholder_dofs = np.zeros(len(elite_data[0]['dofs']) if elite_data else 10)
                        
                        non_elite_data.append({
                            'id': len(elite_data) + len(non_elite_data),
                            'is_elite': False,
                            'objectives': np.array(obj),
                            'dofs': placeholder_dofs,
                            'metrics': np.array(obj),
                            'constraints': [],
                        })
            
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
            
            return True
            
        except Exception as e:
            st.error(f"Error loading MongoDB data: {str(e)}")
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
                        title="Generation",
                        titleside="right"
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
        
        # Elite status
        status = "🏆 Pareto Front Member" if individual['is_elite'] else "👥 Population Member"
        st.markdown(f"**Status:** {status}")
        
        # Objectives
        st.subheader("📊 Objectives")
        obj_df = pd.DataFrame({
            'Objective': self.objective_labels,
            'Value': individual['objectives']
        })
        st.dataframe(obj_df, hide_index=True)
        
        # Metrics (if different from objectives)
        if not np.array_equal(individual['metrics'], individual['objectives']):
            st.subheader("📈 Metrics")
            metrics_df = pd.DataFrame({
                'Metric': self.metric_labels,
                'Value': individual['metrics']
            })
            st.dataframe(metrics_df, hide_index=True)
        
        # Design of Experiments (DOFs)
        st.subheader("🔧 Design Variables (DOFs)")
        n_dofs = len(individual['dofs'])
        dof_names = [f'DOF_{i+1}' for i in range(n_dofs)]
        
        # Check if DOFs are placeholder zeros (for non-elites from MongoDB)
        if individual['is_elite'] or not np.allclose(individual['dofs'], 0):
            # Display actual DOFs
            dof_df = pd.DataFrame({
                'Variable': dof_names,
                'Value': individual['dofs']
            })
            
            # Show DOFs in columns if there are many
            if n_dofs > 10:
                col1, col2 = st.columns(2)
                mid_point = n_dofs // 2
                with col1:
                    st.dataframe(dof_df.iloc[:mid_point], hide_index=True)
                with col2:
                    st.dataframe(dof_df.iloc[mid_point:], hide_index=True)
            else:
                st.dataframe(dof_df, hide_index=True)
            
            # DOF Statistics
            st.subheader("📊 DOF Statistics")
            dof_stats = pd.DataFrame({
                'Statistic': ['Mean', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    np.mean(individual['dofs']),
                    np.std(individual['dofs']),
                    np.min(individual['dofs']),
                    np.max(individual['dofs'])
                ]
            })
            st.dataframe(dof_stats, hide_index=True)
            
            # DOF Visualization
            st.subheader("📊 DOF Distribution")
            fig_dof = px.bar(
                x=dof_names,
                y=individual['dofs'],
                title="Design Variables Values",
                labels={'x': 'DOF', 'y': 'Value'}
            )
            fig_dof.update_layout(height=300)
            st.plotly_chart(fig_dof, use_container_width=True)
        else:
            # DOFs are placeholder (non-elite from MongoDB)
            st.info("DOF details are only available for Pareto front members when loading from MongoDB.")
        
        # Constraints (if any)
        if individual['constraints']:
            st.subheader("⚠️ Constraints")
            const_df = pd.DataFrame({
                'Constraint': [f'Constraint_{i+1}' for i in range(len(individual['constraints']))],
                'Value': individual['constraints']
            })
            st.dataframe(const_df, hide_index=True)

def main():
    st.title("🎯 Interactive Pareto Front Explorer")
    st.markdown("Explore Pareto optimal solutions interactively. Click on points to see detailed information.")
    
    app = InteractiveParetoApp()
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Controls")
    
    # Data loading options
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Load from MongoDB", "Sample Data", "Load from GA Instance"],
        help="Choose data source for the Pareto front visualization"
    )
    
    if data_source == "Sample Data":
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
        if app.db:
            experiments = app.list_experiments()
            if experiments:
                st.sidebar.subheader("📊 Available Experiments")
                
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
                
                if st.sidebar.button("Load Experiment Data"):
                    # Extract experiment ID from selection
                    exp_id = selected_exp.split(" (Gen:")[0]
                    if app.load_mongodb_data(exp_id):
                        st.sidebar.success(f"Loaded data for experiment: {exp_id}")
                        st.session_state.auto_loaded = True
                    else:
                        st.sidebar.error("Failed to load experiment data")
            else:
                st.sidebar.info("No experiments found in database")
                if not app.data:
                    st.sidebar.info("Loading sample data instead...")
                    app.load_sample_data()
    
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
    main() 