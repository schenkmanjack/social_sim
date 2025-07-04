import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import json
import math
import plotly.colors as pc

st.set_page_config(
    page_title="Pareto Front Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("🎯 Pareto Front Explorer")
st.write("Upload your JSON state file to visualize elite solutions across generations")

# Initialize session state for selections
if 'selected_generation' not in st.session_state:
    st.session_state.selected_generation = None
if 'selected_individual_id' not in st.session_state:
    st.session_state.selected_individual_id = None

# Function to create network graph for a duplicate
def create_network_graph(connectivity, agent_colors, duplicate_idx, title):
    """Create a network graph using the connectivity and agent colors"""
    if not connectivity:
        return go.Figure().add_annotation(
            text="No connectivity data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Create networkx graph
    G = nx.Graph()
    
    # Determine total number of agents from agent_colors
    all_agents = set()
    for color_list in agent_colors.values():
        for agent_name in color_list:
            if agent_name.startswith('agent_'):
                agent_num = int(agent_name.split('_')[1])
                all_agents.add(agent_num)
    
    # Also check connectivity for any additional agents
    for edge in connectivity:
        if len(edge) == 2:
            all_agents.add(edge[0])
            all_agents.add(edge[1])
    
    # Add all nodes (agents)
    for agent_num in all_agents:
        G.add_node(agent_num)
    
    # Add edges from connectivity
    for edge in connectivity:
        if len(edge) == 2:
            G.add_edge(edge[0], edge[1])
    
    # Use circular layout for consistent positioning across all graphs
    # This ensures the same agent is always in the same position for easy comparison
    
    # Create consistent circular layout
    pos = {}
    sorted_nodes = sorted(G.nodes())
    n_nodes = len(sorted_nodes)
    
    for i, node in enumerate(sorted_nodes):
        # Position nodes in a circle, starting from top (90 degrees) and going clockwise
        angle = (2 * math.pi * i / n_nodes) - (math.pi / 2)  # Start from top
        pos[node] = (math.cos(angle), math.sin(angle))
    
    # Extract node positions
    node_x = [pos[node][0] for node in sorted(G.nodes())]
    node_y = [pos[node][1] for node in sorted(G.nodes())]
    
    # Determine node colors based on agent colors
    node_colors = []
    node_text = []
    
    for node in sorted(G.nodes()):
        agent_key = f"agent_{node}"
        color = 'lightgray'  # default
        
        # Check which color this agent is in
        if agent_key in agent_colors.get('red', []):
            color = 'red'
        elif agent_key in agent_colors.get('blue', []):
            color = 'blue'
        elif agent_key in agent_colors.get('undefined', []):
            color = 'lightgray'
        
        node_colors.append(color)
        node_text.append(f"Agent {node}")
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='lightgray'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=20,
            color=node_colors,
            line=dict(width=2, color='black')
        ),
        text=[f"{i}" for i in sorted(G.nodes())],
        textposition="middle center",
        textfont=dict(color="white", size=12, family="Arial Black"),
        hovertemplate='<b>Agent %{text}</b><br>' +
                      'Color: %{marker.color}<br>' +
                      '<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='white'
    )
    
    return fig

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload JSON state file",
        type="json",
        help="Upload a JSON file generated by RedBlueStateSaver"
    )

if uploaded_file is not None:
    try:
        # Load JSON data
        data = json.load(uploaded_file)
        
        if 'generations' not in data:
            st.error("Invalid JSON format: missing 'generations' key")
            st.stop()
        
        generations = data['generations']
        st.sidebar.success(f"✅ Loaded {len(generations)} generations")
        
        # Extract all elites from all generations
        all_elites = []
        generation_stats = {}
        filtered_count = 0  # Track individuals filtered out due to high objective values
        
        for gen_data in generations:
            generation = gen_data['generation']
            individuals = gen_data.get('individuals', [])
            
            # Filter only elite individuals
            gen_elites = [ind for ind in individuals if ind.get('is_elite', False)]
            generation_stats[generation] = {
                'total': len(individuals),
                'elites': len(gen_elites)
            }
            
            for individual in gen_elites:
                objectives = individual.get('objectives', [0, 0])
                
                # Convert special values back to float
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
                
                # Skip individuals with any objective >= 10000 (likely failed evaluations)
                if any(obj >= 10000 for obj in objectives if not np.isnan(obj)):
                    filtered_count += 1
                    continue
                
                # Skip infinite/NaN values for plotting
                if not (np.any(np.isinf(objectives)) or np.any(np.isnan(objectives))):
                    # Extract prompt safely - handle both string and dictionary formats
                    raw_prompt = individual.get('prompt', 'No prompt available')
                    
                    # Handle different prompt formats
                    if isinstance(raw_prompt, dict):
                        # Dictionary format: {"prompt": "actual text", "connectivity": [...]}
                        prompt = raw_prompt.get("prompt", "No prompt available")
                    elif isinstance(raw_prompt, str):
                        # String format: "actual prompt text"
                        prompt = raw_prompt
                    else:
                        # Fallback
                        prompt = str(raw_prompt) if raw_prompt is not None else 'No prompt available'
                    
                    # Safely handle prompt for preview - ensure it's a string
                    if prompt is None:
                        prompt = 'No prompt available'
                        prompt_preview = prompt
                    else:
                        prompt_str = str(prompt)
                        prompt_preview = (prompt_str[:100] + "...") if len(prompt_str) > 100 else prompt_str
                    
                    elite_entry = {
                        'generation': generation,
                        'individual_id': individual.get('individual_id', 0),
                        'prompt': prompt,
                        'prompt_preview': prompt_preview,
                        'objectives': objectives
                    }
                    
                    # Add individual objective columns
                    for j in range(len(objectives)):
                        elite_entry[f'obj{j+1}'] = objectives[j]
                    
                    all_elites.append(elite_entry)
        
        if not all_elites:
            st.warning("No valid elite solutions found for plotting")
            st.stop()
        
        # Create DataFrame
        df = pd.DataFrame(all_elites)
        
        # Ensure generation column is properly typed for color mapping
        df['generation'] = pd.to_numeric(df['generation'])
        
        # Debug: Check generation distribution
        if len(all_elites) > 0:
            generation_counts = df['generation'].value_counts().sort_index()
            st.sidebar.write("**Debug: Elites per generation:**")
            for gen, count in generation_counts.items():
                st.sidebar.write(f"Gen {gen}: {count} elites")
            
            # Debug: Check generation range for color mapping
            min_gen = df['generation'].min()
            max_gen = df['generation'].max()
            unique_gens = sorted(df['generation'].unique())
            st.sidebar.write(f"**Generation range:** {min_gen} to {max_gen}")
            st.sidebar.write(f"**Unique generations:** {unique_gens}")
            st.sidebar.write(f"**Generation data type:** {df['generation'].dtype}")
            
            # Debug: Check objective value ranges per generation
            with st.sidebar.expander("Debug: Objective Ranges by Generation"):
                for gen in unique_gens:
                    gen_data = df[df['generation'] == gen]
                    if len(gen_data) > 0:
                        obj1_range = f"{gen_data['obj1'].min():.3f} to {gen_data['obj1'].max():.3f}"
                        obj2_range = f"{gen_data['obj2'].min():.3f} to {gen_data['obj2'].max():.3f}" if 'obj2' in gen_data.columns else "N/A"
                        st.write(f"**Gen {gen}** ({len(gen_data)} elites):")
                        st.write(f"  Obj1: {obj1_range}")
                        st.write(f"  Obj2: {obj2_range}")
            
            # Sample some rows for debugging
            sample_rows = df[['generation', 'individual_id']].head(10)
            with st.sidebar.expander("Debug: Sample Data"):
                st.dataframe(sample_rows)
        
        # Sidebar stats
        with st.sidebar:
            st.header("📊 Statistics")
            st.metric("Total Elites", len(all_elites))
            st.metric("Generations", len(generation_stats))
            if filtered_count > 0:
                st.metric("Filtered Out (≥10000)", filtered_count)
            
            # Generation breakdown
            with st.expander("Generation Breakdown"):
                for gen, stats in generation_stats.items():
                    st.write(f"**Gen {gen}:** {stats['elites']}/{stats['total']} elites")
            
            # Axis Selection
            st.header("📈 Plot Configuration")
            
            # Determine available objectives and their labels
            if all_elites:
                sample_objectives = all_elites[0]['objectives']
                n_objectives = len(sample_objectives)
                
                # Try to get labels from metadata first
                metadata = data.get('metadata', {})
                objective_labels_from_metadata = metadata.get('objective_labels', [])
                
                # Create objective options and labels ONLY for objectives that exist in the data
                objective_options = []
                objective_labels = []
                
                for i in range(n_objectives):  # Only loop through actual number of objectives in data
                    objective_options.append(f'obj{i+1}')
                    
                    # Use metadata labels if available, otherwise use fallbacks
                    if i < len(objective_labels_from_metadata):
                        objective_labels.append(f'Objective {i+1} ({objective_labels_from_metadata[i]})')
                    else:
                        # Fallback labels for backward compatibility - only for objectives that exist
                        fallback_names = ["Split Deviation", "Neither Fraction", "Prompt Length", "Connection Count", "Failure Count"]
                        if i < len(fallback_names):
                            objective_labels.append(f'Objective {i+1} ({fallback_names[i]})')
                        else:
                            objective_labels.append(f'Objective {i+1}')
                
                # Show info about available objectives
                st.info(f"📊 Found {n_objectives} objectives in data")
                if objective_labels_from_metadata:
                    st.info(f"📋 Using labels from metadata: {objective_labels_from_metadata}")
                else:
                    st.warning("⚠️ No labels in metadata, using defaults. Update your state saver to include labels.")
                
                # X-axis selection
                x_axis_idx = st.selectbox(
                    "X-Axis:",
                    range(len(objective_options)),
                    index=0,
                    format_func=lambda x: objective_labels[x],
                    help="Choose which objective to display on the X-axis"
                )
                
                # Y-axis selection  
                y_axis_idx = st.selectbox(
                    "Y-Axis:",
                    range(len(objective_options)),
                    index=1 if len(objective_options) > 1 else 0,
                    format_func=lambda x: objective_labels[x],
                    help="Choose which objective to display on the Y-axis"
                )
                
                # Get selected column names and labels
                x_col = objective_options[x_axis_idx]
                y_col = objective_options[y_axis_idx]
                x_label = objective_labels[x_axis_idx]
                y_label = objective_labels[y_axis_idx]
            else:
                # Fallback if no elites
                x_col, y_col = 'obj1', 'obj2'
                x_label, y_label = 'Objective 1', 'Objective 2'
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Pareto Front Evolution")
            st.write(f"Showing {len(all_elites)} elite solutions across {len(generation_stats)} generations")
            st.write("💡 **Click on any point** to view its details on the right →")
            
            # Create the scatter plot - use continuous color scale
            fig = go.Figure()
            
            # Single scatter plot with continuous color scale
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='markers',
                marker=dict(
                    color=df['generation'],
                    colorscale='viridis',
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white'),
                    colorbar=dict(title="Generation")
                ),
                text=df['prompt_preview'],
                customdata=df[['generation', 'individual_id']],
                hovertemplate=
                    '<b>Generation %{customdata[0]}</b><br>' +
                    'Individual ID: %{customdata[1]}<br>' +
                    f'{x_label}: %{{x:.4f}}<br>' +
                    f'{y_label}: %{{y:.4f}}<br>' +
                    'Prompt: %{text}<br>' +
                    '<extra></extra>',
                name='Elite Solutions',
                showlegend=False
            ))
            
            fig.update_layout(
                title="Elite Solutions Across Generations",
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=600,
                showlegend=True
            )
            
            # Display the plot with selection capability
            selected_points = st.plotly_chart(
                fig, 
                use_container_width=True,
                key="pareto_plot",
                on_select="rerun",
                selection_mode="points"
            )
            
            # Handle point selection
            if selected_points and "selection" in selected_points and "points" in selected_points["selection"]:
                selected_indices = selected_points["selection"]["points"]
                if selected_indices:
                    # Get the first selected point
                    point_idx = selected_indices[0]["point_index"]
                    selected_row = df.iloc[point_idx]
                    
                    # Update session state with selected values
                    st.session_state.selected_generation = selected_row['generation']
                    st.session_state.selected_individual_id = selected_row['individual_id']
                    
                    # Show selection feedback
                    st.success(f"🎯 Selected: Generation {selected_row['generation']}, Individual {selected_row['individual_id']}")
        
        with col2:
            st.subheader("🔍 Solution Details")
            
            # Individual selector with session state integration
            generation_options = sorted(df['generation'].unique())
            
            # Use session state value if available, otherwise default to first generation
            if st.session_state.selected_generation is not None and st.session_state.selected_generation in generation_options:
                default_gen_idx = generation_options.index(st.session_state.selected_generation)
            else:
                default_gen_idx = 0
                
            selected_gen = st.selectbox(
                "Select Generation:", 
                generation_options,
                index=default_gen_idx,
                key="generation_selector"
            )
            
            gen_elites = df[df['generation'] == selected_gen]
            if not gen_elites.empty:
                elite_options = [f"Individual {row['individual_id']}" for _, row in gen_elites.iterrows()]
                elite_ids = [row['individual_id'] for _, row in gen_elites.iterrows()]
                
                # Use session state value if available and valid for this generation
                default_elite_idx = 0
                if (st.session_state.selected_individual_id is not None and 
                    st.session_state.selected_individual_id in elite_ids):
                    default_elite_idx = elite_ids.index(st.session_state.selected_individual_id)
                
                selected_elite_idx = st.selectbox(
                    "Select Individual:", 
                    range(len(elite_options)), 
                    format_func=lambda x: elite_options[x],
                    index=default_elite_idx,
                    key="individual_selector"
                )
                
                # Display selected individual details
                selected_row = gen_elites.iloc[selected_elite_idx]
                
                st.markdown(f"**Generation:** {selected_row['generation']}")
                st.markdown(f"**Individual ID:** {selected_row['individual_id']}")
                
                # Show ALL objective values for this individual
                st.subheader("📊 All Objective Values")
                objectives = selected_row['objectives']
                
                # Create columns for objectives display (max 2 per row)
                n_cols = 2
                n_rows = (len(objectives) + n_cols - 1) // n_cols
                
                for row in range(n_rows):
                    cols = st.columns(n_cols)
                    for col_idx in range(n_cols):
                        obj_idx = row * n_cols + col_idx
                        if obj_idx < len(objectives):
                            with cols[col_idx]:
                                # Get objective label
                                if obj_idx < len(objective_labels):
                                    label = objective_labels[obj_idx].split('(')[0].strip()
                                else:
                                    label = f"Objective {obj_idx + 1}"
                                
                                # Display objective value
                                obj_value = objectives[obj_idx]
                                if obj_value >= 10000:
                                    st.metric(label, "FAILED", delta="≥10000")
                                else:
                                    st.metric(label, f"{obj_value:.4f}")
                
                # Performance assessment based on primary objectives (first two if available)
                if len(objectives) >= 2:
                    obj1, obj2 = objectives[0], objectives[1]
                    if obj1 == 0.0 and obj2 == 0.0:
                        st.success("🎯 Perfect solution!")
                    elif obj1 <= 0.5 and obj2 <= 0.5:
                        st.info("✅ Good solution")
                    else:
                        st.warning("⚠️ Suboptimal solution")
                elif len(objectives) >= 1:
                    obj1 = objectives[0]
                    if obj1 == 0.0:
                        st.success("🎯 Perfect solution!")
                    elif obj1 <= 0.5:
                        st.info("✅ Good solution")
                    else:
                        st.warning("⚠️ Suboptimal solution")
                
                # Prompt
                st.subheader("📝 Optimized Prompt")
                st.text_area(
                    "Prompt:",
                    value=selected_row['prompt'],
                    height=200,
                    disabled=True
                )
                
                # Prompt stats
                prompt_text = str(selected_row['prompt'])
                col_i, col_ii, col_iii = st.columns(3)
                with col_i:
                    st.metric("Characters", len(prompt_text))
                with col_ii:
                    st.metric("Words", len(prompt_text.split()))
                with col_iii:
                    st.metric("Lines", len(prompt_text.split('\n')))
                
                # Network Connectivity Visualization
                st.subheader("🔗 Network Connectivity")
                
                # Find the original individual data with connectivity info
                selected_individual_data = None
                for gen_data in generations:
                    if gen_data['generation'] == selected_row['generation']:
                        for individual in gen_data.get('individuals', []):
                            if individual.get('individual_id') == selected_row['individual_id']:
                                selected_individual_data = individual
                                break
                        break
                
                if selected_individual_data and 'connectivity' in selected_individual_data:
                    connectivity = selected_individual_data['connectivity']
                    agent_colors_per_duplicate = selected_individual_data.get('agent_colors_per_duplicate', [])
                    
                    if agent_colors_per_duplicate:
                        # Display network graphs for each duplicate vertically
                        for duplicate_data in agent_colors_per_duplicate:
                            duplicate_idx = duplicate_data.get('duplicate', 0)
                            agent_colors = {
                                'red': duplicate_data.get('red', []),
                                'blue': duplicate_data.get('blue', []),
                                'undefined': duplicate_data.get('undefined', [])
                            }
                            
                            # Clean the data - move agents that appear in multiple categories to undefined
                            cleaned_agent_colors = {
                                'red': [],
                                'blue': [],
                                'undefined': list(agent_colors['undefined'])  # Start with existing undefined
                            }
                            
                            # Find agents that appear in multiple categories
                            red_set = set(agent_colors['red'])
                            blue_set = set(agent_colors['blue'])
                            undefined_set = set(agent_colors['undefined'])
                            
                            all_agents = red_set.union(blue_set).union(undefined_set)
                            
                            for agent in all_agents:
                                categories = []
                                if agent in red_set:
                                    categories.append('red')
                                if agent in blue_set:
                                    categories.append('blue')
                                if agent in undefined_set:
                                    categories.append('undefined')
                                
                                # If agent appears in multiple categories, move to undefined
                                if len(categories) > 1:
                                    if agent not in cleaned_agent_colors['undefined']:
                                        cleaned_agent_colors['undefined'].append(agent)
                                else:
                                    # Agent appears in only one category, keep it there
                                    if agent in red_set:
                                        cleaned_agent_colors['red'].append(agent)
                                    elif agent in blue_set:
                                        cleaned_agent_colors['blue'].append(agent)
                                    elif agent in undefined_set:
                                        if agent not in cleaned_agent_colors['undefined']:
                                            cleaned_agent_colors['undefined'].append(agent)
                            
                            # Create title with cleaned color counts
                            red_count = len(cleaned_agent_colors['red'])
                            blue_count = len(cleaned_agent_colors['blue'])
                            undefined_count = len(cleaned_agent_colors['undefined'])
                            title = f"Duplicate {duplicate_idx} - Red: {red_count}, Blue: {blue_count}, Undefined: {undefined_count}"
                            
                            # Debug information - show what agents are in each color
                            with st.expander(f"Debug: Duplicate {duplicate_idx} Agent Colors"):
                                st.write("**Red agents:**", cleaned_agent_colors['red'])
                                st.write("**Blue agents:**", cleaned_agent_colors['blue'])  
                                st.write("**Undefined agents:**", cleaned_agent_colors['undefined'])
                            
                            # Create and display the network graph using cleaned data
                            network_fig = create_network_graph(connectivity, cleaned_agent_colors, duplicate_idx, title)
                            st.plotly_chart(network_fig, use_container_width=True, key=f"network_{duplicate_idx}")
                    else:
                        st.info("No agent color data available for visualization")
                elif selected_individual_data and 'connectivity' in selected_individual_data:
                    # Show connectivity structure even without agent colors
                    connectivity = selected_individual_data['connectivity']
                    st.json({"connectivity": connectivity})
                else:
                    st.info("No connectivity data available for this individual")
        
        # Data table at the bottom
        with st.expander("📋 All Elite Solutions Data"):
            # Show a subset of columns for better readability
            display_df = df[['generation', 'individual_id', x_col, y_col, 'prompt_preview']].copy()
            display_df.columns = ['Generation', 'Individual ID', x_label, y_label, 'Prompt Preview']
            st.dataframe(display_df, use_container_width=True)
        
        # Metadata
        metadata = data.get('metadata', {})
        if metadata:
            with st.expander("ℹ️ Experiment Metadata"):
                st.json(metadata)
                
    except Exception as e:
        st.error(f"Error loading JSON: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

else:
    # Welcome message with example
    st.info("👆 Please upload a JSON state file to begin visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 Expected JSON Format")
        st.code("""
{
  "metadata": {
    "created": "2024-01-01T00:00:00",
    "num_agents": 4,
    "steps": 3
  },
  "generations": [
    {
      "generation": 0,
      "individuals": [
        {
          "is_elite": true,
          "individual_id": 0,
          "objectives": [0.1, 0.2],
          "prompt": "Your optimized prompt..."
        }
      ]
    }
  ]
}
        """, language="json")
    
    with col2:
        st.subheader("🎯 What You'll See")
        st.write("""
        - **Interactive scatter plot** showing elite solutions across generations
        - **Color-coded points** by generation (darker = earlier, brighter = later)
        - **Click any point** to instantly view its prompt and details
        - **Generation-by-generation breakdown** of elite solutions
        - **Performance assessment** for RedBlue optimization problem
        """)

# Add footer
st.markdown("---")
st.markdown("*Built for visualizing genetic algorithm optimization results with RedBlueStateSaver*") 