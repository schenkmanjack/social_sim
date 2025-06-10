# 🎯 Interactive Pareto Front Explorer

A sophisticated web-based application for exploring Pareto optimal solutions from your genetic algorithm optimization runs. Built with Streamlit and Plotly for an interactive, notebook-like experience.

## ✨ Features

- **🎨 Interactive Visualization**: Click on any point in the Pareto front to see detailed information
- **📊 Dynamic Axis Selection**: Choose which objectives/metrics to display on X and Y axes
- **🔍 Detailed Individual Information**: View DOFs, objectives, metrics, and constraints for each solution
- **📈 Multiple Plot Types**: Switch between objectives and metrics visualization
- **🎛️ User-Friendly Interface**: Clean sidebar controls and responsive layout
- **🔄 Real-time Updates**: Instant feedback when exploring different solutions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_pareto_app.txt
```

### 2. Launch the App

#### Option A: With Sample Data (Demo)
```bash
streamlit run interactive_pareto_app.py
```

#### Option B: With Your GA Instance
```python
from interactive_pareto_app import InteractiveParetoApp

# After running your genetic algorithm
app = InteractiveParetoApp()
app.load_ga_data(your_ga_instance)

# Then run: streamlit run interactive_pareto_app.py
```

#### Option C: Integration Example
```bash
python example_integration.py
```

## 📱 User Interface

### Left Panel: Pareto Front Visualization
- **Interactive Plot**: Scatter plot showing population (gray) and Pareto front (red)
- **Clickable Points**: Click any point to see details in the right panel
- **Hover Information**: See basic info by hovering over points
- **Zoom/Pan**: Use Plotly controls to explore the plot

### Right Panel: Individual Details
- **📊 Objectives**: All objective function values
- **📈 Metrics**: Additional metrics (if different from objectives)
- **🔧 DOFs**: Design variables with statistics
- **📊 Visualizations**: Bar charts of DOF distributions
- **⚠️ Constraints**: Constraint values (if any)

### Sidebar Controls
- **🎛️ Data Source**: Choose between sample data or GA instance
- **📊 Plot Type**: Switch between objectives and metrics
- **🎯 Axis Selection**: X and Y axis dropdown menus
- **📈 Statistics**: Population and Pareto front size

## 🔧 Integration with Your GA Code

### Method 1: Direct Integration
```python
# In your GA script
from interactive_pareto_app import InteractiveParetoApp

# After optimization
ga = GeneticAlgorithmBase(...)
ga.optimize(...)

# Load into interactive app
app = InteractiveParetoApp()
app.load_ga_data(ga)
```

### Method 2: Modified plot_optimization_results
Replace or enhance your existing plotting with:
```python
def plot_optimization_results_interactive(ga_instance):
    """Launch interactive Pareto explorer instead of static plots"""
    from interactive_pareto_app import launch_interactive_pareto_explorer
    return launch_interactive_pareto_explorer(ga_instance)
```

### Method 3: Post-Processing
```python
# Load saved GA results and explore interactively
ga = load_ga_results()  # Your loading function
app = InteractiveParetoApp()
app.load_ga_data(ga)
```

## 📊 Data Structure Compatibility

The app works with `GeneticAlgorithmBase` instances that have:

- **Individuals**: Objects with `objectives`, `dofs`, `metrics`, `constraints`
- **Population**: List of individuals with `is_elite` status
- **Labels**: `objective_labels` and `metric_labels` attributes
- **PyTorch/NumPy**: Automatic handling of tensor/array conversions

### Expected Individual Structure:
```python
individual.objectives    # np.array or torch.tensor
individual.dofs         # np.array or torch.tensor  
individual.metrics      # np.array or torch.tensor
individual.constraints  # List[float]
individual.is_elite     # bool
```

## 🎨 Customization

### Adding Custom Metrics
```python
# In your evaluator
def evaluate(self, dofs):
    objectives = compute_objectives(dofs)
    metrics = compute_additional_metrics(dofs)  # Custom metrics
    constraints = compute_constraints(dofs)
    return objectives, constraints, metrics
```

### Custom Labels
```python
ga.objective_labels = ["Error Rate", "Complexity", "Energy"]
ga.metric_labels = ["Training Loss", "Validation Loss", "Inference Time"]
```

### Styling
Modify the Streamlit configuration in `interactive_pareto_app.py`:
```python
st.set_page_config(
    page_title="Your Custom Title",
    page_icon="🎯",
    layout="wide"
)
```

## 🐛 Troubleshooting

### Common Issues

1. **"No data loaded"**: Ensure your GA instance has a populated `population` attribute
2. **"Error loading GA data"**: Check that individuals have required attributes (`objectives`, `dofs`)
3. **Empty plot**: Verify that some individuals have `is_elite=True`
4. **Import errors**: Install all dependencies from `requirements_pareto_app.txt`

### Data Validation
```python
# Check your GA data before loading
print(f"Population size: {len(ga.population)}")
print(f"Elite count: {len([ind for ind in ga.population if ind.is_elite])}")
print(f"Objective labels: {ga.objective_labels}")
```

## 📈 Example Use Cases

1. **🔬 Research**: Explore trade-offs between objectives in multi-objective optimization
2. **🏭 Engineering**: Analyze design alternatives and their parameters  
3. **🎯 Hyperparameter Tuning**: Understand relationships between hyperparameters and performance
4. **📊 Presentation**: Create interactive demos for stakeholders
5. **🔍 Debugging**: Investigate GA behavior and convergence patterns

## 🤝 Contributing

To extend the app:

1. **Add new visualizations**: Extend `create_pareto_plot()` method
2. **Enhance details panel**: Modify `display_individual_details()`
3. **Custom data loaders**: Add methods to `InteractiveParetoApp` class
4. **New plot types**: Add options to the sidebar controls

## 📄 License

This interactive Pareto front explorer is part of your genetic algorithm codebase.

---

**🎯 Happy Exploring!** 

Click on those Pareto points and discover the optimal trade-offs in your multi-objective optimization problems! 