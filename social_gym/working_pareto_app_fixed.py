# Auto-install packages if not available
import subprocess
import sys

def install_packages():
    required_packages = [
        'streamlit>=1.20.0',
        'plotly>=5.10.0', 
        'networkx>=2.6.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0'
    ]
    
    for package in required_packages:
        try:
            __import__(package.split('>=')[0].replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try to install missing packages
try:
    import plotly.express as px
except ImportError:
    install_packages()

# Now proceed with normal imports
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

# Add the rest of your app code here...
st.info("✅ All packages loaded successfully! Add your original app code below this line.") 