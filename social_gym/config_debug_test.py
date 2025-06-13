import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import json

# Test different page config options one by one
# Uncomment ONE line at a time to find the problematic option:

# st.set_page_config(page_title="Test")
# st.set_page_config(page_icon="📊") 
# st.set_page_config(layout="wide")
# st.set_page_config(initial_sidebar_state="expanded")

# Full config (this is what was in the original app):
st.set_page_config(
    page_title="Simple Pareto Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Config Debug Test")
st.write("Testing page configuration...")

# Same content as no_config_test to confirm it still works
uploaded_file = st.file_uploader("Upload JSON file", type="json")

if uploaded_file:
    st.success("File uploaded successfully!")
    try:
        data = json.load(uploaded_file)
        st.write(f"JSON keys: {list(data.keys())}")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a file")

# Simple plot test
data = pd.DataFrame({
    'x': np.random.randn(20),
    'y': np.random.randn(20)
})

fig = px.scatter(data, x='x', y='y', title="Test Plot")
st.plotly_chart(fig) 