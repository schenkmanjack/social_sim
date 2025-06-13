import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import json

# NO st.set_page_config() call

st.title("📊 Pareto Explorer (No Config)")
st.write("Testing without page config...")

# File upload
uploaded_file = st.file_uploader(
    "Upload JSON file",
    type="json"
)

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