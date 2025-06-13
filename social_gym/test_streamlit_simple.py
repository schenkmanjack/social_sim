import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.title("🧪 Simple Streamlit Test")
st.write("If you see this, Streamlit is working!")

# Simple plot test
data = pd.DataFrame({
    'x': np.random.randn(50),
    'y': np.random.randn(50)
})

fig = px.scatter(data, x='x', y='y', title="Test Plot")
st.plotly_chart(fig)

st.success("✅ Streamlit is working correctly!") 