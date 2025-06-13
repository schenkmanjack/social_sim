import streamlit as st

st.title("🔍 Minimal Test - Step 1")
st.write("Testing basic Streamlit...")

if st.button("Test Button"):
    st.success("Button works!")

st.write("If you see this, basic Streamlit is working")

# Uncomment these one by one to find the problematic import:
# import numpy as np
# st.write("✅ numpy imported")

# import pandas as pd  
# st.write("✅ pandas imported")

# import plotly.express as px
# st.write("✅ plotly imported")

# import json
# st.write("✅ json imported") 