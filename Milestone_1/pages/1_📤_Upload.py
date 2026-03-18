import streamlit as st
import pandas as pd
from theme import theme_sidebar, apply_theme 
st.set_page_config(layout="wide")

theme_sidebar()   # Show switch at top
apply_theme()     # Apply theme
st.title("Step 1 • Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df

    st.success(f"Dataset Loaded! {df.shape[0]} rows × {df.shape[1]} columns")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Total Nulls", df.isnull().sum().sum())