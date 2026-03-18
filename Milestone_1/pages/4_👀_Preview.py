import streamlit as st
from theme import theme_sidebar, apply_theme 
st.set_page_config(layout="wide")

theme_sidebar()   # Show switch at top
apply_theme()     # Apply theme

st.title("Step 4 • Preview Cleaned Dataset")

if "cleaned_df" not in st.session_state:
    st.warning("Preprocess data first.")
else:
    st.dataframe(st.session_state.cleaned_df, use_container_width=True)