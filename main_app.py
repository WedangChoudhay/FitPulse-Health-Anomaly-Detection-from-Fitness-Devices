import streamlit as st

# import your files
from theme import theme_sidebar, apply_theme
from Milestone_1.preprocessing import run as preprocessing_run
from Milestone_2.pattern_extraction import run as pattern_run
st.set_page_config(layout="wide", page_title="FitPulse")
# st.set_page_config(layout="wide")

st.sidebar.title("📊 Navigation")

module = st.sidebar.selectbox(
    "Choose Module",
    ["Preprocessing", "Pattern Extraction"]
)

if module == "Preprocessing":

    sub_option = st.sidebar.radio(
        "Steps",
        ["Upload", "Null Check", "Preprocess", "Preview", "EDA"]
    )

    preprocessing_run(sub_option)   # PASSING VALUE

elif module == "Pattern Extraction":
    pattern_run()
    