import streamlit as st
import plotly.graph_objects as go
# import your files
from theme import theme_sidebar, apply_theme
from Milestone_1.preprocessing import run as preprocessing_run
from Milestone_2.pattern_extraction import run as pattern_run
from Milestone_3.anomaly_detector import run as anomaly_run
from Milestone_4.dashboard import run as dashboard_run
st.set_page_config(layout="wide", page_title="FitPulse")
# st.set_page_config(layout="wide")

st.sidebar.title("📊 Navigation")

module = st.sidebar.selectbox(
    "Choose Module",
    ["Preprocessing", "Pattern Extraction", "Anomaly Detection", "Dashboard"]
)

if module == "Preprocessing":

    sub_option = st.sidebar.radio(
        "Steps",
        ["Upload", "Null Check", "Preprocess", "Preview", "EDA"]
    )

    preprocessing_run(sub_option)   # PASSING VALUE

elif module == "Pattern Extraction":
    pattern_run()

elif module == "Anomaly Detection":
    anomaly_run()

elif module == "Dashboard":
    dashboard_run()
