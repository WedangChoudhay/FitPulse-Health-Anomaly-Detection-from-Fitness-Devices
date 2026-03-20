import sys
import os

sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from theme import theme_sidebar, apply_theme 
def run(section):
    st.set_page_config(layout="wide")

    theme_sidebar()   # Show switch at top
    apply_theme() 
    if section == "Upload":
        st.title("Step 1 • Upload Dataset")
        st.write("Upload your dataset here")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df

            st.success(f"Dataset Loaded! {df.shape[0]} rows × {df.shape[1]} columns")
    # ===============================
    # HERO SECTION
    # ===============================
        st.markdown("""
        <style>
        .badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 25px;
            color: white;
            font-size: 13px;
            font-weight: 600;
            margin-right: 10px;
            transition: all 0.3s ease;
            cursor: default;
        }

        .badge:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }

        .badge1 { background: linear-gradient(135deg, #00C2CB, #38BDF8); }
        .badge2 { background: linear-gradient(135deg, #38BDF8, #6366F1); }
        .badge3 { background: linear-gradient(135deg, #34D399, #10B981); }

        .hero-box {
            padding: 40px 30px;
            border-radius: 20px;
            background: linear-gradient(
                135deg,
                rgba(0,194,203,0.12),
                rgba(56,189,248,0.12)
            );
            margin-bottom: 30px;
        }
        </style>

        <div class="hero-box">

        <h1 style="font-size: 40px; font-weight: 800; margin-bottom: 15px;">
        💓 FitPulse — Health Anomaly Detection from Fitness Devices
        </h1>

        <p style="font-size: 16px; line-height: 1.7; max-width: 900px;">
        FitPulse is an intelligent health analytics platform designed to monitor, analyze,
        and detect anomalies from wearable fitness device data. By evaluating trends in 
        heart rate, sleep cycles, stress levels, and physical activity, the system identifies 
        irregular behavioral patterns that may indicate early health risks.
        </p>

        <div style="margin-top: 20px;">
            <span class="badge badge1">Real-Time Monitoring</span>
            <span class="badge badge2">Anomaly Detection</span>
            <span class="badge badge3">Advanced EDA</span>
        </div>

        </div>
        """, unsafe_allow_html=True)

    # ===============================
    # KPI METRICS
    # ===============================
        col1, col2, col3 = st.columns(3)

        col1.metric("📊 Data Points Processed", "22,100+")
        col2.metric("👤 Active Users Analyzed", "50")
        col3.metric("⚠ Potential Anomalies Flagged", "12")

    # ===============================
    # HOW IT WORKS SECTION
    # ===============================
        st.markdown("### 🔍 How FitPulse Works")

        st.markdown("""
        - 📥 Collect wearable fitness device data  
        - 🧹 Clean & preprocess timestamps and missing values  
        - 📈 Perform statistical & time-series analysis  
        - 🧠 Detect abnormal health patterns  
        - 📊 Visualize insights via interactive dashboards  
        """)

        st.markdown("---")

        st.success("🚀 Navigate using the sidebar to upload data and run analysis.")
    
    elif section == "Null Check":
        exec(open("Milestone_1/pages/null_check.py", encoding="utf-8").read())

    elif section == "Preprocess":
        exec(open("Milestone_1/pages/preprocess.py", encoding="utf-8").read())

    elif section == "Preview":
        exec(open("Milestone_1/pages/preview.py", encoding="utf-8").read())

    elif section == "EDA":
        exec(open("Milestone_1/pages/eda.py", encoding="utf-8").read())
        
        