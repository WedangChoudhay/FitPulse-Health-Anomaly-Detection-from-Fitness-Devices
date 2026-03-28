import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd 
def run():
    # ===============================
    # PREMIUM UI CSS 🔥
    # ===============================
    st.markdown("""
    <style>
    /* PREMIUM HEADER */
    [data-testid="stHeader"] {
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(0,255,255,0.1);
    }

/* SPACING FIX */
.block-container {
    padding-top: 1.5rem !important;
}
    /* ===== MAIN BACKGROUND ===== */
    .stApp {
        background: radial-gradient(circle at top, #0f172a, #020617);
        color: white;
    }

    /* ===== GLASS CARD ===== */
    .glass {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(25px);
        border-radius: 20px;
        border: 1px solid rgba(0,255,255,0.25);
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 0 40px rgba(0,255,255,0.2);
    }

    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: rgba(0,255,255,0.08);
        border: 1px solid rgba(0,255,255,0.2);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0,255,255,0.2);
    }

    /* ===== TITLE ===== */
    .title {
        font-size: 28px;
        font-weight: 700;
        color: #00FFFF;
    }

    /* ===== BUTTON ===== */
    .stButton>button {
        background: linear-gradient(90deg,#00FFFF,#8B5CF6);
        color: black;
        border-radius: 10px;
    }
    /* DOWNLOAD BUTTON PREMIUM */
    [data-testid="stDownloadButton"] > button {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #00FFFF !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0,255,255,0.3) !important;
        backdrop-filter: blur(10px);
        padding: 12px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    /* HOVER EFFECT */
    [data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(90deg, #00FFFF, #8B5CF6) !important;
        color: black !important;
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(0,255,255,0.4);
    }
    
    /* ===== SIDEBAR FULL UPGRADE ===== */
   /* ===== FINAL CLEAN SELECTBOX ===== */
    section[data-testid="stSidebar"] .stSelectbox > div {
        background: rgba(15, 23, 42, 0.6) !important;
        border-radius: 14px;
        border: 1px solid rgba(0,255,255,0.2);
        padding: 2px;
        box-shadow: 0 0 8px rgba(0,255,255,0.15);
    }

    /* INNER SELECTBOX (MATCH OUTER) */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: transparent !important;
        border-radius: 12px !important;
        color: white !important;
    }

    /* TEXT */
    section[data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: white !important;
        font-weight: 500;
    }

    /* REMOVE INNER BOX EFFECT */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div > div {
        border: none !important;
    }

    /* HOVER (SUBTLE) */
    section[data-testid="stSidebar"] .stSelectbox > div:hover {
        border: 1px solid rgba(0,255,255,0.4);
        box-shadow: 0 0 10px rgba(0,255,255,0.2);
    }
    /* FULL SIDEBAR BACKGROUND FIX */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #0f172a) !important;
    }

    /* ALSO FIX INNER WRAPPER */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #020617, #0f172a) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("🚀 FitPulse Dashboard")

    # ===============================
    # CHECK DATA FROM MILESTONE 2
    # ===============================
    if "processed_df" not in st.session_state:
        st.warning("⚠ Please run Milestone 2 first")
        st.stop()

    df = st.session_state.processed_df.copy()
    # FIX DATE COLUMN ISSUE
    if "Date" not in df.columns and "ActivityDate" in df.columns:
        df["Date"] = pd.to_datetime(df["ActivityDate"])
    # ===============================
    # DATE HANDLING
    # ===============================
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

    # ===============================
    # FILTER BAR 🔥
    # ===============================
    st.markdown("### 📅 Filters")
       
    range_option = st.radio(
       "",
       ["All", "Last 7 Days", "Last 30 Days"],
       horizontal=True
    )
    col1, col2 = st.columns(2)

    with col1:
        numeric_cols = [col for col in df.select_dtypes(include="number").columns if col.lower() != "id"]
        metric = st.selectbox("Select Metric", numeric_cols)
    with col2:
        if "Date" in df.columns:
            start_date = st.date_input("Start Date", df["Date"].min())
            end_date = st.date_input("End Date", df["Date"].max())
    # RANGE FILTER
    with st.spinner("Processing data..."):
        if "Date" in df.columns:
            df = df.sort_values("Date")

            # Always apply date filter first
            df = df[(df["Date"] >= pd.to_datetime(start_date)) &
                    (df["Date"] <= pd.to_datetime(end_date))]

            # Then apply range option
            if range_option == "Last 7 Days":
                df = df.tail(7)

            elif range_option == "Last 30 Days":
                df = df.tail(30)
  
            elif range_option == "All":
                pass   # keep full filtered data
        if df.empty:
           st.warning("⚠ No data available for selected filters")
           return
        st.markdown(f"📊 Showing <b>{len(df)}</b> records", unsafe_allow_html=True)
    # ===============================
    # KPI CARDS 🔥
    # ===============================
    st.markdown("### 📊 Key Insights")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f'<div class="metric-card">⚡ Avg {metric}<br><b>{round(df[metric].mean(),2)}</b></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card">🔥 Max {metric}<br><b>{round(df[metric].max(),2)}</b></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card">📉 Min {metric}<br><b>{round(df[metric].min(),2)}</b></div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("📊 Quick Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Mean Value", round(df[metric].mean(), 2))
    col3.metric("Std Deviation", round(df[metric].std(), 2))

    st.markdown('</div>', unsafe_allow_html=True)
    # ===============================
    # PREMIUM CHART 🔥
    # ===============================
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("### 📈 Trend Analysis")

    fig = go.Figure()
    # fig.update_traces(line=dict(width=4))
    # Main metric
    fig.add_trace(go.Scatter(
        x=df["Date"] if "Date" in df.columns else df.index,
        y=df[metric],
        mode='lines+markers',
        line=dict(color='#00FFFF', width=4),
        marker=dict(size=6),
        line_shape="spline"
    ))

    # OPTIONAL: Add second line (if exists)
    if "Calories" in df.columns and metric != "Calories":
        fig.add_trace(go.Scatter(
            x=df["Date"] if "Date" in df.columns else df.index,
            y=df["Calories"],
            mode='lines',
            line=dict(color='#FF6EC7', width=2),
            name="Calories"
        ))

    fig.update_layout(
       template="plotly_dark",
       paper_bgcolor="rgba(0,0,0,0)",
       plot_bgcolor="rgba(0,0,0,0)",
       title=f"{metric} Trend",
       font=dict(color="white"),
       hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # ===============================
    # 🤖 ML ANOMALY DETECTION (ADVANCED)
    # ===============================
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("### ⚠ AI-Based Anomaly Detection")

    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df[[metric]])

    anomalies = df[df["anomaly"] == -1]

    st.success(f"Detected {len(anomalies)} anomalies using ML")

    # Plot anomalies
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df["Date"] if "Date" in df.columns else df.index,
        y=df[metric],
        mode='lines',
        name="Normal",
        line=dict(color="cyan")
    ))

    fig2.add_trace(go.Scatter(
        x=anomalies["Date"] if "Date" in df.columns else anomalies.index,
        y=anomalies[metric],
        mode='markers',
        name="Anomaly",
        marker=dict(
            color="red",
            size=12,
            line=dict(color="white", width=1)
        )
    ))

    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===============================
    # ANOMALY TABLE
    # ===============================
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("### 🚨 Anomaly Table")
    st.dataframe(anomalies)

    st.markdown('</div>', unsafe_allow_html=True)
    # ===============================
    # EXPORT
    # ===============================
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("### 📤 Export")

    st.download_button(
        "⬇ Download Processed Data",
        df.to_csv(index=False),
        "fitpulse_dashboard.csv"
    )

    st.markdown('</div>', unsafe_allow_html=True)
    #-------------------
    #footer section
    #--------------------
    st.markdown("""
    <div style='
        text-align:center;
        margin-top:40px;
        padding:15px;
        font-size:14px;
        color: rgba(255,255,255,0.6);
        border-top: 1px solid rgba(0,255,255,0.1);
    '>
        ⚡ <b>FitPulse Dashboard</b>  
        <br>
        Built by <b>Wedang Choudhary</b> | Streamlit • ML • Data Analytics  
    </div>
    """, unsafe_allow_html=True)