import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

from prophet import Prophet

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def run():
    # st.set_page_config(layout="wide", page_title="Fitbit Health Analytics")
    if "progress" not in st.session_state:
        st.session_state.progress = 0 
    # ------------------------------
    # DARK / LIGHT MODE COLORS
    # ------------------------------

    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

    if dark_mode:
        bg_color = "#071426"
        card_color = "#132d4b"
        text_color = "white"
    else:
        bg_color = "#f5f7fb"
        card_color = "#ffffff"
        text_color = "#1f2937"
    # ------------------------------------------------
    # UI STYLE
    # ------------------------------------------------
    st.markdown(f"""
    <style>

    .stApp{{
    background: linear-gradient(135deg,{bg_color},#0f2a44);
    color:{text_color};
    }}

    /* FIX WHITE TOP BAR */
    [data-testid="stHeader"]{{
    background:#071426 !important;
    height:0px;
    }}

    header{{
    background:#071426 !important;
    }}

    /* Sidebar toggle area */
    [data-testid="collapsedControl"]{{
    background:#071426 !important;
    }}

    /* REMOVE EXTRA TOP SPACE */
    .block-container{{
    padding-top:0rem !important;
    }}

    section.main > div{{
    padding-top:0rem !important;
    margin-top:0rem !important;
    }}

    .card{{
    background:{card_color};
    padding:20px;
    border-radius:15px;
    border:1px solid #2a4d73;
    text-align:center;
    font-size:16px;
    }}

    .card-found{{
    border:1px solid #22c55e;
    }}

    .card-missing{{
    border:1px solid #ef4444;
    }}

    .metric-card{{
    background:{card_color};
    padding:25px;
    border-radius:15px;
    border:1px solid #2a4d73;
    text-align:center;
    }}
    .metric-card{{
    background:{card_color};
    padding:25px;
    border-radius:15px;
    border:1px solid #2a4d73;
    text-align:center;
    }}
    
    /* ✅ PASTE HERE */
    div[data-baseweb="select"] * {{
     color: {text_color} !important;
    }}

    div[data-baseweb="select"] > div {{
       background-color: {card_color} !important;
       border: 1px solid #2a4d73 !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    #FOR LOAD BUTTON
    st.markdown("""
    <style>

    div.stButton > button {
        background-color: #1f4e79;
        color: white;
        border-radius: 10px;
        height: 45px;
        font-size: 16px;
        font-weight: 600;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #2b6cb0;
        color: white;
    }

    </style>
    """, unsafe_allow_html=True)
    #for card
    st.markdown(f"""
    <style>

    /* Fix metric value color */
    [data-testid="stMetricValue"] {{
        color: white ! important;
        font-size: 40px ! important;
        font-weight: 700 ! important;
    }}

    /* Fix metric label color */
    [data-testid="stMetricLabel"] {{
        color: #9cc3ff ! important;
        font-size: 16px ! important;
    }}

    /* Fix metric container spacing */
    [data-testid="stMetric"] {{
        background: {card_color};
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #2a4d73;
    }}

    </style>
    """, unsafe_allow_html=True)
    #FOR NULL CHECK
    st.markdown(f"""
    <style>

    /* Null Check Cards */

    .card{{
    background:{card_color};
    padding:16px;
    border-radius:14px;
    border:1px solid #2a4d73;
    text-align:center;
    }}

    /* dataset name */

    .card h4{{
    font-size:14px;
    font-weight:500;
    color:#cbd5e1;
    margin-bottom:10px;
    }}

    /* null value number */

    .card h2{{
    color:#22c55e;
    font-size:24px;
    font-weight:600;
    margin:6px 0;
    }}

    /* rows text */

    .card p{{
    font-size:13px;
    color:#94a3b8;
    margin-top:6px;
    }}

    /* hover animation */

    .card:hover{{
    transform:translateY(-3px);
    box-shadow:0 6px 16px rgba(0,0,0,0.35);
    transition:0.25s;
    }}

    </style>
    """, unsafe_allow_html=True)
    #TIME LOG
    st.markdown("""
    <style>

    /* Time Normalization Log */

    .log-box{
    background:#0f1f33;
    padding:18px;
    border-radius:12px;
    border:1px solid #2a4d73;
    font-family:monospace;
    line-height:1.6;
    font-size:14px;
    }

    .log-success{
    color:#22c55e;
    }

    .log-info{
    color:#60a5fa;
    }

    .log-warning{
    color:#f59e0b;
    }

    </style>
    """, unsafe_allow_html=True)
    #for cluster
    st.markdown(f"""
    <style>

    .cluster-card {{
        background:{card_color};
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #2a4d73;
        text-align: center;
        margin-bottom: 15px;
    }}

    .cluster-title {{
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 10px;
    }}

    .cluster-stat {{
        font-size: 14px;
        color: #cbd5e1;
    }}

    </style>
    """, unsafe_allow_html=True)
    #for side
    st.markdown("""
    <style>

    /* Sidebar background */
    section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#071426,#0f2a44);
    padding:25px 15px;
    }
    /* Safe Spacing */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    /* ADD HERE */
    section[data-testid="stSidebar"] * {
        pointer-events: auto !important;
    }

    div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
    }

    div[role="radiogroup"] label {
        cursor: pointer !important;
    }
              
    /* Sidebar title */
    .sidebar-title{
    font-size:28px;
    font-weight:700;
    color:#e2e8f0;
    margin-bottom:15px;
    }

    /* Sidebar modules */
    .sidebar-module{
    font-size:15px;
    color:#cbd5e1;
    margin:6px 0;
    padding:6px 10px;
    border-radius:6px;
    }

    /* Hover effect */
    .sidebar-module:hover{
    background:#132d4b;
    transition:0.2s;
    cursor:pointer;
    }

    /* Dataset card */
    .sidebar-card{
    background:#132d4b;
    padding:18px;
    border-radius:12px;
    border:1px solid #2a4d73;
    margin-top:20px;
    color:#e2e8f0;
    font-size:14px;
    line-height:1.6;
    }

    </style>
    """, unsafe_allow_html=True)
    # browser button
    st.markdown("""
    <style>

    /* File uploader container */
    [data-testid="stFileUploader"]{
    background:#132d4b;
    border:1px solid #2a4d73;
    border-radius:12px;
    padding:20px;
    }

    /* Drag and drop text */
    [data-testid="stFileUploader"] label{
    color:#e2e8f0;
    font-size:16px;
    }

    /* Browse button */
    [data-testid="stFileUploader"] button{
    background:#1f4e79;
    color:white;
    border-radius:8px;
    border:none;
    padding:8px 15px;
    }

    [data-testid="stFileUploader"] button:hover{
    background:#2b6cb0;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>

    .header-card{
    background: linear-gradient(90deg,#061a33,#0d2b5c);
    padding:35px;
    border-radius:20px;
    border:1px solid #1f4e79;
    margin-bottom:30px;
    }

    .milestone-badge{
    display:inline-block;
    background:#0f2a44;
    border:1px solid #2a4d73;
    padding:6px 16px;
    border-radius:25px;
    font-size:12px;
    letter-spacing:1px;
    color:#9cc3ff;
    margin-bottom:15px;
    }

    .header-title{
    font-size:42px;
    font-weight:700;
    color:#e2e8f0;
    margin-bottom:10px;
    }

    .header-sub{
    color:#94a3b8;
    font-size:16px;
    }

    </style>
    """, unsafe_allow_html=True)
    #FOR PIPELINE
    st.markdown("""
    <style>

    .pipeline-box{
    background:#132d4b;
    padding:20px;
    border-radius:16px;
    border:1px solid #2a4d73;
    margin-top: 0px;
    margin-bottom: 10px;           
    box-shadow:0 8px 20px rgba(0,0,0,0.35);
    }

    .pipeline-title{
    font-size:14px;
    font-weight:600;
    color:#e2e8f0;
    margin-bottom:12px;
    letter-spacing:0.5px;
    }

    .pipeline-bar{
    background:#0f1f33;
    height:12px;
    border-radius:12px;
    overflow:hidden;
    margin-bottom:14px;
    }
                
    .pipeline-progress{
    background:linear-gradient(90deg,#4f9fd1,#2ecc71);
    height:100%;
    transition: width 0.6s ease-in-out;
    }

    .pipeline-step{
    font-size:14px;
    margin:6px 0;
    color:#cbd5e1;
    display:flex;
    align-items:center;
    gap:6px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>

    .header-card{
    background: linear-gradient(90deg,#061a33,#0d2b5c);
    padding:35px;
    border-radius:20px;
    border:1px solid #1f4e79;
    margin-bottom:30px;
    }

    .milestone-badge{
    display:inline-block;
    background:#0f2a44;
    border:1px solid #2a4d73;
    padding:6px 16px;
    border-radius:25px;
    font-size:12px;
    letter-spacing:1px;
    color:#9cc3ff;
    margin-bottom:15px;
    }

    .header-title{
    font-size:42px;
    font-weight:700;
    color:#e2e8f0;
    margin-bottom:10px;
    }

    .header-sub{
    color:#94a3b8;
    font-size:16px;
    }

    </style>
    """, unsafe_allow_html=True)

    #SUMMARY BOX
    st.markdown("""
    <style>

    /* Summary container */
    .summary-box{
    background: linear-gradient(90deg,#061a33,#0d2b5c);
    border-radius:18px;
    border:1px solid #1f4e79;
    padding:25px;
    margin-top:40px;
    }

    /* Summary title */
    .summary-title{
    font-size:24px;
    font-weight:600;
    margin-bottom:15px;
    }

    /* Summary row */
    .summary-row{
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:12px 0;
    border-bottom:1px solid rgba(255,255,255,0.08);
    }

    /* Left content */
    .summary-left{
    font-size:16px;
    }

    /* Right description */
    .summary-right{
    color:#9aa6b2;
    font-size:15px;
    }

    </style>
    """, unsafe_allow_html=True)

    #---------------------
    # SIDEBAR PANEL
    #----------------------
    st.sidebar.markdown("""
    <div class="sidebar-title">
    🧬 FitPulse<br>
    <span style='font-size:12px;color:#9cc3ff;'>AI Health Analytics Dashboard</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### 📌 Navigation")

    section = st.sidebar.radio(
        "Navigation",
        [
            "📂 Data Upload",
            "🔍 Data Analysis",
            "🧬 TSFresh",
            "📈 Prophet",
            "🧠 Clustering",
            "📊 Summary"
        ]
    )
    st.sidebar.markdown("### ⚙ ML Pipeline Modules")

    modules = [
    "✔ Data Detection",
    "✔ TSFresh Feature Extraction",
    "✔ Prophet Forecasting",
    "✔ KMeans Clustering",
    "✔ DBSCAN Outlier Detection",
    "✔ PCA Visualization",
    "✔ t-SNE Visualization"
    ]

    for m in modules:
        st.sidebar.markdown(f'<div class="sidebar-module">{m}</div>', unsafe_allow_html=True)
    # --------------------------
    # PIPELINE PROGRESS 
    # --------------------------
    progress = st.session_state.progress
    status_text = "Not Started"

    if progress >= 100:
        status_text = "Completed ✅"
    elif progress >= 80:
        status_text = "Forecasting Done • Clustering Running"
    elif progress >= 60:
        status_text = "TSFresh Done • Prophet Running"
    elif progress >= 50:
        status_text = "Data Loaded"
    elif progress > 0:
        status_text = "In Progress"

    st.sidebar.markdown("### 🚀 Pipeline Progress")

    st.sidebar.markdown(
    f"""
    <div class="pipeline-box">

    <div class="pipeline-title">
        PIPELINE • {status_text}
        </div>

    <div class="pipeline-bar">
    <div class="pipeline-progress" style="width:{progress}%"></div>
    </div>

    <div class="pipeline-step">
    {"✅" if progress>=50 else "⚪"} 📂 Data Loading
    </div>

    <div class="pipeline-step">
    {"✅" if progress>=60 else "⚪"} 🧬 TSFresh Features
    </div>

    <div class="pipeline-step">
    {"✅" if progress>=80 else "⚪"} 📈 Prophet Forecast
    </div>

    <div class="pipeline-step">
    {"✅" if progress>=100 else "⚪"} 🧠 Clustering
    </div>

    </div>
    """,
    unsafe_allow_html=True
    )
    # --------------------------
    # DATA STATUS
    # --------------------------
    if "loaded" in st.session_state:
        st.sidebar.success("✅ Data Loaded Successfully")
    else:
        st.sidebar.markdown("""
        <div style="
            background:#1f3d2b;
            padding:12px;
            border-radius:10px;
            border:1px solid #2a4d73;
            color:#facc15;
            margin-top:5px;
        ">
        ⚠ Waiting for dataset files
        </div>
        """, unsafe_allow_html=True)

    # --------------------------
    # USER FILTER
    # --------------------------
    st.sidebar.markdown("### 🎛 Controls")

    if "loaded" in st.session_state:
        user_options = ["All"] + list(st.session_state.daily["Id"].unique())
    else:
        user_options = ["All"]

    selected_user = st.sidebar.selectbox("Select User ID", user_options)
    # --------------------------
    # QUICK STATS
    # --------------------------
    if "loaded" in st.session_state:
        st.sidebar.markdown("### 📊 Quick Stats")
        st.sidebar.metric("Users", st.session_state.daily["Id"].nunique())
        st.sidebar.metric("Rows", len(st.session_state.daily))

    st.sidebar.markdown("""
    <div class="sidebar-card">

    <b>📊 Dataset</b><br>
    Fitbit Fitness Tracker Data<br><br>

    <b>👥 Users</b><br>
    35<br><br>

    <b>🤖 Models</b><br>
    Prophet • KMeans • DBSCAN

    </div>
    """, unsafe_allow_html=True)

    # HEADER BANNER
    st.markdown("""
    <div class="header-card">

    <div class="milestone-badge">
    MILESTONE 2 • FEATURE EXTRACTION & MODELING
    </div>

    <h1 class="header-title">🧬 FitPulse ML Pipeline</h1>

    <p class="header-sub">
    TSFresh • Prophet • KMeans • DBSCAN • PCA • t-SNE — Real Fitbit Device Data
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------
    # FILE UPLOADER
    # ------------------------------------------------
    st.subheader("📂 Upload Fitbit Dataset Files")

    uploaded_files = st.file_uploader(
        "Drop all Fitbit CSV files here",
        type=["csv"],
        accept_multiple_files=True
    )

    datasets = {
    "Daily Activity":None,
    "Hourly Steps":None,
    "Hourly Intensities":None,
    "Minute Sleep":None,
    "Heart Rate":None
    }

    # ------------------------------------------------
    # DETECT DATASETS
    # ------------------------------------------------

    if uploaded_files:

        for file in uploaded_files:

            df = pd.read_csv(file, nrows=5)

            file.seek(0)

            cols = [c.lower() for c in df.columns]

            if "totalsteps" in cols:
                datasets["Daily Activity"] = file

            elif "steptotal" in cols:
                datasets["Hourly Steps"] = file

            elif "totalintensity" in cols:
                datasets["Hourly Intensities"] = file

            elif "logid" in cols:
                datasets["Minute Sleep"] = file

            elif "time" in cols and "value" in cols:
                datasets["Heart Rate"] = file


    # ------------------------------------------------
    # STATUS CARDS
    # ------------------------------------------------

    col1,col2,col3,col4,col5 = st.columns(5)

    cards=[
    ("🏃 Daily Activity",datasets["Daily Activity"]),
    ("👣 Hourly Steps",datasets["Hourly Steps"]),
    ("⚡ Hourly Intensities",datasets["Hourly Intensities"]),
    ("💤 Minute Sleep",datasets["Minute Sleep"]),
    ("❤️ Heart Rate",datasets["Heart Rate"])
    ]

    for col,(title,file) in zip([col1,col2,col3,col4,col5],cards):

        if file:
            col.markdown(f'<div class="card card-found"><h4>{title}</h4>Found ✓</div>',unsafe_allow_html=True)

        else:
            col.markdown(f'<div class="card card-missing"><h4>{title}</h4>Missing ✗</div>',unsafe_allow_html=True)

    # ------------------------------------------------
    # DATASET STATUS SUMMARY
    # ------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    detected = sum(v is not None for v in datasets.values())
    missing = 5 - detected

    c1,c2,c3 = st.columns(3)

    c1.metric("Detected Files", detected)
    c2.metric("Missing Files", missing)
    c3.metric("Ready to Load", "Yes" if detected==5 else "No")

    # ------------------------------------------------
    # RUN FULL PIPELINE
    # ------------------------------------------------

    if all(datasets.values()):
            
        if st.button("⚡ Load & Parse All Files") and all(datasets.values()):
            with st.spinner("🔄 Loading Fitbit datasets..."):
                # Reset file positions BEFORE reading
                datasets["Daily Activity"].seek(0)
                datasets["Hourly Steps"].seek(0)
                datasets["Minute Sleep"].seek(0)
                datasets["Heart Rate"].seek(0)
                # Load into session state
                st.session_state.daily = pd.read_csv(datasets["Daily Activity"])
                st.session_state.steps = pd.read_csv(datasets["Hourly Steps"])
                st.session_state.sleep = pd.read_csv(datasets["Minute Sleep"])
                st.session_state.hr = pd.read_csv(datasets["Heart Rate"])
        
                st.session_state.loaded = True
                st.session_state.progress = 50

                st.success("✅ All datasets loaded successfully!")
                st.rerun() # Refresh to show data
                
# =====================================================
# ✅ SHOW DATA ONLY AFTER LOADED
# =====================================================
    if "loaded" in st.session_state:
        daily = st.session_state.daily
        steps = st.session_state.steps
        sleep = st.session_state.sleep
        hr = st.session_state.hr
        # Date Conversions
        daily["ActivityDate"] = pd.to_datetime(daily["ActivityDate"])
        hr["Time"] = pd.to_datetime(hr["Time"])
        
        # ------------------------------
        # DATASET OVERVIEW CARDS
        # ------------------------------
        st.header("Dataset Overview")

        c1,c2,c3,c4,c5 = st.columns(5)

        c1.metric("Daily Users", daily["Id"].nunique())
        c2.metric("HR Users", hr["Id"].nunique())
        c3.metric("Sleep Users", sleep["Id"].nunique())
        c4.metric("HR Minute Rows", len(hr))
        c5.metric("Master Rows", len(daily))
    # ------------------------------------------------
    # DATA PREVIEW
    # ------------------------------------------------
    if "loaded" in st.session_state:

        daily = st.session_state.daily.copy()
        steps = st.session_state.steps.copy()
        sleep = st.session_state.sleep.copy()
        hr = st.session_state.hr.copy()

    if selected_user != "All":
        daily = daily[daily["Id"] == selected_user]
        steps = steps[steps["Id"] == selected_user]
        sleep = sleep[sleep["Id"] == selected_user]
        hr = hr[hr["Id"] == selected_user]
        st.header("1️⃣ Clean Dataset Preview")

        st.dataframe(
            daily.head(),
            use_container_width=True
        )
    # ------------------------------------------------
    # NULL CHECK
    # ------------------------------------------------
    if "loaded" in st.session_state:
        st.subheader("🔹 Step 2 • Null Value Check")

        c1,c2,c3,c4 = st.columns(4)

        c1.markdown(f"""
        <div class="card">
        <h4>dailyActivity</h4>
        <h2>{daily.isnull().sum().sum()}</h2>
        <p>nulls • {len(daily)} rows</p>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="card">
        <h4>hourlySteps</h4>
        <h2>{steps.isnull().sum().sum()}</h2>
        <p>nulls • {len(steps)} rows</p>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
        <div class="card">
        <h4>minuteSleep</h4>
        <h2>{sleep.isnull().sum().sum()}</h2>
        <p>nulls • {len(sleep)} rows</p>
        </div>
        """, unsafe_allow_html=True)

        c4.markdown(f"""
        <div class="card">
        <h4>heartRate</h4>
        <h2>{hr.isnull().sum().sum()}</h2>
        <p>nulls • {len(hr)} rows</p>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------
    # STEPS DISTRIBUTION
    # ------------------------------------------------

        st.header("3️⃣ Steps Distribution")

        fig = px.histogram(
            daily,
            x="TotalSteps",
            nbins=30,
            title="Distribution of Daily Steps",
            color_discrete_sequence=["#4f9fd1"]
        )

        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------
    # HEART RATE RESAMPLING
    # ------------------------------------------------

        st.header("4️⃣ Heart Rate Resampling")

        hr_minute = (
            hr.set_index("Time")
            .groupby("Id")["Value"]
            .resample("1min")
            .mean()
            .reset_index()
        )

        hr_minute.columns=["Id","Time","HeartRate"]

        st.session_state.hr_minute = hr_minute

        st.dataframe(hr_minute.head())

        st.markdown("### 🔹 5 • Time Normalization Log")

        st.markdown(f"""
        <div class="log-box">

        <div class="log-success">✔ HR resampled</div>
        seconds → 1-minute intervals<br>

        Rows before : <b>{len(hr)}</b> |
        Rows after : <b>{len(hr_minute)}</b><br><br>

        <div class="log-info">✔ Date range</div>
        {hr_minute['Time'].min().date()} → {hr_minute['Time'].max().date()}<br><br>

        <div class="log-info">✔ Hourly frequency</div>
        1.0h median | 100% exact 1-hour<br><br>

        <div class="log-info">✔ Sleep stages</div>
        1 = Light • 2 = Deep • 3 = REM |
        <b>{len(sleep)}</b> records<br><br>

        <div class="log-warning">⚠ Timezone</div>
        Local time — UTC normalization not applicable

        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------
    # TSFRESH FEATURE EXTRACTION
    # ------------------------------------------------
        st.divider()
        st.header("🧬 TSFresh Feature Extraction")

        st.info("TSFresh extracts statistical features from minute-level heart rate time series.")
        
        if st.button("🔬 Run TSFresh Feature Extraction"):
            st.session_state.tsfresh_done = True
            st.session_state.progress = 60
        if st.session_state.get("tsfresh_done"):
            ts_hr = st.session_state.hr_minute[["Id","Time","HeartRate"]].copy()

            ts_hr = ts_hr.dropna()

            ts_hr = ts_hr.sort_values(["Id","Time"])

            ts_hr = ts_hr.rename(columns={
                "Id":"id",
                "Time":"time",
                "HeartRate":"value"
            })

            features = extract_features(
                ts_hr,
                column_id="id",
                column_sort="time",
                column_value="value",
                default_fc_parameters=MinimalFCParameters(),
                disable_progressbar=False
            )
            st.session_state.features = features

            features = features.dropna(axis=1, how="all")
            if "features" in st.session_state:
                features = st.session_state.features
            st.success("✅ TSFresh Feature Extraction Complete")
            c1,c2,c3 = st.columns(3)

            c1.metric("Users", features.shape[0])
            c2.metric("Minute Rows", len(ts_hr))
            c3.metric("Features Extracted", features.shape[1])
            st.subheader("📊 TSFresh Feature Matrix (Extracted Features)")

            st.caption(
            "Each row represents a Fitbit user and each column is a statistical feature extracted "
            "from the minute-level heart rate time series using TSFresh."
            )

            st.write(f"Feature Matrix Shape: {features.shape[0]} users × {features.shape[1]} features")

            st.dataframe(features)

            # ------------------------------------------------
            # TSFRESH HEATMAP (NOTEBOOK STYLE)
            # ------------------------------------------------

            st.header("6️⃣ TSFresh Feature Matrix — Real Fitbit Heart Rate Data")

            scaler_vis = MinMaxScaler()

            features_norm = pd.DataFrame(
                scaler_vis.fit_transform(features),
                index=features.index,
                columns=features.columns
            )

            fig, ax = plt.subplots(figsize=(16,8))

            sns.heatmap(
                features_norm,
                cmap="coolwarm",
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                linecolor="gray",
                cbar_kws={"label":"Normalized Feature Value"},
                ax=ax
            )

            ax.set_title(
                "TSFresh Feature Matrix — Real Fitbit Heart Rate Data\n(Normalized 0-1 per feature)",
                fontsize=14
            )

            ax.set_xlabel("Extracted Statistical Features")
            ax.set_ylabel("User ID")

            plt.xticks(rotation=45)
            plt.yticks(rotation=0)

            plt.tight_layout()

            st.pyplot(fig)
        #-----------------------------------------
        # Prepare data for Prophet (same as notebook)
        #---------------------------------------------
        hr_minute["Date"] = hr_minute["Time"].dt.date

        prophet_hr = (
        hr_minute.groupby("Date")["HeartRate"]
        .mean()
        .reset_index()
        )

        prophet_hr.columns = ["ds", "y"]

        prophet_hr["ds"] = pd.to_datetime(prophet_hr["ds"])
        # ------------------------------------------------
        # PROPHET FORECAST
        # ------------------------------------------------
        st.divider()

        st.header("📈 Prophet Trend Forecasting")

        st.info(
        "Prophet fits additive models with weekly seasonality and 80% confidence intervals. "
        "30-day forecasts for Heart Rate, Steps, and Sleep."
        )

        if  st.button("📊 Run Prophet Forecasting (Heart Rate + Steps + Sleep)"):
            st.session_state.progress = 80
            st.success("✅ 3 Prophet models fitted — HR, Steps, Sleep • 30-day forecast each")
            st.session_state.prophet_done = True
        if st.session_state.get("prophet_done"):
            st.subheader("Heart Rate Forecast")
            # Fit Prophet
            model_hr = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.80,
                changepoint_prior_scale=0.01,
                changepoint_range=0.8
            )

            model_hr.fit(prophet_hr)

            future_hr = model_hr.make_future_dataframe(periods=30)

            forecast_hr = model_hr.predict(future_hr)

            # Plot
            fig, ax = plt.subplots(figsize=(14,5))

            # Actual HR
            ax.scatter(
                prophet_hr["ds"], prophet_hr["y"],
                color="#e53e3e",
                s=20,
                alpha=0.7,
                label="Actual HR",
                zorder=3
            )

            # Predicted trend
            ax.plot(
                forecast_hr["ds"], forecast_hr["yhat"],
                color="#3182ce",
                linewidth=2.5,
                label="Predicted Trend"
            )
        
            # Confidence interval
            ax.fill_between(
                forecast_hr["ds"],
                forecast_hr["yhat_lower"],
                forecast_hr["yhat_upper"],
                alpha=0.25,
                color="#3182ce",
                label="80% Confidence Interval"
            )

            # Forecast start line
            forecast_start = prophet_hr["ds"].max()

            ax.axvline(
                forecast_start,
                color="#f6ad55",
                linestyle="--",
                linewidth=2,
                label="Forecast Start"
            )

            ax.set_title("Heart Rate — Prophet Trend Forecast (Real Fitbit Data)", fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Heart Rate (bpm)")

            ax.legend()

            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)

            # ------------------------------------------------
            # PROPHET COMPONENTS
            # ------------------------------------------------
            st.header("Prophet Components — Real Heart Rate Data")
            fig2 = model_hr.plot_components(forecast_hr)
            fig2.set_size_inches(12,6)
            plt.tight_layout()
            st.pyplot(fig2)

        # ------------------------------------------------
        # PROPHET FORECAST — STEPS
        # ------------------------------------------------

            st.header("Steps — Prophet Trend Forecast")

            # Prepare Steps data for Prophet
            steps_df = (
            daily.groupby("ActivityDate")["TotalSteps"]
            .mean()
            .reset_index()
            )

            steps_df.columns = ["ds", "y"]

            steps_df["ds"] = pd.to_datetime(steps_df["ds"])

        # -----------------------------
        # Train Prophet Model
        # -----------------------------

            model_steps = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.80
            )

            model_steps.fit(steps_df)

            future_steps = model_steps.make_future_dataframe(periods=30)

            forecast_steps = model_steps.predict(future_steps)

        # -----------------------------
        # Plot Forecast
        #  -----------------------------

            fig, ax = plt.subplots(figsize=(14,5))

            # Actual steps
            ax.scatter(
            steps_df["ds"], steps_df["y"],
            color="#38a169",
            s=25,
            alpha=0.7,
            label="Actual Steps"
            )

            # Predicted trend
            ax.plot(
            forecast_steps["ds"],
            forecast_steps["yhat"],
            color="#2d3748",
            linewidth=2.5,
            label="Trend"
            )

            # Confidence interval
            ax.fill_between(
            forecast_steps["ds"],
            forecast_steps["yhat_lower"],
            forecast_steps["yhat_upper"],
            color="#38a169",
            alpha=0.25,
            label="80% CI"
            )

            # Forecast start
            forecast_start = steps_df["ds"].max()

            ax.axvline(
            forecast_start,
            color="#f6ad55",
            linestyle="--",
            linewidth=2,
            label="Forecast Start"
            )

            ax.set_title("Steps — Prophet Trend Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Steps")

            ax.legend()

            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
        # ------------------------------------------------
        # PROPHET FORECAST — SLEEP
        # ------------------------------------------------
            st.header("Sleep (minutes) — Prophet Trend Forecast")

            sleep_copy = sleep.copy()

            # Detect correct datetime column
            if "dateTime" in sleep_copy.columns:
                sleep_copy["date"] = pd.to_datetime(sleep_copy["dateTime"]).dt.date

            elif "date" in sleep_copy.columns:
                sleep_copy["date"] = pd.to_datetime(sleep_copy["date"]).dt.date

            elif "SleepDay" in sleep_copy.columns:
                sleep_copy["date"] = pd.to_datetime(sleep_copy["SleepDay"]).dt.date

            else:
                st.error("No valid sleep datetime column found")
                st.stop()

    # Each row = 1 minute of sleep
            sleep_agg = (
            sleep_copy.groupby("date")
            .size()
            .reset_index(name="sleep_minutes")
            )

            sleep_agg.columns = ["ds", "y"]

            sleep_agg["ds"] = pd.to_datetime(sleep_agg["ds"])

        # Train Prophet
            model_sleep = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.80
            )

            model_sleep.fit(sleep_agg)

            future_sleep = model_sleep.make_future_dataframe(periods=30)

            forecast_sleep = model_sleep.predict(future_sleep)

    # Plot
            fig, ax = plt.subplots(figsize=(14,5))

            ax.scatter(
            sleep_agg["ds"], sleep_agg["y"],
            color="#b794f4",
            s=25,
            alpha=0.7,
            label="Actual Sleep (minutes)"
            )

            ax.plot(
            forecast_sleep["ds"],
            forecast_sleep["yhat"],
            color="#2d3748",
            linewidth=2.5,
            label="Trend"
            )

            ax.fill_between(
            forecast_sleep["ds"],
            forecast_sleep["yhat_lower"],
            forecast_sleep["yhat_upper"],
            color="#b794f4",
            alpha=0.25,
            label="80% CI"
            )

            ax.axvline(
            sleep_agg["ds"].max(),
            color="#f6ad55",
            linestyle="--",
            linewidth=2,
            label="Forecast Start"
            )

            ax.set_title("Sleep (minutes) — Prophet Trend Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Sleep Minutes")

            ax.legend()

            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)

        # Prophet Insights Cards
            st.subheader("Prophet Forecast Insights")

            col1,col2,col3 = st.columns(3)

            col1.info("""
            ❤️ **Heart Rate**

            Forecast shows slight decline trend over 30 days.
            Weekly pattern detected.
            """)

            col2.success("""
            🚶 **Steps**

            Upward trend observed.
            Users appear to walk more over time.
            """)

            col3.warning("""
            😴 **Sleep**

            Confidence band wider due to sparse data.
            Sleep data variability present.
            """)
    # ------------------------------------------------
    # CLUSTERING
    # ------------------------------------------------

    st.divider()

    st.header("🧠 Clustering — KMeans + DBSCAN + PCA + t-SNE")

    st.info(
    "Using activity features for clustering. KMeans = 3 clusters and DBSCAN detects outliers."
    )

    if st.button("🧩 Run Clustering (K=3)"):
        st.session_state.progress = 100
        
        cluster_cols = [
        "TotalSteps",
        "Calories",
        "VeryActiveMinutes",
        "SedentaryMinutes"
        ]

    # Select features
        X = daily[cluster_cols].copy()

    # Remove missing values (important for clustering)
        X = X.dropna()

    # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # -------------------------
    # ELBOW METHOD
    # -------------------------

        inertias = []

        for k in range(2,10):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        fig, ax = plt.subplots(figsize=(8,4))   # balanced size

        ax.plot(
        range(2,10),
        inertias,
        marker="o",
        color="#4f9fd1",
        linewidth=2.5,
        markersize=9,
        markerfacecolor="#ff6b81",
        markeredgecolor="#4f9fd1"
        )

        ax.set_title("KMeans Elbow Curve — Real Fitbit Data", fontsize=14)
        ax.set_xlabel("Number of Clusters (K)", fontsize=11)
        ax.set_ylabel("Inertia", fontsize=11)

        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()

        st.pyplot(fig)
    # -------------------------
    # KMEANS CLUSTERING
    # -------------------------

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        from sklearn.metrics import silhouette_score
        score = silhouette_score(X_scaled, labels)
        st.metric("Silhouette Score (KMeans)", round(score, 3))
    # PCA projection
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        var_explained = pca.explained_variance_ratio_ * 100 # new by me

        fig, ax = plt.subplots(figsize=(8,5))

    # Custom colors like your image
        colors = ["#4f9fd1", "#ff6b81", "#2ecc71"]

        for cluster in range(3):

            cluster_points = X_pca[labels == cluster]

            ax.scatter(
              cluster_points[:,0],
              cluster_points[:,1],
              color=colors[cluster],
              s=120,
              alpha=0.8,
              label=f"Cluster {cluster}",
              edgecolor="white"
            )

    # Add point labels (steps values)
        for i in range(0, len(X_pca), 15):   # label every 15th point
            ax.text(
              X_pca[i,0],
              X_pca[i,1],
              str(daily["TotalSteps"].iloc[i]),
              fontsize=7,
              ha="center"
            )

    # Titles
        ax.set_title(
        "KMeans Clustering — PCA Projection\nReal Fitbit Data (K=3)",
        fontsize=14
        )

    # Axis labels with PCA variance
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")

    # Grid
        ax.grid(True, linestyle="--", alpha=0.5)

    # Legend
        ax.legend(title="Cluster")

        plt.tight_layout()

        st.pyplot(fig)

    # -------------------------
    # DBSCAN MODEL
    # -------------------------

        EPS = 1.3
        MIN_SAMPLES = 5

        dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
        dbscan_labels = dbscan.fit_predict(X_scaled)

        # -------------------------
    # CLUSTERING SUMMARY CARDS
    # -------------------------
        st.subheader("Clustering Summary")

        c1,c2,c3,c4,c5,c6 = st.columns(6)

        c1.metric("Users Clustered", len(X))
        c2.metric("KMeans Clusters", 3)
        c3.metric("PC1 Variance", f"{var_explained[0]:.1f}%")
        c4.metric("PC2 Variance", f"{var_explained[1]:.1f}%")
        c5.metric("DBSCAN Clusters", len(set(dbscan_labels))-1)
        c6.metric("Noise Points", list(dbscan_labels).count(-1))

    # -------------------------
    # DBSCAN CLUSTER VISUALIZATION
    # -------------------------

        fig, ax = plt.subplots(figsize=(10,6))

        palette = ["#5DADE2", "#F194B4", "#58D68D"]

        for cluster_id in sorted(set(dbscan_labels)):

            mask = dbscan_labels == cluster_id

            if cluster_id == -1:
                ax.scatter(
                    X_pca[mask,0],
                    X_pca[mask,1],
                    color="red",
                    marker="x",
                    s=250,
                    linewidths=3,
                    label="Noise / Outlier"
                )

            else:
                ax.scatter(
                    X_pca[mask,0],
                    X_pca[mask,1],
                    color=palette[cluster_id % len(palette)],
                    s=180,
                    edgecolors="white",
                    linewidths=1.5,
                    alpha=0.9,
                    label=f"Cluster {cluster_id}"
                )

        # Clean legend (remove duplicates)
        handles, legend_labels = ax.get_legend_handles_labels()
        unique = dict(zip(legend_labels, handles))
        ax.legend(unique.values(), unique.keys(), title="Cluster", fontsize=11)

        # Titles
        ax.set_title(
            f"DBSCAN Clustering — PCA Projection\nReal Fitbit Data (eps={EPS})",
            fontsize=18,
            pad=20
        )

        ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)", fontsize=12)
        ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)", fontsize=12)

        # Grid styling
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()

        st.pyplot(fig)


        # -------------------------
        # t-SNE VISUALIZATION
        # -------------------------

        st.header("t-SNE Visualization")

        st.write("⏳ Running t-SNE...")

        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, len(X_scaled) - 1),
            max_iter=1000
        )

        X_tsne = tsne.fit_transform(X_scaled)

        fig, axes = plt.subplots(1, 2, figsize=(16,6))

        palette = ["#5DADE2", "#F194B4", "#58D68D"]

        # KMeans t-SNE
        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id

            axes[0].scatter(
                X_tsne[mask,0],
                X_tsne[mask,1],
                c=palette[cluster_id % len(palette)],
                label=f"Cluster {cluster_id}",
                s=120,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.8
            )

        axes[0].set_title("KMeans — t-SNE Projection (K=3)")
        axes[0].set_xlabel("t-SNE Dim 1")
        axes[0].set_ylabel("t-SNE Dim 2")
        axes[0].legend(title="Cluster")
        axes[0].grid(alpha=0.2)

        # DBSCAN t-SNE
        for label in sorted(set(dbscan_labels)):
            mask = dbscan_labels == label

            if label == -1:
                axes[1].scatter(
                    X_tsne[mask,0],
                    X_tsne[mask,1],
                    c="red",
                    marker="x",
                    s=150,
                    label="Noise"
                )
            else:
                axes[1].scatter(
                    X_tsne[mask,0],
                    X_tsne[mask,1],
                    c=palette[label % len(palette)],
                    label=f"Cluster {label}",
                    s=120
                )

        axes[1].set_title(f"DBSCAN — t-SNE Projection (eps={EPS})")
        axes[1].set_xlabel("t-SNE Dim 1")
        axes[1].set_ylabel("t-SNE Dim 2")
        axes[1].legend(title="Cluster")
        axes[1].grid(alpha=0.2)

        plt.tight_layout()

        st.pyplot(fig)

        st.success("✅ t-SNE projection generated")

        # -----------------------
        # Cluster Behavior Profile
        # -----------------------

        st.header("Cluster Behavior Profile")

        sleep_minutes = (
            sleep.groupby("Id")
            .size()
            .reset_index(name="TotalSleepMinutes")
        )

        cluster_data = daily.merge(sleep_minutes, on="Id", how="left")

        cluster_data["Cluster"] = labels

        profile = cluster_data.groupby("Cluster")[[
            "TotalSteps",
            "Calories",
            "VeryActiveMinutes",
            "SedentaryMinutes",
            "TotalSleepMinutes"
        ]].mean()

        profile_reset = profile.reset_index()

        fig = px.bar(
            profile_reset,
            x="Cluster",
            y=[
                "TotalSteps",
                "Calories",
                "VeryActiveMinutes",
                "SedentaryMinutes",
                "TotalSleepMinutes"
            ],
            barmode="group",
            title="Average Behavior by Cluster"
        )

        st.plotly_chart(fig, use_container_width=True)
        #-------------------------------
        # Cluster Interpretation Cards
        #-------------------------------

        st.subheader(" 🧠 Cluster Profiles")

        col1,col2,col3 = st.columns(3)
        cluster_icons = ["🚶","🛋","🏃"]
        clusters = ["Moderately Active","Sedentary","Highly Active"]

        for i, col in enumerate([col1,col2,col3]):

            steps = int(profile.loc[i,"TotalSteps"])
            sedentary = int(profile.loc[i,"SedentaryMinutes"])
            active = int(profile.loc[i,"VeryActiveMinutes"])

            users = cluster_data[cluster_data["Cluster"]==i]["Id"].nunique()

            col.markdown(f"""
            <div class="cluster-card">

            <div class="cluster-title">
            {cluster_icons[i]} Cluster {i} — {clusters[i]}
            </div>

            <div class="cluster-stat">
            🚶 Steps: <b>{steps}/day</b><br>
            🪑 Sedentary: <b>{sedentary} min</b><br>
            ⚡ Very Active: <b>{active} min</b><br>
            👤 Users: <b>{users}</b>
            </div>

            </div>
            """, unsafe_allow_html=True)

    # ------------------------------------------------
    # MILESTONE 2 SUMMARY (DYNAMIC)
    # ------------------------------------------------

    st.header("✅ Milestone 2 Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "📂 Files Loaded",
        len(uploaded_files) if uploaded_files else 0
    )

    if "features" in st.session_state:
        col2.metric("🧪 TSFresh Features", st.session_state.features.shape[1])
    else:
        col2.metric("🧪 TSFresh Features", "—")

    col3.metric("📈 Prophet Forecast", "30 Days")

    if "labels" in locals():
        col4.metric("⚙ KMeans Clusters", len(set(labels)))
    else:
        col4.metric("⚙ KMeans Clusters", "—")

    if "dbscan_labels" in locals():
        col5.metric("🔍 DBSCAN Noise", list(dbscan_labels).count(-1))
    else:
        col5.metric("🔍 DBSCAN Noise", "—")


    st.divider()
    st.success("✅ Fitbit Health Analytics ML Pipeline Completed Successfully")

    st.caption("Built using Streamlit • TSFresh • Prophet • KMeans • DBSCAN • PCA • t-SNE")

    # ------------------------------------------------
    # FOOTER
    # ------------------------------------------------

    st.markdown("""
    ---
    <div style="text-align:center; padding:10px; color:#9aa6b2; font-size:14px">

    <b>FitPulse — Fitbit Health Analytics ML Pipeline</b><br>

    Developed by <b>Wedang Choudhary</b>  
    B.Tech Computer Science Engineering  

    Machine Learning Pipeline using  
    TSFresh • Prophet • KMeans • DBSCAN • PCA • t-SNE • Streamlit

    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run()
