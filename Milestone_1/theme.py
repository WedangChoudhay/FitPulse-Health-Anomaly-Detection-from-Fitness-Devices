import streamlit as st


# ===============================
# INITIALIZE THEME SAFELY
# ===============================
def init_theme():
    if "theme" not in st.session_state:
        st.session_state.theme = "light"


# ===============================
# SIDEBAR TOGGLE (GLOBAL)
# ===============================
def theme_sidebar():

    init_theme()   # MUST be above usage

    st.sidebar.markdown("## 🎨 Theme")

    dark_mode = st.sidebar.toggle(
        "🌙 Dark Mode",
        key="dark_toggle",
        value=(st.session_state.theme == "dark")
    )

    if dark_mode:
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"


# ===============================
# APPLY THEME
# ===============================
def apply_theme():

    init_theme()

    if st.session_state.theme == "dark":

        st.markdown("""
        <style>

        /* ===== MAIN BACKGROUND ===== */
        .stApp {
            background: linear-gradient(
                180deg,
                #0F172A 0%,
                #111827 100%
            );
            color: #F1F5F9;
        }

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {
            background-color: #1E293B;
        }

        section[data-testid="stSidebar"] * {
            color: #F1F5F9 !important;
        }

        /* ===== HEADINGS ===== */
        h1, h2, h3 {
            color: #00E0D1 !important;
        }

        /* ===== PAGE CONTAINER CARD ===== */
        .block-container {
            background-color: #1E293B;
            padding: 25px;
            border-radius: 15px;
            border-left: 6px solid #00E0D1;
        }

        /* ===== BUTTONS ===== */
        .stButton>button {
            background: linear-gradient(90deg, #00E0D1, #06B6D4);
            color: black;
            border-radius: 8px;
            font-weight: 600;
        }

        </style>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <style>

        /* ===== MAIN BACKGROUND ===== */
        .stApp {
            background: linear-gradient(
                180deg,
                #F5F7FA 0%,
                #E0F2F1 100%
            );
            color: #1E293B;
        }

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
        }

        section[data-testid="stSidebar"] * {
            color: #2E3A59 !important;
        }

        /* ===== HEADINGS ===== */
        h1, h2, h3 {
            color: #00C2CB !important;
        }

        /* ===== PAGE CONTAINER CARD ===== */
        .block-container {
            background-color: #FFFFFF;
            padding: 25px;
            border-radius: 15px;
            border-left: 6px solid #00C2CB;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        }

        /* ===== BUTTONS ===== */
        .stButton>button {
            background: linear-gradient(90deg, #00C2CB, #38BDF8);
            color: white;
            border-radius: 8px;
            font-weight: 600;
        }

        </style>
        """, unsafe_allow_html=True)