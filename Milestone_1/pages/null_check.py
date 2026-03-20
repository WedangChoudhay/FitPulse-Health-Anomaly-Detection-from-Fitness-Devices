import streamlit as st
import plotly.express as px
# from theme import theme_sidebar, apply_theme 
# st.set_page_config(layout="wide")

# theme_sidebar()   # Show switch at top
# apply_theme() 

st.header("🔍 Null Value Analysis")

if "df" not in st.session_state:
    st.warning("Please upload dataset first.")
else:
    df = st.session_state.df

    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]

    if len(null_counts) == 0:
        st.success("No Null Values Found 🎉")
    else:

        # ===============================
        # THEME ACCENT COLOR
        # ===============================
        accent = "#00E0D1" if st.session_state.theme == "dark" else "#00C2CB"
        text_color = "#E2E8F0" if st.session_state.theme == "dark" else "#2E3A59"

        # ===============================
        # CUSTOM NULL CARDS
        # ===============================
        st.subheader("Columns with Missing Values")

        for col in null_counts.index:
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, {accent}, #38BDF8);
                padding: 16px 20px;
                border-radius: 12px;
                color: white;
                font-weight: 600;
                margin-bottom: 12px;
                box-shadow: 0 6px 15px rgba(0,0,0,0.08);
            ">
                {col}: {null_counts[col]} null values
            </div>
            """, unsafe_allow_html=True)

        # ===============================
        # PLOTLY BAR CHART (THEMED)
        # ===============================
        st.subheader("Null Value Count Chart")

        fig = px.bar(
            x=null_counts.values,
            y=null_counts.index,
            orientation='h',
            color_discrete_sequence=[accent]
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=text_color,
            xaxis_title="Number of Null Values",
            yaxis_title=""
        )

        fig.update_traces(marker=dict(line=dict(width=0)))

        st.plotly_chart(fig, use_container_width=True)