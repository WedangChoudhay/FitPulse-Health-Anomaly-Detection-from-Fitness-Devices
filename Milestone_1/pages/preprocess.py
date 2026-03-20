import streamlit as st
import numpy as np
import pandas as pd
# from theme import theme_sidebar, apply_theme 
# st.set_page_config(layout="wide")

# theme_sidebar()   # Show switch at top
# apply_theme()     # Apply theme
st.title("Step 3 • Preprocess Data")

if "df" not in st.session_state:
    st.warning("Upload dataset first.")
else:
    df = st.session_state.df

    st.markdown('<div class="section-header">Step 3 • Preprocess Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if st.button("Run Preprocessing"):

            df_before = df.copy()

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])

            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].interpolate()

            if "Workout_Type" in df.columns:
                df["Workout_Type"] = df["Workout_Type"].fillna("No Workout")

            st.session_state.cleaned_df = df
            st.session_state.processed_df = df   # ← ADDED FOR MILESTONE 2

            st.success("Preprocessing Completed")

            col1, col2 = st.columns(2)
            col1.write("Before:")
            col1.write(df_before.isnull().sum())
            col2.write("After:")
            col2.write(df.isnull().sum())

    st.markdown('</div>', unsafe_allow_html=True)