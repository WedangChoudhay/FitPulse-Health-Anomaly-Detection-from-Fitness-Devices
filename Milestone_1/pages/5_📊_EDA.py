import streamlit as st
import eda
from theme import theme_sidebar, apply_theme 
st.set_page_config(layout="wide")

theme_sidebar()   # Show switch at top
apply_theme()     # Apply theme
st.title("Step 5 • Run EDA")

if "cleaned_df" not in st.session_state:
    st.warning("Preprocess data first.")
else:
    if st.button("Run EDA"):

        figures, user_summary = eda.run_eda(
            st.session_state.cleaned_df
        )

        st.subheader("📊 Distribution Analysis")
        st.pyplot(figures[0])

        st.subheader("📦 Outlier Detection")
        st.pyplot(figures[1])

        st.subheader("🔥 Correlation Heatmap")
        st.pyplot(figures[2])

        st.subheader("📈 Heart Rate Trend")
        st.pyplot(figures[3])

        st.subheader("🏋 Workout Type Distribution")
        st.pyplot(figures[4])

        st.subheader("👥 User-Level Average Summary")
        st.dataframe(user_summary, use_container_width=True)

        st.success("✅ EDA Completed Successfully!")