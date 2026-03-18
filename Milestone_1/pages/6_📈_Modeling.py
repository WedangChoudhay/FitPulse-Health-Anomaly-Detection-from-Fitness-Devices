import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from theme import theme_sidebar, apply_theme
st.set_page_config(layout="wide")

theme_sidebar()
apply_theme()

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

from prophet import Prophet

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

st.title("📈 • Feature Extraction & Modeling")

# ======================================
# LOAD DATA FROM MILESTONE 1
# ======================================

if "processed_df" not in st.session_state:
    st.warning("⚠ Please run preprocessing first.")
    st.stop()

df = st.session_state["processed_df"].copy()

st.success("✅ Preprocessed data loaded")

# ======================================
# TSFRESH FEATURE EXTRACTION
# ======================================

st.header("🔹 TSFresh Feature Extraction")

ts_df = df[["User_ID","Date","Heart_Rate (bpm)"]].dropna()

ts_df = ts_df.sort_values(["User_ID","Date"])

ts_df = ts_df.rename(columns={
    "User_ID":"id",
    "Date":"time",
    "Heart_Rate (bpm)":"value"
})

features = extract_features(
    ts_df,
    column_id="id",
    column_sort="time",
    column_value="value",
    default_fc_parameters=MinimalFCParameters()
)

features = features.dropna(axis=1, how="all")

st.subheader("TSFresh Feature Matrix")
st.dataframe(features.head())

# Heatmap
heatmap_data = (features - features.mean()) / features.std()
fig1 = plt.figure(figsize=(12,5))

sns.heatmap(
    features.iloc[:20],
    cmap="coolwarm",
    annot=False,
    linewidths=0.3
)

plt.title("TSFresh Feature Matrix Heatmap")
plt.xlabel("Extracted Statistical Features")
plt.ylabel("User ID")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

st.pyplot(fig1)
# ======================================
# PROPHET TREND MODELING
# ======================================

st.header("🔹 Prophet Trend Modeling")

df["Date"] = pd.to_datetime(df["Date"])

prophet_hr = df.groupby("Date")["Heart_Rate (bpm)"].mean().reset_index()

prophet_hr.columns = ["ds","y"]

model_hr = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    interval_width=0.80
)

model_hr.fit(prophet_hr)

future = model_hr.make_future_dataframe(periods=90)

forecast = model_hr.predict(future)

fig, ax = plt.subplots(figsize=(12,5))

# Actual data points
ax.scatter(
    prophet_hr["ds"],
    prophet_hr["y"],
    color="pink",
    s=12,
    alpha=0.6,
    label="Actual Heart Rate"
)

# Predicted trend
ax.plot(
    forecast["ds"],
    forecast["yhat"],
    color="#1f77b4",
    linewidth=2,
    label="Predicted Trend"
)

# Confidence interval
ax.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    alpha=0.25,
    color="#1f77b4",
    label="80% Confidence Interval"
)
# Forecast start maker
forecast_start = prophet_hr["ds"].max()

ax.axvline(
    forecast_start,
    color="orange",
    linestyle="--",
    linewidth=2,
    label="Forecast Start"
)

ax.set_title("Heart Rate Forecast using Prophet")
ax.set_xlabel("Date")
ax.set_ylabel("Heart Rate (bpm)")
ax.legend()

# plt.xticks(rotation=45)
# plt.tight_layout()
st.pyplot(fig)

# Prophet components
fig3 = model_hr.plot_components(forecast)
st.pyplot(fig3)
# ======================================
# PROPHET MODEL — HOURS SLEPT
# ======================================

st.subheader("Sleep Trend Forecast")

sleep_df = df.groupby("Date")["Hours_Slept"].mean().reset_index()
sleep_df.columns = ["ds","y"]

model_sleep = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    interval_width=0.80
)

model_sleep.fit(sleep_df)

future_sleep = model_sleep.make_future_dataframe(periods=90)

forecast_sleep = model_sleep.predict(future_sleep)

fig_sleep, ax = plt.subplots(figsize=(12,5))

ax.scatter(
    sleep_df["ds"],
    sleep_df["y"],
    color="black",
    s=12,
    alpha=0.6,
    label="Actual Sleep Hours"
)

ax.plot(
    forecast_sleep["ds"],
    forecast_sleep["yhat"],
    color="#2ca02c",
    linewidth=2,
    label="Predicted Trend"
)

ax.fill_between(
    forecast_sleep["ds"],
    forecast_sleep["yhat_lower"],
    forecast_sleep["yhat_upper"],
    alpha=0.25,
    color="#2ca02c",
    label="80% Confidence Interval"
)

forecast_start = sleep_df["ds"].max()

ax.axvline(
    forecast_start,
    color="orange",
    linestyle="--",
    linewidth=2,
    label="Forecast Start"
)

ax.set_title("Sleep Forecast using Prophet")
ax.set_xlabel("Date")
ax.set_ylabel("Hours_Slept")
ax.legend()

st.pyplot(fig_sleep)
# ======================================
# PROPHET MODEL — STEPS TAKEN
# ======================================

st.subheader("Steps Trend Forecast")

steps_df = df.groupby("Date")["Steps_Taken"].mean().reset_index()
steps_df.columns = ["ds","y"]

model_steps = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    interval_width=0.80
)

model_steps.fit(steps_df)

future_steps = model_steps.make_future_dataframe(periods=90)

forecast_steps = model_steps.predict(future_steps)

fig_steps, ax = plt.subplots(figsize=(12,5))

ax.scatter(
    steps_df["ds"],
    steps_df["y"],
    color="black",
    s=12,
    alpha=0.6,
    label="Actual Steps"
)

ax.plot(
    forecast_steps["ds"],
    forecast_steps["yhat"],
    color="#9467bd",
    linewidth=2,
    label="Predicted Trend"
)

ax.fill_between(
    forecast_steps["ds"],
    forecast_steps["yhat_lower"],
    forecast_steps["yhat_upper"],
    alpha=0.25,
    color="#9467bd",
    label="80% Confidence Interval"
)

forecast_start = steps_df["ds"].max()

ax.axvline(
    forecast_start,
    color="orange",
    linestyle="--",
    linewidth=2,
    label="Forecast Start"
)

ax.set_title("Steps Forecast using Prophet")
ax.set_xlabel("Date")
ax.set_ylabel("Steps_Taken")
ax.legend()

st.pyplot(fig_steps)

# ======================================
# CLUSTERING FEATURES
# ======================================

st.header("🔹 Behavioral Clustering")

numeric_cols = [
"Hours_Slept",
"Active_Minutes",
"Heart_Rate (bpm)",
"Steps_Taken",
"Calories_Burned",
"Stress_Level (1-10)"
]

cluster_features = df.groupby("User_ID")[numeric_cols].mean()

cluster_features = cluster_features.dropna()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(cluster_features)

# ======================================
# KMEANS
# ======================================

kmeans = KMeans(n_clusters=3, random_state=42)

kmeans_labels = kmeans.fit_predict(X_scaled)

# ======================================
# DBSCAN
# ======================================

dbscan = DBSCAN(eps=1.5,min_samples=3)

dbscan_labels = dbscan.fit_predict(X_scaled)

# ======================================
# PCA VISUALIZATION
# ======================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

fig4, ax = plt.subplots()

scatter = ax.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=kmeans_labels,
    cmap="Set2",
    s=80
)

ax.set_title("KMeans Clustering (PCA Projection)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

# Add legend
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

st.pyplot(fig4)

# DBSCAN PCA

fig5, ax = plt.subplots()

scatter = ax.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=dbscan_labels,
    cmap="Set1",
    s=80
)

ax.set_title("DBSCAN Clustering (PCA Projection)")

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

st.pyplot(fig5)

# ======================================
# t-SNE VISUALIZATION
# ======================================

st.header("🔹 t-SNE Projection")

tsne = TSNE(n_components=2,random_state=42)

X_tsne = tsne.fit_transform(X_scaled)

fig6, ax = plt.subplots()

scatter = ax.scatter(
    X_tsne[:,0],
    X_tsne[:,1],
    c=kmeans_labels,
    cmap="Set2",
    s=80
)

ax.set_title("t-SNE Projection of User Behavior")

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

st.pyplot(fig6)

# ======================================
# CLUSTER PROFILE
# ======================================

profile = cluster_features.copy()

profile["Cluster"] = kmeans_labels

cluster_profile = profile.groupby("Cluster")[numeric_cols].mean()

st.subheader("Cluster Profiles")

st.dataframe(cluster_profile)

profile_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())

fig7 = profile_norm.T.plot(
    kind="bar",
    figsize=(12,5),
    colormap="coolwarm",
    edgecolor="white"
).get_figure()

plt.title("Normalized Cluster Profiles", fontsize=14)

plt.xlabel("Health Features")
plt.ylabel("Scaled Value (0–1)")

plt.xticks(rotation=45, ha="right")

plt.legend(title="Cluster", loc="upper right")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

st.pyplot(fig7)
st.success("✅ Milestone 2 Completed Successfully")