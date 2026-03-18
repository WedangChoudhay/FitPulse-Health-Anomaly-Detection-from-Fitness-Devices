# =========================================
# PROFESSIONAL EDA MODULE (PRODUCTION READY)
# =========================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def run_eda(df):

    # -------------------------------
    # THEME COLORS (Auto Adaptive)
    # -------------------------------
    PRIMARY = "#00C2CB"
    SOFT_BLUE = "#38BDF8"
    SOFT_ORANGE = "#FB923C"
    SOFT_GREEN = "#34D399"
    PURPLE = "#A78BFA"
    PINK = "#F472B6"

    TEXT = "#2E3A59"
    BG = "#FFFFFF"

    sns.set_style("whitegrid")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = [
        "Steps_Taken",
        "Calories_Burned",
        "Hours_Slept",
        "Active_Minutes",
        "Heart_Rate (bpm)",
        "Stress_Level (1-10)"
    ]

    figures = []

    # =====================================
    # 1️⃣ CLEAN DISTRIBUTIONS (Not Clustered)
    # =====================================
    fig1, axes = plt.subplots(3, 2, figsize=(16, 11))
    axes = axes.flatten()

    colors = [PRIMARY, SOFT_BLUE, SOFT_ORANGE,
              SOFT_GREEN, PURPLE, PINK]

    for i, col in enumerate(numeric_cols):
        sns.histplot(
            df[col],
            kde=True,
            bins=25,
            ax=axes[i],
            color=colors[i],
            edgecolor="white",
            alpha=0.95
        )
        axes[i].set_title(f"Distribution of {col}", fontsize=12, color=TEXT)
        axes[i].tick_params(axis='both', labelsize=9)
        axes[i].grid(alpha=0.3)

    plt.tight_layout(pad=3)
    figures.append(fig1)

    # =====================================
    # 2️⃣ PROFESSIONAL BOXPLOTS
    # =====================================
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 11))
    axes2 = axes2.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(
            x=df[col],
            ax=axes2[i],
            color=colors[i],
            width=0.4
        )
        axes2[i].set_title(f"Boxplot of {col}", fontsize=12, color=TEXT)
        axes2[i].grid(alpha=0.3)

    plt.tight_layout(pad=3)
    figures.append(fig2)

    # =====================================
    # 3️⃣ MODERN HEATMAP
    # =====================================
    corr_matrix = df[numeric_cols].corr()

    fig3, ax3 = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=0.7,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax3
    )

    ax3.set_title("Correlation Heatmap", fontsize=14, color=TEXT)
    figures.append(fig3)

    # =====================================
    # 4️⃣ SMOOTH TIME SERIES
    # =====================================
    user_id = df["User_ID"].iloc[0]
    user_data = df[df["User_ID"] == user_id].sort_values("Date")

    fig4, ax4 = plt.subplots(figsize=(14, 6))

    ax4.plot(
        user_data["Date"],
        user_data["Heart_Rate (bpm)"],
        color=PRIMARY,
        linewidth=2.5
    )

    ax4.fill_between(
        user_data["Date"],
        user_data["Heart_Rate (bpm)"],
        color=PRIMARY,
        alpha=0.15
    )

    ax4.set_title(f"Heart Rate Trend - User {user_id}",
                  fontsize=14,
                  color=TEXT)

    ax4.grid(alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    figures.append(fig4)

    # =====================================
    # 5️⃣ CLEAN PIE CHART
    # =====================================
    workout_counts = df["Workout_Type"].value_counts()

    fig5, ax5 = plt.subplots(figsize=(7, 7))

    ax5.pie(
        workout_counts,
        labels=workout_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=[PRIMARY, SOFT_ORANGE, SOFT_GREEN, PURPLE],
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )

    ax5.set_title("Workout Type Distribution", fontsize=14, color=TEXT)

    figures.append(fig5)

    # =====================================
    # 6️⃣ USER SUMMARY TABLE
    # =====================================
    user_summary = (
        df.groupby("User_ID")[numeric_cols]
        .mean()
        .reset_index()
    )

    return figures, user_summary