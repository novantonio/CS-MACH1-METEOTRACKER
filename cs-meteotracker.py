"""
app.py
------
CS-MACH1 AirLogger pipeline — Streamlit single-file app.

Layout
------
For every uploaded CSV:
  • 3×3 figure: one panel per parameter (Temp, Hum, Alt, Press, DP, θ, HDX, Speed, Radiation)
    Each panel shows raw data + rolling mean + mean/median lines

After all individual files:
  • Summary time-series: one plot per parameter, x=date, y=daily mean across all files
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────

PARAMS = [
    ("Temp[°C]",    "Temperature",    "°C",    "steelblue"),
    ("Hum[%]",      "Humidity",       "%",     "mediumseagreen"),
    ("Alt[m]",      "Altitude",       "m",     "slategray"),
    ("Press[mbar]", "Pressure",       "mbar",  "mediumpurple"),
    ("DP[°C]",      "Dew Point",      "°C",    "cadetblue"),
    ("θ[K]",        "Pot. Temp θ",    "K",     "sandybrown"),
    ("HDX[°C]",     "Heat Discomfort","°C",    "indianred"),
    ("Speed[km/h]", "Wind Speed",     "km/h",  "royalblue"),
    ("Radiation[]", "Radiation",      "a.u.",  "goldenrod"),
]

PARAM_COLS = [p[0] for p in PARAMS]

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CS-MACH1 AirLogger Pipeline",
    page_icon="🌬️",
    layout="wide",
)

st.title("🌬️ CS-MACH1: AirLogger Environmental Data Explorer")
st.caption(
    "Upload one or more AirLogger CSV files to visualise all environmental "
    "parameters and compare their daily means across sessions."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    window_size = st.slider("Rolling window (samples)", min_value=1, max_value=30, value=5)
    st.divider()
    n_files = len(st.session_state.get("uploaded_files", []))
    if n_files > 0:
        st.success(f"📂 {n_files} file{'s' if n_files != 1 else ''} loaded")
    else:
        st.info("📂 No files loaded yet")
    st.divider()
    start_button = st.button("▶️ Start Processing", type="primary", use_container_width=True)
    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ── File uploader ──────────────────────────────────────────────────────────────

uploaded_files = st.file_uploader(
    "Upload one or more AirLogger CSV files, then press **Start Processing**",
    type=["csv"],
    accept_multiple_files=True,
)
if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files

# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_airlog_csv(file) -> pd.DataFrame:
    """
    Parse an AirLogger CSV.
    Expected columns include: Time, Temp[°C], Hum[%], Alt[m], Press[mbar],
    DP[°C], θ[K], HDX[°C], Speed[km/h], Radiation[]
    """
    df = pd.read_csv(file)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce", utc=False)
    # strip timezone info for simplicity
    if df["Time"].dt.tz is not None:
        df["Time"] = df["Time"].dt.tz_localize(None)
    # coerce numeric columns
    for col in PARAM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Time"])
    df = df.sort_values("Time").reset_index(drop=True)
    return df


def add_rolling(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df[col].rolling(window=window, min_periods=1).mean()

# ── 3×3 Figure ─────────────────────────────────────────────────────────────────

def make_3x3_figure(df: pd.DataFrame, label: str, window: int) -> plt.Figure:
    """
    3×3 grid — one panel per parameter.
    Each panel:
      • raw data (thin, transparent)
      • rolling mean (bold, tomato)
      • mean dashed line (crimson)
      • median dashed line (darkorange)
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle(f"📋 {label}", fontsize=14, fontweight="bold", y=1.01)

    for ax, (col, name, unit, color) in zip(axes.flat, PARAMS):
        if col not in df.columns or df[col].isna().all():
            ax.set_visible(False)
            continue

        series = df[col].dropna()
        times = df.loc[series.index, "Time"]
        rolling = add_rolling(df.loc[series.index], col, window)

        p_mean = series.mean()
        p_med  = series.median()

        # raw
        ax.plot(
            times, series,
            alpha=0.30, linewidth=0.7, color=color,
            label="Raw data",
        )
        # rolling mean
        ax.plot(
            times, rolling,
            linewidth=2, color="tomato",
            label=f"Rolling mean (w={window})",
        )
        # mean / median
        ax.axhline(p_mean, color="crimson",    linewidth=1.4, linestyle="--",
                   label=f"Mean   {p_mean:.2f} {unit}")
        ax.axhline(p_med,  color="darkorange", linewidth=1.4, linestyle="--",
                   label=f"Median {p_med:.2f} {unit}")

        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("Time", fontsize=8)
        ax.set_ylabel(f"{name} ({unit})", fontsize=8)
        ax.set_title(f"{name}", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.tight_layout()
    return fig

# ── Summary figure (all files, daily means) ────────────────────────────────────

def make_summary_figure(all_means: pd.DataFrame) -> plt.Figure:
    """
    One figure with 9 subplots (3×3).
    x = date (day), y = mean of all uploaded sessions for that day.
    Each session is plotted as a scatter point + the overall daily mean as a line.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle("📊 Summary — Daily Mean Across All Sessions", fontsize=14,
                 fontweight="bold", y=1.01)

    colors = plt.cm.tab10(np.linspace(0, 1, max(all_means["source"].nunique(), 1)))
    sources = all_means["source"].unique()
    color_map = dict(zip(sources, colors))

    for ax, (col, name, unit, default_color) in zip(axes.flat, PARAMS):
        if col not in all_means.columns:
            ax.set_visible(False)
            continue

        # one scatter per source file
        for src in sources:
            sub = all_means[all_means["source"] == src].dropna(subset=[col])
            if sub.empty:
                continue
            ax.scatter(
                sub["date"], sub[col],
                color=color_map[src], s=60, zorder=3,
                label=src, alpha=0.8,
            )

        # daily mean across all sources
        daily = all_means.groupby("date")[col].mean().reset_index()
        ax.plot(
            daily["date"], daily[col],
            color="crimson", linewidth=2, linestyle="--",
            label="Daily mean", zorder=4,
        )

        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel(f"{name} ({unit})", fontsize=8)
        ax.set_title(f"{name}", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))

        if len(sources) <= 8:
            ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    return fig

# ── Main processing ────────────────────────────────────────────────────────────

if start_button and "uploaded_files" in st.session_state:
    raw_files = st.session_state["uploaded_files"]
    total = len(raw_files)
    logger_dfs: dict[str, pd.DataFrame] = {}

    progress = st.progress(0, text="Starting…")

    for i, f in enumerate(raw_files):
        progress.progress(int(i / total * 100), text=f"Processing {f.name} …")
        try:
            df = parse_airlog_csv(f)
            logger_dfs[f.name] = df
        except Exception as exc:
            st.warning(f"⚠️ Failed to parse **{f.name}**: {exc}")

    progress.progress(100, text="✅ Done!")

    if not logger_dfs:
        st.error("No valid files found.")
        st.stop()

    st.session_state["logger_dfs"] = logger_dfs
    st.session_state["window_size"] = window_size

# ── Display ────────────────────────────────────────────────────────────────────

if "logger_dfs" in st.session_state:
    logger_dfs: dict[str, pd.DataFrame] = st.session_state["logger_dfs"]
    win = st.session_state.get("window_size", window_size)

    # ── Per-file section ──────────────────────────────────────────────────────
    for fname, df in logger_dfs.items():
        st.subheader(f"📄 {fname}")

        # Quick metric row
        cols = st.columns(len(PARAMS))
        for c, (col, name, unit, _) in zip(cols, PARAMS):
            if col in df.columns and not df[col].isna().all():
                c.metric(name, f"{df[col].mean():.2f} {unit}")

        # 3×3 figure
        label = Path(fname).stem
        fig = make_3x3_figure(df, label=label, window=win)
        st.pyplot(fig)
        plt.close(fig)
        st.divider()

    # ── Summary section ───────────────────────────────────────────────────────
    st.header("📊 Summary — All Sessions")

    # Build a long dataframe with one row per (source, day)
    mean_rows = []
    for fname, df in logger_dfs.items():
        df2 = df.copy()
        df2["date"] = df2["Time"].dt.normalize()   # floor to day
        daily = df2.groupby("date")[PARAM_COLS].mean(numeric_only=True).reset_index()
        daily["source"] = Path(fname).stem
        mean_rows.append(daily)

    if mean_rows:
        all_means = pd.concat(mean_rows, ignore_index=True)

        # Table
        st.subheader("Session statistics")
        rows = []
        for fname, df in logger_dfs.items():
            row = {"File": fname, "Start": str(df["Time"].iloc[0])[:19],
                   "End": str(df["Time"].iloc[-1])[:19], "N samples": len(df)}
            for col, name, unit, _ in PARAMS:
                if col in df.columns:
                    row[f"{name} mean ({unit})"] = round(df[col].mean(), 2)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Summary figure
        fig_s = make_summary_figure(all_means)
        st.pyplot(fig_s)
        plt.close(fig_s)

st.markdown("---")
st.caption("CS-MACH1 AirLogger Pipeline · built with Streamlit & Matplotlib")
