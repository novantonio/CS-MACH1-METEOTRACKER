"""
CS-MACH1 AirLogger Environmental Data Explorer
Streamlit app: upload one or more AirLogger CSV files, visualise all
environmental parameters, and compare daily/session summaries.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.airlogger import (
    PARAMS,
    compute_metrics,
    make_3x3_figure,
    make_trajectory_map,
    parse_airlog_csv,
)

# ── Page config ─────────────────────────────────────────────────────────

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

# ── Session state ───────────────────────────────────────────────────────
# logger_dfs: parsed, unmodified per-file DataFrames (kept separate from
# any future cleaned/QC'd version, mirroring the CS-MACH1 QC pipelines).

if "logger_dfs" not in st.session_state:
    st.session_state["logger_dfs"] = {}

# ── Sidebar ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    window_size = st.slider("Rolling window (samples)", min_value=1, max_value=30, value=5)
    st.divider()

    n_files = len(st.session_state["logger_dfs"])
    if n_files > 0:
        st.success(f"📂 {n_files} file{'s' if n_files != 1 else ''} loaded")
    else:
        st.info("📂 No files loaded yet")

    st.divider()
    start_button = st.button("▶️ Start Processing", type="primary", use_container_width=True)
    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ── File uploader ───────────────────────────────────────────────────────

uploaded_files = st.file_uploader(
    "Upload one or more AirLogger CSV files, then press **Start Processing**",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files

# ── Processing ──────────────────────────────────────────────────────────

if start_button:
    files = st.session_state.get("uploaded_files", [])
    if not files:
        st.warning("Please upload at least one CSV file before starting.")
    else:
        logger_dfs: dict[str, pd.DataFrame] = {}
        progress = st.progress(0.0, text="Parsing files...")

        for i, f in enumerate(files):
            try:
                df = parse_airlog_csv(f)
                logger_dfs[f.name] = df
            except Exception as exc:
                st.warning(f"⚠️ Failed to parse **{f.name}**: {exc}")
            progress.progress((i + 1) / len(files), text=f"Parsed {f.name}")

        progress.empty()
        st.session_state["logger_dfs"] = logger_dfs

        if logger_dfs:
            st.success(f"✅ Parsed {len(logger_dfs)} file(s).")
        else:
            st.error("No valid files could be parsed.")

# ── Results ─────────────────────────────────────────────────────────────

logger_dfs = st.session_state.get("logger_dfs", {})

if not logger_dfs:
    st.info("No data yet — upload CSV files and press **Start Processing**.")
else:
    # summary table across all sessions
    st.subheader("📊 Session Summary")
    summary_rows = []
    for fname, df in logger_dfs.items():
        row = {"file": fname, "n_samples": len(df),
               "start": df["Time"].min(), "end": df["Time"].max()}
        row.update(compute_metrics(df))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)
    st.download_button(
        "⬇️ Download summary CSV",
        summary_df.to_csv(index=False).encode("utf-8"),
        file_name="airlogger_summary.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("📄 Per-file Analysis")

    fnames = list(logger_dfs.keys())
    tabs = st.tabs(fnames)

    for tab, fname in zip(tabs, fnames):
        with tab:
            df = logger_dfs[fname]
            label = Path(fname).stem

            metrics = compute_metrics(df)
            cols = st.columns(len(metrics) or 1)
            for c, (name, val) in zip(cols, metrics.items()):
                unit = next(u for _, n, u, _ in PARAMS if n == name)
                c.metric(name, f"{val:.2f} {unit}")

            traj_fig = make_trajectory_map(df, label)
            if traj_fig is not None:
                st.markdown("**📍 Trajectory Map**")
                st.plotly_chart(traj_fig, use_container_width=True, key=f"map_{fname}")
            else:
                st.caption("No valid Lat/Lon data found for this file.")

            st.markdown("**📈 Parameters over Time**")
            fig_time = make_3x3_figure(df, label=label, window=window_size, x_col="Time")
            st.pyplot(fig_time)
            plt.close(fig_time)

            if "Lon" in df.columns and not df["Lon"].isna().all():
                st.markdown("**📈 Parameters over Longitude**")
                fig_lon = make_3x3_figure(df, label=label, window=window_size, x_col="Lon")
                st.pyplot(fig_lon)
                plt.close(fig_lon)

            with st.expander("Raw data"):
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "⬇️ Download parsed CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{label}_parsed.csv",
                    mime="text/csv",
                    key=f"dl_{fname}",
                )
