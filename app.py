"""
CS-MACH1 AirLogger Environmental Data Explorer
Streamlit app: upload one or more AirLogger CSV files, visualise all
environmental parameters, and compare daily/session summaries.

Single-file version: parsing/plotting utilities live here directly
(previously in src/airlogger.py) for a simpler, standalone deployment.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

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

# ── Constants ────────────────────────────────────────────────────────────

PARAMS = [
    ("Temp[°C]",    "Temperature",     "°C",   "steelblue"),
    ("Hum[%]",      "Humidity",        "%",    "mediumseagreen"),
    ("Alt[m]",      "Altitude",        "m",    "slategray"),
    ("Press[mbar]", "Pressure",        "mbar", "mediumpurple"),
    ("DP[°C]",      "Dew Point",       "°C",   "cadetblue"),
    ("θ[K]",        "Pot. Temp θ",     "K",    "sandybrown"),
    ("HDX[°C]",     "Heat Discomfort", "°C",   "indianred"),
    ("Speed[km/h]", "Wind Speed",      "km/h", "royalblue"),
    ("Radiation[]", "Radiation",       "a.u.", "goldenrod"),
]

PARAM_COLS = [p[0] for p in PARAMS]


# ── Parsing ──────────────────────────────────────────────────────────────

def parse_airlog_csv(file_obj) -> pd.DataFrame:
    """
    Parse an AirLogger CSV.

    `file_obj` can be a path (str/Path) or a Streamlit UploadedFile
    (file-like object) - both work transparently with pandas.

    Expected columns include: Time, Temp[°C], Hum[%], Alt[m], Press[mbar],
    DP[°C], θ[K], HDX[°C], Speed[km/h], Radiation[], and optionally Lat/Lon.
    """
    df = pd.read_csv(file_obj)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce", utc=False)
    if df["Time"].dt.tz is not None:
        df["Time"] = df["Time"].dt.tz_localize(None)

    for col in PARAM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return df


def add_rolling(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df[col].rolling(window=window, min_periods=1).mean()


# ── Figures ──────────────────────────────────────────────────────────────

def make_3x3_figure(df: pd.DataFrame, label: str, window: int,
                     x_col: str = "Time") -> plt.Figure:
    """
    3x3 grid, one panel per parameter, plotted against `x_col`
    ("Time" or "Lon"). Each panel shows raw data, rolling mean,
    mean and median reference lines.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle(f"📋 {label}", fontsize=14, fontweight="bold", y=1.01)

    for ax, (col, name, unit, color) in zip(axes.flat, PARAMS):
        if col not in df.columns or df[col].isna().all():
            ax.set_visible(False)
            continue

        series = df[col].dropna()
        x_values = df.loc[series.index, x_col]
        rolling = add_rolling(df.loc[series.index], col, window)

        p_mean = series.mean()
        p_med = series.median()

        ax.plot(x_values, series, alpha=0.30, linewidth=0.7, color=color,
                label="Raw data")
        ax.plot(x_values, rolling, linewidth=2, color="tomato",
                label=f"Rolling mean (w={window})")
        ax.axhline(p_mean, color="crimson", linewidth=1.4, linestyle="--",
                   label=f"Mean   {p_mean:.2f} {unit}")
        ax.axhline(p_med, color="darkorange", linewidth=1.4, linestyle="--",
                   label=f"Median {p_med:.2f} {unit}")

        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel(x_col, fontsize=8)
        ax.set_ylabel(f"{name} ({unit})", fontsize=8)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if x_col == "Time":
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.tight_layout()
    return fig


def make_trajectory_map_folium(df: pd.DataFrame, label: str) -> folium.Map | None:
    """
    Folium trajectory map (Lat/Lon) with start/end markers, or None if no
    valid coordinates are present. Renders in Streamlit via
    streamlit_folium.st_folium(map, ...).

    Folium is used instead of Cartopy on purpose: Cartopy ships a compiled
    C extension (cartopy.trace) that needs a prebuilt wheel matching the
    exact Python version plus system GEOS/PROJ libraries, which frequently
    breaks on Streamlit Community Cloud. Folium (Leaflet.js under the hood)
    has no compiled dependency and is already used across the other
    CS-MACH1 apps.
    """
    if "Lat" not in df.columns or "Lon" not in df.columns:
        return None
    if df["Lat"].isna().all() or df["Lon"].isna().all():
        return None

    track_df = df.dropna(subset=["Lat", "Lon"])
    coords = list(zip(track_df["Lat"], track_df["Lon"]))
    if not coords:
        return None

    center_lat = track_df["Lat"].mean()
    center_lon = track_df["Lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                    tiles="CartoDB positron")

    # track line
    folium.PolyLine(
        locations=coords,
        color="blue",
        weight=2,
        opacity=0.7,
        tooltip=f"Track — {label}",
    ).add_to(m)

    # raw points as small markers
    for lat, lon in coords:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    # start marker
    folium.Marker(
        location=[coords[0][0], coords[0][1]],
        popup="Start",
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)

    # end marker
    folium.Marker(
        location=[coords[-1][0], coords[-1][1]],
        popup="End",
        tooltip="End",
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
    ).add_to(m)

    m.fit_bounds(coords)
    return m


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Mean value per available parameter, for summary tables/exports."""
    metrics = {}
    for col, name, unit, _ in PARAMS:
        if col in df.columns and not df[col].isna().all():
            metrics[name] = round(df[col].mean(), 2)
    return metrics


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

            traj_map = make_trajectory_map_folium(df, label)
            if traj_map is not None:
                st.markdown("**📍 Trajectory Map**")
                st_folium(traj_map, use_container_width=True, height=450, key=f"map_{fname}")
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
