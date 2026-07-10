"""
CS-MACH1 AirLogger Environmental Data Explorer
Streamlit app: upload one or more AirLogger CSV files, visualise all
environmental parameters, and compare daily/session summaries.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st





warnings.filterwarnings("ignore")

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


def make_trajectory_map(df: pd.DataFrame, label: str) -> plt.Figure | None:
    """Cartopy trajectory map (Lat/Lon) with start/end markers, or None
    if no valid coordinates are present."""
    if "Lat" not in df.columns or "Lon" not in df.columns:
        return None
    if df["Lat"].isna().all() or df["Lon"].isna().all():
        return None

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.plot(df["Lon"], df["Lat"], marker="o", markersize=3, linestyle="-",
            alpha=0.7, color="blue", transform=ccrs.PlateCarree())
    ax.scatter(df["Lon"].iloc[0], df["Lat"].iloc[0], color="green",
               marker="^", s=100, label="Start", zorder=5,
               transform=ccrs.PlateCarree())
    ax.scatter(df["Lon"].iloc[-1], df["Lat"].iloc[-1], color="red",
               marker="v", s=100, label="End", zorder=5,
               transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    ax.set_title(f"Trajectory for {label}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                 alpha=0.6, linestyle="--")
    ax.legend()
    fig.tight_layout()
    return fig


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Mean value per available parameter, for summary tables/exports."""
    metrics = {}
    for col, name, unit, _ in PARAMS:
        if col in df.columns and not df[col].isna().all():
            metrics[name] = round(df[col].mean(), 2)
    return metrics

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
                st.pyplot(traj_fig)
                plt.close(traj_fig)
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
