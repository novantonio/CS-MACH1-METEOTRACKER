"""
app.py
------
CS-MACH1 AirLogger pipeline — Streamlit single-file app.

Features
--------
• Upload multiple AirLogger CSV files
• Per-file 3×3 dashboard (9 core parameters: raw + rolling mean + mean/median)
• Extended parameter groups: Air Quality (CO₂, O₃, CO, SO₂, NO₂, AQI),
  Particulate Matter (PM1.0 – PM10 mass & number)
• Interactive GPS track map (folium, colour-coded by temperature or any param)
• PNG export of every figure
• Cross-session summary: daily-mean plot for every parameter group
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Parameter definitions
# ══════════════════════════════════════════════════════════════════════════════

CORE_PARAMS = [
    ("Temp[°C]",    "Temperature",      "°C",   "steelblue"),
    ("Hum[%]",      "Humidity",         "%",    "mediumseagreen"),
    ("Alt[m]",      "Altitude",         "m",    "slategray"),
    ("Press[mbar]", "Pressure",         "mbar", "mediumpurple"),
    ("DP[°C]",      "Dew Point",        "°C",   "cadetblue"),
    ("θ[K]",        "Pot. Temp θ",      "K",    "sandybrown"),
    ("HDX[°C]",     "Heat Discomfort",  "°C",   "indianred"),
    ("Speed[km/h]", "Wind Speed",       "km/h", "royalblue"),
    ("Radiation[]", "Radiation",        "a.u.", "goldenrod"),
]

AQ_PARAMS = [
    ("CO2[ppm]",  "CO₂",  "ppm",  "dimgray"),
    ("O3[ppb]",   "O₃",   "ppb",  "darkcyan"),
    ("CO[ppm]",   "CO",   "ppm",  "saddlebrown"),
    ("SO2[ppb]",  "SO₂",  "ppb",  "olive"),
    ("NO2[ppb]",  "NO₂",  "ppb",  "firebrick"),
    ("AQI[]",     "AQI",  "",     "darkviolet"),
]

PM_PARAMS = [
    ("mass PM1.0[μg/m3]",   "PM1.0 mass",    "μg/m³", "coral"),
    ("mass PM2.5[μg/m3]",   "PM2.5 mass",    "μg/m³", "orangered"),
    ("mass PM4[μg/m3]",     "PM4 mass",      "μg/m³", "tomato"),
    ("mass PM10[μg/m3]",    "PM10 mass",     "μg/m³", "crimson"),
    ("number PM0.5[#/cm3]", "PM0.5 count",   "#/cm³", "peru"),
    ("number PM1.0[#/cm3]", "PM1.0 count",   "#/cm³", "chocolate"),
    ("number PM2.5[#/cm3]", "PM2.5 count",   "#/cm³", "sienna"),
    ("number PM10[#/cm3]",  "PM10 count",    "#/cm³", "maroon"),
    ("Typical Part Size[μm]","Typical P. Size","μm",   "rosybrown"),
]

EXTRA_PARAMS = [
    ("Air density[kg/m3]", "Air Density", "kg/m³", "teal"),
    ("Tir[°C]",            "Tir",         "°C",    "salmon"),
    ("Tsur[°C]",           "Tsur",        "°C",    "palevioletred"),
    ("EAQ[]",              "EAQ",         "",      "seagreen"),
    ("FAQ[]",              "FAQ",         "",      "darkolivegreen"),
]

ALL_PARAMS   = CORE_PARAMS + AQ_PARAMS + PM_PARAMS + EXTRA_PARAMS
ALL_PARAM_COLS = [p[0] for p in ALL_PARAMS]

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CS-MACH1 AirLogger Pipeline",
    page_icon="🌬️",
    layout="wide",
)

st.title("🌬️ CS-MACH1: AirLogger Environmental Data Explorer")
st.caption(
    "Upload one or more AirLogger CSV files · GPS track map · "
    "Core / Air Quality / PM dashboard · PNG export · Cross-session summary"
)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    window_size = st.slider("Rolling window (samples)", min_value=1, max_value=30, value=5)
    map_color_param = st.selectbox(
        "GPS map colour by",
        options=[p[0] for p in CORE_PARAMS],
        format_func=lambda c: next(p[1] for p in CORE_PARAMS if p[0] == c),
        index=0,
    )
    st.divider()
    n_files = len(st.session_state.get("uploaded_files", []))
    st.success(f"📂 {n_files} file{'s' if n_files != 1 else ''} loaded") if n_files else st.info("📂 No files loaded yet")
    st.divider()
    start_button = st.button("▶️ Start Processing", type="primary", use_container_width=True)
    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# File uploader
# ══════════════════════════════════════════════════════════════════════════════

uploaded_files = st.file_uploader(
    "Upload one or more AirLogger CSV files, then press **Start Processing**",
    type=["csv"],
    accept_multiple_files=True,
)
if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def parse_airlog_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce", utc=False)
    if df["Time"].dt.tz is not None:
        df["Time"] = df["Time"].dt.tz_localize(None)
    for col in ALL_PARAM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for geo in ("Lat", "Lon"):
        if geo in df.columns:
            df[geo] = pd.to_numeric(df[geo], errors="coerce")
    return df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)


def available_params(df: pd.DataFrame, param_list: list) -> list:
    return [p for p in param_list if p[0] in df.columns and df[p[0]].notna().any()]


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def download_btn(fig: plt.Figure, filename: str, label: str = "⬇️ Download PNG"):
    st.download_button(label, data=fig_to_png_bytes(fig),
                       file_name=filename, mime="image/png")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting: NxM parameter grid
# ══════════════════════════════════════════════════════════════════════════════

def make_param_grid(
    df: pd.DataFrame,
    param_list: list,
    title: str,
    window: int,
    ncols: int = 3,
) -> plt.Figure | None:
    avail = available_params(df, param_list)
    if not avail:
        return None

    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for idx, (col, name, unit, color) in enumerate(avail):
        ax = axes[idx // ncols][idx % ncols]
        series  = df[col].dropna()
        times   = df.loc[series.index, "Time"]
        rolling = series.rolling(window=window, min_periods=1).mean()
        p_mean, p_med = series.mean(), series.median()
        u = f" ({unit})" if unit else ""

        ax.plot(times, series,  alpha=0.30, linewidth=0.7, color=color, label="Raw data")
        ax.plot(times, rolling, linewidth=2, color="tomato",
                label=f"Rolling mean (w={window})")
        ax.axhline(p_mean, color="crimson",    linewidth=1.4, linestyle="--",
                   label=f"Mean   {p_mean:.2f}{u}")
        ax.axhline(p_med,  color="darkorange", linewidth=1.4, linestyle="--",
                   label=f"Median {p_med:.2f}{u}")

        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("Time", fontsize=8)
        ax.set_ylabel(f"{name}{u}", fontsize=8)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    for idx in range(len(avail), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# GPS map (folium)
# ══════════════════════════════════════════════════════════════════════════════

def make_gps_map(df: pd.DataFrame, color_col: str) -> folium.Map | None:
    if "Lat" not in df.columns or "Lon" not in df.columns:
        return None
    gdf = df.dropna(subset=["Lat", "Lon"])
    if gdf.empty:
        return None

    center = [gdf["Lat"].mean(), gdf["Lon"].mean()]
    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    if color_col in gdf.columns and gdf[color_col].notna().any():
        vals = gdf[color_col].fillna(gdf[color_col].mean())
        vmin, vmax = vals.min(), vals.max()
        cmap_fn = cm.get_cmap("RdYlBu_r")
        norm    = mcolors.Normalize(vmin=vmin, vmax=vmax)
        hex_colors = [mcolors.to_hex(cmap_fn(norm(v))) for v in vals]
    else:
        hex_colors = ["steelblue"] * len(gdf)

    coords = list(zip(gdf["Lat"], gdf["Lon"]))
    folium.PolyLine(coords, color="gray", weight=2, opacity=0.5).add_to(m)

    for (lat, lon), hc, (_, row) in zip(coords, hex_colors, gdf.iterrows()):
        parts = [f"<b>Time:</b> {row['Time']}"]
        for col, name, unit, _ in CORE_PARAMS:
            if col in row.index and pd.notna(row[col]):
                parts.append(f"<b>{name}:</b> {row[col]:.2f} {unit}")
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=hc,
            fill=True, fill_color=hc, fill_opacity=0.85,
            popup=folium.Popup("<br>".join(parts), max_width=240),
        ).add_to(m)

    folium.Marker(coords[0],  popup="▶ Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(coords[-1], popup="■ End",
                  icon=folium.Icon(color="red",   icon="stop")).add_to(m)

    # Colorbar legend as HTML
    if color_col in gdf.columns and gdf[color_col].notna().any():
        col_name = next((p[1] for p in ALL_PARAMS if p[0] == color_col), color_col)
        col_unit = next((p[2] for p in ALL_PARAMS if p[0] == color_col), "")
        legend_html = f"""
        <div style="position:fixed;bottom:30px;left:50px;z-index:1000;
                    background:white;padding:8px 12px;border-radius:6px;
                    box-shadow:2px 2px 6px rgba(0,0,0,.3);font-size:12px;">
          <b>{col_name} {col_unit}</b><br>
          <span style="color:#313695">■</span> {vmin:.1f}
          &nbsp;━━▶&nbsp;
          <span style="color:#a50026">■</span> {vmax:.1f}
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ══════════════════════════════════════════════════════════════════════════════
# Cross-session summary figure
# ══════════════════════════════════════════════════════════════════════════════

def make_summary_figure(
    all_means: pd.DataFrame,
    param_list: list,
    title: str,
    ncols: int = 3,
) -> plt.Figure | None:
    avail = [p for p in param_list
             if p[0] in all_means.columns and all_means[p[0]].notna().any()]
    if not avail:
        return None

    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    sources  = all_means["source"].unique()
    palette  = cm.tab10(np.linspace(0, 1, max(len(sources), 1)))
    cmap     = dict(zip(sources, palette))

    for idx, (col, name, unit, _) in enumerate(avail):
        ax = axes[idx // ncols][idx % ncols]
        u  = f" ({unit})" if unit else ""

        for src in sources:
            sub = all_means[all_means["source"] == src].dropna(subset=[col])
            if not sub.empty:
                ax.scatter(sub["date"], sub[col], color=cmap[src],
                           s=60, zorder=3, label=src, alpha=0.85)

        daily = all_means.groupby("date")[col].mean().reset_index()
        ax.plot(daily["date"], daily[col], color="crimson",
                linewidth=2, linestyle="--", label="Daily mean", zorder=4)

        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel(f"{name}{u}", fontsize=8)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
        if len(sources) <= 8:
            ax.legend(fontsize=6, loc="upper right")

    for idx in range(len(avail), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Processing
# ══════════════════════════════════════════════════════════════════════════════

if start_button and "uploaded_files" in st.session_state:
    raw_files = st.session_state["uploaded_files"]
    total     = len(raw_files)
    logger_dfs: dict[str, pd.DataFrame] = {}
    progress = st.progress(0, text="Starting…")

    for i, f in enumerate(raw_files):
        progress.progress(int(i / total * 100), text=f"Processing {f.name} …")
        try:
            logger_dfs[f.name] = parse_airlog_csv(f)
        except Exception as exc:
            st.warning(f"⚠️ Failed to parse **{f.name}**: {exc}")

    progress.progress(100, text="✅ Done!")
    if not logger_dfs:
        st.error("No valid files found.")
        st.stop()

    st.session_state["logger_dfs"]        = logger_dfs
    st.session_state["window_size"]       = window_size
    st.session_state["map_color_param"]   = map_color_param


# ══════════════════════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════════════════════

if "logger_dfs" in st.session_state:
    logger_dfs: dict[str, pd.DataFrame] = st.session_state["logger_dfs"]
    win     = st.session_state.get("window_size", window_size)
    map_col = st.session_state.get("map_color_param", map_color_param)

    # ────────────────────────────────────────────────────────────────────────
    # Per-file section
    # ────────────────────────────────────────────────────────────────────────
    for fname, df in logger_dfs.items():
        stem = Path(fname).stem
        st.subheader(f"📄 {fname}")

        # Quick metrics row
        avail_core = available_params(df, CORE_PARAMS)
        metric_cols = st.columns(max(len(avail_core), 1))
        for c, (col, name, unit, _) in zip(metric_cols, avail_core):
            c.metric(name, f"{df[col].mean():.2f} {unit}")

        # GPS map
        gps_map = make_gps_map(df, map_col)
        if gps_map:
            st.markdown("#### 🗺️ GPS Track")
            col_name = next((p[1] for p in ALL_PARAMS if p[0] == map_col), map_col)
            st.caption(f"Colour = {col_name} · click a point for all values")
            st_folium(gps_map, width="100%", height=420, returned_objects=[])
        else:
            st.info("No GPS data (Lat/Lon) found in this file.")

        # Core 3×3
        st.markdown("#### 🌡️ Core Parameters")
        fig_core = make_param_grid(df, CORE_PARAMS, f"Core — {stem}", win)
        if fig_core:
            st.pyplot(fig_core)
            download_btn(fig_core, f"{stem}_core.png", "⬇️ Download Core Parameters PNG")
            plt.close(fig_core)

        # Air Quality
        avail_aq = available_params(df, AQ_PARAMS)
        st.markdown("#### 🏭 Air Quality")
        if avail_aq:
            fig_aq = make_param_grid(df, AQ_PARAMS, f"Air Quality — {stem}", win)
            if fig_aq:
                st.pyplot(fig_aq)
                download_btn(fig_aq, f"{stem}_airquality.png", "⬇️ Download Air Quality PNG")
                plt.close(fig_aq)
        else:
            st.info("Air quality sensors not active in this file (CO₂, O₃, CO, SO₂, NO₂, AQI).")

        # Particulate Matter
        avail_pm = available_params(df, PM_PARAMS)
        st.markdown("#### 💨 Particulate Matter")
        if avail_pm:
            fig_pm = make_param_grid(df, PM_PARAMS, f"PM — {stem}", win)
            if fig_pm:
                st.pyplot(fig_pm)
                download_btn(fig_pm, f"{stem}_pm.png", "⬇️ Download PM PNG")
                plt.close(fig_pm)
        else:
            st.info("Particulate matter sensors not active in this file (PM1.0 – PM10).")

        # Extra
        avail_ex = available_params(df, EXTRA_PARAMS)
        if avail_ex:
            st.markdown("#### 🔬 Additional Parameters")
            fig_ex = make_param_grid(df, EXTRA_PARAMS, f"Extra — {stem}", win)
            if fig_ex:
                st.pyplot(fig_ex)
                download_btn(fig_ex, f"{stem}_extra.png", "⬇️ Download Extra PNG")
                plt.close(fig_ex)

        st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # Summary section
    # ────────────────────────────────────────────────────────────────────────
    st.header("📊 Summary — All Sessions")

    # Build all_means dataframe
    mean_rows = []
    for fname, df in logger_dfs.items():
        df2        = df.copy()
        df2["date"] = df2["Time"].dt.normalize()
        daily      = df2.groupby("date")[ALL_PARAM_COLS].mean(numeric_only=True).reset_index()
        daily["source"] = Path(fname).stem
        mean_rows.append(daily)
    all_means = pd.concat(mean_rows, ignore_index=True) if mean_rows else pd.DataFrame()

    # Statistics table
    rows = []
    for fname, df in logger_dfs.items():
        row = {
            "File":      fname,
            "Start":     str(df["Time"].iloc[0])[:19],
            "End":       str(df["Time"].iloc[-1])[:19],
            "N samples": len(df),
        }
        for col, name, unit, _ in CORE_PARAMS:
            if col in df.columns and df[col].notna().any():
                row[f"{name} mean ({unit})"] = round(df[col].mean(), 2)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if not all_means.empty:
        # Core summary
        st.markdown("#### 🌡️ Core Parameters — Daily Mean")
        fig_sc = make_summary_figure(all_means, CORE_PARAMS,
                                     "Core Parameters — Daily Mean (all sessions)")
        if fig_sc:
            st.pyplot(fig_sc)
            download_btn(fig_sc, "summary_core.png", "⬇️ Download Summary Core PNG")
            plt.close(fig_sc)

        # AQ summary
        fig_sa = make_summary_figure(all_means, AQ_PARAMS,
                                     "Air Quality — Daily Mean (all sessions)")
        if fig_sa:
            st.markdown("#### 🏭 Air Quality — Daily Mean")
            st.pyplot(fig_sa)
            download_btn(fig_sa, "summary_airquality.png", "⬇️ Download Summary AQ PNG")
            plt.close(fig_sa)

        # PM summary
        fig_sp = make_summary_figure(all_means, PM_PARAMS,
                                     "Particulate Matter — Daily Mean (all sessions)")
        if fig_sp:
            st.markdown("#### 💨 Particulate Matter — Daily Mean")
            st.pyplot(fig_sp)
            download_btn(fig_sp, "summary_pm.png", "⬇️ Download Summary PM PNG")
            plt.close(fig_sp)

    # Combined GPS map (if >1 file)
    if len(logger_dfs) > 1:
        combined     = pd.concat(list(logger_dfs.values()), ignore_index=True)
        combined_map = make_gps_map(combined, map_col)
        if combined_map:
            st.markdown("#### 🗺️ Combined GPS Tracks — All Sessions")
            st_folium(combined_map, width="100%", height=500, returned_objects=[])

st.markdown("---")
st.caption("CS-MACH1 AirLogger Pipeline · Streamlit + Matplotlib + Folium")
