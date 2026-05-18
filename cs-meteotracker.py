# CS-MACH1 MeteoTracker — Complete Working app.py

from __future__ import annotations
import io
import warnings
from datetime import datetime
from pathlib import Path

import folium
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from streamlit_folium import st_folium

from cs_mach1_theme import apply_cs_mach1_theme, cs_mach1_footer

warnings.filterwarnings("ignore")
matplotlib.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════════════════════════════════════════
# Branding
# ══════════════════════════════════════════════════════════════════════════════
apply_cs_mach1_theme(
    page_title="CS-MACH1 MeteoTracker",
    page_icon="logo.png",
    main_title="🌬️ CS-MACH1: MeteoTracker Environmental Data Explorer",
    subtitle="Upload one or more MeteoTracker Logger CSV files",
    logo_path="logo.png",
    logo_width=200,
)

# ══════════════════════════════════════════════════════════════════════════════
# Parameter definitions
# ══════════════════════════════════════════════════════════════════════════════
CORE_PARAMS = [
    ("Temp[°C]", "Temperature", "°C", "steelblue"),
    ("Hum[%]", "Humidity", "%", "mediumseagreen"),
    ("Alt[m]", "Altitude", "m", "slategray"),
    ("Press[mbar]", "Pressure", "mbar", "mediumpurple"),
    ("DP[°C]", "Dew Point", "°C", "cadetblue"),
    ("θ[K]", "Pot. Temp θ", "K", "sandybrown"),
    ("HDX[°C]", "Heat Discomfort", "°C", "indianred"),
    ("Speed[km/h]", "Wind Speed", "km/h", "royalblue"),
    ("Radiation[]", "Radiation", "a.u.", "goldenrod"),
]

AQ_PARAMS = [
    ("CO2[ppm]", "CO₂", "ppm", "dimgray"),
    ("O3[ppb]", "O₃", "ppb", "darkcyan"),
    ("CO[ppm]", "CO", "ppm", "saddlebrown"),
    ("SO2[ppb]", "SO₂", "ppb", "olive"),
    ("NO2[ppb]", "NO₂", "ppb", "firebrick"),
    ("AQI[]", "AQI", "", "darkviolet"),
]

PM_PARAMS = [
    ("mass PM1.0[μg/m3]", "PM1.0 mass", "μg/m³", "coral"),
    ("mass PM2.5[μg/m3]", "PM2.5 mass", "μg/m³", "orangered"),
    ("mass PM4[μg/m3]", "PM4 mass", "μg/m³", "tomato"),
    ("mass PM10[μg/m3]", "PM10 mass", "μg/m³", "crimson"),
    ("number PM0.5[#/cm3]", "PM0.5 count", "#/cm³", "peru"),
    ("number PM1.0[#/cm3]", "PM1.0 count", "#/cm³", "chocolate"),
    ("number PM2.5[#/cm3]", "PM2.5 count", "#/cm³", "sienna"),
    ("number PM10[#/cm3]", "PM10 count", "#/cm³", "maroon"),
    ("Typical Part Size[μm]", "Typical P. Size", "μm", "rosybrown"),
]

EXTRA_PARAMS = [
    ("Air density[kg/m3]", "Air Density", "kg/m³", "teal"),
    ("Tir[°C]", "Tir", "°C", "salmon"),
    ("Tsur[°C]", "Tsur", "°C", "palevioletred"),
    ("EAQ[]", "EAQ", "", "seagreen"),
    ("FAQ[]", "FAQ", "", "darkolivegreen"),
]

ALL_PARAMS = CORE_PARAMS + AQ_PARAMS + PM_PARAMS + EXTRA_PARAMS
ALL_PARAM_COLS = [p[0] for p in ALL_PARAMS]
BRAND_BLUE = "#00A6D6"

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("logo.png", width=160)
    st.markdown("### ⚙️ Settings")
    
    window_size = st.slider("Rolling window (samples)", min_value=1, max_value=30, value=5)
    
    map_color_param = st.selectbox(
        "Map colour parameter",
        options=[p[0] for p in CORE_PARAMS],
        format_func=lambda c: next((p[1] for p in CORE_PARAMS if p[0] == c), c),
        index=0,
    )

    st.divider()
    if st.button("🧹 Reset All", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # PDF Report Button in Sidebar
    if st.session_state.get("logger_dfs"):
        if st.button("📄 Generate Full PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating comprehensive PDF report..."):
                pdf_bytes = generate_pdf_report(
                    logger_dfs=st.session_state["logger_dfs"],
                    window=window_size,
                    map_color_col=map_color_param,
                )
                st.download_button(
                    label="⬇️ Download Full PDF Report",
                    data=pdf_bytes,
                    file_name=f"CS_MACH1_MeteoTracker_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# File uploader
# ══════════════════════════════════════════════════════════════════════════════
uploaded_files = st.file_uploader(
    "Upload one or more MeteoTracker CSV files",
    type=["csv"],
    accept_multiple_files=True,
    key="uploader"
)

if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files

# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════
def parse_airlog_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    if df["Time"].dt.tz is not None:
        df["Time"] = df["Time"].dt.tz_localize(None)
    
    for col in ALL_PARAM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    for geo in ("Lat", "Lon"):
        if geo in df.columns:
            df[geo] = pd.to_numeric(df[geo], errors="coerce")
    
    return df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)


def available(df: pd.DataFrame, param_list: list) -> list:
    return [p for p in param_list if p[0] in df.columns and df[p[0]].notna().any()]


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def dl_btn(fig: plt.Figure, filename: str, label: str = "⬇️ Download PNG"):
    st.download_button(label, data=fig_to_png_bytes(fig), file_name=filename, mime="image/png")


def _plot_generic(ax, df, col, name, unit, color, window):
    series = df[col].dropna()
    if series.empty:
        ax.set_visible(False)
        return
    times = df.loc[series.index, "Time"]
    
    no_roll_cols = {"DP[°C]", "θ[K]", "HDX[°C]", "Speed[km/h]", "Radiation[]"}
    
    u = f" ({unit})" if unit else ""
    
    ax.plot(times, series, alpha=0.35, linewidth=0.8, color=color, label="Raw data")
    
    if col not in no_roll_cols:
        rolling = series.rolling(window=window, min_periods=1).mean()
        ax.plot(times, rolling, linewidth=2, color="#E8524A", label=f"Rolling mean (w={window})")
    
    ax.axhline(series.mean(), color="crimson", linewidth=1.3, linestyle="--", label=f"Mean {series.mean():.2f}{u}")
    ax.axhline(series.median(), color="darkorange", linewidth=1.3, linestyle="--", label=f"Median {series.median():.2f}{u}")
    
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.75)
    ax.set_ylabel(f"{name}{u}", fontsize=8)
    ax.set_title(name, fontsize=9, fontweight="bold", color="#333333")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def make_param_grid(df: pd.DataFrame, param_list: list, title: str, window: int, ncols: int = 3):
    avail = available(df, param_list)
    if not avail:
        return None
    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.8 * nrows), squeeze=False)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(title, fontsize=14, fontweight="bold", color=BRAND_BLUE, y=0.98)
    
    for idx, (col, name, unit, color) in enumerate(avail):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor("#fafafa")
        _plot_generic(ax, df, col, name, unit, color, window)
    
    for idx in range(len(avail), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    
    fig.tight_layout()
    return fig


def make_folium_map(df: pd.DataFrame, color_col: str) -> folium.Map | None:
    if "Lat" not in df.columns or "Lon" not in df.columns:
        return None
    gdf = df.dropna(subset=["Lat", "Lon"])
    if gdf.empty:
        return None
    
    center = [gdf["Lat"].mean(), gdf["Lon"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    
    if color_col in gdf.columns and gdf[color_col].notna().any():
        vals = gdf[color_col].fillna(gdf[color_col].mean())
        vmin, vmax = vals.min(), vals.max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_fn = cm.get_cmap("RdYlBu_r")
        hex_colors = [mcolors.to_hex(cmap_fn(norm(v))) for v in vals]
    else:
        hex_colors = [BRAND_BLUE] * len(gdf)
    
    coords = list(zip(gdf["Lat"], gdf["Lon"]))
    folium.PolyLine(coords, color="#aaaaaa", weight=2.5, opacity=0.6).add_to(m)
    
    for (lat, lon), color, (_, row) in zip(coords, hex_colors, gdf.iterrows()):
        popup_text = f"<b>Time:</b> {row['Time']}<br>"
        for c, name, unit, _ in CORE_PARAMS:
            if c in row and pd.notna(row[c]):
                popup_text += f"<b>{name}:</b> {row[c]:.2f} {unit}<br>"
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_text, max_width=280),
        ).add_to(m)
    
    return m


def generate_pdf_report(logger_dfs: dict, window: int, map_color_col: str) -> bytes:
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Title Page
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.axis("off")
        plt.text(0.5, 0.72, "CS-MACH1 MeteoTracker Report", ha="center", va="center", 
                fontsize=28, color=BRAND_BLUE, fontweight="bold")
        plt.text(0.5, 0.58, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                ha="center", fontsize=16)
        plt.text(0.5, 0.50, f"Files analyzed: {len(logger_dfs)}", ha="center", fontsize=16)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        
        for fname, df in logger_dfs.items():
            stem = Path(fname).stem
            for params, title in [
                (CORE_PARAMS, f"Core Parameters — {stem}"),
                (AQ_PARAMS, f"Air Quality — {stem}"),
                (PM_PARAMS, f"Particulate Matter — {stem}"),
                (EXTRA_PARAMS, f"Extra Parameters — {stem}")
            ]:
                fig = make_param_grid(df, params, title, window)
                if fig:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
    
    pdf_buffer.seek(0)
    return pdf_buffer.read()

# ══════════════════════════════════════════════════════════════════════════════
# Processing
# ══════════════════════════════════════════════════════════════════════════════
if st.button("▶️ Start Processing", type="primary", use_container_width=True) and "uploaded_files" in st.session_state:
    raw_files = st.session_state["uploaded_files"]
    logger_dfs = {}
    progress = st.progress(0, text="Processing files...")
    
    for i, f in enumerate(raw_files):
        progress.progress(int((i + 1) / len(raw_files) * 100), text=f"Processing {f.name}...")
        try:
            logger_dfs[f.name] = parse_airlog_csv(f)
        except Exception as e:
            st.warning(f"⚠️ Failed to parse {f.name}: {e}")
    
    if logger_dfs:
        st.session_state["logger_dfs"] = logger_dfs
        st.session_state["window_size"] = window_size
        st.session_state["map_color_param"] = map_color_param
        st.success(f"✅ {len(logger_dfs)} file(s) processed successfully!")
    else:
        st.error("No valid files could be processed.")

# ══════════════════════════════════════════════════════════════════════════════
# Display Results
# ══════════════════════════════════════════════════════════════════════════════
if "logger_dfs" in st.session_state and st.session_state["logger_dfs"]:
    logger_dfs = st.session_state["logger_dfs"]
    win = st.session_state.get("window_size", 5)
    map_col = st.session_state.get("map_color_param", "Temp[°C]")

    for i, (fname, df) in enumerate(logger_dfs.items()):
        stem = Path(fname).stem
        st.markdown(f"### 📄 {fname}")
        st.caption(f"{df['Time'].iloc[0]} → {df['Time'].iloc[-1]} | {len(df):,} samples")

        # Quick metrics
        avail = available(df, CORE_PARAMS)
        cols = st.columns(max(len(avail), 1))
        for c, (col, name, unit, _) in zip(cols, avail):
            c.metric(name, f"{df[col].mean():.2f} {unit}")

        # Parameter plots
        for param_list, title in [
            (CORE_PARAMS, "🌡️ Core Parameters"),
            (AQ_PARAMS, "🏭 Air Quality"),
            (PM_PARAMS, "💨 Particulate Matter"),
            (EXTRA_PARAMS, "🔬 Extra Parameters")
        ]:
            fig = make_param_grid(df, param_list, f"{title} — {stem}", win)
            if fig:
                st.markdown(f"#### {title}")
                st.pyplot(fig)
                dl_btn(fig, f"{stem}_{title.lower().replace(' ', '_').replace('🌡️','').replace('🏭','').replace('💨','').replace('🔬','')}.png")
                plt.close(fig)

        # Map
        st.markdown("#### 🗺️ Interactive GPS Map")
        fmap = make_folium_map(df, map_col)
        if fmap:
            st_folium(fmap, width="100%", height=480, key=f"folium_map_{i}_{stem}")
        else:
            st.info("No GPS data available for this session.")

        st.divider()

    # Combined view for multiple files
    if len(logger_dfs) > 1:
        combined = pd.concat(list(logger_dfs.values()), ignore_index=True)
        st.markdown("#### 🗺️ Combined GPS Tracks — All Sessions")
        combined_map = make_folium_map(combined, map_col)
        if combined_map:
            st_folium(combined_map, width="100%", height=520, key="combined_map")

# Footer
st.markdown("---")
cs_mach1_footer("CS-MACH1 MeteoTracker Pipeline")
