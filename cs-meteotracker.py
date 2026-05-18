# CS-MACH1 MeteoTracker — Complete Refactored app.py

from __future__ import annotations

import io
import warnings
from datetime import datetime
from pathlib import Path

import contextily as ctx
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
from pyproj import Transformer
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
_PARAM_LOOKUP = {p[0]: p for p in ALL_PARAMS}

BRAND_BLUE = "#00A6D6"

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("logo.png", width=160)

    st.markdown("### ⚙️ Settings")

    window_size = st.slider(
        "Rolling window (samples)",
        min_value=1,
        max_value=30,
        value=5,
    )

    map_color_param = st.selectbox(
        "Map colour parameter",
        options=[p[0] for p in CORE_PARAMS],
        format_func=lambda c: next(p[1] for p in CORE_PARAMS if p[0] == c),
        index=0,
    )

    st.divider()

    n_files = len(st.session_state.get("uploaded_files", []))

    if n_files:
        st.success(f"📂 {n_files} file{'s' if n_files != 1 else ''} loaded")
    else:
        st.info("📂 No files loaded yet")

    st.divider()

    start_button = st.button(
        "▶️ Start Processing",
        type="primary",
        use_container_width=True,
    )

    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# File uploader
# ══════════════════════════════════════════════════════════════════════════════

uploaded_files = st.file_uploader(
    "Upload one or more MeteoTracker CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files

# ══════════════════════════════════════════════════════════════════════════════
# Parsing helpers
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

    return (
        df.dropna(subset=["Time"])
        .sort_values("Time")
        .reset_index(drop=True)
    )



def available(df: pd.DataFrame, param_list: list) -> list:
    return [
        p for p in param_list
        if p[0] in df.columns and df[p[0]].notna().any()
    ]



def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()

    fig.savefig(
        buf,
        format="png",
        dpi=130,
        bbox_inches="tight",
    )

    buf.seek(0)

    return buf.read()



def dl_btn(fig: plt.Figure, filename: str, label: str = "⬇️ Download PNG"):
    st.download_button(
        label,
        data=fig_to_png_bytes(fig),
        file_name=filename,
        mime="image/png",
    )

# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════════════════


def _style_ax(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    ax.grid(True, alpha=0.25, color="#dddddd")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))



def _plot_generic(ax, df, col, name, unit, color, window):
    """
    Generic parameter plotter.

    Rolling mean DISABLED for:
    - Dew Point
    - Potential Temperature
    - Heat Discomfort
    - Wind Speed
    - Radiation
    """

    series = df[col].dropna()

    if series.empty:
        ax.set_visible(False)
        return

    times = df.loc[series.index, "Time"]

    no_roll_cols = {
        "DP[°C]",
        "θ[K]",
        "HDX[°C]",
        "Speed[km/h]",
        "Radiation[]",
    }

    u = f" ({unit})" if unit else ""

    # Raw signal
    ax.plot(
        times,
        series,
        alpha=0.35,
        linewidth=0.8,
        color=color,
        label="Raw data",
    )

    # Rolling only where desired
    if col not in no_roll_cols:
        rolling = series.rolling(window=window, min_periods=1).mean()

        ax.plot(
            times,
            rolling,
            linewidth=2,
            color="#E8524A",
            label=f"Rolling mean (w={window})",
        )

    # Statistics
    ax.axhline(
        series.mean(),
        color="crimson",
        linewidth=1.3,
        linestyle="--",
        label=f"Mean {series.mean():.2f}{u}",
    )

    ax.axhline(
        series.median(),
        color="darkorange",
        linewidth=1.3,
        linestyle="--",
        label=f"Median {series.median():.2f}{u}",
    )

    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.75)

    ax.set_ylabel(f"{name}{u}", fontsize=8)

    ax.set_title(
        name,
        fontsize=9,
        fontweight="bold",
        color="#333333",
    )

    ax.grid(True, alpha=0.3)

    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M")
    )

# ══════════════════════════════════════════════════════════════════════════════
# Generic parameter grid
# ══════════════════════════════════════════════════════════════════════════════


def make_param_grid(
    df: pd.DataFrame,
    param_list: list,
    title: str,
    window: int,
    ncols: int = 3,
) -> plt.Figure | None:

    avail = [
        p for p in param_list
        if p[0] in df.columns and df[p[0]].notna().any()
    ]

    if not avail:
        return None

    nrows = int(np.ceil(len(avail) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    fig.patch.set_facecolor("#ffffff")

    fig.suptitle(
        title,
        fontsize=13,
        fontweight="bold",
        color=BRAND_BLUE,
        y=1.01,
    )

    for idx, (col, name, unit, color) in enumerate(avail):

        ax = axes[idx // ncols][idx % ncols]

        ax.set_facecolor("#fafafa")

        _plot_generic(
            ax,
            df,
            col,
            name,
            unit,
            color,
            window,
        )

    for idx in range(len(avail), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()

    return fig
# ══════════════════════════════════════════════════════════════════════════════
# Altitude plotter
# ══════════════════════════════════════════════════════════════════════════════

def _plot_altitude(ax, df):
    """
    Altitude:
    raw step-plot only
    NO rolling mean
    """

    col = "Alt[m]"

    if col not in df.columns or df[col].isna().all():
        ax.set_visible(False)
        return

    series = df[col].dropna()

    times = df.loc[
        series.index,
        "Time"
    ]

    # Step plot
    ax.step(
        times,
        series,
        where="mid",
        linewidth=1.3,
        color="slategray",
        label="Altitude",
    )

    # Filled area
    ax.fill_between(
        times,
        series,
        step="mid",
        alpha=0.18,
        color="slategray",
    )

    # Mean line
    ax.axhline(
        series.mean(),
        color="crimson",
        linewidth=1.3,
        linestyle="--",
        label=f"Mean {series.mean():.1f} m",
    )

    # Median line
    ax.axhline(
        series.median(),
        color="darkorange",
        linewidth=1.3,
        linestyle="--",
        label=f"Median {series.median():.1f} m",
    )

    ax.legend(
        fontsize=6.5,
        loc="upper right",
    )

    ax.set_ylabel(
        "Altitude (m)",
        fontsize=8,
    )

    ax.set_title(
        "Altitude",
        fontsize=9,
        fontweight="bold",
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.tick_params(
        axis="x",
        rotation=25,
        labelsize=7,
    )

    ax.tick_params(
        axis="y",
        labelsize=7,
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M")
    )
    
# ══════════════════════════════════════════════════════════════════════════════
# Interactive Folium map
# ══════════════════════════════════════════════════════════════════════════════


def make_folium_map(df: pd.DataFrame, color_col: str) -> folium.Map | None:

    if "Lat" not in df.columns or "Lon" not in df.columns:
        return None

    gdf = df.dropna(subset=["Lat", "Lon"])

    if gdf.empty:
        return None

    center = [gdf["Lat"].mean(), gdf["Lon"].mean()]

    m = folium.Map(
        location=center,
        zoom_start=15,
        tiles="CartoDB positron",
    )

    if color_col in gdf.columns and gdf[color_col].notna().any():

        vals = gdf[color_col].fillna(gdf[color_col].mean())

        vmin, vmax = vals.min(), vals.max()

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        cmap_fn = cm.get_cmap("RdYlBu_r")

        hex_colors = [mcolors.to_hex(cmap_fn(norm(v))) for v in vals]

        col_label = next(
            (p[1] for p in ALL_PARAMS if p[0] == color_col),
            color_col,
        )

        col_unit = next(
            (p[2] for p in ALL_PARAMS if p[0] == color_col),
            "",
        )

    else:
        hex_colors = [BRAND_BLUE] * len(gdf)
        vmin = vmax = col_label = col_unit = None

    coords = list(zip(gdf["Lat"], gdf["Lon"]))

    folium.PolyLine(
        coords,
        color="#aaaaaa",
        weight=2,
        opacity=0.5,
    ).add_to(m)

    for (lat, lon), hc, (_, row) in zip(coords, hex_colors, gdf.iterrows()):

        parts = [f"<b>Time:</b> {row['Time']}"]

        for c, name, unit, _ in CORE_PARAMS:
            if c in row.index and pd.notna(row[c]):
                parts.append(f"<b>{name}:</b> {row[c]:.2f} {unit}")

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=hc,
            fill=True,
            fill_color=hc,
            fill_opacity=0.85,
            popup=folium.Popup("<br>".join(parts), max_width=260),
        ).add_to(m)

    folium.Marker(
        coords[0],
        popup="▶ Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    folium.Marker(
        coords[-1],
        popup="■ End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    if col_label:

        legend_html = f"""
        <div style="position:fixed;bottom:30px;left:50px;z-index:1000;
                    background:white;padding:8px 14px;border-radius:8px;
                    box-shadow:2px 2px 8px rgba(0,0,0,.25);font-size:12px;
                    border-left:4px solid {BRAND_BLUE};">
          <b>{col_label} {col_unit}</b><br>
          <span style="color:#313695">■</span> {vmin:.1f}
          &nbsp;━━▶&nbsp;
          <span style="color:#a50026">■</span> {vmax:.1f}
        </div>
        """

        m.get_root().html.add_child(folium.Element(legend_html))

    return m

# ══════════════════════════════════════════════════════════════════════════════
# Summary figure
# ══════════════════════════════════════════════════════════════════════════════


def make_summary_figure(
    all_means: pd.DataFrame,
    param_list: list,
    title: str,
    ncols: int = 3,
) -> plt.Figure | None:

    avail = [
        p for p in param_list
        if p[0] in all_means.columns and all_means[p[0]].notna().any()
    ]

    if not avail:
        return None

    nrows = int(np.ceil(len(avail) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    fig.patch.set_facecolor("#ffffff")

    fig.suptitle(
        title,
        fontsize=13,
        fontweight="bold",
        color=BRAND_BLUE,
        y=1.01,
    )

    sources = all_means["source"].unique()

    palette = cm.tab10(np.linspace(0, 1, max(len(sources), 1)))

    cmap = dict(zip(sources, palette))

    for idx, (col, name, unit, _) in enumerate(avail):

        ax = axes[idx // ncols][idx % ncols]

        ax.set_facecolor("#fafafa")

        u = f" ({unit})" if unit else ""

        for src in sources:

            sub = all_means[
                all_means["source"] == src
            ].dropna(subset=[col])

            if not sub.empty:
                ax.scatter(
                    sub["date"],
                    sub[col],
                    color=cmap[src],
                    s=60,
                    zorder=3,
                    label=src,
                    alpha=0.85,
                )

        daily = all_means.groupby("date")[col].mean().reset_index()

        ax.plot(
            daily["date"],
            daily[col],
            color="crimson",
            linewidth=2,
            linestyle="--",
            label="Daily mean",
            zorder=4,
        )

        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel(f"{name}{u}", fontsize=8)

        ax.set_title(
            name,
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )

        ax.grid(True, alpha=0.25, color="#dddddd")

        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))

        if len(sources) <= 8:
            ax.legend(fontsize=6, loc="upper right", framealpha=0.7)

    for idx in range(len(avail), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()

    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PDF export
# ══════════════════════════════════════════════════════════════════════════════


def generate_pdf_report(
    logger_dfs: dict[str, pd.DataFrame],
    window: int,
    map_color_col: str,
) -> bytes:

    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:

        fig = plt.figure(figsize=(11.69, 8.27))

        fig.patch.set_facecolor("white")

        plt.axis("off")

        plt.text(
            0.5,
            0.72,
            "CS-MACH1 MeteoTracker Report",
            ha="center",
            va="center",
            fontsize=28,
            color=BRAND_BLUE,
            fontweight="bold",
        )

        plt.text(
            0.5,
            0.60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ha="center",
            fontsize=16,
        )

        plt.text(
            0.5,
            0.50,
            f"Files analyzed: {len(logger_dfs)}",
            ha="center",
            fontsize=16,
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for fname, df in logger_dfs.items():

            stem = Path(fname).stem

            fig_core = make_param_grid(
                df,
                CORE_PARAMS,
                f"Core Parameters — {stem}",
                window,
            )

            if fig_core:
                pdf.savefig(fig_core, bbox_inches="tight")
                plt.close(fig_core)

            fig_aq = make_param_grid(
                df,
                AQ_PARAMS,
                f"Air Quality — {stem}",
                window,
            )

            if fig_aq:
                pdf.savefig(fig_aq, bbox_inches="tight")
                plt.close(fig_aq)

            fig_pm = make_param_grid(
                df,
                PM_PARAMS,
                f"Particulate Matter — {stem}",
                window,
            )

            if fig_pm:
                pdf.savefig(fig_pm, bbox_inches="tight")
                plt.close(fig_pm)

            fig_ex = make_param_grid(
                df,
                EXTRA_PARAMS,
                f"Extra Parameters — {stem}",
                window,
            )

            if fig_ex:
                pdf.savefig(fig_ex, bbox_inches="tight")
                plt.close(fig_ex)

        mean_rows = []

        for fname, df in logger_dfs.items():

            df2 = df.copy()

            df2["date"] = df2["Time"].dt.normalize()

            daily = (
                df2.groupby("date")[ALL_PARAM_COLS]
                .mean(numeric_only=True)
                .reset_index()
            )

            daily["source"] = Path(fname).stem

            mean_rows.append(daily)

        all_means = (
            pd.concat(mean_rows, ignore_index=True)
            if mean_rows else pd.DataFrame()
        )

        if not all_means.empty:

            fig_sc = make_summary_figure(
                all_means,
                CORE_PARAMS,
                "Core Parameters — Daily Mean",
            )

            if fig_sc:
                pdf.savefig(fig_sc, bbox_inches="tight")
                plt.close(fig_sc)

            fig_sa = make_summary_figure(
                all_means,
                AQ_PARAMS,
                "Air Quality — Daily Mean",
            )

            if fig_sa:
                pdf.savefig(fig_sa, bbox_inches="tight")
                plt.close(fig_sa)

            fig_sp = make_summary_figure(
                all_means,
                PM_PARAMS,
                "Particulate Matter — Daily Mean",
            )

            if fig_sp:
                pdf.savefig(fig_sp, bbox_inches="tight")
                plt.close(fig_sp)

    pdf_buffer.seek(0)

    return pdf_buffer.read()

# ══════════════════════════════════════════════════════════════════════════════
# Processing
# ══════════════════════════════════════════════════════════════════════════════

if start_button and "uploaded_files" in st.session_state:

    raw_files = st.session_state["uploaded_files"]

    total = len(raw_files)

    logger_dfs: dict[str, pd.DataFrame] = {}

    progress = st.progress(0, text="Starting…")

    for i, f in enumerate(raw_files):

        progress.progress(
            int(i / total * 100),
            text=f"Processing {f.name} …",
        )

        try:
            logger_dfs[f.name] = parse_airlog_csv(f)

        except Exception as exc:
            st.warning(f"⚠️ Failed to parse {f.name}: {exc}")

    progress.progress(100, text="✅ Done!")

    if not logger_dfs:
        st.error("No valid files found.")
        st.stop()

    st.session_state["logger_dfs"] = logger_dfs
    st.session_state["window_size"] = window_size
    st.session_state["map_color_param"] = map_color_param

# ══════════════════════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════════════════════

if "logger_dfs" in st.session_state:

    logger_dfs: dict[str, pd.DataFrame] = st.session_state["logger_dfs"]

    win = st.session_state.get("window_size", window_size)

    map_col = st.session_state.get("map_color_param", map_color_param)

    for fname, df in logger_dfs.items():

        stem = Path(fname).stem

        st.markdown(f"### 📄 {fname}")

        st.markdown(
            f"<div style='font-size:13px;color:{BRAND_BLUE};font-weight:600;'>"
            f"Session: {str(df['Time'].iloc[0])[:19]} → {str(df['Time'].iloc[-1])[:19]} "
            f"· {len(df)} samples</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        avail_core = available(df, CORE_PARAMS)

        cols = st.columns(max(len(avail_core), 1))

        for c, (col, name, unit, _) in zip(cols, avail_core):
            c.metric(name, f"{df[col].mean():.2f} {unit}")

        st.markdown("#### 🌡️ Core Parameters")

        fig_core = make_param_grid(
            df,
            CORE_PARAMS,
            f"Core — {stem}",
            win,
        )

        if fig_core:
            st.pyplot(fig_core)
            dl_btn(fig_core, f"{stem}_core.png", "⬇️ Download Core PNG")
            plt.close(fig_core)

        st.markdown("#### 🏭 Air Quality")

        fig_aq = make_param_grid(
            df,
            AQ_PARAMS,
            f"Air Quality — {stem}",
            win,
        )

        if fig_aq:
            st.pyplot(fig_aq)
            dl_btn(fig_aq, f"{stem}_aq.png", "⬇️ Download AQ PNG")
            plt.close(fig_aq)
        else:
            st.info("Air quality sensors not active in this file.")

        st.markdown("#### 💨 Particulate Matter")

        fig_pm = make_param_grid(
            df,
            PM_PARAMS,
            f"Particulate Matter — {stem}",
            win,
        )

        if fig_pm:
            st.pyplot(fig_pm)
            dl_btn(fig_pm, f"{stem}_pm.png", "⬇️ Download PM PNG")
            plt.close(fig_pm)
        else:
            st.info("PM sensors not active in this file.")

        fig_ex = make_param_grid(
            df,
            EXTRA_PARAMS,
            f"Extra Parameters — {stem}",
            win,
        )

        if fig_ex:
            st.markdown("#### 🔬 Additional Parameters")
            st.pyplot(fig_ex)
            dl_btn(fig_ex, f"{stem}_extra.png", "⬇️ Download Extra PNG")
            plt.close(fig_ex)

        st.markdown("#### 🗺️ Interactive GPS Map")

        fmap = make_folium_map(df, map_col)

        if fmap:
            col_name = next((p[1] for p in CORE_PARAMS if p[0] == map_col), map_col)

            st.caption(f"Colour = {col_name} · click any point for values")

            st_folium(
                fmap,
                width="100%",
                height=430,
                returned_objects=[],
            )

        else:
            st.info("No GPS data in this file.")

        st.divider()

    st.markdown("## 📊 Summary — All Sessions")

    mean_rows = []

    for fname, df in logger_dfs.items():

        df2 = df.copy()

        df2["date"] = df2["Time"].dt.normalize()

        daily = (
            df2.groupby("date")[ALL_PARAM_COLS]
            .mean(numeric_only=True)
            .reset_index()
        )

        daily["source"] = Path(fname).stem

        mean_rows.append(daily)

    all_means = (
        pd.concat(mean_rows, ignore_index=True)
        if mean_rows else pd.DataFrame()
    )

    rows = []

    for fname, df in logger_dfs.items():

        row = {
            "File": fname,
            "Start": str(df["Time"].iloc[0])[:19],
            "End": str(df["Time"].iloc[-1])[:19],
            "N": len(df),
        }

        for col, name, unit, _ in CORE_PARAMS:
            if col in df.columns and df[col].notna().any():
                row[f"{name} ({unit})"] = round(df[col].mean(), 2)

        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if not all_means.empty:

        st.markdown("#### 🌡️ Core Parameters — Daily Mean")

        fig_sc = make_summary_figure(
            all_means,
            CORE_PARAMS,
            "Core Parameters — Daily Mean (all sessions)",
        )

        if fig_sc:
            st.pyplot(fig_sc)
            dl_btn(fig_sc, "summary_core.png", "⬇️ Download Summary Core PNG")
            plt.close(fig_sc)

        fig_sa = make_summary_figure(
            all_means,
            AQ_PARAMS,
            "Air Quality — Daily Mean (all sessions)",
        )

        if fig_sa:
            st.markdown("#### 🏭 Air Quality — Daily Mean")
            st.pyplot(fig_sa)
            dl_btn(fig_sa, "summary_airquality.png", "⬇️ Download AQ Summary PNG")
            plt.close(fig_sa)

        fig_sp = make_summary_figure(
            all_means,
            PM_PARAMS,
            "Particulate Matter — Daily Mean (all sessions)",
        )

        if fig_sp:
            st.markdown("#### 💨 Particulate Matter — Daily Mean")
            st.pyplot(fig_sp)
            dl_btn(fig_sp, "summary_pm.png", "⬇️ Download PM Summary PNG")
            plt.close(fig_sp)

    if len(logger_dfs) > 1:

        combined = pd.concat(list(logger_dfs.values()), ignore_index=True)

        combined_map = make_folium_map(combined, map_col)

        if combined_map:
            st.markdown("#### 🗺️ Combined GPS Tracks — All Sessions")

            st_folium(
                combined_map,
                width="100%",
                height=500,
                returned_objects=[],
            )

    st.markdown("## 📄 Export Report")

    with st.spinner("Generating PDF report..."):

        pdf_bytes = generate_pdf_report(
            logger_dfs=logger_dfs,
            window=win,
            map_color_col=map_col,
        )

    st.download_button(
        label="⬇️ Download Full PDF Report",
        data=pdf_bytes,
        file_name=f"CS_MACH1_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Additional climatology plots
# ══════════════════════════════════════════════════════════════════════════════

if "logger_dfs" in st.session_state:
    st.markdown("## 🌈 Temperature Climatology")

    combined_frames = []
    for fname, df in logger_dfs.items():
        if "Temp[°C]" in df.columns:
            tmp = df.copy()
            tmp["source"] = Path(fname).stem
            combined_frames.append(tmp)

    if combined_frames:
        combined_df = pd.concat(combined_frames, ignore_index=True)

        combined_df["Time"] = pd.to_datetime(combined_df["Time"])
        combined_df["Hour"] = combined_df["Time"].dt.hour
        combined_df["day_of_year"] = combined_df["Time"].dt.dayofyear

        min_temp = combined_df["Temp[°C]"].min()
        max_temp = combined_df["Temp[°C]"].max()

        cmap = plt.colormaps["rainbow"]
        norm = mcolors.Normalize(vmin=min_temp, vmax=max_temp)

        # ────────────────────────────────────────────────────────────────────
        # Scatter: Day of Year vs Hour
        # ────────────────────────────────────────────────────────────────────
        fig_hour, ax = plt.subplots(figsize=(15, 7))

        scatter = ax.scatter(
            combined_df["day_of_year"],
            combined_df["Hour"],
            c=combined_df["Temp[°C]"],
            cmap=cmap,
            norm=norm,
            s=18,
            alpha=0.75,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Temp [°C]")

        ax.set_xlabel("Day of Year (1–366)")
        ax.set_ylabel("Hour of Day")
        ax.set_title("Temperature vs Day of Year and Hour")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 367)
        ax.set_ylim(0, 24)

        fig_hour.tight_layout()

        st.pyplot(fig_hour)
        dl_btn(
            fig_hour,
            "temperature_dayofyear_hour.png",
            "⬇️ Download DayOfYear-Hour Scatter PNG",
        )

        plt.close(fig_hour)

        # ────────────────────────────────────────────────────────────────────
        # Scatter: Day of Year vs Temperature
        # ────────────────────────────────────────────────────────────────────
        fig_temp, ax = plt.subplots(figsize=(15, 7))

        scatter2 = ax.scatter(
            combined_df["day_of_year"],
            combined_df["Temp[°C]"],
            c=combined_df["Temp[°C]"],
            cmap=cmap,
            norm=norm,
            s=18,
            alpha=0.75,
        )

        cbar2 = plt.colorbar(scatter2, ax=ax)
        cbar2.set_label("Temp [°C]")

        ax.set_xlabel("Day of Year (1–366)")
        ax.set_ylabel("Temperature [°C]")
        ax.set_title("Temperature Distribution by Day of Year")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 367)
        ax.set_ylim(min_temp, max_temp)

        fig_temp.tight_layout()

        st.pyplot(fig_temp)
        dl_btn(
            fig_temp,
            "temperature_dayofyear_temp.png",
            "⬇️ Download DayOfYear-Temperature Scatter PNG",
        )

        plt.close(fig_temp)

# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")

cs_mach1_footer(
    "CS-MACH1 MeteoTracker Pipeline"
)

