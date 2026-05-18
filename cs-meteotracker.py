"""
app.py
------
CS-MACH1 AirLogger pipeline — Streamlit single-file app.

Layout per file
---------------
4 × 4 matplotlib figure:
  [0,0]  Static map snapshot (CartoDB Positron via contextily)
  [0,1]  Temp[°C]
  [0,2]  Hum[%]
  [0,3]  Press[mbar]
  [1,0]  DP[°C]
  [1,1]  θ[K]
  [1,2]  HDX[°C]
  [1,3]  Radiation[]
  [2,0]  Alt[m]        ← step-plot only, no rolling mean
  [2,1]  Speed[km/h]   ← raw + Beaufort arrows, no rolling mean
  [2,2]  Extra A       (Air density or first available extra)
  [2,3]  Extra B       (second extra or blank)
  [3,0–3] Air Quality / PM panels (filled dynamically if sensors active)

After all files
---------------
  • Cross-session summary: daily-mean per parameter group
  • Interactive folium GPS map (colour by param, click popup)
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import contextily as ctx
import folium
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pyproj import Transformer
from streamlit_folium import st_folium

from cs_mach1_theme import apply_cs_mach1_theme, cs_mach1_footer

warnings.filterwarnings("ignore")
matplotlib.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════════════════════════════════════════
# Branding
# ══════════════════════════════════════════════════════════════════════════════

apply_cs_mach1_theme(
    page_title  = "CS-MACH1 AirLogger Pipeline",
    page_icon   = "logo.png",
    main_title  = "🌬️ CS-MACH1: AirLogger Environmental Data Explorer",
    subtitle    = (
        "Upload AirLogger CSV files · 4×4 dashboard (map + 9 params + Beaufort) "
        "· Air Quality · Particulate Matter · PNG export · Cross-session summary"
    ),
    logo_path   = "logo.png",
    logo_width  = 200,
)

# ══════════════════════════════════════════════════════════════════════════════
# Parameter definitions
# ══════════════════════════════════════════════════════════════════════════════

CORE_PARAMS = [
    ("Temp[°C]",    "Temperature",     "°C",   "steelblue"),
    ("Hum[%]",      "Humidity",        "%",    "mediumseagreen"),
    ("Press[mbar]", "Pressure",        "mbar", "mediumpurple"),
    ("DP[°C]",      "Dew Point",       "°C",   "cadetblue"),
    ("θ[K]",        "Pot. Temp θ",     "K",    "sandybrown"),
    ("HDX[°C]",     "Heat Discomfort", "°C",   "indianred"),
    ("Radiation[]", "Radiation",       "a.u.", "goldenrod"),
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
    ("mass PM1.0[μg/m3]",    "PM1.0 mass",     "μg/m³", "coral"),
    ("mass PM2.5[μg/m3]",    "PM2.5 mass",     "μg/m³", "orangered"),
    ("mass PM4[μg/m3]",      "PM4 mass",       "μg/m³", "tomato"),
    ("mass PM10[μg/m3]",     "PM10 mass",      "μg/m³", "crimson"),
    ("number PM0.5[#/cm3]",  "PM0.5 count",    "#/cm³", "peru"),
    ("number PM1.0[#/cm3]",  "PM1.0 count",    "#/cm³", "chocolate"),
    ("number PM2.5[#/cm3]",  "PM2.5 count",    "#/cm³", "sienna"),
    ("number PM10[#/cm3]",   "PM10 count",     "#/cm³", "maroon"),
    ("Typical Part Size[μm]","Typical P. Size", "μm",   "rosybrown"),
]

EXTRA_PARAMS = [
    ("Air density[kg/m3]", "Air Density", "kg/m³", "teal"),
    ("Tir[°C]",            "Tir",         "°C",    "salmon"),
    ("Tsur[°C]",           "Tsur",        "°C",    "palevioletred"),
    ("EAQ[]",              "EAQ",         "",      "seagreen"),
    ("FAQ[]",              "FAQ",         "",      "darkolivegreen"),
]

ALL_PARAMS     = CORE_PARAMS + AQ_PARAMS + PM_PARAMS + EXTRA_PARAMS
ALL_PARAM_COLS = [p[0] for p in ALL_PARAMS] + ["Alt[m]", "Speed[km/h]"]

_PARAM_LOOKUP  = {p[0]: p for p in ALL_PARAMS}

# Brand colour for plot accents
BRAND_BLUE = "#00A6D6"

# ══════════════════════════════════════════════════════════════════════════════
# Beaufort helpers
# ══════════════════════════════════════════════════════════════════════════════

_BFT_THRESHOLDS = [1, 5, 11, 19, 28, 38, 49, 61, 74, 88, 102, 117]
_BFT_COLORS = [
    "#d0f0ff", "#b0e0ff", "#80d0f0", "#50c0e0", "#20a0d0",
    "#1080b0", "#005090", "#003060", "#ff9900", "#ff6000",
    "#ff2000", "#cc0000", "#800000",
]
_BFT_LABELS = [
    "Calm", "Light air", "Light breeze", "Gentle breeze", "Moderate breeze",
    "Fresh breeze", "Strong breeze", "Near gale", "Gale",
    "Severe gale", "Storm", "Violent storm", "Hurricane",
]

def kmh_to_beaufort(speed: float) -> int:
    speed = max(0.0, float(speed))
    for b, t in enumerate(_BFT_THRESHOLDS):
        if speed < t:
            return b
    return 12

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
# Parsing
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


def available(df: pd.DataFrame, param_list: list) -> list:
    return [p for p in param_list if p[0] in df.columns and df[p[0]].notna().any()]


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def dl_btn(fig: plt.Figure, filename: str, label: str = "⬇️ Download PNG"):
    st.download_button(label, data=fig_to_png_bytes(fig),
                       file_name=filename, mime="image/png")

# ══════════════════════════════════════════════════════════════════════════════
# Panel plotters
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax):
    """Apply CS-MACH1 brand accent to axis spines."""
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.25, color="#dddddd")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def _plot_generic(ax, df, col, name, unit, color, window):
    series  = df[col].dropna()
    if series.empty:
        ax.set_visible(False); return
    times   = df.loc[series.index, "Time"]
    rolling = series.rolling(window=window, min_periods=1).mean()
    p_mean, p_med = series.mean(), series.median()
    u = f" ({unit})" if unit else ""

    ax.plot(times, series, alpha=0.28, linewidth=0.7, color=color, label="Raw")
    ax.plot(times, rolling, linewidth=2, color="#E8524A", label=f"Rolling (w={window})")
    ax.axhline(p_mean, color="crimson",    linewidth=1.3, linestyle="--",
               label=f"Mean {p_mean:.2f}{u}")
    ax.axhline(p_med,  color="darkorange", linewidth=1.3, linestyle="--",
               label=f"Med  {p_med:.2f}{u}")
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.7)
    ax.set_ylabel(f"{name}{u}", fontsize=8)
    ax.set_title(name, fontsize=9, fontweight="bold", color="#333333")
    _style_ax(ax)


def _plot_altitude(ax, df):
    col = "Alt[m]"
    if col not in df.columns or df[col].isna().all():
        ax.set_visible(False); return
    series = df[col].dropna()
    times  = df.loc[series.index, "Time"]

    ax.step(times, series, where="mid", linewidth=1.3, color="slategray", label="Altitude")
    ax.fill_between(times, series, step="mid", alpha=0.18, color="slategray")
    ax.axhline(series.mean(),   color="crimson",    linewidth=1.3, linestyle="--",
               label=f"Mean {series.mean():.1f} m")
    ax.axhline(series.median(), color="darkorange", linewidth=1.3, linestyle="--",
               label=f"Med  {series.median():.1f} m")
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.7)
    ax.set_ylabel("Altitude (m)", fontsize=8)
    ax.set_title("Altitude", fontsize=9, fontweight="bold", color="#333333")
    _style_ax(ax)


def _plot_speed_beaufort(ax, df):
    col = "Speed[km/h]"
    if col not in df.columns or df[col].isna().all():
        ax.set_visible(False); return

    series = df[col].dropna().clip(lower=0)
    times  = df.loc[series.index, "Time"]
    t_num  = mdates.date2num(times)

    ax.plot(times, series, linewidth=1.2, color="royalblue", alpha=0.55, label="Speed")
    ax.axhline(series.mean(),   color="crimson",    linewidth=1.3, linestyle="--",
               label=f"Mean {series.mean():.1f} km/h")
    ax.axhline(series.median(), color="darkorange", linewidth=1.3, linestyle="--",
               label=f"Med  {series.median():.1f} km/h")

    ymax = max(series.max() * 1.40, 10)
    arrow_base = ymax * 0.76

    step = max(1, len(series) // 20)
    for i in range(0, len(series), step):
        spd = series.iloc[i]
        b   = kmh_to_beaufort(spd)
        c   = _BFT_COLORS[b]
        xv  = t_num[series.index[i]]
        arrow_len = (b / 12) * ymax * 0.18 + ymax * 0.03
        ax.annotate(
            "", xy=(xv, arrow_base + arrow_len), xytext=(xv, arrow_base),
            arrowprops=dict(
                arrowstyle="-|>", color=c,
                lw=1.5 + b * 0.15, mutation_scale=8 + b,
            ),
        )
        ax.text(xv, arrow_base - ymax * 0.05, f"B{b}",
                ha="center", va="top", fontsize=5.5,
                color=c, fontweight="bold")

    ax.set_ylim(0, ymax)
    ax.set_ylabel("Speed (km/h)", fontsize=8)
    ax.set_title("Wind Speed + Beaufort", fontsize=9, fontweight="bold", color="#333333")

    unique_b = sorted({kmh_to_beaufort(v) for v in series})
    patches  = [mpatches.Patch(color=_BFT_COLORS[b], label=f"B{b} {_BFT_LABELS[b]}")
                for b in unique_b]
    ax.legend(handles=patches, fontsize=5.5, loc="upper right",
              title="Beaufort", title_fontsize=6, framealpha=0.75)
    _style_ax(ax)

# ══════════════════════════════════════════════════════════════════════════════
# Static map panel (CartoDB Positron via contextily)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_static_map(ax, df, color_col: str):
    if "Lat" not in df.columns or "Lon" not in df.columns:
        ax.text(0.5, 0.5, "No GPS data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title("GPS Track", fontsize=9, fontweight="bold")
        return

    gdf = df.dropna(subset=["Lat", "Lon"])
    if gdf.empty:
        ax.text(0.5, 0.5, "No GPS data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title("GPS Track", fontsize=9, fontweight="bold")
        return

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(gdf["Lon"].values, gdf["Lat"].values)

    if color_col in gdf.columns and gdf[color_col].notna().any():
        vals    = gdf[color_col].fillna(gdf[color_col].mean()).values
        vmin, vmax = vals.min(), vals.max()
        norm    = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_fn = cm.get_cmap("RdYlBu_r")
        colors  = [cmap_fn(norm(v)) for v in vals]
        col_label = next((p[1] for p in ALL_PARAMS if p[0] == color_col), color_col)
        col_unit  = next((p[2] for p in ALL_PARAMS if p[0] == color_col), "")
    else:
        colors    = [BRAND_BLUE] * len(xs)
        vmin = vmax = col_label = col_unit = None

    ax.plot(xs, ys, color="#aaaaaa", linewidth=1.2, alpha=0.6, zorder=1)
    ax.scatter(xs, ys, c=colors, s=20, zorder=2, linewidths=0)
    ax.plot(xs[0],  ys[0],  marker="^", color="limegreen", markersize=10,
            zorder=3, markeredgecolor="black", markeredgewidth=0.7, label="Start")
    ax.plot(xs[-1], ys[-1], marker="s", color="red",       markersize=10,
            zorder=3, markeredgecolor="black", markeredgewidth=0.7, label="End")

    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.CartoDB.Positron,
                        zoom="auto", attribution=False)
    except Exception:
        ax.set_facecolor("#f2f0eb")

    ax.set_axis_off()
    ax.set_title("GPS Track — CartoDB Positron", fontsize=9,
                 fontweight="bold", color="#333333")
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85)

    if vmin is not None and vmax is not None and vmax != vmin:
        sm = cm.ScalarMappable(cmap="RdYlBu_r",
                               norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.032, pad=0.01)
        u = f" {col_unit}" if col_unit else ""
        cbar.set_label(f"{col_label}{u}", fontsize=6.5)
        cbar.ax.tick_params(labelsize=6)

# ══════════════════════════════════════════════════════════════════════════════
# 4×4 dashboard figure
# ══════════════════════════════════════════════════════════════════════════════

_GRID_LAYOUT = [
    (0, 0, "__MAP__"),
    (0, 1, "Temp[°C]"),
    (0, 2, "Hum[%]"),
    (0, 3, "Press[mbar]"),
    (1, 0, "DP[°C]"),
    (1, 1, "θ[K]"),
    (1, 2, "HDX[°C]"),
    (1, 3, "Radiation[]"),
    (2, 0, "__ALT__"),
    (2, 1, "__SPEED__"),
    (2, 2, "__EXTRA_A__"),
    (2, 3, "__EXTRA_B__"),
    (3, 0, "__AQ_PM_0__"),
    (3, 1, "__AQ_PM_1__"),
    (3, 2, "__AQ_PM_2__"),
    (3, 3, "__AQ_PM_3__"),
]


def make_4x4_figure(df: pd.DataFrame, stem: str, window: int,
                    map_color_col: str) -> plt.Figure:
    avail_aq_pm = [p for p in (AQ_PARAMS + PM_PARAMS)
                   if p[0] in df.columns and df[p[0]].notna().any()]
    avail_extra = [p for p in EXTRA_PARAMS
                   if p[0] in df.columns and df[p[0]].notna().any()]
    aq_pm_queue = list(avail_aq_pm)
    extra_queue = list(avail_extra)

    fig, axes = plt.subplots(
        4, 4,
        figsize=(22, 18),
        gridspec_kw={"hspace": 0.48, "wspace": 0.32},
    )
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(
        stem, fontsize=15, fontweight="bold",
        color=BRAND_BLUE, y=1.005,
    )

    for row, col_idx, key in _GRID_LAYOUT:
        ax = axes[row][col_idx]
        ax.set_facecolor("#fafafa")

        if key == "__MAP__":
            _plot_static_map(ax, df, map_color_col)

        elif key == "__ALT__":
            _plot_altitude(ax, df)

        elif key == "__SPEED__":
            _plot_speed_beaufort(ax, df)

        elif key == "__EXTRA_A__":
            if extra_queue:
                p = extra_queue.pop(0)
                _plot_generic(ax, df, p[0], p[1], p[2], p[3], window)
            else:
                ax.set_visible(False)

        elif key == "__EXTRA_B__":
            if extra_queue:
                p = extra_queue.pop(0)
                _plot_generic(ax, df, p[0], p[1], p[2], p[3], window)
            else:
                ax.set_visible(False)

        elif key.startswith("__AQ_PM_"):
            if aq_pm_queue:
                p = aq_pm_queue.pop(0)
                _plot_generic(ax, df, p[0], p[1], p[2], p[3], window)
            else:
                ax.set_visible(False)

        else:
            if key in _PARAM_LOOKUP and key in df.columns and df[key].notna().any():
                p = _PARAM_LOOKUP[key]
                _plot_generic(ax, df, p[0], p[1], p[2], p[3], window)
            else:
                ax.set_visible(False)

    return fig

# ══════════════════════════════════════════════════════════════════════════════
# Interactive folium GPS map
# ══════════════════════════════════════════════════════════════════════════════

def make_folium_map(df: pd.DataFrame, color_col: str) -> folium.Map | None:
    if "Lat" not in df.columns or "Lon" not in df.columns:
        return None
    gdf = df.dropna(subset=["Lat", "Lon"])
    if gdf.empty:
        return None

    center = [gdf["Lat"].mean(), gdf["Lon"].mean()]
    m = folium.Map(location=center, zoom_start=15, tiles="CartoDB positron")

    if color_col in gdf.columns and gdf[color_col].notna().any():
        vals       = gdf[color_col].fillna(gdf[color_col].mean())
        vmin, vmax = vals.min(), vals.max()
        norm       = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_fn    = cm.get_cmap("RdYlBu_r")
        hex_colors = [mcolors.to_hex(cmap_fn(norm(v))) for v in vals]
        col_label  = next((p[1] for p in ALL_PARAMS if p[0] == color_col), color_col)
        col_unit   = next((p[2] for p in ALL_PARAMS if p[0] == color_col), "")
    else:
        hex_colors = [BRAND_BLUE] * len(gdf)
        vmin = vmax = col_label = col_unit = None

    coords = list(zip(gdf["Lat"], gdf["Lon"]))
    folium.PolyLine(coords, color="#aaaaaa", weight=2, opacity=0.5).add_to(m)

    for (lat, lon), hc, (_, row) in zip(coords, hex_colors, gdf.iterrows()):
        spd   = row.get("Speed[km/h]", None)
        b_str = f" · B{kmh_to_beaufort(spd)} {_BFT_LABELS[kmh_to_beaufort(spd)]}" \
                if pd.notna(spd) else ""
        parts = [f"<b>Time:</b> {row['Time']}{b_str}"]
        for c, name, unit, _ in CORE_PARAMS + [
            ("Alt[m]", "Altitude", "m", ""),
            ("Speed[km/h]", "Speed", "km/h", ""),
        ]:
            if c in row.index and pd.notna(row[c]):
                parts.append(f"<b>{name}:</b> {row[c]:.2f} {unit}")
        folium.CircleMarker(
            location=[lat, lon], radius=5,
            color=hc, fill=True, fill_color=hc, fill_opacity=0.85,
            popup=folium.Popup("<br>".join(parts), max_width=260),
        ).add_to(m)

    folium.Marker(coords[0],  popup="▶ Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(coords[-1], popup="■ End",
                  icon=folium.Icon(color="red",   icon="stop")).add_to(m)

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
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

    return m

# ══════════════════════════════════════════════════════════════════════════════
# Cross-session summary figure
# ══════════════════════════════════════════════════════════════════════════════

def make_summary_figure(all_means: pd.DataFrame, param_list: list,
                        title: str, ncols: int = 3) -> plt.Figure | None:
    avail = [p for p in param_list
             if p[0] in all_means.columns and all_means[p[0]].notna().any()]
    if not avail:
        return None

    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                              squeeze=False)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(title, fontsize=13, fontweight="bold", color=BRAND_BLUE, y=1.01)

    sources = all_means["source"].unique()
    palette = cm.tab10(np.linspace(0, 1, max(len(sources), 1)))
    cmap    = dict(zip(sources, palette))

    for idx, (col, name, unit, _) in enumerate(avail):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor("#fafafa")
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
        ax.set_title(name, fontsize=9, fontweight="bold", color="#333333")
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
        st.error("No valid files found."); st.stop()

    st.session_state["logger_dfs"]      = logger_dfs
    st.session_state["window_size"]     = window_size
    st.session_state["map_color_param"] = map_color_param

# ══════════════════════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════════════════════

if "logger_dfs" in st.session_state:
    logger_dfs: dict[str, pd.DataFrame] = st.session_state["logger_dfs"]
    win     = st.session_state.get("window_size", window_size)
    map_col = st.session_state.get("map_color_param", map_color_param)

    # ── Per-file ──────────────────────────────────────────────────────────────
    for fname, df in logger_dfs.items():
        stem = Path(fname).stem

        st.markdown(f"### 📄 {fname}")
        st.markdown(
            f"<div style='font-size:13px;color:{BRAND_BLUE};font-weight:600;'>"
            f"Session: {str(df['Time'].iloc[0])[:19]}  →  {str(df['Time'].iloc[-1])[:19]} "
            f"· {len(df)} samples</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Quick metrics
        avail_core = available(df, CORE_PARAMS)
        cols = st.columns(max(len(avail_core), 1))
        for c, (col, name, unit, _) in zip(cols, avail_core):
            c.metric(name, f"{df[col].mean():.2f} {unit}")

        st.markdown("<br>", unsafe_allow_html=True)

        # 4×4 dashboard
        with st.spinner("Rendering 4×4 dashboard…"):
            fig44 = make_4x4_figure(df, stem=stem, window=win,
                                    map_color_col=map_col)
        st.pyplot(fig44)
        dl_btn(fig44, f"{stem}_dashboard.png", "⬇️ Download 4×4 Dashboard PNG")
        plt.close(fig44)

        # AQ / PM status
        avail_aq = available(df, AQ_PARAMS)
        avail_pm = available(df, PM_PARAMS)
        if not avail_aq:
            st.info("ℹ️ Air quality sensors not active in this file (CO₂, O₃, CO, SO₂, NO₂, AQI).")
        if not avail_pm:
            st.info("ℹ️ Particulate matter sensors not active in this file (PM1.0 – PM10).")

        # Interactive map
        st.markdown("#### 🗺️ Interactive GPS Map")
        fmap = make_folium_map(df, map_col)
        if fmap:
            col_name = next((p[1] for p in CORE_PARAMS if p[0] == map_col), map_col)
            st.caption(f"Colour = {col_name} · click any point for values + Beaufort scale")
            st_folium(fmap, width="100%", height=430, returned_objects=[])
        else:
            st.info("No GPS data in this file.")

        st.divider()

    # ── Summary ───────────────────────────────────────────────────────────────
    st.markdown(f"## 📊 Summary — All Sessions")

    mean_rows = []
    for fname, df in logger_dfs.items():
        df2          = df.copy()
        df2["date"]  = df2["Time"].dt.normalize()
        daily        = df2.groupby("date")[ALL_PARAM_COLS].mean(numeric_only=True).reset_index()
        daily["source"] = Path(fname).stem
        mean_rows.append(daily)
    all_means = pd.concat(mean_rows, ignore_index=True) if mean_rows else pd.DataFrame()

    # Statistics table
    rows = []
    for fname, df in logger_dfs.items():
        row = {
            "File":  fname,
            "Start": str(df["Time"].iloc[0])[:19],
            "End":   str(df["Time"].iloc[-1])[:19],
            "N":     len(df),
        }
        for col, name, unit, _ in CORE_PARAMS + [
            ("Alt[m]",      "Alt",   "m",    ""),
            ("Speed[km/h]", "Speed", "km/h", ""),
        ]:
            if col in df.columns and df[col].notna().any():
                row[f"{name} ({unit})"] = round(df[col].mean(), 2)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if not all_means.empty:
        all_core = CORE_PARAMS + [
            ("Alt[m]",      "Altitude",   "m",    "slategray"),
            ("Speed[km/h]", "Wind Speed", "km/h", "royalblue"),
        ]
        st.markdown("#### 🌡️ Core Parameters — Daily Mean")
        fig_sc = make_summary_figure(all_means, all_core,
                                     "Core Parameters — Daily Mean (all sessions)")
        if fig_sc:
            st.pyplot(fig_sc)
            dl_btn(fig_sc, "summary_core.png", "⬇️ Download Summary Core PNG")
            plt.close(fig_sc)

        fig_sa = make_summary_figure(all_means, AQ_PARAMS,
                                     "Air Quality — Daily Mean (all sessions)")
        if fig_sa:
            st.markdown("#### 🏭 Air Quality — Daily Mean")
            st.pyplot(fig_sa)
            dl_btn(fig_sa, "summary_airquality.png", "⬇️ Download AQ Summary PNG")
            plt.close(fig_sa)

        fig_sp = make_summary_figure(all_means, PM_PARAMS,
                                     "Particulate Matter — Daily Mean (all sessions)")
        if fig_sp:
            st.markdown("#### 💨 Particulate Matter — Daily Mean")
            st.pyplot(fig_sp)
            dl_btn(fig_sp, "summary_pm.png", "⬇️ Download PM Summary PNG")
            plt.close(fig_sp)

    # Combined interactive map
    if len(logger_dfs) > 1:
        combined     = pd.concat(list(logger_dfs.values()), ignore_index=True)
        combined_map = make_folium_map(combined, map_col)
        if combined_map:
            st.markdown("#### 🗺️ Combined GPS Tracks — All Sessions")
            st_folium(combined_map, width="100%", height=500, returned_objects=[])

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════

cs_mach1_footer(
    "CS-MACH1 AirLogger Pipeline · Streamlit + Matplotlib + Contextily + Folium"
)
