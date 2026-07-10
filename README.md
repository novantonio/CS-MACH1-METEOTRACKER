# CS-MACH1 AirLogger Environmental Data Explorer

Streamlit app for the [CS-MACH1](https://cordis.europa.eu/project/id/101214613) citizen
science project: upload one or more AirLogger CSV files and interactively explore
temperature, humidity, pressure, wind speed and other environmental parameters
recorded during a session, with rolling-mean smoothing and optional GPS trajectory
mapping (Cartopy).

## Features

- Multi-file upload (`st.file_uploader`, `accept_multiple_files=True`)
- Per-file parsing with numeric coercion and timestamp cleaning
- 3x3 parameter grid (raw signal + rolling mean + mean/median reference lines),
  plotted against time or, if available, longitude
- Cartopy trajectory map (start/end markers) when `Lat`/`Lon` columns are present
- Cross-session summary table with per-parameter means, downloadable as CSV
- Per-file parsed CSV export

## Project structure

```
CS-MACH1-AIRLOGGER/
├── app.py                # Streamlit entrypoint (UI + session_state)
├── src/
│   └── airlogger.py       # Framework-agnostic parsing/plotting (reusable in Colab)
├── requirements.txt
└── README.md
```

## Expected CSV columns

`Time` plus any of: `Temp[°C]`, `Hum[%]`, `Alt[m]`, `Press[mbar]`, `DP[°C]`,
`θ[K]`, `HDX[°C]`, `Speed[km/h]`, `Radiation[]`, and optionally `Lat`/`Lon`
for trajectory mapping.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

`src/airlogger.py` has no Streamlit dependency, so the same parsing/plotting
functions can be reused directly in a Colab notebook or batch script — only
`app.py` handles upload, session state, and layout.
