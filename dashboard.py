"""
dashboard.py — Streamlit SOC dashboard for AegisVision ANPR.

FIXES vs. previous version:
1. The detections table read fields the API never returns (license_plate,
   confidence_score, snapshot_url, clip_url, alert_category) — so every row
   showed UNREADABLE / N/A / —. Now uses the real API fields:
   plate_number, confidence, vehicle_type, evidence_url, camera_id.
2. The "Traffic Enforcement Hub" posted to /alerts/trigger and /telemetry/log,
   endpoints that DON'T EXIST in the backend (guaranteed 404). Replaced with a
   manual detection entry form that posts to the real /detections/add endpoint.
3. Timestamp conversion crashed on tz-aware or unparseable values
   (tz_localize on an already-localized series raises). Now guarded.
4. Metric cards now show real numbers from /analytics instead of a hardcoded
   "142ms" and a fake tunnel label; the traffic chart plots real hourly counts
   instead of [0, 0, 0].

Run with:  streamlit run dashboard.py   (backend must be running on :8000)
"""

import html
import textwrap
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# ============================================================================
# 1. API BACKEND ENDPOINT CONFIGURATION
# ============================================================================
API_BASE_URL = "http://localhost:8000"
HEADERS = {"ngrok-skip-browser-warning": "true"}


def fetch_real_logs():
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/detections", headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return pd.DataFrame(data) if data else None
        st.error(f"⚠️ Could not fetch live logs. Status code: {r.status_code}")
        return None
    except requests.RequestException as e:
        st.error(f"🔌 Failed to connect to log server: {e}")
        return None


def fetch_analytics():
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/analytics", headers=HEADERS, timeout=10)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def post_manual_detection(payload: dict):
    try:
        r = requests.post(f"{API_BASE_URL}/api/v1/detections/add",
                          json=payload, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        st.error(f"⚠️ Backend returned {r.status_code}: {r.text[:300]}")
        return None
    except requests.RequestException as e:
        st.error(f"🔌 Failed to communicate with API server: {e}")
        return None


# ============================================================================
# 2. SOC INTERFACE STYLING — design tokens
# ============================================================================
st.set_page_config(page_title="AegisVision | Command Matrix", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg: #060A10;
        --panel: #0D1420;
        --panel-2: #121B2A;
        --border: #1E2A3A;
        --text: #E8EEF5;
        --muted: #7C8B9E;
        --accent: #2DD4E8;
        --accent-dim: rgba(45, 212, 232, 0.12);
        --amber: #FFB020;
        --red: #FF4D6A;
        --green: #34D399;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: var(--panel);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

    h1, h2, h3, .soc-title { font-family: 'Space Grotesk', sans-serif !important; }

    .soc-topbar {
        position: relative;
        overflow: hidden;
        background-color: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.85rem 1.2rem;
        margin-bottom: 1.4rem;
    }
    .soc-topbar::before {
        content: "";
        position: absolute;
        top: 0; left: -30%;
        width: 30%; height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        animation: scan-sweep 3.2s linear infinite;
        opacity: 0.85;
    }
    @keyframes scan-sweep {
        0%   { left: -30%; }
        100% { left: 100%; }
    }
    .soc-topbar-row { display: flex; align-items: center; justify-content: space-between; }
    .soc-topbar-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        color: var(--muted);
    }
    .soc-topbar-label strong { color: var(--text); letter-spacing: 0.08em; }
    .live-dot {
        display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        background: var(--green); margin-right: 6px;
        box-shadow: 0 0 8px var(--green);
        animation: pulse-dot 1.6s ease-in-out infinite;
    }
    @keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

    .soc-title {
        font-size: 1.7rem; font-weight: 700; color: var(--text);
        margin-bottom: 0.9rem; letter-spacing: -0.01em;
    }

    .metric-row { display: flex; gap: 0.9rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 190px;
        background-color: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.95rem 1.1rem;
        transition: border-color 0.15s ease;
    }
    .metric-card:hover { border-color: var(--accent); }
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem; letter-spacing: 0.07em;
        color: var(--muted); text-transform: uppercase; margin-bottom: 0.35rem;
    }
    .metric-value { font-size: 1.5rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }

    .aegis-table-wrap {
        border: 1px solid var(--border); border-radius: 10px;
        overflow: hidden; margin-bottom: 1rem;
    }
    table.aegis-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
    table.aegis-table thead th {
        background-color: var(--panel-2);
        color: var(--muted);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase;
        text-align: left; padding: 0.65rem 0.9rem; border-bottom: 1px solid var(--border);
    }
    table.aegis-table tbody td {
        padding: 0.6rem 0.9rem; border-bottom: 1px solid var(--border);
        color: var(--text); vertical-align: middle;
    }
    table.aegis-table tbody tr:last-child td { border-bottom: none; }
    table.aegis-table tbody tr:hover { background-color: var(--panel); }
    .mono { font-family: 'JetBrains Mono', monospace; color: var(--muted); font-size: 0.82rem; }

    .chip {
        display: inline-block; font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em;
        padding: 0.22rem 0.55rem; border-radius: 999px; white-space: nowrap;
    }
    .chip-low    { background: rgba(52, 211, 153, 0.12); color: var(--green); border: 1px solid rgba(52, 211, 153, 0.35); }
    .chip-med    { background: rgba(255, 176, 32, 0.12); color: var(--amber); border: 1px solid rgba(255, 176, 32, 0.35); }
    .chip-high   { background: rgba(255, 77, 106, 0.12); color: var(--red); border: 1px solid rgba(255, 77, 106, 0.35); }
    .chip-video  { background: var(--accent-dim); color: var(--accent); border: 1px solid rgba(45, 212, 232, 0.35); }
    .chip-photo  { background: rgba(124, 139, 158, 0.12); color: var(--muted); border: 1px solid var(--border); }
    .chip-none   { background: transparent; color: var(--muted); border: 1px dashed var(--border); }
    .evidence-link { color: var(--accent); text-decoration: none; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
    .evidence-link:hover { text-decoration: underline; }
    .evidence-dash { color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }

    .plate-chip {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem; font-weight: 700; letter-spacing: 0.09em;
        color: var(--bg);
        background: var(--accent);
        padding: 0.28rem 0.6rem;
        border-radius: 4px;
        border: 1px solid var(--accent);
    }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar Route Control
st.sidebar.markdown("## 🛡️ AegisVision")
st.sidebar.caption("Data Science Command Engine")
st.sidebar.divider()
selected_room = st.sidebar.radio(
    "Navigation Control Rooms",
    ["🏠 Executive Dashboard", "🏍️ Traffic Enforcement Hub"],
)

st.markdown(
    f"""<div class="soc-topbar"><div class="soc-topbar-row">
    <div class="soc-topbar-label"><strong>AEGISVISION</strong> &nbsp;·&nbsp; CENTRALIZED API GATEWAY ROUTER</div>
    <div class="soc-topbar-label"><span class="live-dot"></span>ONLINE &nbsp;|&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div></div>""",
    unsafe_allow_html=True,
)

# ============================================================================
# 3. ROUTED CONTROL DASHBOARD ROOMS
# ============================================================================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 30:
    st.session_state.last_refresh = time.time()
    st.rerun()


def license_plate_chip(plate):
    if isinstance(plate, str) and plate.strip():
        return f'<span class="plate-chip">{html.escape(plate.strip().upper())}</span>'
    return '<span class="chip chip-none">UNREADABLE</span>'


def confidence_chip(score):
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return '<span class="chip chip-none">N/A</span>'
    pct = score * 100 if score <= 1 else score
    cls = "chip-low" if pct >= 90 else ("chip-med" if pct >= 75 else "chip-high")
    return f'<span class="chip {cls}">{pct:.1f}%</span>'


def vehicle_chip(vtype):
    if isinstance(vtype, str) and vtype.strip():
        return f'<span class="chip chip-video">{html.escape(vtype.strip().upper())}</span>'
    return '<span class="chip chip-none">—</span>'


def evidence_link(url, label):
    if isinstance(url, str) and url:
        full_url = url if url.startswith("http") else f"{API_BASE_URL}{url}"
        return f'<a class="evidence-link" href="{html.escape(full_url)}" target="_blank">{label}</a>'
    return '<span class="evidence-dash">—</span>'


def render_detection_table(df: pd.DataFrame):
    """FIXED: reads the fields the API actually returns —
    plate_number, vehicle_type, confidence, evidence_url."""
    rows_html = []
    for _, row in df.iterrows():
        ts = html.escape(str(row.get("timestamp", "—")))
        det_id = html.escape(str(row.get("id", "—")))
        cam_id = html.escape(str(row.get("camera_id", "—")))
        row_html = f"""<tr>
<td class="mono">{ts}</td>
<td class="mono">#{det_id}</td>
<td class="mono">{cam_id}</td>
<td>{vehicle_chip(row.get("vehicle_type"))}</td>
<td>{license_plate_chip(row.get("plate_number"))}</td>
<td>{confidence_chip(row.get("confidence"))}</td>
<td>{evidence_link(row.get("evidence_url"), "🖼️ Evidence")}</td>
</tr>"""
        rows_html.append(row_html)

    # No leading whitespace — Markdown treats 4+ leading spaces as a code block
    table_html = textwrap.dedent(f"""\
<div class="aegis-table-wrap">
<table class="aegis-table">
<thead>
<tr>
<th>Time Detected</th>
<th>ID</th>
<th>Cam ID</th>
<th>Vehicle</th>
<th>License Plate</th>
<th>AI Confidence</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</div>
""")
    st.markdown(table_html, unsafe_allow_html=True)


if selected_room == "🏠 Executive Dashboard":
    st.markdown("<div class='soc-title'>🏠 System Executive Dashboard</div>", unsafe_allow_html=True)

    if st.button("🔄 Fetch Live Data"):
        st.rerun()

    with st.spinner("🔄 Streaming live detection logs..."):
        live_df = fetch_real_logs()
        stats = fetch_analytics()

    if live_df is not None and not live_df.empty and "timestamp" in live_df.columns:
        raw_ts = pd.to_datetime(live_df["timestamp"], errors="coerce", utc=True)
        live_df["_hour"] = raw_ts.dt.tz_convert("Asia/Kolkata").dt.floor("h")
        live_df["timestamp"] = (
        raw_ts.dt.tz_convert("Asia/Kolkata").dt.strftime("%b %d, %I:%M %p")
    )

        total = stats["total_detections"] if stats else len(live_df)
        today = stats["today_count"] if stats else "—"
        breakdown = stats["vehicle_breakdown"] if stats else {}

        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-label">Total Detections</div>
                    <div class="metric-value">{total}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Detections Today</div>
                    <div class="metric-value">{today}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Cars / Bikes / Trucks</div>
                    <div class="metric-value" style="font-size:1.05rem;">
                        {breakdown.get("cars", "—")} / {breakdown.get("bikes", "—")} / {breakdown.get("trucks", "—")}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True,
        )

        render_detection_table(live_df.drop(columns=["_hour"]))

        # FIXED: real hourly traffic volume instead of a hardcoded [0, 0, 0]
        hourly = live_df.groupby("_hour").size().rename("Traffic Volume")
        if not hourly.empty:
            st.line_chart(hourly)
    else:
        st.info("🛡️ System Nominal: No detections in the database yet.")


elif selected_room == "🏍️ Traffic Enforcement Hub":
    st.markdown("<div class='soc-title'>🏍️ Traffic Enforcement Hub</div>", unsafe_allow_html=True)
    st.caption("Manually log a detection into the database (same endpoint the AI pipeline uses).")

    # FIXED: previously posted to /alerts/trigger and /telemetry/log, which
    # don't exist in the backend (always 404). Now posts to /detections/add.
    with st.form("manual_detection_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            plate = st.text_input("License Plate", placeholder="MH03DY5705")
            camera_id = st.text_input("Camera ID", value="CAM-01-MAIN")
        with col_b:
            vehicle_type = st.selectbox("Vehicle Type", ["car", "bike", "bus", "truck"])
            confidence = st.slider("Confidence", 0.0, 1.0, 0.90, 0.01)
        evidence = st.text_input("Evidence URL (optional)", placeholder="/static/crops/MH03DY5705.jpg")
        submitted = st.form_submit_button("🚨 Log Detection")

    if submitted:
        if not plate.strip():
            st.warning("Plate number is required.")
        else:
            payload = {
                "camera_id": camera_id.strip() or "CAM-01-MAIN",
                "plate_number": plate.strip().upper(),
                "vehicle_type": vehicle_type,
                "confidence": float(confidence),
                "evidence_url": evidence.strip() or None,
            }
            with st.spinner("⏳ Writing entry to database..."):
                api_response = post_manual_detection(payload)
            if api_response:
                st.success("Entry successfully written to the database!")
                st.json(api_response)