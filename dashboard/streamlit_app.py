"""
streamlit_app.py
----------------
Real-time border surveillance dashboard.

Run:
    streamlit run dashboard/streamlit_app.py

Features:
    • KPI cards   — total alerts, HIGH count, avg score, uptime
    • Priority pie chart
    • Anomaly-score trend line with threshold lines
    • Detection counts bar chart (persons / vehicles / weapons per hour)
    • Alert feed table (sortable, filterable)
    • Sidebar filters — priority, date range
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json
import os
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Border Surveillance AI",
    page_icon  = "🚨",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH = os.environ.get("ALERTS_DB_PATH", "alerts.db")

PRIORITY_COLORS = {
    "HIGH":   "#FF3131",
    "MEDIUM": "#FFD600",
    "LOW":    "#39FF14",
}


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)   # refresh every 30 seconds
def load_alerts(db_path: str = DB_PATH) -> pd.DataFrame:
    if not os.path.exists(db_path):
        # Return empty DataFrame with correct columns when DB doesn't exist yet
        return pd.DataFrame(columns=[
            "id","alert_id","timestamp","alert_type","confidence",
            "anomaly_score","priority","frame_path","location","objects_detected"])
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM alerts ORDER BY timestamp DESC", conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)   # strip tz → naive
        df["hour"]      = df["timestamp"].dt.floor("h") 
        df["objects_list"] = df["objects_detected"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else [])
    return df


def seed_demo_data(db_path: str = DB_PATH, n: int = 80):
    """Insert synthetic alerts so the dashboard is non-empty on first run."""
    import random, uuid
    random.seed(42)
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT, timestamp TEXT, alert_type TEXT,
                confidence REAL, anomaly_score REAL, priority TEXT,
                frame_path TEXT, location TEXT, objects_detected TEXT
            )""")
        count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        if count >= n:
            return
        now = datetime.utcnow()
        rows = []
        for i in range(n):
            score    = random.uniform(0.1, 0.99)
            priority = "HIGH" if score > 0.7 else ("MEDIUM" if score > 0.4 else "LOW")
            ts       = now - timedelta(hours=random.uniform(0, 24))
            rows.append((
                str(uuid.uuid4())[:8].upper(),
                ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                random.choice(["intrusion","crowd_spike","vehicle_surge","motion_anomaly"]),
                round(random.uniform(0.5, 0.99), 3),
                round(score, 3),
                priority,
                f"data/processed/frame_{i:05d}.jpg",
                f"sector_{random.randint(1,6):02d}",
                json.dumps(random.sample(["person","vehicle","weapon"], k=random.randint(1,3))),
            ))
        conn.executemany(
            "INSERT INTO alerts (alert_id,timestamp,alert_type,confidence,"
            "anomaly_score,priority,frame_path,location,objects_detected) VALUES (?,?,?,?,?,?,?,?,?)",
            rows)
        conn.commit()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/security-checked.png", width=80)
    st.title("🎛️ Filters")

    priority_filter = st.multiselect(
        "Priority Level",
        options=["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )

    hours_back = st.slider("Look-back window (hours)", 1, 72, 24)

    alert_type_filter = st.selectbox(
        "Alert Type",
        ["All", "intrusion", "crowd_spike", "vehicle_surge", "motion_anomaly"],
    )

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh (30 s)", value=True)
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()

    st.divider()
    st.caption("Border Surveillance AI · GTU Internship 2026")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🚨 Border Surveillance AI — Live Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC  |  "
           f"DB: `{DB_PATH}`")
st.divider()

# Seed demo data if DB is empty (makes the dashboard useful immediately)
seed_demo_data(DB_PATH)

df = load_alerts(DB_PATH)

if df.empty:
    st.warning("No alerts found. Run the pipeline to generate data:\n"
               "```\npython src/run_pipeline.py --video data/sample/test_video.mp4\n```")
    st.stop()

# ── Apply filters ─────────────────────────────────────────────────────────────
cutoff = datetime.utcnow() - timedelta(hours=hours_back)
df_f   = df[df["priority"].isin(priority_filter)]
df_f   = df_f[df_f["timestamp"] >= pd.Timestamp(cutoff, tz=None)]
if alert_type_filter != "All":
    df_f = df_f[df_f["alert_type"] == alert_type_filter]


# ── KPI CARDS ────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
total   = len(df_f)
n_high  = len(df_f[df_f["priority"] == "HIGH"])
n_med   = len(df_f[df_f["priority"] == "MEDIUM"])
avg_sc  = df_f["anomaly_score"].mean() if total > 0 else 0
fpr_est = round(len(df_f[df_f["priority"] == "LOW"]) / total, 2) if total > 0 else 0

k1.metric("📋 Total Alerts",        total)
k2.metric("🔴 HIGH",                n_high)
k3.metric("🟡 MEDIUM",              n_med)
k4.metric("📈 Avg Anomaly Score",   f"{avg_sc:.3f}")
k5.metric("🎯 Est. FP Rate",        f"{fpr_est:.1%}")

st.divider()


# ── ROW 1: Pie + Score Trend ──────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Priority Distribution")
    if not df_f.empty:
        pct  = df_f["priority"].value_counts().reset_index()
        pct.columns = ["priority", "count"]
        fig_pie = px.pie(
            pct, values="count", names="priority",
            color="priority",
            color_discrete_map=PRIORITY_COLORS,
            hole=0.4,
        )
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No data for selected filters.")

with col2:
    st.subheader("Anomaly Score Trend")
    if not df_f.empty:
        trend = df_f.sort_values("timestamp").tail(200)
        fig_line = px.scatter(
            trend, x="timestamp", y="anomaly_score",
            color="priority",
            color_discrete_map=PRIORITY_COLORS,
            opacity=0.7,
        )
        fig_line.add_hline(y=0.7, line_dash="dash", line_color="#FF3131",
                           annotation_text="HIGH threshold")
        fig_line.add_hline(y=0.4, line_dash="dash", line_color="#FFD600",
                           annotation_text="MEDIUM threshold")
        fig_line.update_layout(height=300, margin=dict(t=10, b=0))
        st.plotly_chart(fig_line, use_container_width=True)


# ── ROW 2: Detections per hour ────────────────────────────────────────────────
st.subheader("Detection Count per Hour")
if not df_f.empty and "hour" in df_f.columns:
    hour_data = df_f.groupby(["hour", "priority"]).size().reset_index(name="count")
    fig_bar   = px.bar(
        hour_data, x="hour", y="count", color="priority",
        color_discrete_map=PRIORITY_COLORS, barmode="stack",
    )
    fig_bar.update_layout(height=280, margin=dict(t=10, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()


# ── ALERT TABLE ───────────────────────────────────────────────────────────────
st.subheader(f"📋 Alert Feed  ({len(df_f)} records)")

display_cols = ["timestamp", "alert_id", "alert_type", "priority",
                "anomaly_score", "confidence", "location"]
display_cols = [c for c in display_cols if c in df_f.columns]

def color_priority(val):
    colors = {"HIGH": "background-color:#3D0000;color:#FF3131",
              "MEDIUM": "background-color:#2D2600;color:#FFD600",
              "LOW": "background-color:#002D00;color:#39FF14"}
    return colors.get(val, "")

_s = df_f[display_cols].head(100).style.format({"anomaly_score": "{:.4f}", "confidence": "{:.4f}"})
styled = (_s.map if hasattr(_s, "map") else _s.applymap)(color_priority, subset=["priority"])

st.dataframe(styled, use_container_width=True, height=400)


# ── AUTO-REFRESH ──────────────────────────────────────────────────────────────
if auto_refresh:
    import time
    time.sleep(0.1)   # give the page time to render
    # Streamlit re-runs on cache expiry (ttl=30s) — no explicit rerun needed
