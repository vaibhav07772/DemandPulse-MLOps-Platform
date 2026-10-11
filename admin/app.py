"""DemandPulse AI — MLOps Admin Dashboard.

Streamlit dashboard for monitoring experiments, model registry,
and prediction logs. Runs alongside the main FastAPI app.

Usage:
    streamlit run admin/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st
from mlflow.tracking import MlflowClient

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
PRED_LOG = ROOT / "data" / "logs" / "predictions.jsonl"
EXPERIMENT = "demandpulse-forecasting"
REGISTERED_MODEL = "demandpulse-forecaster"

st.set_page_config(
    page_title="DemandPulse Admin",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Data helpers ---
@st.cache_resource
def _client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_URI)
    return MlflowClient()


@st.cache_data(ttl=10)
def load_runs(limit: int = 100) -> pd.DataFrame:
    """Load top-level parent runs only (nested candidate runs are excluded)."""
    client = _client()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return pd.DataFrame()

    all_runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=limit * 3,
    )
    parent_runs = [r for r in all_runs if r.data.tags.get("mlflow.parentRunId") is None]
    runs = parent_runs[:limit]

    rows = []
    for r in runs:
        start = r.info.start_time
        end = r.info.end_time
        rows.append(
            {
                "run_id": r.info.run_id,
                "name": r.data.tags.get("mlflow.runName", ""),
                "status": r.info.status,
                "best_model": r.data.tags.get("best_model", ""),
                "env": r.data.tags.get("environment", ""),
                "wape": r.data.metrics.get("best_wape"),
                "mae": r.data.metrics.get("best_mae"),
                "rmse": r.data.metrics.get("best_rmse"),
                "start_time": pd.to_datetime(start, unit="ms") if start else None,
                "duration_s": round((end - start) / 1000, 2) if (start and end) else None,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=10)
def load_registry() -> pd.DataFrame:
    client = _client()
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    except Exception:
        return pd.DataFrame()
    rows = []
    for v in versions:
        rows.append(
            {
                "version": int(v.version),
                "stage": v.current_stage,
                "status": v.status,
                "created": pd.to_datetime(v.creation_timestamp, unit="ms"),
                "run_id": v.run_id,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("version", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=5)
def load_predictions() -> pd.DataFrame:
    if not PRED_LOG.exists():
        return pd.DataFrame()
    records = []
    with PRED_LOG.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# --- Sidebar ---
with st.sidebar:
    st.title("📊 DemandPulse Admin")
    st.caption("MLOps monitoring for demand forecasting")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Experiments", "Model Registry", "Predictions"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("**Tracking URI**")
    st.code(MLFLOW_URI, language="text")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# --- Load data ---
try:
    runs_df = load_runs()
    registry_df = load_registry()
    preds_df = load_predictions()
    load_error = None
except Exception as exc:
    runs_df = pd.DataFrame()
    registry_df = pd.DataFrame()
    preds_df = pd.DataFrame()
    load_error = f"{type(exc).__name__}: {exc}"

if load_error:
    st.error(f"⚠️ Data load failed: {load_error}")
    st.info(
        "Tip: Run a forecast from the main app first so MLflow has something to read. "
        "If the issue persists, verify `mlflow.db` exists in the project root."
    )
    st.stop()


# --- Pages ---
if page == "Overview":
    st.title("📊 Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total runs", len(runs_df))
    best_wape = (
        f"{runs_df['wape'].min():.2f}%"
        if not runs_df.empty and runs_df["wape"].notna().any()
        else "—"
    )
    col2.metric("Best WAPE", best_wape)
    col3.metric("Model versions", len(registry_df))
    col4.metric("Total predictions", len(preds_df))

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Latest training run")
        if not runs_df.empty:
            latest = runs_df.iloc[0]
            st.markdown(f"**Run name:** `{latest['name']}`")
            st.markdown(f"**Best model:** `{latest['best_model']}`")
            st.markdown(f"**Status:** {latest['status']}")
            st.markdown(f"**WAPE:** {latest['wape']}%")
            st.markdown(f"**MAE:** {latest['mae']}")
            st.markdown(f"**RMSE:** {latest['rmse']}")
            st.markdown(f"**Duration:** {latest['duration_s']}s")
            st.markdown(f"**Environment:** `{latest['env']}`")
        else:
            st.info("No runs logged yet. Run a forecast from the main app.")

    with right:
        st.subheader("Registry head")
        if not registry_df.empty:
            st.dataframe(
                registry_df[["version", "stage", "status"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No registered models yet.")


elif page == "Experiments":
    st.title("🧪 Experiments")
    st.caption(f"Experiment: `{EXPERIMENT}`")

    if runs_df.empty:
        st.info("No runs logged yet.")
    else:
        chart_df = runs_df.dropna(subset=["wape"])
        if not chart_df.empty:
            fig = px.bar(
                chart_df,
                x="name",
                y="wape",
                color="best_model",
                title="WAPE across recent runs (lower is better)",
                labels={"wape": "WAPE (%)", "name": "Run name"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("All runs")
        display = runs_df[
            [
                "name",
                "best_model",
                "status",
                "wape",
                "mae",
                "rmse",
                "duration_s",
                "start_time",
            ]
        ].copy()
        st.dataframe(display, use_container_width=True, hide_index=True)


elif page == "Model Registry":
    st.title("📦 Model Registry")
    st.caption(f"Registered model: `{REGISTERED_MODEL}`")

    if registry_df.empty:
        st.info("No versions registered yet. Run a forecast to register the first model.")
    else:
        st.dataframe(registry_df, use_container_width=True, hide_index=True)

        st.subheader("Version timeline")
        timeline = registry_df.sort_values("version").copy()
        timeline["version_str"] = timeline["version"].astype(str)
        fig = px.scatter(
            timeline,
            x="created",
            y="version_str",
            color="stage",
            title="Model version history",
            labels={"version_str": "Version", "created": "Created at"},
        )
        fig.update_traces(marker={"size": 18, "line": {"width": 1, "color": "black"}})
        st.plotly_chart(fig, use_container_width=True)


elif page == "Predictions":
    st.title("🔮 Predictions")

    if preds_df.empty:
        st.info("No predictions logged yet. Run a forecast from the main app.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total forecasts", len(preds_df))
        col2.metric(
            "Unique products",
            preds_df["product"].nunique() if "product" in preds_df else 0,
        )
        avg_wape = f"{preds_df['wape'].mean():.2f}%" if "wape" in preds_df else "—"
        col3.metric("Avg WAPE", avg_wape)

        st.divider()

        if "timestamp" in preds_df.columns and "wape" in preds_df.columns:
            timeline = preds_df.sort_values("timestamp")
            fig = px.line(
                timeline,
                x="timestamp",
                y="wape",
                color="product",
                markers=True,
                title="WAPE over time (per product)",
                labels={"wape": "WAPE (%)", "timestamp": "Time"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent predictions")
        cols = [
            c
            for c in [
                "timestamp",
                "product",
                "store",
                "horizon",
                "model",
                "segment",
                "wape",
                "forecast_total",
            ]
            if c in preds_df.columns
        ]
        st.dataframe(
            preds_df.sort_values("timestamp", ascending=False)[cols].head(100),
            use_container_width=True,
            hide_index=True,
        )


st.sidebar.divider()
st.sidebar.caption("Part of **DemandPulse AI MLOps**")
st.sidebar.caption("Built by **Vaibhav Singh Rajput** 🚀")
