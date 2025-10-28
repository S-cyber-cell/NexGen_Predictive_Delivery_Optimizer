# app.py
# NexGen Predictive Delivery Optimizer — Final Developer-Controlled Version
# Author: Arnav Bhatia (GitHub: S-cyber-cell)


import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")
DATA_PATH = Path(__file__).parent / "data"

REQUIRED_FILES = {
    "orders": "orders.csv",
    "delivery_performance": "delivery_performance.csv",
    "routes_distance": "routes_distance.csv",
    "vehicle_fleet": "vehicle_fleet.csv",
    "warehouse_inventory": "warehouse_inventory.csv",
    "customer_feedback": "customer_feedback.csv",
    "cost_breakdown": "cost_breakdown.csv",
}

# ---------------------------------------------------------------------
# LOAD ALL DATASETS AUTOMATICALLY
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_csv(name):
    path = DATA_PATH / name
    if not path.exists():
        st.error(f"Missing {name} in data folder.")
        return pd.DataFrame()
    return pd.read_csv(path)

data = {k: load_csv(v) for k, v in REQUIRED_FILES.items()}

# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------
def build_feature_table(ds):
    base = ds.get("orders", pd.DataFrame()).copy()
    if base.empty:
        return pd.DataFrame()

    perf = ds.get("delivery_performance", pd.DataFrame())
    if not perf.empty:
        base = base.merge(perf, on="order_id", how="left")

    routes = ds.get("routes_distance", pd.DataFrame())
    if not routes.empty:
        base = base.merge(routes, on="order_id", how="left")

    fleet = ds.get("vehicle_fleet", pd.DataFrame())
    if not fleet.empty and "vehicle_id" in base.columns:
        base = base.merge(fleet, on="vehicle_id", how="left")

    cost = ds.get("cost_breakdown", pd.DataFrame())
    if not cost.empty:
        base = base.merge(cost, on="order_id", how="left")

    # Derived features
    if "promised_delivery_hours" in base and "actual_delivery_hours" in base:
        base["lateness_hours"] = (
            base["actual_delivery_hours"] - base["promised_delivery_hours"]
        )
        base["is_late"] = (base["lateness_hours"] > 0.5).astype(int)
    else:
        base["is_late"] = 0

    base["priority_num"] = base.get("priority", "Standard").map(
        {"Economy": 0, "Standard": 1, "Express": 2}
    )

    base["traffic_delay_flag"] = (
        pd.to_numeric(base.get("traffic_delay_hours", 0), errors="coerce").fillna(0) > 0.25
    ).astype(int)

    base["weather_impact_flag"] = base.get("weather_impact", "").astype(str).str.lower().isin(
        ["rain", "storm", "snow", "bad", "adverse", "rainy"]
    ).astype(int)

    return base


def prepare_data(tab):
    feat_cols = [
        "priority_num", "distance_km", "traffic_delay_hours", "traffic_delay_flag",
        "weather_impact_flag", "order_value", "toll_charges", "delivery_cost",
        "fuel_consumption_l", "vehicle_age_years", "fuel_efficiency_kmpl",
        "promised_delivery_hours"
    ]
    df = tab.dropna(subset=["is_late"]).copy()
    y = df["is_late"].astype(int)
    X = pd.DataFrame(index=df.index)
    for c in feat_cols:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            X[c] = 0
    return X, y


def prob_to_label(p, thr):
    return (p >= thr).astype(int)


# ---------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------
st.title("Predictive Delivery Optimizer")
st.caption("All datasets loaded automatically from developer-provided data folder")

# Quick health check
st.subheader("Data Overview")
for name, df in data.items():
    st.write(f"**{name}** — {len(df)} rows, {len(df.columns)} columns")

ft = build_feature_table(data)

tab1, tab2, tab3 = st.tabs(["EDA", "Model Training", "Predictions"])

# -------------------- EDA --------------------
with tab1:
    if ft.empty:
        st.error("No data found in ./data folder.")
    else:
        st.subheader("Exploratory Data Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Priority mix")
            fig, ax = plt.subplots()
            ft["priority"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

        with col2:
            st.write("Lateness distribution")
            fig, ax = plt.subplots()
            ft["lateness_hours"].dropna().plot(kind="hist", bins=20, ax=ax)
            st.pyplot(fig)

        with col3:
            st.write("Distance vs lateness")
            fig, ax = plt.subplots()
            ax.scatter(ft["distance_km"], ft["lateness_hours"], alpha=0.4)
            ax.set_xlabel("Distance (km)")
            ax.set_ylabel("Lateness (hrs)")
            st.pyplot(fig)

# -------------------- MODEL --------------------
with tab2:
    if ft.empty:
        st.error("No data available for training.")
    else:
        st.subheader("Train Random Forest Model")

        X, y = prepare_data(ft)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        n_estimators = st.slider("Trees", 100, 500, 300, 50)
        max_depth = st.slider("Max Depth", 3, 15, 8, 1)
        threshold = st.slider("Alert threshold", 0.1, 0.9, 0.5, 0.05)

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = prob_to_label(proba, threshold)

        report = pd.DataFrame(
            classification_report(y_test, preds, output_dict=True, zero_division=0)
        ).T.round(3)
        st.dataframe(report, use_container_width=True)

        auc = roc_auc_score(y_test, proba)
        st.metric("ROC AUC", f"{auc:.3f}")

        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, preds)
        ax.imshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        st.session_state["model"] = model
        st.session_state["X_cols"] = X.columns
        st.session_state["feature_table"] = ft

# -------------------- PREDICTIONS --------------------
with tab3:
    st.subheader("Predict At-Risk Orders")
    if "model" not in st.session_state:
        st.info("Train model first.")
    else:
        model = st.session_state["model"]
        cols = st.session_state["X_cols"]
        ft = st.session_state["feature_table"]

        X_pred = ft[cols].fillna(0)
        proba = model.predict_proba(X_pred)[:, 1]
        ft["late_risk_prob"] = proba
        ft["predicted_late"] = (proba > 0.5).astype(int)

        st.dataframe(
            ft[["order_id", "late_risk_prob", "predicted_late"]].sort_values(
                "late_risk_prob", ascending=False
            ).head(50),
            use_container_width=True,
        )

        total_pred = ft["predicted_late"].sum()
        st.metric("Predicted late orders", int(total_pred))
        st.metric("Average risk probability", f"{ft['late_risk_prob'].mean():.2f}")
        st.success("Predictions generated automatically using developer data.")

st.caption("Developed by Arnav Bhatia (GitHub: S-cyber-cell)")
