# app.py
# NexGen Logistics — Predictive Delivery Optimizer
# Streamlit prototype for OFI Services case challenge
#
# How to run (after placing CSVs in ./data):
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Predictive Delivery Optimizer — NexGen Logistics",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILES = {
    "orders": "orders.csv",
    "delivery_performance": "delivery_performance.csv",
    "routes_distance": "routes_distance.csv",
    "vehicle_fleet": "vehicle_fleet.csv",
    "warehouse_inventory": "warehouse_inventory.csv",
    "customer_feedback": "customer_feedback.csv",
    "cost_breakdown": "cost_breakdown.csv",
}

DATA_PATH = Path(__file__).parent / "data"

@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_or_buffer)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as e:
        st.warning(f"Could not read: {e}")
        return pd.DataFrame()

def load_datasets(sources: Dict[str, object]) -> Dict[str, pd.DataFrame]:
    return {k: load_csv(v) if v else pd.DataFrame() for k, v in sources.items()}

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "left") -> pd.DataFrame:
    if left.empty:
        return left
    if right.empty:
        return left
    inter = set(left.columns).intersection(right.columns)
    if on not in inter:
        for cand in ["order_id", "orderid", "id"]:
            if cand in inter:
                on = cand
                break
        else:
            return left
    return left.merge(right, how=how, on=on)

def build_feature_table(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = d.get("orders", pd.DataFrame()).copy()
    if base.empty:
        return pd.DataFrame()

    needed = ["order_id","order_date","priority","product_category","origin","destination",
              "customer_segment","order_value","special_handling"]
    for col in needed:
        if col not in base.columns:
            base[col] = np.nan

    perf = d.get("delivery_performance", pd.DataFrame()).copy()
    if not perf.empty:
        for c in ["promised_delivery_hours","actual_delivery_hours","status","rating","delivery_cost"]:
            if c not in perf.columns:
                perf[c] = np.nan
    tab = safe_merge(base, perf, on="order_id", how="left")

    routes = d.get("routes_distance", pd.DataFrame()).copy()
    if not routes.empty:
        for c in ["distance_km","fuel_consumption_l","toll_charges","traffic_delay_hours","weather_impact"]:
            if c not in routes.columns:
                routes[c] = np.nan
    tab = safe_merge(tab, routes, on="order_id", how="left")

    fleet = d.get("vehicle_fleet", pd.DataFrame()).copy()
    if not fleet.empty:
        key = "vehicle_id" if "vehicle_id" in set(tab.columns).intersection(fleet.columns) else None
        if key:
            for c in ["vehicle_type","capacity_kg","fuel_efficiency_kmpl","vehicle_age_years","co2_g_per_km"]:
                if c not in fleet.columns:
                    fleet[c] = np.nan
            tab = safe_merge(tab, fleet, on=key, how="left")

    cost = d.get("cost_breakdown", pd.DataFrame()).copy()
    if not cost.empty and "order_id" in cost.columns:
        tab = safe_merge(tab, cost, on="order_id", how="left")

    # Target label
    if "actual_delivery_hours" in tab.columns and "promised_delivery_hours" in tab.columns:
        tab["lateness_hours"] = pd.to_numeric(tab["actual_delivery_hours"], errors="coerce") - pd.to_numeric(tab["promised_delivery_hours"], errors="coerce")
        tab["is_late"] = (tab["lateness_hours"] > 0.5).astype("Int64")
    else:
        tab["is_late"] = pd.Series([pd.NA] * len(tab), dtype="Int64")

    # Derived features
    if "traffic_delay_hours" in tab.columns:
        tab["traffic_delay_flag"] = (pd.to_numeric(tab["traffic_delay_hours"], errors="coerce").fillna(0) > 0.25).astype(int)
    else:
        tab["traffic_delay_flag"] = 0

    if "weather_impact" in tab.columns:
        tab["weather_impact_flag"] = tab["weather_impact"].astype(str).str.lower().isin(
            ["rain","storm","heavy_rain","flood","snow","bad","adverse","1","true","yes"]
        ).astype(int)
    else:
        tab["weather_impact_flag"] = 0

    tab["priority"] = tab["priority"].astype(str).str.title()
    priority_map = {"Express":2, "Standard":1, "Economy":0}
    tab["priority_num"] = tab["priority"].map(priority_map).fillna(1).astype(int)

    for c in ["distance_km","order_value","toll_charges","delivery_cost","fuel_consumption_l",
              "promised_delivery_hours","actual_delivery_hours","vehicle_age_years","fuel_efficiency_kmpl"]:
        if c in tab.columns:
            tab[c] = pd.to_numeric(tab[c], errors="coerce")

    return tab

def prepare_model_data(tab: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = tab.dropna(subset=["is_late"]).copy()
    y = df["is_late"].astype(int)

    # Numeric features
    features = [
        "priority_num","distance_km","traffic_delay_hours","traffic_delay_flag",
        "weather_impact_flag","order_value","toll_charges","delivery_cost",
        "fuel_consumption_l","vehicle_age_years","fuel_efficiency_kmpl","promised_delivery_hours"
    ]
    X = pd.DataFrame(index=df.index)
    for c in features:
        if c in df.columns:
            X[c] = df[c].fillna(0)
    # Categorical one-hots
    for cat in ["product_category","origin","destination"]:
        if cat in df.columns:
            dummies = pd.get_dummies(df[cat].astype(str), prefix=cat, dummy_na=True)
            X = pd.concat([X, dummies], axis=1)

    X = X.fillna(0)
    return X, y

def prob_to_label(p, threshold: float):
    return (p >= threshold).astype(int)

def business_impact(n_pred: int, avoid_rate: float, penalty_per_late: float) -> float:
    return n_pred * avoid_rate * penalty_per_late

# --------------------- Sidebar ---------------------
st.sidebar.header("Data")
use_local = st.sidebar.checkbox("Use CSVs from ./data folder", value=True)
uploads = {}
if not use_local:
    for key, fname in DATA_FILES.items():
        uploads[key] = st.sidebar.file_uploader(f"Upload {fname}", type="csv")

paths = {k: (DATA_PATH / v if use_local else None) for k, v in DATA_FILES.items()}
sources = {}
for k, p in paths.items():
    if use_local and p and p.exists():
        sources[k] = p
    elif not use_local and uploads.get(k) is not None:
        sources[k] = uploads[k]
    else:
        sources[k] = None

data = load_datasets(sources)

st.title("Predictive Delivery Optimizer")
st.caption("NexGen Logistics — turning delivery data into proactive action")

# --------------------- Data health ---------------------
with st.expander("Quick data health check", expanded=False):
    for name, df in data.items():
        st.write(f"**{name}** — rows: {len(df)}, cols: {len(df.columns)}")
        if not df.empty:
            st.dataframe(df.head(5), use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(["Overview & EDA", "Model", "At-Risk Orders", "Downloads"])

with tab1:
    st.subheader("Exploratory view")
    feature_table = build_feature_table(data)
    if feature_table.empty:
        st.info("Load data to view EDA.")
    else:
        colA, colB, colC = st.columns(3)

        with colA:
            st.write("Priority mix")
            if "priority" in feature_table.columns:
                counts = feature_table["priority"].value_counts(dropna=False)
                fig, ax = plt.subplots()
                counts.plot(kind="bar", ax=ax)
                ax.set_ylabel("Orders")
                ax.set_xlabel("Priority")
                st.pyplot(fig)

        with colB:
            st.write("Lateness (hours) — histogram")
            if "lateness_hours" in feature_table.columns:
                fig, ax = plt.subplots()
                feature_table["lateness_hours"].dropna().plot(kind="hist", bins=20, ax=ax)
                ax.set_xlabel("Actual - Promised (hours)")
                st.pyplot(fig)

        with colC:
            st.write("Distance vs Lateness")
            if "distance_km" in feature_table.columns and "lateness_hours" in feature_table.columns:
                fig, ax = plt.subplots()
                ax.scatter(feature_table["distance_km"], feature_table["lateness_hours"], alpha=0.5)
                ax.set_xlabel("Distance (km)")
                ax.set_ylabel("Lateness (hours)")
                st.pyplot(fig)

        st.markdown("---")
        st.write("Top routes / lanes")
        if "origin" in feature_table.columns and "destination" in feature_table.columns:
            lanes = feature_table.groupby(["origin","destination"], dropna=False).size().sort_values(ascending=False).head(10)
            st.dataframe(lanes.reset_index(name="orders"), use_container_width=True)

with tab2:
    st.subheader("Train a Random Forest")
    feature_table = build_feature_table(data)
    if feature_table.empty or feature_table["is_late"].isna().all():
        st.info("Need labeled data (promised vs actual delivery hours) to train the model.")
    else:
        X, y = prepare_model_data(feature_table)

        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        n_estimators = st.slider("Trees", 100, 600, 300, 50)
        max_depth = st.slider("Max depth", 3, 20, 10, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:,1]

        threshold = st.slider("Alert threshold (probability of being late)", 0.1, 0.9, 0.5, 0.05)
        y_pred = prob_to_label(proba, threshold)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Classification report")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(report).T.round(3)
            st.dataframe(rep_df, use_container_width=True)

        with col2:
            try:
                auc = roc_auc_score(y_test, proba)
            except ValueError:
                auc = float("nan")
            st.metric("ROC AUC", f"{auc:.3f}" if auc==auc else "n/a")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            ax.imshow(cm, interpolation="nearest")
            ax.set_title("Confusion matrix")
            for (i, j), z in np.ndenumerate(cm):
                ax.text(j, i, str(z), ha='center', va='center')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)

        # Feature importance
        st.write("Feature importance")
        importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots()
        importances.iloc[::-1].plot(kind="barh", ax=ax)
        ax.set_xlabel("Gini importance")
        st.pyplot(fig)

        st.session_state["model"] = clf
        st.session_state["model_features"] = X.columns.tolist()
        st.session_state["feature_table"] = feature_table

with tab3:
    st.subheader("At-risk orders (for proactive action)")
    if "model" not in st.session_state:
        st.info("Train the model first.")
    else:
        clf = st.session_state["model"]
        features = st.session_state["model_features"]
        feature_table = st.session_state["feature_table"].copy()

        # At-risk = rows without label (e.g., in-transit) or all rows depending on user choice
        scope = st.radio("Prediction scope", ["Only unlabeled (in-transit / missing actuals)", "All orders"], index=0)
        if scope.startswith("Only"):
            pred_df = feature_table[feature_table["is_late"].isna()].copy()
        else:
            pred_df = feature_table.copy()

        # Build X with same columns
        X_pred = pd.DataFrame(index=pred_df.index)
        for c in features:
            if c in pred_df.columns:
                X_pred[c] = pred_df[c]
            else:
                X_pred[c] = 0
        X_pred = X_pred.fillna(0)

        proba = clf.predict_proba(X_pred)[:,1]
        threshold_pred = st.slider("Operational alert threshold", 0.3, 0.9, 0.6, 0.05)
        labels = prob_to_label(proba, threshold_pred)

        pred_df["late_risk_prob"] = proba
        pred_df["predicted_late"] = labels

        st.write("Predicted late orders")
        st.dataframe(pred_df.sort_values("late_risk_prob", ascending=False).head(50), use_container_width=True)

        # Simple actions
        st.markdown("**Suggested quick actions**")
        st.write("- Expedite Express upgrades for high-probability late orders")
        st.write("- Switch carrier on lanes with repeated delays")
        st.write("- Avoid peak traffic hours on high-delay routes")
        st.write("- Assign newer, fuel-efficient vehicles to longer lanes")

        # Business impact model
        st.markdown("---")
        st.write("Estimated business impact from proactive interventions")
        avoid_rate = st.slider("Avoidance rate (fraction of predicted-late you can save)", 0.0, 1.0, 0.3, 0.05)
        penalty = st.number_input("Penalty / cost per late order (₹)", value=250.0, min_value=0.0, step=10.0)
        total_savings = business_impact(int(pred_df["predicted_late"].sum()), avoid_rate, penalty)

        colx, coly = st.columns(2)
        with colx:
            st.metric("Predicted-late (alerts)", int(pred_df["predicted_late"].sum()))
        with coly:
            st.metric("Estimated savings (₹)", f"{total_savings:,.0f}")

        st.session_state["predictions"] = pred_df

with tab4:
    st.subheader("Export")
    # Export feature table
    ft = build_feature_table(data)
    if not ft.empty:
        ft_csv = ft.to_csv(index=False).encode("utf-8")
        st.download_button("Download feature table (CSV)", ft_csv, "feature_table.csv", "text/csv")

    # Export predictions
    pred_df = st.session_state.get("predictions")
    if pred_df is not None and not pred_df.empty:
        pred_csv = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions (CSV)", pred_csv, "predicted_at_risk_orders.csv", "text/csv")

st.caption("Note: This is a prototype. Results depend on data quality, coverage, and labeling completeness.")
