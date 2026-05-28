"""
utils.py — ML & Analytics Backend
Data cleaning, K-Means segmentation, churn prediction (LR / RF / XGB), SHAP.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import streamlit as st

# ============================================================================
# 1. DATA CLEANING & PREPROCESSING
# ============================================================================

FEATURE_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
CAT_COLS = ["Geography", "Gender"]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]
TARGET = "Exited"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and basic cleaning."""
    df = df.copy()
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)
    for col in CAT_COLS:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
        ],
        remainder="drop",
    )


def prepare_features(df: pd.DataFrame):
    """Return X_processed (ndarray), y, fitted preprocessor, and feature names."""
    df = clean_dataframe(df)
    X = df[FEATURE_COLS]
    y = df[TARGET].values
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_COLS).tolist()
    feature_names = NUM_COLS + cat_feature_names
    return X_processed, y, preprocessor, feature_names


# ============================================================================
# 2. K-MEANS SEGMENTATION
# ============================================================================

CLUSTER_LABELS = {
    0: "💎 High-Value Engaged",
    1: "⚠️ At-Risk Seniors",
    2: "🌱 New & Growing",
    3: "😴 Dormant Low-Balance",
}

CLUSTER_PLAYBOOKS = {
    0: "**Premium Retention**: Assign dedicated relationship managers, offer exclusive investment products, priority support lines, and loyalty rewards tied to AUM growth.",
    1: "**Proactive Outreach**: Schedule quarterly wellness calls, simplify digital banking UX, offer retirement planning consultations and fee waivers for long-tenure clients.",
    2: "**Engagement Acceleration**: Onboard with gamified savings goals, cross-sell credit cards and insurance, provide financial literacy content and referral bonuses.",
    3: "**Re-activation Campaign**: Send win-back offers (cashback on transactions), reduce friction in mobile app, trigger push notifications for idle accounts >30 days.",
}


@st.cache_data(show_spinner=False)
def run_kmeans_analysis(df: pd.DataFrame, k_range: tuple = (2, 9)):
    """Run elbow + silhouette analysis and fit optimal K-Means."""
    df = clean_dataframe(df)
    cluster_features = ["Age", "Tenure", "Balance", "CreditScore", "EstimatedSalary", "NumOfProducts"]
    X_cluster = df[cluster_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    inertias, silhouettes = [], []
    K_vals = list(range(k_range[0], k_range[1]))
    for k in K_vals:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    optimal_k = K_vals[np.argmax(silhouettes)]
    # Clamp to 4 for business interpretability
    optimal_k = min(optimal_k, 4)

    km_final = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    df = df.copy()
    df["Cluster"] = km_final.fit_predict(X_scaled)
    df["ClusterLabel"] = df["Cluster"].map(
        lambda c: CLUSTER_LABELS.get(c, f"Cluster {c}")
    )

    return df, K_vals, inertias, silhouettes, optimal_k


def elbow_silhouette_chart(K_vals, inertias, silhouettes, optimal_k):
    """Plotly dual-axis chart: Elbow (inertia) + Silhouette Score."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=K_vals, y=inertias, mode="lines+markers", name="Inertia (Elbow)",
        line=dict(color="#636EFA", width=3), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=K_vals, y=silhouettes, mode="lines+markers", name="Silhouette Score",
        yaxis="y2", line=dict(color="#EF553B", width=3, dash="dot"), marker=dict(size=8),
    ))
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="#00CC96",
                  annotation_text=f"Optimal k={optimal_k}")
    fig.update_layout(
        title="Elbow Method & Silhouette Score",
        xaxis_title="Number of Clusters (k)",
        yaxis=dict(title=dict(text="Inertia", font=dict(color="#636EFA"))),
        yaxis2=dict(title=dict(text="Silhouette Score", font=dict(color="#EF553B")),
                    overlaying="y", side="right"),
        template="plotly_white", height=400, legend=dict(x=0.4, y=1.15, orientation="h"),
    )
    return fig


def cluster_scatter(df: pd.DataFrame):
    """Balance vs Tenure scatter colored by cluster."""
    fig = px.scatter(
        df, x="Tenure", y="Balance", color="ClusterLabel",
        hover_data=["Age", "CreditScore", "NumOfProducts"],
        title="Customer Segments: Balance vs. Tenure",
        template="plotly_white", height=480,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.65))
    return fig


def cluster_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster aggregation with business labels."""
    actions = {
        "💎 High-Value Engaged": "VIP retention program",
        "⚠️ At-Risk Seniors": "Proactive outreach calls",
        "🌱 New & Growing": "Cross-sell & onboarding",
        "😴 Dormant Low-Balance": "Win-back campaign",
    }
    summary = df.groupby("ClusterLabel").agg(
        Customers=("Exited", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Balance=("Balance", "mean"),
        Avg_Salary=("EstimatedSalary", "mean"),
        Churn_Rate=("Exited", "mean"),
    ).reset_index()
    summary["Avg_Age"] = summary["Avg_Age"].round(1)
    summary["Avg_Balance"] = summary["Avg_Balance"].round(0)
    summary["Avg_Salary"] = summary["Avg_Salary"].round(0)
    summary["Churn_Rate"] = (summary["Churn_Rate"] * 100).round(1)
    summary["Retention Action"] = summary["ClusterLabel"].map(actions).fillna("General engagement")
    return summary


# ============================================================================
# 3. CHURN PREDICTION MODELS
# ============================================================================

@st.cache_resource(show_spinner=False)
def train_all_models(_df: pd.DataFrame):
    """Train LR, RF, XGB. Return models dict, metrics, preprocessor, feature_names, test sets."""
    df = _df.copy()
    X_processed, y, preprocessor, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  eval_metric="logloss", random_state=42),
    }

    results = {}
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        }
        fitted_models[name] = model

    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})

    # Best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
    best_model = fitted_models[best_name]

    # SHAP (use TreeExplainer for tree models, otherwise KernelExplainer)
    if isinstance(best_model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        # For RF, shap_values can be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer = shap.LinearExplainer(best_model, X_train)
        shap_values = explainer.shap_values(X_test)

    return {
        "fitted_models": fitted_models,
        "best_name": best_name,
        "best_model": best_model,
        "metrics_df": metrics_df,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "explainer": explainer,
        "shap_values": shap_values,
    }


# ============================================================================
# 4. VISUALIZATIONS — PREDICTIVE TAB
# ============================================================================

def roc_curves_plot(fitted_models: dict, X_test, y_test):
    """Overlay ROC curves for all models."""
    fig = go.Figure()
    colors = {"Logistic Regression": "#636EFA", "Random Forest": "#00CC96", "XGBoost": "#EF553B"}
    for name, model in fitted_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc_val:.3f})",
            line=dict(color=colors.get(name, "#AB63FA"), width=2.5),
        ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(dash="dash", color="grey", width=1)))
    fig.update_layout(
        title="ROC Curves — Model Comparison",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        template="plotly_white", height=420,
        legend=dict(x=0.45, y=0.05),
    )
    return fig


def confusion_matrix_plot(model, X_test, y_test, model_name: str):
    """Heatmap confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm, text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Retained", "Churned"], y=["Retained", "Churned"],
        color_continuous_scale="Blues", title=f"Confusion Matrix — {model_name}",
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


def shap_feature_importance_plot(shap_values, feature_names, top_n: int = 10):
    """Plotly horizontal bar chart of mean |SHAP| values (top N features)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
    importance = importance.nlargest(top_n, "importance").sort_values("importance")

    fig = px.bar(
        importance, x="importance", y="feature", orientation="h",
        title=f"SHAP Feature Importance (Top {top_n})",
        labels={"importance": "Mean |SHAP value|", "feature": ""},
        color="importance", color_continuous_scale="Reds",
    )
    fig.update_layout(template="plotly_white", height=420, showlegend=False)
    return fig


def shap_waterfall_plot(explainer, shap_values, X_test, feature_names, idx: int = 0):
    """Return a SHAP waterfall figure for a single observation."""
    if hasattr(explainer, "expected_value"):
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = ev[1] if len(ev) > 1 else ev[0]
    else:
        ev = 0.0

    sv = shap_values[idx]
    order = np.argsort(np.abs(sv))[::-1][:10]

    features = [feature_names[i] for i in order]
    values = [sv[i] for i in order]
    colors = ["#EF553B" if v > 0 else "#636EFA" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title=f"SHAP Waterfall — Sample #{idx} (base={ev:.3f})",
        xaxis_title="SHAP value (impact on churn probability)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white", height=420,
    )
    return fig


# ============================================================================
# 5. LIVE SIMULATOR
# ============================================================================

def predict_single_customer(customer_dict: dict, model, preprocessor, feature_names,
                            X_train=None):
    """Predict churn probability for a single customer and return SHAP breakdown."""
    row = pd.DataFrame([customer_dict])
    X = preprocessor.transform(row[FEATURE_COLS])
    proba = model.predict_proba(X)[0][1]

    # SHAP for this single instance
    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        exp = shap.TreeExplainer(model)
        sv = exp.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
    else:
        # LinearExplainer needs training data as background
        if X_train is not None:
            exp = shap.LinearExplainer(model, X_train)
        else:
            exp = shap.LinearExplainer(model, X)
        sv = exp.shap_values(X)

    sv = sv[0]
    top_idx = np.argsort(np.abs(sv))[::-1][:3]
    top_factors = [(feature_names[i], sv[i]) for i in top_idx]

    return proba, top_factors


def gauge_chart(proba: float):
    """Plotly gauge for churn probability."""
    pct = round(proba * 100, 1)
    if pct < 25:
        color = "#2ECC71"
    elif pct < 50:
        color = "#F1C40F"
    elif pct < 75:
        color = "#E67E22"
    else:
        color = "#E74C3C"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 48}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 25], "color": "#D5F5E3"},
                {"range": [25, 50], "color": "#FEF9E7"},
                {"range": [50, 75], "color": "#FDEBD0"},
                {"range": [75, 100], "color": "#FADBD8"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": pct},
        },
        title={"text": "Churn Probability", "font": {"size": 18}},
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30))
    return fig


def risk_label(proba: float) -> tuple[str, str]:
    """Return (emoji+label, retention playbook) for a churn probability."""
    pct = proba * 100
    if pct < 25:
        return (
            "🟢 Faible (<25%)",
            "**Maintain Engagement**: Continue current service quality. Consider loyalty program enrollment and periodic satisfaction surveys.",
        )
    elif pct < 50:
        return (
            "🟡 Modéré (25–50%)",
            "**Preventive Action**: Schedule a relationship check-in within 30 days. Offer a personalized product bundle or fee reduction. Monitor activity closely.",
        )
    elif pct < 75:
        return (
            "🟠 Élevé (50–75%)",
            "**Urgent Intervention**: Assign a dedicated advisor immediately. Offer a competitive retention package (rate match, fee waiver, cashback). Initiate a root-cause survey.",
        )
    else:
        return (
            "🔴 Critique (>75%)",
            "**Critical Save Protocol**: Escalate to retention team within 24 h. Authorize maximum discount authority. Personal call from branch manager. Prepare win-back offer if churn occurs.",
        )
