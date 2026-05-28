"""
app.py — Bank Customer Churn Prediction & Retention Intelligence Dashboard
Streamlit frontend with 4 tabs: Executive View, Segmentation, Models, Simulator.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pipeline import ensure_db, query_kpis, query_churn_donut, query_churn_by_geography, \
    query_churn_by_products, query_cumulative_churn_by_tenure, query_all_customers, build_where_clause
from utils import (
    run_kmeans_analysis, elbow_silhouette_chart, cluster_scatter,
    cluster_summary_table, CLUSTER_PLAYBOOKS,
    train_all_models, roc_curves_plot, confusion_matrix_plot,
    shap_feature_importance_plot, shap_waterfall_plot,
    predict_single_customer, gauge_chart, risk_label,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #0f3460;
    }
    [data-testid="stMetric"] label {color: #a8b2d1 !important; font-size: 0.82rem;}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {color: #e6f1ff !important;}
    div[data-testid="stTabs"] button[data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
    .playbook-card {
        background: #f0f4ff; border-radius: 10px; padding: 16px;
        border-left: 4px solid #3366ff; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Initialise DB ────────────────────────────────────────────────────────────
ensure_db()

# ── Sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/3d-fluency/94/bank-building.png", width=72)
st.sidebar.title("Churn Dashboard")
st.sidebar.markdown("---")

all_geo = ["France", "Germany", "Spain"]
all_gender = ["Male", "Female"]
all_products = [1, 2, 3, 4]
all_active = [0, 1]

sel_geo = st.sidebar.multiselect("Geography", all_geo, default=all_geo)
sel_gender = st.sidebar.multiselect("Gender", all_gender, default=all_gender)
sel_products = st.sidebar.multiselect("Num. of Products", all_products, default=all_products)
sel_active = st.sidebar.multiselect("Active Member", all_active, default=all_active,
                                     format_func=lambda x: "Yes" if x == 1 else "No")

where = build_where_clause(sel_geo, sel_gender, sel_products, sel_active)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Vue Exécutive",
    "Segmentation K-Means",
    "Modèles Prédictifs",
    "Simulateur Anti-Churn",
])

# ============================================================================
# TAB 1 — VUE EXÉCUTIVE
# ============================================================================
with tab1:
    kpis = query_kpis(where)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{int(kpis['total_customers']):,}")
    c2.metric("Global Churn Rate", f"{kpis['churn_rate']}%")
    c3.metric("Avg Balance (Churned)", f"${kpis['avg_balance_churned']:,.0f}" if kpis['avg_balance_churned'] else "N/A")
    c4.metric("Revenue At Risk", f"${kpis['revenue_at_risk']:,.0f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # Donut chart
    with col_left:
        donut_df = query_churn_donut(where)
        fig_donut = px.pie(
            donut_df, names="status", values="count", hole=0.55,
            color="status", color_discrete_map={"Churned": "#EF553B", "Retained": "#00CC96"},
            title="Churn vs Retention",
        )
        fig_donut.update_traces(textinfo="percent+label", pull=[0.04, 0])
        fig_donut.update_layout(template="plotly_white", height=380, showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    # Churn by Geography
    with col_right:
        geo_df = query_churn_by_geography(where)
        fig_geo = px.bar(
            geo_df, x="Geography", y="churn_rate_pct", color="Geography",
            text="churn_rate_pct", title="Churn Rate by Geography (%)",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_geo.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_geo.update_layout(template="plotly_white", height=380, showlegend=False,
                              yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig_geo, use_container_width=True)

    col_bl, col_br = st.columns(2)

    # Churn by Number of Products
    with col_bl:
        prod_df = query_churn_by_products(where)
        fig_prod = px.bar(
            prod_df, x="NumOfProducts", y="churn_rate_pct", text="churn_rate_pct",
            title="Churn Rate by Number of Products (%)",
            color="churn_rate_pct", color_continuous_scale="OrRd",
        )
        fig_prod.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_prod.update_layout(template="plotly_white", height=380, showlegend=False,
                               xaxis_title="Products", yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig_prod, use_container_width=True)

    # Cumulative churn by Tenure
    with col_br:
        tenure_df = query_cumulative_churn_by_tenure(where)
        tenure_df["cum_churned"] = tenure_df["churned"].cumsum()
        fig_tenure = px.line(
            tenure_df, x="Tenure", y="cum_churned", markers=True,
            title="Cumulative Churn by Tenure (months)",
            labels={"cum_churned": "Cumulative Churned", "Tenure": "Tenure (months)"},
        )
        fig_tenure.update_traces(line=dict(color="#636EFA", width=3), marker=dict(size=8))
        fig_tenure.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig_tenure, use_container_width=True)

# ============================================================================
# TAB 2 — SEGMENTATION K-MEANS
# ============================================================================
with tab2:
    raw_df = query_all_customers(where)
    if len(raw_df) < 20:
        st.warning("Not enough data for clustering with current filters. Please broaden your selection.")
    else:
        seg_df, K_vals, inertias, silhouettes, optimal_k = run_kmeans_analysis(raw_df)

        st.subheader("Elbow Method & Silhouette Analysis")
        st.plotly_chart(elbow_silhouette_chart(K_vals, inertias, silhouettes, optimal_k),
                        use_container_width=True)

        st.subheader("Customer Scatter — Balance vs. Tenure")
        st.plotly_chart(cluster_scatter(seg_df), use_container_width=True)

        st.subheader("Cluster Summary")
        summary = cluster_summary_table(seg_df)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.subheader("Retention Playbooks by Segment")
        for label, playbook in CLUSTER_PLAYBOOKS.items():
            st.markdown(f'<div class="playbook-card"><strong>{label}</strong><br/>{playbook}</div>',
                        unsafe_allow_html=True)

# ============================================================================
# TAB 3 — MODÈLES PRÉDICTIFS
# ============================================================================
with tab3:
    full_df = query_all_customers()  # Train on full unfiltered data
    with st.spinner("Training models — this may take a moment on first load…"):
        res = train_all_models(full_df)

    st.subheader("Model Comparison")
    metrics = res["metrics_df"].copy()
    # Highlight best row
    st.dataframe(
        metrics.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                                     color="#c6efce"),
        use_container_width=True, hide_index=True,
    )
    st.info(f" **Best model by ROC-AUC**: {res['best_name']}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(roc_curves_plot(res["fitted_models"], res["X_test"], res["y_test"]),
                        use_container_width=True)
    with col_r:
        st.plotly_chart(confusion_matrix_plot(res["best_model"], res["X_test"], res["y_test"],
                                               res["best_name"]),
                        use_container_width=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.plotly_chart(
            shap_feature_importance_plot(res["shap_values"], res["feature_names"]),
            use_container_width=True,
        )
    with col_s2:
        st.plotly_chart(
            shap_waterfall_plot(res["explainer"], res["shap_values"],
                                res["X_test"], res["feature_names"], idx=0),
            use_container_width=True,
        )

# ============================================================================
# TAB 4 — SIMULATEUR ANTI-CHURN EN TEMPS RÉEL
# ============================================================================
with tab4:
    st.subheader("Real-Time Churn Simulator")
    st.markdown("Adjust customer attributes below to see the predicted churn probability and personalized retention recommendations.")

    # Make sure models are trained
    full_df = query_all_customers()
    res = train_all_models(full_df)

    col_form1, col_form2, col_form3 = st.columns(3)

    with col_form1:
        sim_age = st.slider("Age", 18, 92, 38)
        sim_tenure = st.slider("Tenure (years)", 0, 10, 5)
        sim_credit = st.slider("Credit Score", 300, 850, 650)
        sim_salary = st.slider("Estimated Salary ($)", 0, 250_000, 100_000, step=5_000)

    with col_form2:
        sim_balance = st.slider("Balance ($)", 0, 260_000, 75_000, step=1_000)
        sim_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
        sim_geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
        sim_gender = st.selectbox("Gender", ["Male", "Female"])

    with col_form3:
        sim_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x else "No")
        sim_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x else "No")

    customer = {
        "CreditScore": sim_credit, "Geography": sim_geo, "Gender": sim_gender,
        "Age": sim_age, "Tenure": sim_tenure, "Balance": sim_balance,
        "NumOfProducts": sim_products, "HasCrCard": sim_card,
        "IsActiveMember": sim_active, "EstimatedSalary": sim_salary,
    }

    proba, top_factors = predict_single_customer(
        customer, res["best_model"], res["preprocessor"], res["feature_names"],
        X_train=res["X_train"],
    )

    col_gauge, col_risk = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(gauge_chart(proba), use_container_width=True)

    with col_risk:
        label, playbook = risk_label(proba)
        st.markdown(f"### Risk Level: {label}")
        st.markdown(playbook)

        st.markdown("#### Top 3 Personalized Risk Factors")
        for feat, val in top_factors:
            direction = "⬆️ increases" if val > 0 else "⬇decreases"
            st.markdown(f"- **{feat}**: SHAP = `{val:+.3f}` ({direction} churn risk)")

# ── Footer ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · SQLite · Scikit-learn · XGBoost · SHAP")
