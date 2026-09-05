"""
app/components.py
------------------------------------------------------------
Reusable UI building blocks: metric strip, risk gauge, and the
model-insights charts. Keeps app/main.py focused on page layout.
------------------------------------------------------------
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

GAUGE_COLORS = {"LOW": "#34d399", "MEDIUM": "#fbbf24", "HIGH": "#fb7185"}

_TRANSPARENT_LAYOUT = dict(
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "#f5f6fb"},
)


def render_metric_strip(st, metrics: dict):
    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cols = st.columns(len(labels))
    for col, label, key in zip(cols, labels, keys):
        col.markdown(
            f"""<div class="metric-card"><div class="metric-label">{label}</div>
                    <div class="metric-value">{metrics[key]:.2f}</div></div>""",
            unsafe_allow_html=True,
        )


def risk_gauge_figure(probability: float, risk_level: str) -> go.Figure:
    color = GAUGE_COLORS[risk_level]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"color": "#f5f6fb", "size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9aa3b8"},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(52,211,153,0.20)"},
                    {"range": [35, 66], "color": "rgba(251,191,36,0.20)"},
                    {"range": [66, 100], "color": "rgba(251,113,133,0.20)"},
                ],
            },
        )
    )
    fig.update_layout(height=230, **_TRANSPARENT_LAYOUT)
    return fig


def model_comparison_figure(per_model_f1: dict) -> go.Figure:
    names = list(per_model_f1.keys())
    scores = [v * 100 for v in per_model_f1.values()]
    colors = ["#7c5cff" if n != "Stacking" else "#22d3ee" for n in names]
    fig = go.Figure(go.Bar(x=names, y=scores, marker_color=colors, text=[f"{s:.1f}" for s in scores], textposition="outside"))
    fig.update_layout(height=340, yaxis=dict(range=[0, max(scores) + 15], gridcolor="rgba(255,255,255,0.08)"), **_TRANSPARENT_LAYOUT)
    return fig


def confusion_matrix_figure(cm) -> go.Figure:
    cm = np.array(cm)
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Purples",
        x=["Predicted: Stay", "Predicted: Leave"], y=["Actual: Stay", "Actual: Leave"],
    )
    fig.update_layout(height=340, coloraxis_showscale=False, **_TRANSPARENT_LAYOUT)
    return fig


def feature_importance_figure(feature_importance_df, top_n: int = 15) -> go.Figure:
    top_feats = feature_importance_df.head(top_n).sort_values("Importance")
    fig = go.Figure(
        go.Bar(
            x=top_feats["Importance"], y=top_feats["Feature"], orientation="h",
            marker=dict(color=top_feats["Importance"], colorscale="Viridis"),
        )
    )
    fig.update_layout(height=420, xaxis=dict(gridcolor="rgba(255,255,255,0.08)"), **_TRANSPARENT_LAYOUT)
    return fig


def attrition_distribution_figure(attrition_counts: dict) -> go.Figure:
    stay, leave = attrition_counts.get("No", 0), attrition_counts.get("Yes", 0)
    fig = go.Figure(
        go.Bar(
            x=["Stay", "Leave"], y=[stay, leave],
            marker_color=["#22d3ee", "#f472b6"], text=[stay, leave], textposition="outside",
        )
    )
    fig.update_layout(height=420, yaxis=dict(gridcolor="rgba(255,255,255,0.08)"), **_TRANSPARENT_LAYOUT)
    return fig
