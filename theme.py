"""
app/theme.py
------------------------------------------------------------
CSS for the advanced dark glassmorphism theme, applied once at
the top of app/main.py.
------------------------------------------------------------
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

:root{
    --bg-0:#080b14; --bg-1:#0d1220;
    --card:rgba(255,255,255,0.045); --card-border:rgba(255,255,255,0.09);
    --accent-1:#7c5cff; --accent-2:#22d3ee; --accent-3:#f472b6;
    --good:#34d399; --warn:#fbbf24; --bad:#fb7185;
    --text-hi:#f5f6fb; --text-lo:#9aa3b8;
}
html, body, [class*="css"]{ font-family:'Manrope', sans-serif; }
.stApp{
    background:
        radial-gradient(circle at 15% 0%, rgba(124,92,255,0.16), transparent 42%),
        radial-gradient(circle at 85% 15%, rgba(34,211,238,0.12), transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(244,114,182,0.08), transparent 40%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
    color: var(--text-hi);
}
section[data-testid="stSidebar"]{
    background: rgba(8,11,20,0.85); border-right: 1px solid var(--card-border);
    backdrop-filter: blur(10px);
}
h1, h2, h3{ font-family:'Sora', sans-serif !important; letter-spacing: -0.01em; }
.hero{
    padding: 34px 38px; border-radius: 22px;
    background: linear-gradient(120deg, rgba(124,92,255,0.20), rgba(34,211,238,0.10) 60%, rgba(244,114,182,0.10));
    border: 1px solid var(--card-border); margin-bottom: 26px; position: relative; overflow: hidden;
}
.hero:before{ content:""; position:absolute; inset:0;
    background: radial-gradient(circle at 90% -10%, rgba(255,255,255,0.10), transparent 55%); }
.hero h1{
    font-size: 2.05rem; margin: 0 0 6px 0;
    background: linear-gradient(90deg, #ffffff, #b9c2ff 55%, #7dd3fc);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p{ color: var(--text-lo); font-size: 0.98rem; margin: 0; max-width: 720px; }
.badge-row{ margin-top:14px; display:flex; gap:10px; flex-wrap:wrap; }
.badge{ display:inline-flex; align-items:center; gap:6px; padding: 6px 14px; border-radius: 999px;
    background: rgba(255,255,255,0.06); border: 1px solid var(--card-border); font-size: 0.78rem; color: var(--text-lo); }
.glass-card{ background: var(--card); border: 1px solid var(--card-border); border-radius: 18px;
    padding: 22px 24px; backdrop-filter: blur(6px); margin-bottom: 18px; }
.metric-card{ background: var(--card); border: 1px solid var(--card-border); border-radius: 16px; padding: 18px 20px; }
.metric-label{ color: var(--text-lo); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-value{ font-family:'Sora', sans-serif; font-size: 1.7rem; font-weight: 700; color: var(--text-hi); margin-top: 4px;}
.risk-pill{ display:inline-block; padding: 5px 16px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.03em; }
.risk-LOW{ background: rgba(52,211,153,0.15); color: var(--good); border:1px solid rgba(52,211,153,0.35);}
.risk-MEDIUM{ background: rgba(251,191,36,0.15); color: var(--warn); border:1px solid rgba(251,191,36,0.35);}
.risk-HIGH{ background: rgba(251,113,133,0.15); color: var(--bad); border:1px solid rgba(251,113,133,0.35);}
.factor-chip{ display:inline-block; padding:6px 12px; margin: 4px 6px 0 0; border-radius: 10px;
    background: rgba(255,255,255,0.05); border: 1px solid var(--card-border); font-size: 0.82rem; color: var(--text-hi); }
.section-title{ font-family:'Sora', sans-serif; font-weight:700; font-size:1.05rem; color: var(--text-hi); margin-bottom: 2px; }
.section-sub{ color: var(--text-lo); font-size: 0.85rem; margin-bottom: 14px;}
.stButton>button{ background: linear-gradient(90deg, var(--accent-1), var(--accent-2)); color: #0a0e1a;
    font-weight: 700; border: none; border-radius: 12px; padding: 0.6rem 1.4rem; transition: transform 0.15s ease; }
.stButton>button:hover{ transform: translateY(-1px); filter: brightness(1.05); }
hr{ border-color: var(--card-border); }
[data-testid="stMetricValue"]{ color: var(--text-hi); }
</style>
"""


def apply_theme(st):
    st.markdown(CSS, unsafe_allow_html=True)


def render_hero(st):
    st.markdown(
        """
<div class="hero">
    <h1>🧭 Employee Turnover Risk Prediction and Early Intervention System</h1>
    <p>An ensemble machine-learning system (Random Forest, Decision Tree, KNN and
    Logistic Regression, combined via a Logistic-Regression stacking meta-model)
    that flags employees at elevated attrition risk so HR can act early.</p>
    <div class="badge-row">
        <span class="badge">🧠 Stacking Classifier</span>
        <span class="badge">📊 IQR-robust preprocessing</span>
        <span class="badge">⚡ Live inference</span>
        <span class="badge">🔒 Local &amp; private</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
