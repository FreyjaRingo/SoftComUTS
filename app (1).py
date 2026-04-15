import streamlit as st
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Soft Computing Prediktor Depresi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #fdf6f0 0%, #fce9e1 40%, #f7e6f0 100%);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1b2e 0%, #1a0f1a 100%);
        border-right: 1px solid rgba(212, 163, 115, 0.3);
    }
    [data-testid="stSidebar"] * {
        color: #f0d9c8 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e8b89a !important;
        font-family: 'Playfair Display', serif !important;
    }
    [data-testid="stSidebar"] .stSlider > label {
        color: #e8b89a !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }

    /* Slider track rail (background) */
    [data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSlider"] div,
    [data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div {
        background: rgba(255, 255, 255, 0.18) !important;
    }
    /* Slider filled portion */
    [data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSlider"] div[style*="left"],
    [data-testid="stSidebar"] [data-baseweb="slider"] div[class*="Fill"] {
        background: #d4836d !important;
    }
    /* Slider thumb */
    [data-testid="stSidebar"] [role="slider"] {
        background: #f0b090 !important;
        border: 3px solid #fff !important;
        box-shadow: 0 0 0 3px rgba(212, 131, 109, 0.5), 0 2px 6px rgba(0,0,0,0.4) !important;
        width: 20px !important;
        height: 20px !important;
    }
    /* Slider tick/min-max labels */
    [data-testid="stSidebar"] [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stTickBarMax"] {
        color: #c8a48a !important;
        font-size: 0.72rem !important;
    }

    /* ── Main header ── */
    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #6b2d3e;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        color: #7a4a36;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .battle-badge {
        display: inline-block;
        background: linear-gradient(135deg, #b5838d, #c77daa);
        color: white;
        font-family: 'Playfair Display', serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        padding: 4px 16px;
        border-radius: 30px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* ── Model cards ── */
    .model-card {
        background: rgba(255, 255, 255, 0.82);
        border-radius: 18px;
        padding: 1.4rem 1.2rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(181, 131, 141, 0.25);
        box-shadow: 0 4px 20px rgba(107, 45, 62, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(107, 45, 62, 0.13);
    }
    .model-card-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .model-card-subtitle {
        font-size: 0.8rem;
        color: #6b4848;
        margin-bottom: 1rem;
        font-style: italic;
    }
    .model-card-nb1 { border-top: 4px solid #4472c4; }
    .model-card-nb2 { border-top: 4px solid #e07b39; }
    .model-card-nb3 { border-top: 4px solid #2e7d32; }

    .title-nb1 { color: #1a3a6b; }
    .title-nb2 { color: #7a3000; }
    .title-nb3 { color: #1b4a1f; }

    /* ── Prediction badges ── */
    .pred-depresi {
        background: linear-gradient(135deg, #c62828, #e53935);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 10px 20px;
        border-radius: 12px;
        text-align: center;
        margin: 0.8rem 0;
        letter-spacing: 0.3px;
    }
    .pred-normal {
        background: linear-gradient(135deg, #2e7d32, #43a047);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 10px 20px;
        border-radius: 12px;
        text-align: center;
        margin: 0.8rem 0;
        letter-spacing: 0.3px;
    }

    /* ── Metric rows ── */
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px dashed rgba(181, 131, 141, 0.2);
        font-size: 0.82rem;
    }
    .metric-label { color: #5a3030; font-weight: 400; }
    .metric-value { font-weight: 600; color: #2a1010; }

    /* ── Score bar ── */
    .score-bar-bg {
        background: #e8c8b8;
        border-radius: 8px;
        height: 8px;
        margin-top: 6px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 8px;
        border-radius: 8px;
        transition: width 0.6s ease;
    }

    /* ── Winner banner ── */
    .winner-banner {
        background: linear-gradient(135deg, #6b2d3e, #b5838d);
        color: white;
        font-family: 'Playfair Display', serif;
        text-align: center;
        padding: 1rem 2rem;
        border-radius: 16px;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1.5rem 0 0.5rem 0;
        letter-spacing: 0.3px;
    }

    /* ── Info box ── */
    .info-box {
        background: rgba(255, 255, 255, 0.65);
        border-radius: 14px;
        padding: 1rem 1.3rem;
        border-left: 4px solid #b5838d;
        margin: 0.6rem 0;
        font-size: 0.85rem;
        color: #3d2020;
        line-height: 1.6;
    }

    /* ── Sidebar button ── */
    .stButton > button {
        background: linear-gradient(135deg, #b5838d, #c77daa) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        width: 100% !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px rgba(181, 131, 141, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #9e6b77, #b06695) !important;
        box-shadow: 0 6px 20px rgba(181, 131, 141, 0.55) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.6) !important;
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        color: #6b2d3e !important;
    }

    hr { border-color: rgba(181, 131, 141, 0.25) !important; }

    /* ── Main content headings ── */
    .main [data-testid="stMarkdownContainer"] h3,
    .main [data-testid="stMarkdownContainer"] h2,
    .main [data-testid="stMarkdownContainer"] h1,
    section[data-testid="stMain"] h1,
    section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3 {
        color: #6b2d3e !important;
        font-family: 'Playfair Display', serif !important;
    }

    /* ── Expander header text ── */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        color: #6b2d3e !important;
        font-weight: 600 !important;
    }

    /* ── st.metric ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.55);
        border-radius: 10px;
        padding: 0.4rem 0.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #5a3a2a !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #2a1010 !important;
    }
</style>
""", unsafe_allow_html=True)

METRICS_NB1 = {
    "accuracy":  0.9742,
    "precision": 0.0000,
    "recall":    0.0000,
    "f1_score":  0.0000,
}
METRICS_NB2 = {
    "accuracy":  0.9742,
    "precision": 0.5000,
    "recall":    0.4194,
    "f1_score":  0.4557,
}
METRICS_NB3 = {
    "accuracy":  0.9825,
    "precision": 0.7143,
    "recall":    0.5161,
    "f1_score":  0.5993,
}


def _load_json_safe(path):
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None

_m1 = _load_json_safe("metrics_notebook1.json")
_m3 = _load_json_safe("metrics_notebook3.json")
if _m1:
    METRICS_NB1.update({k: v for k, v in _m1.items() if k in METRICS_NB1})
if _m3:
    METRICS_NB3.update({k: v for k, v in _m3.items() if k in METRICS_NB3})

@st.cache_data(show_spinner=False)
def load_mf_params():
    initial_params = {
        'sleep_R':  [3.0, 3.0, 5.0, 7.0],
        'sleep_M':  [5.0, 7.5, 10.0],
        'sleep_T':  [9.0, 10.5, 12.0, 12.0],
        'social_R': [0.0, 0.0, 2.0, 4.0],
        'social_M': [2.0, 5.0, 8.0],
        'social_T': [6.0, 8.0, 12.0, 12.0],
        'screen_R': [0.0, 0.0, 0.5, 1.5],
        'screen_M': [0.5, 2.0, 3.5],
        'screen_T': [2.5, 3.5, 5.0, 5.0],
        'dep_R':    [0.0, 0.0, 0.2, 0.45],
        'dep_M':    [0.3, 0.5, 0.7],
        'dep_T':    [0.55, 0.75, 1.0, 1.0],
    }
    ga_params = dict(initial_params)

    _i = _load_json_safe("initial_mf_params.json")
    if _i:
        initial_params.update({k: v for k, v in _i.items()
                                if isinstance(v, list) and k in initial_params})

    _g = _load_json_safe("ga_optimized_params.json")
    if _g:
        ga_params.update({k: v for k, v in _g.items()
                          if isinstance(v, list) and k in ga_params})
        perf = _g.get("performance", {})
        if perf:
            METRICS_NB2["f1_score"]  = perf.get("notebook2_f1",        METRICS_NB2["f1_score"])
            METRICS_NB2["accuracy"]  = perf.get("notebook2_accuracy",   METRICS_NB2["accuracy"])
            METRICS_NB2["precision"] = perf.get("notebook2_precision",  METRICS_NB2["precision"])
            METRICS_NB2["recall"]    = perf.get("notebook2_recall",     METRICS_NB2["recall"])

    return initial_params, ga_params

INITIAL_PARAMS, GA_PARAMS = load_mf_params()
RULES = [
    (0, 2, 2, 2), (0, 2, 1, 2), (0, 1, 2, 2),
    (0, 1, 1, 1), (0, 0, 2, 1), (0, 0, 0, 1),
    (1, 2, 2, 2), (1, 2, 1, 1), (1, 1, 2, 1),
    (1, 0, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1),
    (2, 2, 2, 2), (2, 2, 1, 1), (2, 1, 2, 1),
    (2, 0, 0, 1), (2, 1, 0, 0), (2, 0, 0, 0),
]

def trimf(x: float, params) -> float:
    a, b, c = params
    if x <= a or x >= c:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (c - x) / (c - b) if c != b else 1.0

def trapmf(x: float, params) -> float:
    a, b, c, d = params
    if x <= a or x >= d:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    elif x <= c:
        return 1.0
    else:
        return (d - x) / (d - c) if d != c else 1.0

def gaussmf(x: float, center: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((x - center) / (sigma + 1e-8)) ** 2))

def _fuzzify(val: float, p: dict, prefix: str):
    results = []
    for sfx in ['R', 'M', 'T']:
        key = f"{prefix}_{sfx}"
        par = p[key]
        if len(par) == 4:
            results.append(trapmf(val, par))
        else:
            results.append(trimf(val, par))
    return results

def _defuzz_centroid(agg: np.ndarray, x_dep: np.ndarray) -> float:
    denom = agg.sum()
    if denom < 1e-9:
        return 0.25
    return float(np.dot(agg, x_dep) / denom)

def _mamdani_core(sleep_val, social_val, screen_val, params):
    x_dep = np.linspace(0, 1, 101)
    dep_mfs = []
    for sfx in ['R', 'M', 'T']:
        key = f"dep_{sfx}"
        par = params[key]
        if len(par) == 4:
            dep_mfs.append(np.array([trapmf(x, par) for x in x_dep]))
        else:
            dep_mfs.append(np.array([trimf(x, par) for x in x_dep]))

    mu_sleep  = _fuzzify(sleep_val,  params, 'sleep')
    mu_social = _fuzzify(social_val, params, 'social')
    mu_screen = _fuzzify(screen_val, params, 'screen')

    agg = np.zeros(len(x_dep))
    for (s_idx, a_idx, c_idx, d_idx) in RULES:
        firing = min(mu_sleep[s_idx], mu_social[a_idx], mu_screen[c_idx])
        if firing > 1e-9:
            cut = np.minimum(firing, dep_mfs[d_idx])
            agg = np.maximum(agg, cut)

    return _defuzz_centroid(agg, x_dep)


def predict_manual_fis(sleep_val: float, social_val: float, screen_val: float):
    score = _mamdani_core(
        np.clip(sleep_val,  3.0, 12.0),
        np.clip(social_val, 0.0, 12.0),
        np.clip(screen_val, 0.0,  5.0),
        INITIAL_PARAMS,
    )
    label = 1 if score >= 0.5 else 0
    return score, label


def predict_ga_fis(sleep_val: float, social_val: float, screen_val: float):
    score = _mamdani_core(
        np.clip(sleep_val,  3.0, 12.0),
        np.clip(social_val, 0.0, 12.0),
        np.clip(screen_val, 0.0,  5.0),
        GA_PARAMS,
    )
    label = 1 if score >= 0.5 else 0
    return score, label


def _get_anfis_params():
    _g = _load_json_safe("ga_optimized_params.json")
    if _g and "anfis_init" in _g:
        ai = _g["anfis_init"]
        centers, sigmas = [], []
        for var in ['sleep', 'social', 'screen']:
            row_c, row_s = [], []
            for sfx in ['R', 'M', 'T']:
                info = ai.get(var, {}).get(sfx, {})
                row_c.append(float(info.get('center', 0.5)))
                row_s.append(float(info.get('sigma',  0.3)))
            centers.append(row_c)
            sigmas.append(row_s)
        return np.array(centers), np.array(sigmas)

    centers, sigmas = [], []
    for prefix in ['sleep', 'social', 'screen']:
        row_c, row_s = [], []
        for sfx in ['R', 'M', 'T']:
            par = GA_PARAMS.get(f"{prefix}_{sfx}", [0, 0.5, 1])
            if len(par) == 4:
                c = (par[1] + par[2]) / 2
                w = par[-1] - par[0]
            else:
                c = par[1]
                w = par[-1] - par[0]
            row_c.append(c)
            row_s.append(max(w / 4, 0.1))
        centers.append(row_c)
        sigmas.append(row_s)
    return np.array(centers), np.array(sigmas)

_ANFIS_CENTERS, _ANFIS_SIGMAS = _get_anfis_params()
_X_MINS = np.array([0.0, 0.0, 0.0])
_X_MAXS = np.array([24.0, 24.0, 10.0])
_CONSEQUENTS = None

def _build_consequents():
    dep_vals = {0: 0.2, 1: 0.5, 2: 0.85}
    cons = []
    for s in range(3):
        for a in range(3):
            for c in range(3):
                matched_d = None
                for (si, ai, ci, di) in RULES:
                    if si == s and ai == a and ci == c:
                        matched_d = di
                        break
                if matched_d is None:
                    matched_d = min(round((s + a + c) / 3), 2)
                cons.append(dep_vals[matched_d])
    return np.array(cons)

_CONSEQUENTS = _build_consequents()

def predict_anfis(sleep_val: float, social_val: float, screen_val: float):
    x_raw  = np.array([sleep_val, social_val, screen_val], dtype=float)
    span   = _X_MAXS - _X_MINS
    x_norm = np.clip((x_raw - _X_MINS) / span, 0.0, 1.0)

    centers_norm = np.zeros_like(_ANFIS_CENTERS)
    sigmas_norm  = np.zeros_like(_ANFIS_SIGMAS)
    for i in range(3):
        s = span[i] if span[i] > 0 else 1.0
        centers_norm[i] = np.clip((_ANFIS_CENTERS[i] - _X_MINS[i]) / s, 0.05, 0.95)
        sigmas_norm[i]  = np.clip(_ANFIS_SIGMAS[i] / s, 0.05, 0.5)

    mu = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            mu[i, j] = gaussmf(x_norm[i], centers_norm[i, j], sigmas_norm[i, j])

    rule_strength = []
    for s in range(3):
        for a in range(3):
            for c in range(3):
                rule_strength.append(mu[0, s] * mu[1, a] * mu[2, c])
    rule_strength = np.array(rule_strength)

    total = rule_strength.sum()
    w_bar = rule_strength / total if total >= 1e-9 else np.ones(27) / 27

    raw_out = float(np.dot(w_bar, _CONSEQUENTS))
    score   = 1.0 / (1.0 + np.exp(-10 * (raw_out - 0.5)))
    label   = 1 if score >= 0.5 else 0
    return score, label


def render_model_card(title, subtitle, card_class, title_class, score, label,
                      metrics, color_hex):
    pred_html = (
        '<div class="pred-depresi">Terdeteksi: DEPRESI</div>'
        if label == 1 else
        '<div class="pred-normal">Terdeteksi: TIDAK DEPRESI</div>'
    )
    delta_from_threshold = score - 0.5
    delta_sign = "+" if delta_from_threshold >= 0 else ""
    bar_pct    = int(score * 100)
    bar_color  = "#c62828" if score >= 0.5 else "#2e7d32"

    metrics_html = ""
    for k, v in metrics.items():
        metrics_html += f'<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px dashed rgba(181,131,141,0.2);font-size:0.82rem;"><span style="color:#5a3030;font-weight:400;">{k}</span><span style="font-weight:600;color:#2a1010;">{v:.4f}</span></div>'

    html = f"""
    <div class="model-card {card_class}">
        <div class="model-card-title {title_class}">{title}</div>
        <div class="model-card-subtitle">{subtitle}</div>
        {pred_html}
        <div style="margin: 0.6rem 0 1rem 0;">
            <div style="display:flex;justify-content:space-between;font-size:0.78rem;
                        color:#5a3030;margin-bottom:4px;">
                <span>Skor Crisp / Probabilitas</span>
                <span><b style="color:{bar_color}">{score:.4f}</b>
                &nbsp;(Δ: {delta_sign}{delta_from_threshold:.4f})</span>
            </div>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{bar_pct}%;background:{bar_color};"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.7rem;
                        color:#6b3a3a;margin-top:3px;">
                <span>0.0 — Aman</span>
                <span style="color:#8b4040;">▲ threshold 0.5</span>
                <span>1.0 — Depresi</span>
            </div>
        </div>
        <hr style="margin:0.5rem 0;border-color:rgba(181,131,141,0.2);">
        <div style="font-size:0.78rem;font-weight:600;color:#5a2030;margin-bottom:0.4rem;">
            Metrik Historis (pada dataset penuh)
        </div>
        {metrics_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def plot_mf_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor='#fdf6f0')
    fig.suptitle(
        "Perbandingan Membership Functions: Manual (NB1) → GA-Tuned (NB2) → ANFIS Gaussian (NB3)",
        fontsize=11, fontweight='bold', color='#3d1f1f', y=1.02
    )

    vars_config = [
        ("Sleep Hours",              np.linspace(3, 12, 300), 'sleep',  3.0, 12.0),
        ("Social Media Hours",       np.linspace(0, 12, 300), 'social', 0.0, 12.0),
        ("Screen Time Before Sleep", np.linspace(0, 5,  300), 'screen', 0.0,  5.0),
    ]
    colors = {'R': '#1565C0', 'M': '#e07b39', 'T': '#2e7d32'}
    labels = {'R': 'Rendah',  'M': 'Sedang',  'T': 'Tinggi'}

    for ax, (var_name, x_arr, prefix, x_min, x_max) in zip(axes, vars_config):
        ax.set_facecolor('#fff9f6')
        for sfx in ['R', 'M', 'T']:
            c   = colors[sfx]
            lbl = labels[sfx]

            par1 = INITIAL_PARAMS.get(f"{prefix}_{sfx}", [0, 0.5, 1])
            y1 = [trapmf(x, par1) if len(par1) == 4 else trimf(x, par1) for x in x_arr]
            ax.plot(x_arr, y1, color=c, lw=1.0, ls=':', alpha=0.5)

            par2 = GA_PARAMS.get(f"{prefix}_{sfx}", par1)
            y2 = [trapmf(x, par2) if len(par2) == 4 else trimf(x, par2) for x in x_arr]
            ax.plot(x_arr, y2, color=c, lw=1.5, ls='--', alpha=0.7)

            idx    = ['sleep', 'social', 'screen'].index(prefix)
            sfx_i  = ['R', 'M', 'T'].index(sfx)
            span   = x_max - x_min if x_max > x_min else 1.0
            c_norm = np.clip((_ANFIS_CENTERS[idx, sfx_i] - x_min) / span, 0.05, 0.95)
            s_norm = np.clip(_ANFIS_SIGMAS[idx, sfx_i] / span, 0.05, 0.5)
            c_orig = c_norm * span + x_min
            s_orig = s_norm * span
            y3 = [gaussmf(x, c_orig, s_orig) for x in x_arr]
            ax.plot(x_arr, y3, color=c, lw=2.5, ls='-', alpha=0.95, label=lbl)
            ax.fill_between(x_arr, y3, alpha=0.08, color=c)

        ax.set_title(var_name, fontsize=10, fontweight='bold', color='#3d1f1f', pad=8)
        ax.set_xlabel("Nilai Input", fontsize=8, color='#6b3a2a')
        ax.set_ylabel("μ(x)",        fontsize=8, color='#6b3a2a')
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7.5, loc='upper right')
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    leg_handles = [
        mpatches.Patch(color='gray', alpha=0.4,  label='NB1 — Manual (titik-titik)'),
        mpatches.Patch(color='gray', alpha=0.65, label='NB2 — GA-Tuned (putus-putus)'),
        mpatches.Patch(color='gray', alpha=0.9,  label='NB3 — ANFIS Gaussian (solid)'),
    ]
    fig.legend(handles=leg_handles, loc='lower center', ncol=3,
               fontsize=8.5, framealpha=0.6, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    return fig



st.markdown('<div class="main-header">Perbandingan Prediktor Depresi</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">'
    'Soft Computing Prediktor Depresi Remaja &nbsp;·&nbsp; '
    'Manual FIS &nbsp;vs&nbsp; GA-Tuned FIS &nbsp;vs&nbsp; ANFIS Neuro-Fuzzy'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="text-align:center">'
    '<span class="battle-badge">Notebook 1 &nbsp;·&nbsp; Notebook 2 &nbsp;·&nbsp; Notebook 3</span>'
    '</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Parameter Input")
    st.markdown(
        '<div style="font-size:0.8rem;color:#c8a48a;margin-bottom:1rem;line-height:1.5;">'
        'Atur nilai input menggunakan slider di bawah, '
        'kemudian klik tombol untuk menjalankan analisis.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


    st.session_state.setdefault('sl_sleep',  6.0)
    st.session_state.setdefault('sl_social', 4.0)
    st.session_state.setdefault('sl_screen', 2.0)

    if 'apply_preset' in st.session_state:
        _p = st.session_state.pop('apply_preset')
        if _p == 'sehat':
            st.session_state['sl_sleep']  = 8.5
            st.session_state['sl_social'] = 1.5
            st.session_state['sl_screen'] = 0.5
        elif _p == 'risiko':
            st.session_state['sl_sleep']  = 4.5
            st.session_state['sl_social'] = 9.0
            st.session_state['sl_screen'] = 4.5
        st.rerun()


    sleep_hours = st.slider(
        "Jam Tidur per Hari",
        min_value=0.0, max_value=24.0, step=0.5,
        key='sl_sleep',
        help="Rekomendasi WHO untuk remaja: 8–10 jam per hari."
    )
    social_hours = st.slider(
        "Penggunaan Media Sosial (jam/hari)",
        min_value=0.0, max_value=24.0, step=0.5,
        key='sl_social',
        help="Total durasi penggunaan media sosial dalam sehari."
    )
    screen_hours = st.slider(
        "Layar Sebelum Tidur (jam)",
        min_value=0.0, max_value=10.0, step=0.25,
        key='sl_screen',
        help="Durasi penggunaan layar menjelang tidur. Berpengaruh pada produksi melatonin."
    )

    st.markdown("---")


    st.markdown(
        '<div style="font-size:0.8rem;color:#c8a48a;margin-bottom:0.5rem;">(Klik Profil apabila mau menggunakan template)</div>',
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2)
    if col_a.button("Profil Sehat", use_container_width=True):
        st.session_state['apply_preset'] = 'sehat'
        st.rerun()
    if col_b.button("Profil Berisiko", use_container_width=True):
        st.session_state['apply_preset'] = 'risiko'
        st.rerun()


    _s, _a, _c = (
        st.session_state.get('sl_sleep', 6.0),
        st.session_state.get('sl_social', 4.0),
        st.session_state.get('sl_screen', 2.0),
    )
    if _s == 8.5 and _a == 1.5 and _c == 0.5:
        st.info("Preset aktif: profil dengan tidur cukup, media sosial minimal, dan layar sebentar.")
    elif _s == 4.5 and _a == 9.0 and _c == 4.5:
        st.warning("Preset aktif: profil dengan kurang tidur, media sosial berlebihan, dan layar terlalu lama.")

    st.markdown("---")
    run_analysis = st.button("Jalankan Analisis", use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#9a7a6a;line-height:1.7;">'
        '<b>File yang diperlukan:</b><br>'
        '&nbsp;· <code>initial_mf_params.json</code><br>'
        '&nbsp;· <code>ga_optimized_params.json</code><br>'
        '&nbsp;· <code>metrics_notebook1.json</code> <i>(opsional)</i><br>'
        '&nbsp;· <code>metrics_notebook3.json</code> <i>(opsional)</i><br><br>'
        'Letakkan semua file di direktori yang sama dengan <code>app.py</code>.'
        '</div>',
        unsafe_allow_html=True,
    )


if not run_analysis:
    st.markdown("""
    <div class="info-box">
        <b>Selamat datang di The Intelligence Battle</b><br>
        Aplikasi ini membandingkan tiga pendekatan soft computing untuk mengevaluasi risiko depresi remaja
        berdasarkan tiga variabel: pola tidur, durasi penggunaan media sosial, dan waktu layar sebelum tidur.<br><br>
        Atur nilai input di sidebar sebelah kiri, lalu klik <b>Jalankan Analisis</b> untuk melihat
        perbandingan hasil ketiga model.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="model-card model-card-nb1">
            <div class="model-card-title title-nb1">Notebook 1 — Manual FIS</div>
            <div class="model-card-subtitle">Rule-based, Knowledge-driven</div>
            <div class="info-box">
            Parameter membership function ditentukan <b>secara manual</b> berdasarkan
            literatur kesehatan WHO dan psikologi remaja.<br><br>
            Menggunakan fungsi keanggotaan <b>segitiga &amp; trapesium</b>
            dengan inferensi Mamdani dan defuzzifikasi centroid.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="model-card model-card-nb2">
            <div class="model-card-title title-nb2">Notebook 2 — GA-Tuned FIS</div>
            <div class="model-card-subtitle">Evolutionary, Data-driven</div>
            <div class="info-box">
            Parameter MF dioptimasi menggunakan <b>Genetic Algorithm</b> (PyGAD)
            dengan F1-Score sebagai fungsi fitness.<br><br>
            Rule base tetap sama, namun posisi kurva MF disesuaikan secara
            <b>data-driven</b> agar lebih sesuai distribusi data aktual.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="model-card model-card-nb3">
            <div class="model-card-title title-nb3">Notebook 3 — ANFIS</div>
            <div class="model-card-subtitle">Neuro-Fuzzy, Gradient-based</div>
            <div class="info-box">
            <b>Adaptive Neuro-Fuzzy Inference System</b> menggabungkan
            jaringan saraf tiruan dengan logika fuzzy.<br><br>
            Menggunakan <b>Gaussian MF</b> yang differentiable, output Sugeno orde-0,
            dan sigmoid. Diinisialisasi dari parameter hasil GA.
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


with st.spinner("Menjalankan inferensi tiga model..."):
    score1, label1 = predict_manual_fis(sleep_hours, social_hours, screen_hours)
    score2, label2 = predict_ga_fis(sleep_hours, social_hours, screen_hours)
    score3, label3 = predict_anfis(sleep_hours, social_hours, screen_hours)

st.markdown("### Input yang Dianalisis")
ic1, ic2, ic3, ic4 = st.columns(4)
ic1.metric("Jam Tidur",          f"{sleep_hours:.1f} jam")
ic2.metric("Media Sosial",       f"{social_hours:.1f} jam/hari")
ic3.metric("Layar Sebelum Tidur", f"{screen_hours:.1f} jam")
ic4.metric("Threshold",          "0.5000")

st.markdown("---")
st.markdown("### Hasil Prediksi Tiga Model")

col1, col2, col3 = st.columns(3)

with col1:
    render_model_card(
        title="Manual FIS",
        subtitle="Notebook 1 — Human Expert Knowledge",
        card_class="model-card-nb1",
        title_class="title-nb1",
        score=score1,
        label=label1,
        metrics={
            "Accuracy":  METRICS_NB1["accuracy"],
            "Precision": METRICS_NB1["precision"],
            "Recall":    METRICS_NB1["recall"],
            "F1-Score":  METRICS_NB1["f1_score"],
        },
        color_hex="#4472c4",
    )
    st.metric(
        label="Skor Crisp — Manual FIS",
        value=f"{score1:.4f}",
        delta=f"{score1 - 0.5:+.4f} vs threshold",
        delta_color="inverse" if label1 == 0 else "normal",
    )

with col2:
    render_model_card(
        title="GA-Tuned FIS",
        subtitle="Notebook 2 — Evolutionary Optimization",
        card_class="model-card-nb2",
        title_class="title-nb2",
        score=score2,
        label=label2,
        metrics={
            "Accuracy":  METRICS_NB2["accuracy"],
            "Precision": METRICS_NB2["precision"],
            "Recall":    METRICS_NB2["recall"],
            "F1-Score":  METRICS_NB2["f1_score"],
        },
        color_hex="#e07b39",
    )
    st.metric(
        label="Skor Crisp — GA-Tuned FIS",
        value=f"{score2:.4f}",
        delta=f"{score2 - 0.5:+.4f} vs threshold",
        delta_color="inverse" if label2 == 0 else "normal",
    )

with col3:
    render_model_card(
        title="ANFIS / Neuro-Fuzzy",
        subtitle="Notebook 3 — Gradient-based Learning",
        card_class="model-card-nb3",
        title_class="title-nb3",
        score=score3,
        label=label3,
        metrics={
            "Accuracy":  METRICS_NB3["accuracy"],
            "Precision": METRICS_NB3["precision"],
            "Recall":    METRICS_NB3["recall"],
            "F1-Score":  METRICS_NB3["f1_score"],
        },
        color_hex="#2e7d32",
    )
    st.metric(
        label="Probabilitas ANFIS",
        value=f"{score3:.4f}",
        delta=f"{score3 - 0.5:+.4f} vs threshold",
        delta_color="inverse" if label3 == 0 else "normal",
    )

f1_scores   = [METRICS_NB1["f1_score"], METRICS_NB2["f1_score"], METRICS_NB3["f1_score"]]
model_names = ["Manual FIS (NB1)", "GA-Tuned FIS (NB2)", "ANFIS (NB3)"]
winner_idx  = int(np.argmax(f1_scores))
winner_name = model_names[winner_idx]
winner_f1   = f1_scores[winner_idx]

all_agree = (label1 == label2 == label3)
if all_agree:
    verdict_str = "DEPRESI" if label1 == 1 else "TIDAK DEPRESI"
    agreement_html = (
        f'<span style="background:rgba(46,125,50,0.85);padding:3px 12px;'
        f'border-radius:8px;font-size:0.85rem;font-weight:600;">'
        f'Semua model sepakat: {verdict_str}</span>'
    )
else:
    agreement_html = (
        '<span style="background:rgba(224,123,57,0.85);padding:3px 12px;'
        'border-radius:8px;font-size:0.85rem;font-weight:600;">'
        'Terdapat perbedaan prediksi antar model</span>'
    )

st.markdown(f"""
<div class="winner-banner">
    F1-Score Tertinggi: {winner_name} &nbsp;(F1 = {winner_f1:.4f})
    <br><div style="margin-top:8px;font-size:0.88rem;font-weight:400;">{agreement_html}</div>
</div>
""", unsafe_allow_html=True)

v1, v2, v3 = st.columns(3)
verdicts = [
    ("NB1 — Manual FIS",      label1, score1, "#4472c4"),
    ("NB2 — GA-Tuned FIS",    label2, score2, "#e07b39"),
    ("NB3 — ANFIS",           label3, score3, "#2e7d32"),
]
for col, (name, lbl, scr, clr) in zip([v1, v2, v3], verdicts):
    verdict_text = "DEPRESI" if lbl == 1 else "TIDAK DEPRESI"
    verdict_color = "#c62828" if lbl == 1 else "#1b5e20"
    col.markdown(
        f'<div style="text-align:center;padding:0.7rem;background:rgba(255,255,255,0.65);'
        f'border-radius:10px;border-top:3px solid {clr};">'
        f'<b style="color:{clr};font-size:0.82rem;">{name}</b><br>'
        f'<span style="font-size:0.95rem;font-weight:700;color:{verdict_color};">{verdict_text}</span><br>'
        f'<span style="color:#5a3030;font-size:0.75rem;">Skor: {scr:.4f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


st.markdown("---")
with st.expander("Lihat Perbandingan Membership Functions — Evolusi NB1 → NB2 → NB3"):
    st.markdown("""
    <div class="info-box">
    <b>Apa yang ditampilkan grafik ini?</b><br>
    Grafik memperlihatkan bagaimana kurva membership function berevolusi di setiap tahap pipeline:<br>
    &nbsp;· <b>Titik-titik</b> — Manual FIS (NB1), ditentukan secara manual oleh pakar<br>
    &nbsp;· <b>Putus-putus</b> — GA-Tuned (NB2), dioptimasi oleh Genetic Algorithm<br>
    &nbsp;· <b>Garis solid</b> — ANFIS (NB3), Gaussian MF yang dilatih dengan backpropagation<br><br>
    Pergeseran kurva menunjukkan bagaimana pendekatan data-driven dan gradient-based
    merevisi parameter awal pakar agar lebih sesuai distribusi data aktual.
    </div>
    """, unsafe_allow_html=True)

    fig = plot_mf_comparison()
    st.pyplot(fig, width='stretch')
    plt.close(fig)

    st.markdown(
        '<div style="font-size:0.78rem;color:#5a3a2a;text-align:center;margin-top:0.5rem;">'
        'Catatan: grafik GA dan ANFIS menggunakan nilai dari file JSON di direktori yang sama. '
        'Pastikan <code>ga_optimized_params.json</code> tersedia agar parameter yang ditampilkan akurat.'
        '</div>',
        unsafe_allow_html=True,
    )


with st.expander("Penjelasan Teknis — Cara Kerja Masing-Masing Model"):
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div class="info-box">
        <b>Manual FIS (Mamdani)</b><br>

        &nbsp; 1. **Fuzzifikasi** — Hitung derajat keanggotaan μ(x) dari setiap input
           menggunakan fungsi trapesium/segitiga (parameter dari pakar).<br>
        &nbsp; 2. **Rule Evaluation** — Firing strength = min(μ_sleep, μ_social, μ_screen)
           untuk setiap 18 rule IF–THEN.<br>
        &nbsp; 3. **Agregasi** — Gabungkan semua rule aktif dengan max-min composition.<br>
        &nbsp; 4. **Defuzzifikasi** — Centroid of area menghasilkan skor crisp ∈ [0,1].<br>
        &nbsp; 5. **Klasifikasi** — Skor ≥ 0.5 → Depresi (1), sebaliknya Normal (0).<br>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="info-box">
        <b>GA-Tuned FIS</b><br>

        Arsitektur identik dengan Manual FIS, namun titik-titik pembentuk MF
        (a, b, c, d) dioptimasi menggunakan Genetic Algorithm:

        - **Populasi**: 50 kromosom (array float 44D)
        - **Fitness**: F1-Score pada dataset penuh
        - **Seleksi**: Steady-State Selection
        - **Crossover**: Uniform crossover
        - **Mutasi**: 15% gen per generasi
        - **Repair**: np.sort() + ε-gap agar MF tetap valid secara geometri
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="info-box">
        <b>ANFIS (Neuro-Fuzzy)</b><br>

        Arsitektur 5 layer ANFIS:

        - **L1 Fuzzifikasi** — Gaussian MF: μ = exp(-(x-c)²/2σ²)
        - **L2 Rule Strength** — Product T-norm (lebih smooth dari min)
        - **L3 Normalisasi** — w̄ₖ = wₖ / Σwⱼ
        - **L4 Defuzzifikasi** — Sugeno orde-0: Σ(w̄ₖ·cₖ)
        - **L5 Sigmoid** — P(Depresi) = σ(output)

        Parameter μ, σ, dan cₖ dilatih dengan Adam optimizer (backpropagation),
        dengan inisialisasi dari center MF hasil GA (warm start).
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:0.75rem;color:#7a4a4a;padding:0.5rem 0 1rem 0;">'
    '<b>The Intelligence Battle</b> &nbsp;·&nbsp; Soft Computing Pipeline &nbsp;·&nbsp; '
    'Manual FIS → GA-Tuned FIS → ANFIS<br>'
    '<span style="color:#9a6a6a;">Dataset: Teen Mental Health &nbsp;·&nbsp; '
    'Variabel: sleep_hours, daily_social_media_hours, screen_time_before_sleep</span>'
    '</div>',
    unsafe_allow_html=True,
)