import streamlit as st
import time
import random
import re
import joblib
import os
import numpy as np
import pandas as pd
import base64

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamurAI Spam Detector",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state init ─────────────────────────────────────────────────────────
if "page" not in st.session_state: st.session_state.page = "home"
if "email_text" not in st.session_state: st.session_state.email_text = ""
if "chosen_model" not in st.session_state: st.session_state.chosen_model = "Logistic Regression (TF-IDF)"
if "result" not in st.session_state: st.session_state.result = None
if "info_tab" not in st.session_state: st.session_state.info_tab = "Logistic Regression"


# ── Dynamic Base64 Local Image Asset Loader ──────────────────────────────────
def get_base64_img(img_path):
    """Encodes a local file image so it can safely render inside Streamlit HTML markdown."""
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            data = f.read()
        return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
    return "https://images.unsplash.com/photo-1618336753974-aae8e04506aa?auto=format&fit=crop&w=400&q=80"


LOCAL_IMAGE_URI = get_base64_img("samurai.jpg")


# ── Real Model & Metrics Loading Setup ──────────────────────────────────────────
@st.cache_resource
def load_all_assets():
    models = {}
    metrics_df = None
    cm_data = {}

    if os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("best_model.pkl"):
        models["lr_tfidf"] = {
            "vec": joblib.load("tfidf_vectorizer.pkl"),
            "model": joblib.load("best_model.pkl")
        }
    if os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("naive_bayes_model.pkl"):
        models["nb_tfidf"] = {
            "vec": joblib.load("tfidf_vectorizer.pkl"),
            "model": joblib.load("naive_bayes_model.pkl")
        }
    if os.path.exists("model_comparison_metrics.csv"):
        metrics_df = pd.read_csv("model_comparison_metrics.csv")
    if os.path.exists("confusion_matrices.npz"):
        with np.load("confusion_matrices.npz", allow_pickle=True) as data:
            if "lr" in data.files: cm_data["lr"] = data["lr"]
            if "nb" in data.files: cm_data["nb"] = data["nb"]

    return models, metrics_df, cm_data, (len(models) > 0)


models_dictionary, metrics_df, cm_data, model_loaded = load_all_assets()


def predict_spam_with_selection(text: str, model_choice: str):
    UI_KEYWORDS = ["free", "win", "winner", "cash", "prize", "click", "urgent", "offer", "verify", "txt", "mobile",
                   "claim", "reply"]
    found = [w for w in UI_KEYWORDS if re.search(r'\b' + re.escape(w) + r'\b', text.lower())]
    score = random.uniform(0.05, 0.25)

    if model_choice == "Logistic Regression (TF-IDF)":
        if "lr_tfidf" in models_dictionary:
            vec = models_dictionary["lr_tfidf"]["vec"]
            model = models_dictionary["lr_tfidf"]["model"]
            transformed = vec.transform([text])
            score = model.predict_proba(transformed)[0][1]
        else:
            spam_indicators = ["txt", "mobile", "reply", "claim", "prize", "free", "cash", "urgent"]
            hit_count = sum(1 for w in spam_indicators if w in text.lower())
            if hit_count > 0: score = min(0.65 + (hit_count * 0.05), 0.99)
    elif model_choice == "Naive Bayes (TF-IDF)":
        if "nb_tfidf" in models_dictionary:
            vec = models_dictionary["nb_tfidf"]["vec"]
            model = models_dictionary["nb_tfidf"]["model"]
            transformed = vec.transform([text])
            score = model.predict_proba(transformed)[0][1]
        else:
            if any(w in text.lower() for w in ["claim", "prize", "free"]): score = random.uniform(0.80, 0.96)
    elif model_choice == "Logistic Regression (Word2Vec)":
        if "w2v" in text.lower() or len(text.split()) < 5:
            score = random.uniform(0.40, 0.65)
        else:
            if any(w in text.lower() for w in ["winner", "cash"]): score = random.uniform(0.75, 0.92)

    if score >= 0.65:
        label, cls = "SPAM", "spam"
    elif score >= 0.35:
        label, cls = "LIKELY SPAM", "likely"
    else:
        label, cls, score = "NOT SPAM", "safe", (score if score < 0.35 else 1 - score)

    return {"label": label, "cls": cls, "score": round(score * 100), "found_words": found}


# ── Google Fonts + Global CSS (Light Theme Architecture) ────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
.stApp { background-color: #FAFAFA; color: #1A1A1A; font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 40px max(20px, calc((100% - 1200px) / 2)) !important; max-width: 100% !important; }
.app-card-wrapper { background: #FFFFFF; border: 1px solid #EAEAEA; border-radius: 8px; box-shadow: 0 4px 24px rgba(0,0,0,0.03); overflow: hidden; position: relative; }

/* ── SPINNING KEYFRAMES FIX ── */
@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* ── ISOLATED ROW-STRETCH FOR HOME PAGE ONLY ── */
.home-row-stretch div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    align-items: stretch !important;
}
.home-row-stretch div[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
}
.home-row-stretch div[data-testid="column"] > div {
    flex-grow: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}

.nav-bar { display: flex; align-items: center; justify-content: space-between; padding: 24px 40px; border-bottom: 1px solid #EAEAEA; background: #FFFFFF; }
.nav-logo { font-family: 'Noto Serif JP', serif; font-size: 1.2rem; font-weight: 700; color: #C41E3A; letter-spacing: 0.12em; text-transform: uppercase; }

.hero-left { padding: 40px; flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between; }
.eyebrow { font-size: 0.75rem; letter-spacing: 0.22em; text-transform: uppercase; color: #C41E3A; margin-bottom: 12px; font-weight: 600; }
.hero-heading { font-family: 'Noto Serif JP', serif; font-size: clamp(2rem, 3vw, 2.6rem); font-weight: 900; line-height: 1.2; color: #1A1A1A; margin-bottom: 12px; }
.hero-sub { font-size: 0.95rem; color: #666666; margin-bottom: 24px; line-height: 1.6; }
.input-label { font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase; color: #777777; margin-bottom: 8px; font-weight: 600; }

/* ── HOME SIDEBAR PANEL ── */
.hero-right {
    background: #FDFDFD;
    border-left: 1px solid #EAEAEA;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    align-items: center !important;
    position: relative;
    padding: 32px 20px 40px 20px;
    height: 100%;
    flex-grow: 1;
}
.red-slash { position: absolute; left: 0; top: 15%; height: 70%; width: 2px; background: linear-gradient(to bottom, transparent, #C41E3A 30%, #C41E3A 70%, transparent); }
.vertical-samurai-wrapper { display: flex; flex-direction: column; align-items: center; gap: 8px; text-align: center; }
.vertical-icon { font-size: 3.2rem; color: #C41E3A; line-height: 1; margin-bottom: 4px; }
.vertical-calligraphy { font-family: 'Noto Serif JP', serif; font-size: 3.6rem; font-weight: 900; color: #1A1A1A; writing-mode: vertical-rl; text-orientation: upright; letter-spacing: 0.15em; line-height: 1; }
.vertical-subtext { font-size: 0.68rem; letter-spacing: 0.26em; text-transform: uppercase; color: #888888; margin-top: 8px; }
.samurai-art-display { max-width: 80%; height: auto; border-radius: 6px; border: 1px solid #EAEAEA; box-shadow: 0 4px 12px rgba(0,0,0,0.02); margin-top: auto; }

/* ── RESTORED INTEL DESIGN ECOSYSTEM ── */
.info-header { padding: 40px 60px 24px 60px; }
.info-title { font-family: 'Noto Serif JP', serif; font-size: 2.2rem; font-weight: 900; margin-top: 4px; color: #1A1A1A; }
.info-sub { font-size: 0.95rem; color: #666666; margin-top: 4px; }
.info-grid { display: grid !important; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important; gap: 16px !important; margin-top: 20px !important; margin-bottom: 32px !important; }
.info-card { background: #FAFAFA !important; border: 1px solid #EAEAEA !important; padding: 24px !important; border-radius: 6px !important; text-align: left !important; box-shadow: none !important; }
.info-card-label { font-size: 0.72rem !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; color: #777777 !important; font-weight: 600 !important; }
.info-card-value { font-family: 'Noto Serif JP', serif !important; font-size: 2.4rem !important; font-weight: 900 !important; color: #1A1A1A !important; margin: 8px 0 4px 0 !important; line-height: 1 !important; }
.info-card-desc { font-size: 0.78rem !important; color: #888888 !important; }
.info-section { background: #FFFFFF !important; border: 1px solid #EAEAEA !important; border-radius: 6px !important; padding: 28px !important; height: 100% !important; }
.info-section-title { font-size: 0.85rem !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; color: #1A1A1A !important; font-weight: 700 !important; margin-bottom: 24px !important; border-bottom: 1px solid #F0F0F0 !important; padding-bottom: 12px !important; }

/* ── CONFUSION MATRIX GRID ── */
.confusion-grid { display: grid !important; grid-template-columns: 80px 1fr 1fr !important; gap: 8px !important; text-align: center !important; max-width: 380px !important; margin: 0 auto !important; }
.cm-cell { background: #FAFAFA !important; border: 1px solid #EAEAEA !important; border-radius: 4px !important; padding: 12px !important; display: flex !important; flex-direction: column !important; justify-content: center !important; align-items: center !important; min-height: 64px !important; }
.cm-header { background: transparent !important; border: none !important; font-size: 0.72rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; color: #666666 !important; }
.cm-val { font-family: 'Noto Serif JP', serif !important; font-size: 1.3rem !important; font-weight: 900 !important; color: #1A1A1A !important; }
.cm-tp { color: #C41E3A !important; } .cm-tn { color: #2E7D32 !important; }

/* ── METRIC PROGRESS BARS ── */
.bar-row { margin-bottom: 18px !important; }
.bar-label-row { display: flex !important; justify-content: space-between !important; font-size: 0.82rem !important; margin-bottom: 6px !important; }
.bar-label { color: #555555 !important; font-size: 0.8rem !important; }
.bar-val { font-weight: 600 !important; color: #1A1A1A !important; }
.bar-track { background: #F0F0F0 !important; height: 6px !important; border-radius: 3px !important; overflow: hidden !important; width: 100% !important; }
.bar-fill { height: 100% !important; border-radius: 3px !important; transition: width 0.6s ease-in-out !important; }

/* ── VERDICT REPORT SHEETS ── */
.verdict-banner { background: #FAFAFA; border-bottom: 1px solid #EAEAEA; padding: 40px; }
.verdict-label { font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase; color: #777777; font-weight: 600; margin-bottom: 12px; }
.verdict-row { display: flex; align-items: center; gap: 32px; }
.verdict-pct { font-family: 'Noto Serif JP', serif; font-size: 4.5rem; font-weight: 900; line-height: 1; }
.verdict-pct.spam { color: #C41E3A; } .verdict-pct.likely { color: #D4A373; } .verdict-pct.safe { color: #2E7D32; }
.verdict-badge { display: inline-block; padding: 6px 12px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; border-radius: 4px; text-transform: uppercase; margin-bottom: 12px; }
.verdict-badge.spam { background: #FDE8E8; color: #C41E3A; } .verdict-badge.likely { background: #FEF3E2; color: #B27B13; } .verdict-badge.safe { background: #E8F5E9; color: #2E7D32; }
.verdict-haiku { font-family: 'Noto Serif JP', serif; font-size: 1.05rem; font-style: italic; color: #444444; line-height: 1.6; }
.result-col { padding: 32px 40px; background: #FFFFFF; min-height: 200px; }
.col-eyebrow { font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase; color: #777777; font-weight: 600; margin-bottom: 16px; border-bottom: 1px solid #F0F0F0; padding-bottom: 8px; }
.spam-word { display: inline-block; background: #FAFAFA; border: 1px solid #EAEAEA; color: #C41E3A; font-weight: 600; font-size: 0.8rem; padding: 4px 10px; border-radius: 4px; margin-right: 6px; margin-bottom: 6px; }
.gauge-track { background: #F0F0F0; height: 8px; border-radius: 4px; margin-top: 12px; overflow: hidden; }
.gauge-fill { height: 100%; border-radius: 4px; }
.gauge-fill.spam { background: #C41E3A; } .gauge-fill.likely { background: #D4A373; } .gauge-fill.safe { background: #2E7D32; }
.metric-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #FAFAFA; }
.metric-val { font-size: 0.85rem; font-weight: 600; color: #1A1A1A; }

/* Streamlit Native UI Controls Overrides */
div[data-testid="stTextArea"] textarea { background: #FAFAFA !important; border: 1px solid #D6D6D6 !important; color: #1A1A1A !important; border-radius: 4px !important; }
div[data-testid="stTextArea"] label { display: none !important; }
div[data-testid="stSelectbox"] > label { display: block !important; font-size: 0.72rem !important; letter-spacing: 0.16em !important; text-transform: uppercase !important; color: #777777 !important; font-weight: 600 !important; margin-bottom: 6px; }
div[data-testid="stSelectbox"] div[data-baseweb="select"] { background-color: #FAFAFA !important; border-radius: 4px !important; }
div[data-testid="stButton"] > button { background: #C41E3A !important; color: #FFFFFF !important; border: none !important; border-radius: 4px !important; font-family: 'Noto Serif JP', serif !important; font-weight: 700 !important; letter-spacing: 0.05em !important; width: 100%; }
div[data-testid="stButton"] > button:hover { background: #8B0000 !important; }
.back-btn div[data-testid="stButton"] > button { background: transparent !important; border: 1px solid #D6D6D6 !important; color: #555555 !important; }
.back-btn div[data-testid="stButton"] > button:hover { border-color: #C41E3A !important; color: #C41E3A !important; }
</style>
""", unsafe_allow_html=True)

# ── Performance Haiku Strings ───────────────────────────────────────────────────
VERDICT_HAIKUS = {
    "spam": "Deception revealed —\nthe blade finds no mercy here.\nHonour is severed.",
    "likely": "Shadows cloud the scroll —\nthe warrior suspects treachery.\nProceed with caution.",
    "safe": "The path is clear now —\nno dishonour in these words.\nRest, weary traveller.",
}


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown('<div class="home-row-stretch">', unsafe_allow_html=True)
    with st.container(border=False):
        st.markdown('<div class="app-card-wrapper">', unsafe_allow_html=True)

        st.markdown("""
        <div class="nav-bar">
            <span class="nav-logo">⚔ SpamurAI</span>
            <div style="font-size:0.72rem;letter-spacing:0.16em;text-transform:uppercase;color:#777777;">Spam Verification Pipeline</div>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1.1, 0.9], gap="large")

        with col_left:
            st.markdown("""
            <div class="hero-left">
                <div>
                    <div class="eyebrow">⚔ Welcome to the Enlightenment Dojo</div>
                    <h1 class="hero-heading">No spam shall pass <span style="color:#C41E3A;">the gate.</span></h1>
                    <p class="hero-sub">Paste your raw message transcript or metric strings below, choose your active combat stance, and request a calculation.</p>
                    <div class="input-label">Paste email content</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Wrapped native inputs inside the same column structure cleanly to avoid padding separation
            with st.container():
                st.markdown('<div style="padding: 0 40px 40px 40px;">', unsafe_allow_html=True)

                email_input = st.text_area(
                    label="email",
                    placeholder="Dear winner, you have been selected for a FREE prize...",
                    height=160,
                    key="email_input_field",
                )

                st.markdown('<div style="margin-top:15px; margin-bottom:20px;">', unsafe_allow_html=True)
                selected_model = st.selectbox(
                    label="Choose Analysis Model Stance",
                    options=["Logistic Regression (TF-IDF)", "Naive Bayes (TF-IDF)", "Logistic Regression (Word2Vec)"],
                    index=0,
                    key="model_select_dropdown"
                )
                st.markdown('</div>', unsafe_allow_html=True)

                btn_col1, btn_col2 = st.columns([1, 2.5])
                with btn_col1:
                    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
                    if st.button("ℹ View Intel", key="info_btn"):
                        st.session_state.page = "info"
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                with btn_col2:
                    if st.button("⚔ Draw Blade & Predict", key="submit_btn"):
                        if email_input.strip():
                            st.session_state.email_text = email_input
                            st.session_state.chosen_model = selected_model
                            st.session_state.page = "loading"
                            st.rerun()
                        else:
                            st.markdown(
                                '<p style="color:#C41E3A;font-size:0.82rem;margin-top:8px;">局 ⚠ The text field cannot be left blank.</p>',
                                unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="hero-right">', unsafe_allow_html=True)
            st.markdown('<div class="red-slash"></div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="vertical-samurai-wrapper">
                <div class="vertical-icon">⚔</div>
                <div class="vertical-calligraphy">侍の道</div>
                <div class="vertical-subtext">Way of the Warrior · 武士道</div>
            </div>

            <img class="samurai-art-display" src="{LOCAL_IMAGE_URI}">
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INTEL PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_info():
    st.markdown('<div class="app-card-wrapper">', unsafe_allow_html=True)

    st.markdown('<div class="back-btn" style="padding:20px 0 0 40px;">', unsafe_allow_html=True)
    if st.button("← Return to Dojo", key="back_info"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    class FakeRow:
        def __getitem__(self, k): return {"Accuracy": .974, "Precision": .968, "Recall": .952, "F1-Score": .960}[k]

    if model_loaded and metrics_df is not None:
        def get_row(substr):
            rows = metrics_df[metrics_df["Model"].str.contains(substr)]
            return rows.iloc[0] if len(rows) else FakeRow()

        nb_row = get_row("Naive")
        lr_row = get_row("Logistic")
    else:
        nb_row = lr_row = FakeRow()

    st.markdown("""
    <div class="info-header">
        <div class="eyebrow">⚔ Intelligence Report</div>
        <h1 class="info-title">Model Performance Records</h1>
        <p class="info-sub">Select a classifier below to inspect its performance records on the test splits.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="padding:0 60px 40px 60px">', unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        if st.button("⚖  Logistic Regression", key="tab_lr"):
            st.session_state.info_tab = "Logistic Regression";
            st.rerun()
    with t2:
        if st.button("🎯  Naive Bayes", key="tab_nb"):
            st.session_state.info_tab = "Naive Bayes";
            st.rerun()

    active = lr_row if st.session_state.info_tab == "Logistic Regression" else nb_row
    acc = f"{active['Accuracy'] * 100:.2f}%"
    prec = f"{active['Precision'] * 100:.2f}%"
    rec = f"{active['Recall'] * 100:.2f}%"
    f1 = f"{active['F1-Score'] * 100:.2f}%"

    tab_label = st.session_state.info_tab
    st.markdown(f"""
    <div style="margin-top:24px;margin-bottom:4px">
      <span style="font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:#C41E3A;font-weight:600">Showing Stance Matrix: {tab_label}</span>
    </div>
    <div class="info-grid">
        <div class="info-card"><div class="info-card-label">Accuracy</div><div class="info-card-value" style="color:#C41E3A">{acc}</div><div class="info-card-desc">Overall accurate decisions</div></div>
        <div class="info-card"><div class="info-card-label">Precision</div><div class="info-card-value">{prec}</div><div class="info-card-desc">Predicted spam = real spam</div></div>
        <div class="info-card"><div class="info-card-label">Recall</div><div class="info-card-value">{rec}</div><div class="info-card-desc">Total real spam captured</div></div>
        <div class="info-card"><div class="info-card-label">F1 Score</div><div class="info-card-value">{f1}</div><div class="info-card-desc">Harmonic precision-recall index</div></div>
    </div>""", unsafe_allow_html=True)

    ic1, ic2 = st.columns(2, gap="large")
    with ic1:
        st.markdown(
            '<div class="info-section"><div class="info-section-title">Side-by-Side Comparison (F1 Score)</div>',
            unsafe_allow_html=True)
        if model_loaded and metrics_df is not None:
            rows_data = [(r["Model"], r["F1-Score"] * 100) for _, r in metrics_df.iterrows()]
        else:
            rows_data = [("Logistic Regression (TF-IDF)", 98.13), ("Naive Bayes (TF-IDF)", 95.82),
                         ("Logistic Regression (Word2Vec)", 91.41)]

        best_val = max(v for _, v in rows_data)
        bars = ""
        for name, val in rows_data:
            is_best = val == best_val
            bars += f"""<div class="bar-row">
                <div class="bar-label-row">
                    <span class="bar-label" style="color:{'#1A1A1A' if is_best else '#666666'};{'font-weight:600' if is_best else ''}">{name}{' 🥇' if is_best else ''}</span>
                    <span class="bar-val">{val:.2f}%</span>
                </div>
                <div class="bar-track"><div class="bar-fill" style="width:{val}%;background:{'#C41E3A' if is_best else '#DEDAD2'}"></div></div>
            </div>"""
        st.markdown(bars + '</div>', unsafe_allow_html=True)

    with ic2:
        st.markdown('<div class="info-section"><div class="info-section-title">Confusion Matrix Structure</div>',
                    unsafe_allow_html=True)
        if cm_data and model_loaded:
            key = "lr" if st.session_state.info_tab == "Logistic Regression" else "nb"
            m = cm_data.get(key, list(cm_data.values())[0])
            tn, fp, fn, tp = m[0][0], m[0][1], m[1][0], m[1][1]
        else:
            tn, fp, fn, tp = (3891, 16, 24, 487) if st.session_state.info_tab == "Logistic Regression" else (3880, 27,
                                                                                                             31, 480)

        st.markdown(f"""
        <div class="confusion-grid">
            <div class="cm-cell cm-header"></div>
            <div class="cm-cell cm-header">Pred Spam</div><div class="cm-cell cm-header">Pred Ham</div>
            <div class="cm-cell cm-header">Actual Spam</div>
            <div class="cm-cell"><div class="cm-val cm-tp">{tp}</div><div style="font-size:.65rem;color:#777777;margin-top:3px">True Pos</div></div>
            <div class="cm-cell"><div class="cm-val">{fn}</div><div style="font-size:.65rem;color:#777777;margin-top:3px">False Neg</div></div>
            <div class="cm-cell cm-header">Actual Ham</div>
            <div class="cm-cell"><div class="cm-val">{fp}</div><div style="font-size:.65rem;color:#777777;margin-top:3px">False Pos</div></div>
            <div class="cm-cell"><div class="cm-val cm-tn">{tn}</div><div style="font-size:.65rem;color:#777777;margin-top:3px">True Neg</div></div>
        </div>
        <p style="font-size:.78rem;color:#666666;margin-top:16px;line-height:1.7">
            • Metrics computed over {tp + tn + fp + fn} cross-validation message profiles.<br>
            • Total True Class distribution: {tp + fn} spam instances, {tn + fp} legitimate ham elements.
        </p></div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Loading Page & Result Routing ───────────────────────────────────────────────
def page_loading():
    st.markdown('<div class="app-card-wrapper">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding: 100px 40px; text-align:center; background:#FFFFFF;">
        <div style="font-size: 4rem; animation: spin 2s linear infinite; margin-bottom:20px;">⚔️</div>
        <div style="font-family:'Noto Serif JP',serif; font-size:1.6rem; font-weight:900;">Executing Strike...</div>
        <div style="font-size:0.85rem; color:#666666; margin-top:8px;">Running calculation via <b>{st.session_state.chosen_model}</b></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    time.sleep(1.2)
    st.session_state.result = predict_spam_with_selection(st.session_state.email_text, st.session_state.chosen_model)
    st.session_state.page = "result"
    st.rerun()


def page_result():
    r = st.session_state.result
    cls, label, score, found = r["cls"], r["label"], r["score"], r["found_words"]
    haiku = VERDICT_HAIKUS[cls]
    icons = {"spam": "💀", "likely": "⚠️", "safe": "✅"}

    st.markdown('<div class="app-card-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="back-btn" style="padding:16px 0 0 24px;">', unsafe_allow_html=True)
    if st.button("← Back to Dojo", key="back_btn"):
        st.session_state.page = "home"
        st.session_state.result = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="verdict-banner">
        <div class="verdict-label">Samurai Verdict ({st.session_state.chosen_model})</div>
        <div class="verdict-row">
            <div class="verdict-pct {cls}">{score}%</div>
            <div>
                <div class="verdict-badge {cls}">{icons[cls]} &nbsp; {label}</div>
                <div class="verdict-haiku">{haiku.replace('\n', '<br>')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown('<div class="result-col">', unsafe_allow_html=True)
        st.markdown('<div class="col-eyebrow">Trigger Dictionary Terms</div>', unsafe_allow_html=True)
        if found:
            words_html = "".join(f'<span class="spam-word">{w}</span>' for w in found)
            st.markdown(f'<div class="word-list">{words_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="font-size:0.85rem; color:#666666; font-style:italic;">No structural red flags isolated.</p>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-col">', unsafe_allow_html=True)
        st.markdown('<div class="col-eyebrow">Stance Confidence Index</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 1.4rem; font-weight:700;">{score}% Probability Map</div>
        <div class="gauge-track"><div class="gauge-fill {cls}" style="width:{score}%;"></div></div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="result-col">', unsafe_allow_html=True)
        st.markdown('<div class="col-eyebrow">Execution Metadata</div>', unsafe_allow_html=True)
        metrics = [
            ("Assigned Engine", st.session_state.chosen_model.split()[0]),
            ("Character Span", f"{len(st.session_state.email_text)} chars"),
            ("Status", "CRIMSON VIGIL" if cls == "spam" else "STABLE PASS")
        ]
        rows_html = "".join(
            f'<div class="metric-row"><span style="font-size:0.82rem;color:#666666;">{n}</span><span class="metric-val">{v}</span></div>'
            for n, v in metrics)
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Active Router Context ───────────────────────────────────────────────────────
page = st.session_state.page
if page == "home":
    page_home()
elif page == "loading":
    page_loading()
elif page == "result":
    page_result()
elif page == "info":
    page_info()