import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import random
import json
import hashlib
from datetime import datetime
import pandas as pd
from collections import Counter, defaultdict

st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide")

IMAGE_SIZE   = 128
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/blood_reports", exist_ok=True)

def load_data(fname, default):
    ensure_dirs()
    p = f"data/{fname}"
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except:
            pass
    return default

def save_data(fname, data):
    ensure_dirs()
    with open(f"data/{fname}", "w") as f:
        json.dump(data, f, indent=2, default=str)

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

for k, v in [
    ("logged_in", False), ("username", ""), ("auth_mode", "login"),
    ("last_scan", None), ("confirm_delete", None),
    ("confirm_del_pat", None), ("editing_patient", None),
    ("viewing_blood_report", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --bg:      #04080f;
  --surface: #0b1220;
  --border:  #1a2a40;
  --cyan:    #00e5ff;
  --cdim:    #007a99;
  --white:   #e8f4f8;
  --muted:   #6a8a9a;
  --red:     #ff4d6d;
  --green:   #00e676;
  --amber:   #ffab40;
}

.stApp { background: var(--bg); color: var(--white); font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0; padding-bottom: 3rem; max-width: 1180px; }

.hero-wrap {
  position: relative; overflow: hidden;
  display: flex; flex-direction: column; align-items: center;
  padding: 3.2rem 1rem 2.4rem; text-align: center;
}
.hero-glow {
  position: absolute; width: 700px; height: 700px; border-radius: 50%;
  background: radial-gradient(circle, rgba(0,229,255,.09) 0%, transparent 68%);
  top: 50%; left: 50%; transform: translate(-50%,-50%); pointer-events: none;
}
.pulse-ring-wrap {
  position: relative; width: 88px; height: 88px;
  display: flex; align-items: center; justify-content: center; margin-bottom: 1.4rem;
}
.pulse-ring-wrap::before, .pulse-ring-wrap::after {
  content: ''; position: absolute; border-radius: 50%;
  border: 1.5px solid rgba(0,229,255,.7); animation: ring 2.6s ease-out infinite;
}
.pulse-ring-wrap::before { width: 100%; height: 100%; }
.pulse-ring-wrap::after  { width: 155%; height: 155%; animation-delay: .9s; opacity:.45; }
@keyframes ring {
  0%   { transform: scale(.8); opacity: .9; }
  100% { transform: scale(1.7); opacity: 0;  }
}
.brain { font-size: 3rem; filter: drop-shadow(0 0 16px var(--cyan)); }
.hero-title {
  font-family: 'Orbitron', sans-serif; font-weight: 900;
  font-size: clamp(2.2rem, 5vw, 3.6rem);
  letter-spacing: .14em; line-height: 1.1; margin: 0 0 .35rem; color: var(--white);
}
.hero-title .accent { color: var(--cyan); text-shadow: 0 0 18px var(--cyan), 0 0 50px rgba(0,229,255,.28); }
.hero-sub { font-weight: 300; font-size: .95rem; letter-spacing: .22em; text-transform: uppercase; color: var(--muted); margin: 0 0 1.8rem; }
.badges { display: flex; gap: .55rem; flex-wrap: wrap; justify-content: center; }
.badge {
  background: rgba(0,229,255,.07); border: 1px solid rgba(0,229,255,.22);
  border-radius: 20px; padding: .26rem .9rem;
  font-size: .7rem; letter-spacing: .08em; color: var(--cyan); font-weight: 500;
}
.divider { width: 100%; height: 1px; margin: 0 0 1.6rem; background: linear-gradient(90deg, transparent, var(--border), transparent); }

.stTabs [data-baseweb="tab-list"] { gap: 0; background: var(--surface); border-radius: 10px; padding: 4px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
  border-radius: 7px; color: var(--muted); background: transparent; border: none;
  font-family: 'DM Sans', sans-serif; font-weight: 500; font-size: .87rem; letter-spacing: .04em; padding: .48rem 1.2rem;
}
.stTabs [aria-selected="true"] { background: rgba(0,229,255,.12) !important; color: var(--cyan) !important; box-shadow: inset 0 0 0 1px rgba(0,229,255,.3); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.4rem; margin-bottom: .8rem; position: relative; overflow: hidden;
}
.card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--cyan), transparent); }
.card-label { font-family: 'Orbitron', sans-serif; font-size: .62rem; letter-spacing: .18em; color: var(--cyan); text-transform: uppercase; margin-bottom: .7rem; }

.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.4rem 1rem; text-align: center; position: relative; overflow: hidden; }
.stat-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--cyan), transparent); }
.stat-num { font-family: 'Orbitron', sans-serif; font-size: 2rem; font-weight: 900; color: var(--cyan); line-height: 1; }
.stat-label { font-size: .72rem; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; margin-top: .4rem; }
.stat-icon { font-size: 1.4rem; margin-bottom: .4rem; }

.sec {
  font-family: 'Orbitron', sans-serif; font-size: .88rem; font-weight: 700;
  color: var(--white); letter-spacing: .1em; margin: 1.4rem 0 .8rem;
  display: flex; align-items: center; gap: .6rem;
}
.sec::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.rbox { border-radius: 11px; padding: .9rem 1.2rem; margin-top: .6rem; border-left: 4px solid; font-size: .9rem; font-weight: 500; }
.r-green { background: rgba(0,230,118,.08); border-color: var(--green); color: var(--green); }
.r-red   { background: rgba(255,77,109,.08); border-color: var(--red);   color: var(--red); }
.r-amber { background: rgba(255,171,64,.08); border-color: var(--amber); color: var(--amber); }
.r-cyan  { background: rgba(0,229,255,.07);  border-color: var(--cyan);  color: var(--cyan); }

.empty-state {
  min-height: 200px; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: .8rem;
  border: 1px dashed var(--border); border-radius: 14px; background: var(--surface);
}

.stTextInput input, .stNumberInput input, .stTextArea textarea {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important; color: var(--white) !important; font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
  border-color: var(--cyan) !important; box-shadow: 0 0 0 2px rgba(0,229,255,.15) !important;
}
label { color: var(--muted) !important; font-size: .82rem !important; letter-spacing: .05em; }

/* ══ ALL regular buttons — dark teal style ══ */
div.stButton > button {
  background: linear-gradient(135deg, #006680, #008fa6) !important;
  color: #ffffff !important; font-family: 'Orbitron', sans-serif !important;
  font-size: .74rem !important; font-weight: 700 !important; letter-spacing: .1em !important;
  border: none !important; border-radius: 8px !important; padding: .6rem 2rem !important;
  text-transform: uppercase !important; transition: all .2s !important;
}
div.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 22px rgba(0,229,255,.28) !important;
  background: linear-gradient(135deg, #007a99, #00b8d9) !important;
}

/* ══ EDIT (pencil) icon button — cyan outline style ══ */
.edit-btn > div.stButton > button {
  background: linear-gradient(135deg, #031a22, #042a38) !important;
  border: 1.5px solid rgba(0,229,255,.55) !important;
  color: #00e5ff !important;
  width: 38px !important; height: 38px !important;
  min-width: 38px !important; max-width: 38px !important;
  padding: 0 !important; font-size: 1rem !important;
  letter-spacing: 0 !important; text-transform: none !important;
  font-family: inherit !important; display: inline-flex !important;
  align-items: center !important; justify-content: center !important;
  box-shadow: 0 0 8px rgba(0,229,255,.12) !important;
}
.edit-btn > div.stButton > button:hover {
  background: rgba(0,229,255,.12) !important;
  border-color: var(--cyan) !important;
  box-shadow: 0 0 16px rgba(0,229,255,.3) !important;
  transform: translateY(-1px) !important;
}

/* ══ DELETE (trash) icon button — red outline style ══ */
.del-btn > div.stButton > button {
  background: linear-gradient(135deg, #220308, #380612) !important;
  border: 1.5px solid rgba(255,77,109,.55) !important;
  color: #ff4d6d !important;
  width: 38px !important; height: 38px !important;
  min-width: 38px !important; max-width: 38px !important;
  padding: 0 !important; font-size: 1rem !important;
  letter-spacing: 0 !important; text-transform: none !important;
  font-family: inherit !important; display: inline-flex !important;
  align-items: center !important; justify-content: center !important;
  box-shadow: 0 0 8px rgba(255,77,109,.12) !important;
}
.del-btn > div.stButton > button:hover {
  background: rgba(255,77,109,.18) !important;
  border-color: var(--red) !important;
  box-shadow: 0 0 16px rgba(255,77,109,.3) !important;
  transform: translateY(-1px) !important;
}

/* ══ VIEW BLOOD REPORT button — amber style ══ */
.view-report-btn > div.stButton > button {
  background: linear-gradient(135deg, #1a0d00, #2a1a00) !important;
  border: 1.5px solid rgba(255,171,64,.55) !important;
  color: #ffab40 !important;
  font-size: .72rem !important; padding: .45rem 1.1rem !important;
  letter-spacing: .06em !important; font-weight: 700 !important;
  box-shadow: 0 0 8px rgba(255,171,64,.1) !important;
}
.view-report-btn > div.stButton > button:hover {
  background: rgba(255,171,64,.12) !important;
  border-color: var(--amber) !important;
  box-shadow: 0 4px 14px rgba(255,171,64,.25) !important;
  transform: translateY(-1px) !important;
}

/* ══ Blood report viewer panel ══ */
.blood-report-panel {
  background: #0c1118; border: 1px solid rgba(255,171,64,.3);
  border-radius: 12px; padding: 1.2rem 1.4rem; margin-top: .6rem;
}

/* ══ Number input +/- buttons — force dark ══ */
.stNumberInput [data-testid="stNumberInputField"] { background: var(--surface) !important; }
.stNumberInput button {
  background: #0b1220 !important;
  border: 1px solid #1a2a40 !important;
  color: #00e5ff !important;
  border-radius: 6px !important;
  min-width: unset !important;
  padding: .2rem .45rem !important;
  font-size: .9rem !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  font-family: inherit !important;
  font-weight: 400 !important;
  box-shadow: none !important;
}
.stNumberInput button:hover {
  background: rgba(0,229,255,.1) !important;
  border-color: #00e5ff !important;
  transform: none !important;
  box-shadow: none !important;
}

/* ══ Save button — dark green ══ */
.save-btn > div.stButton > button,
.save-btn > div.stFormSubmitButton > button {
  background: #061a0e !important;
  border: 1px solid #00e676 !important;
  color: #00e676 !important;
  font-weight: 700 !important;
}
.save-btn > div.stButton > button:hover,
.save-btn > div.stFormSubmitButton > button:hover {
  background: rgba(0,230,118,.12) !important;
  transform: none !important;
}

/* ══ Cancel button — dark red ══ */
.cancel-btn > div.stButton > button,
.cancel-btn > div.stFormSubmitButton > button {
  background: #1a0306 !important;
  border: 1px solid #ff4d6d !important;
  color: #ff4d6d !important;
}
.cancel-btn > div.stButton > button:hover,
.cancel-btn > div.stFormSubmitButton > button:hover {
  background: rgba(255,77,109,.12) !important;
  transform: none !important;
}

/* ══ Danger confirm button — dark red ══ */
.danger-btn > div.stButton > button {
  background: #1a0306 !important;
  border: 1px solid #ff4d6d !important;
  color: #ff4d6d !important;
}
.danger-btn > div.stButton > button:hover {
  background: rgba(255,77,109,.15) !important;
  transform: none !important;
}

/* ══ Form submit buttons ══ */
div.stFormSubmitButton > button {
  background: linear-gradient(135deg, #006680, #008fa6) !important;
  color: #ffffff !important; font-family: 'Orbitron', sans-serif !important;
  font-size: .74rem !important; font-weight: 700 !important; letter-spacing: .1em !important;
  border: none !important; border-radius: 8px !important; padding: .6rem 2rem !important;
  text-transform: uppercase !important; transition: all .2s !important;
}
div.stFormSubmitButton > button:hover {
  background: linear-gradient(135deg, #007a99, #00b8d9) !important;
  box-shadow: 0 6px 22px rgba(0,229,255,.28) !important;
}

/* ══ File uploader — dark ══ */
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1px dashed var(--cdim) !important;
  border-radius: 12px !important; padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: var(--cyan) !important; }
[data-testid="stFileUploader"] section { background: transparent !important; }
[data-testid="stFileUploader"] button {
  background: #061420 !important;
  border: 1px solid var(--border) !important;
  color: var(--cyan) !important;
  border-radius: 7px !important;
  font-size: .72rem !important;
  padding: .4rem 1rem !important;
  text-transform: uppercase !important;
  letter-spacing: .06em !important;
}
[data-testid="stFileUploader"] button:hover {
  border-color: var(--cyan) !important;
  background: rgba(0,229,255,.08) !important;
  transform: none !important;
  box-shadow: none !important;
}

/* ══ Expander — dark ══ */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] > details > summary {
  background: var(--surface) !important;
  color: var(--cyan) !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] > details[open] > summary {
  border-radius: 12px 12px 0 0 !important;
}

.stProgress > div > div { background: linear-gradient(90deg, #006680, #00e5ff) !important; border-radius: 4px; }
.stImage img { border: 1px solid var(--border); border-radius: 10px; }
.stSpinner > div { border-top-color: var(--cyan) !important; }

.patient-row {
  display: flex; align-items: center; gap: .9rem; padding: .65rem 1rem;
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
}

.login-card { background: var(--surface); border: 1px solid var(--border); border-radius: 18px; padding: 2.5rem 2rem; position: relative; overflow: hidden; }
.login-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, transparent, var(--cyan), transparent); }
.login-title { font-family: 'Orbitron', sans-serif; font-size: 1rem; color: var(--cyan); letter-spacing: .12em; text-align: center; margin-bottom: .3rem; }
.login-sub { font-size: .8rem; color: var(--muted); text-align: center; margin-bottom: 1.6rem; }

.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: .6rem 1.2rem; background: var(--surface); border-bottom: 1px solid var(--border);
  margin-bottom: 0; position: sticky; top: 0; z-index: 999;
}
.topbar-brand { font-family: 'Orbitron', sans-serif; font-size: .9rem; color: var(--white); letter-spacing: .1em; }
.topbar-brand span { color: var(--cyan); }
.topbar-user { font-size: .8rem; color: var(--muted); }

div[data-baseweb="select"] > div { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; color: var(--white) !important; }
.stDataFrame { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.stRadio > label { color: var(--muted) !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--white) !important; }

.scan-card { background: #0d1828; border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: .6rem; }
.scan-card-header { font-family: 'Orbitron', sans-serif; font-size: .7rem; letter-spacing: .1em; color: var(--muted); margin-bottom: .8rem; }
.scan-detail-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: .8rem; margin-top: .8rem; }
.scan-detail-item { display: flex; flex-direction: column; gap: .2rem; }
.scan-detail-label { font-size: .7rem; color: var(--muted); letter-spacing: .06em; text-transform: uppercase; }
.scan-detail-value { font-size: .9rem; color: var(--white); font-weight: 500; }

.pill { display: inline-block; border-radius: 20px; padding: .2rem .75rem; font-size: .72rem; font-weight: 600; letter-spacing: .05em; margin-right: .3rem; }
.pill-yes  { background: rgba(255,77,109,.15);  color: var(--red);   border: 1px solid rgba(255,77,109,.4); }
.pill-no   { background: rgba(0,230,118,.12);   color: var(--green); border: 1px solid rgba(0,230,118,.35); }
.pill-info { background: rgba(0,229,255,.10);   color: var(--cyan);  border: 1px solid rgba(0,229,255,.3); }

.edit-panel {
  background: #0c1622; border: 1px solid rgba(0,229,255,.3); border-top: none;
  border-radius: 0 0 10px 10px; padding: 1.2rem 1.4rem 1.4rem; margin-bottom: .5rem;
}
.del-panel {
  background: rgba(255,77,109,.04); border: 1px solid rgba(255,77,109,.25); border-top: none;
  border-radius: 0 0 10px 10px; padding: 1rem 1.4rem; margin-bottom: .5rem;
}
</style>
""", unsafe_allow_html=True)


def dark_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    for obj in (fig, ax):
        try: obj.set_facecolor('#0b1220')
        except: pass
    ax.tick_params(colors='#6a8a9a', labelsize=8)
    ax.xaxis.label.set_color('#6a8a9a')
    ax.yaxis.label.set_color('#6a8a9a')
    ax.title.set_color('#e8f4f8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2a40')
    return fig, ax

def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    return np.array(image) / 255.0

@st.cache_resource
def build_model(num_classes):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.optimizers import Adam
    base = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
    for layer in base.layers: layer.trainable = False
    for layer in base.layers[-4:-1]: layer.trainable = True
    m = Sequential([Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)), base,
                    Flatten(), Dropout(0.3), Dense(128, activation='relu'),
                    Dropout(0.2), Dense(num_classes, activation='softmax')])
    m.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return m

@st.cache_resource
def load_saved_model(path):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    try: return load_model(path, compile=False)
    except: pass
    try:
        from tensorflow.keras import layers as kl
        orig = kl.InputLayer.from_config
        @classmethod
        def patched(cls, config):
            config.pop("batch_shape", None); config.pop("optional", None)
            if "shape" not in config: config["shape"] = (IMAGE_SIZE, IMAGE_SIZE, 3)
            return orig.__func__(cls, config)
        kl.InputLayer.from_config = patched
        try: return load_model(path, compile=False)
        finally: kl.InputLayer.from_config = orig
    except: pass
    try:
        m = build_model(len(CLASS_LABELS)); m.load_weights(path); return m
    except Exception as e: raise RuntimeError(f"Could not load model: {e}")

def delete_scan(scan_id, scans, patients):
    scans[:] = [s for s in scans if s.get("id") != scan_id]
    for p in patients:
        if "scans" in p and scan_id in p["scans"]: p["scans"].remove(scan_id)
    save_data("scans.json", scans); save_data("patients.json", patients)

def delete_patient(patient_id, patients, scans):
    pat = next((p for p in patients if p.get("id") == patient_id), None)
    if pat:
        scans[:] = [s for s in scans if s.get("patient_name") != pat["name"]]
        save_data("scans.json", scans)
    patients[:] = [p for p in patients if p.get("id") != patient_id]
    save_data("patients.json", patients)


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ═══════════════════════════════════════════════════════════════════════════════
def show_auth():
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-glow"></div>
      <div class="pulse-ring-wrap"><span class="brain">🧠</span></div>
      <h1 class="hero-title">NEURO<span class="accent">SCAN</span> AI</h1>
      <p class="hero-sub">Diagnostic Centre Management System</p>
      <div class="badges">
        <span class="badge">⚡ VGG16 Architecture</span>
        <span class="badge">🎯 4-Class Detection</span>
        <span class="badge">🔬 MRI Analysis</span>
        <span class="badge">🏥 Centre Management</span>
      </div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        users = load_data("users.json", {})
        mode  = st.session_state.auth_mode
        if mode == "login":
            st.markdown("""
            <div class="login-card">
              <div class="login-title">🔑 SIGN IN</div>
              <div class="login-sub">Welcome back to NeuroScan AI</div>
            </div>""", unsafe_allow_html=True)
            uname = st.text_input("Username", placeholder="Enter your username")
            pwd   = st.text_input("Password", type="password", placeholder="Enter your password")
            if st.button("🔑  Sign In", use_container_width=True):
                if uname in users and users[uname]["password"] == hash_pw(pwd):
                    st.session_state.logged_in = True; st.session_state.username = uname; st.rerun()
                else:
                    st.markdown('<div class="rbox r-red">❌ Invalid username or password.</div>', unsafe_allow_html=True)
            st.markdown('<p style="text-align:center;color:var(--muted);font-size:.83rem;margin-top:1rem">No account yet?</p>', unsafe_allow_html=True)
            if st.button("✨  Create Account", use_container_width=True):
                st.session_state.auth_mode = "signup"; st.rerun()
        else:
            st.markdown("""
            <div class="login-card">
              <div class="login-title">✨ CREATE ACCOUNT</div>
              <div class="login-sub">Join the NeuroScan AI platform</div>
            </div>""", unsafe_allow_html=True)
            s1, s2   = st.columns(2)
            new_u    = s1.text_input("Username", placeholder="Choose username")
            new_name = s2.text_input("Full Name", placeholder="Your full name")
            new_role = st.selectbox("Role", ["Radiologist", "Doctor", "Technician", "Admin"])
            new_pwd  = st.text_input("Password", type="password", placeholder="Min 6 characters")
            conf_pwd = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
            if st.button("✨  Create Account", use_container_width=True):
                if not all([new_u, new_name, new_pwd, conf_pwd]):
                    st.markdown('<div class="rbox r-amber">⚠ All fields required.</div>', unsafe_allow_html=True)
                elif new_u in users:
                    st.markdown('<div class="rbox r-red">❌ Username taken.</div>', unsafe_allow_html=True)
                elif new_pwd != conf_pwd:
                    st.markdown('<div class="rbox r-red">❌ Passwords do not match.</div>', unsafe_allow_html=True)
                elif len(new_pwd) < 6:
                    st.markdown('<div class="rbox r-amber">⚠ Min 6 characters.</div>', unsafe_allow_html=True)
                else:
                    users[new_u] = {"name": new_name, "role": new_role,
                                    "password": hash_pw(new_pwd), "created": datetime.now().isoformat()}
                    save_data("users.json", users)
                    st.markdown('<div class="rbox r-green">✅ Account created! Sign in.</div>', unsafe_allow_html=True)
                    st.session_state.auth_mode = "login"; st.rerun()
            st.markdown('<p style="text-align:center;color:var(--muted);font-size:.83rem;margin-top:1rem">Already have an account?</p>', unsafe_allow_html=True)
            if st.button("← Back to Sign In", use_container_width=True):
                st.session_state.auth_mode = "login"; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
def show_app():
    users     = load_data("users.json", {})
    user_info = users.get(st.session_state.username, {})
    patients  = load_data("patients.json", [])
    scans     = load_data("scans.json", [])

    tb1, tb2 = st.columns([5, 1])
    with tb1:
        st.markdown(f"""
        <div class="topbar">
          <div class="topbar-brand">NEURO<span>SCAN</span> AI &nbsp;·&nbsp;
            <span style="color:var(--muted);font-size:.78rem;font-family:'DM Sans',sans-serif">Diagnostic Centre Platform</span>
          </div>
          <div class="topbar-user">
            👤 &nbsp;<strong style="color:var(--white)">{user_info.get('name', st.session_state.username)}</strong>
            &nbsp;·&nbsp; {user_info.get('role','User')}
          </div>
        </div>""", unsafe_allow_html=True)
    with tb2:
        st.markdown("<div style='margin-top:.3rem'></div>", unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False; st.session_state.username = ""; st.rerun()

    st.markdown("""
    <div class="hero-wrap" style="padding:2.4rem 1rem 1.8rem">
      <div class="hero-glow"></div>
      <div class="pulse-ring-wrap"><span class="brain">🧠</span></div>
      <h1 class="hero-title">NEURO<span class="accent">SCAN</span> AI</h1>
      <p class="hero-sub">Brain Tumor Detection &amp; Classification System</p>
      <div class="badges">
        <span class="badge">⚡ VGG16 Architecture</span>
        <span class="badge">🎯 4-Class Detection</span>
        <span class="badge">🔬 MRI Analysis</span>
        <span class="badge">📊 Real-time Inference</span>
      </div>
    </div>
    <div class="divider"></div>""", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs([
        "🏠  Home", "🔍  Scan Analysis",
        "👥  Patient Records", "📋  Patient Reports", "📊  Dashboard",
    ])

    # ══ TAB 1 — HOME ══════════════════════════════════════════════════════════
    with t1:
        st.markdown('<div class="sec">🏥 Centre Overview</div>', unsafe_allow_html=True)
        total_p = len(patients); total_s = len(scans)
        tumor_s = sum(1 for s in scans if s.get("diagnosis") != "notumor"); clear_s = total_s - tumor_s
        c1, c2, c3, c4 = st.columns(4)
        for col, icon, num, label in [
            (c1,"👥",total_p,"Total Patients"),(c2,"🔬",total_s,"Scans Processed"),
            (c3,"🔴",tumor_s,"Tumors Detected"),(c4,"✅",clear_s,"Clear Scans"),
        ]:
            col.markdown(f'<div class="stat-card"><div class="stat-icon">{icon}</div><div class="stat-num">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec">🕐 Recent Scans</div>', unsafe_allow_html=True)
        recent = sorted(scans, key=lambda x: x.get("date",""), reverse=True)[:6]
        if recent:
            for s in recent:
                diag = s.get("diagnosis",""); color = "#00e676" if diag=="notumor" else "#ff4d6d"
                st.markdown(f"""
                <div class="patient-row">
                  <div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#006680,#00e5ff);
                       display:flex;align-items:center;justify-content:center;font-weight:700;color:#000;font-size:.85rem;flex-shrink:0">
                    {s.get('patient_name','?')[0].upper()}
                  </div>
                  <div style="flex:1;min-width:0">
                    <div style="font-weight:500;color:var(--white);font-size:.9rem">{s.get('patient_name','Unknown')}</div>
                    <div style="font-size:.74rem;color:var(--muted)">{s.get('date','')}</div>
                  </div>
                  <div style="color:{color};font-family:'Orbitron',sans-serif;font-size:.68rem;text-align:right">
                    {diag.upper()}<br>
                    <span style="color:var(--muted);font-family:'DM Sans',sans-serif;font-size:.72rem">{s.get('confidence',0)*100:.1f}%</span>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state" style="min-height:140px"><span style="font-size:2rem;opacity:.2">🔬</span><p style="color:#6a8a9a;font-size:.88rem;margin:0">No scans yet.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="rbox r-cyan" style="font-size:.82rem;margin-top:1rem">💡 Go to <strong>Scan Analysis</strong> to upload an MRI and run AI detection.</div>', unsafe_allow_html=True)

    # ══ TAB 2 — SCAN ANALYSIS ═════════════════════════════════════════════════
    with t2:
        st.markdown('<div class="sec">🔍 MRI Scan Analysis</div>', unsafe_allow_html=True)
        left, right = st.columns([1,1], gap="large")
        with left:
            st.markdown('<div class="card"><div class="card-label">Configuration</div>', unsafe_allow_html=True)
            model_path = st.text_input("Model path", value="model.h5")
            uploaded   = st.file_uploader("Upload MRI Scan", type=["jpg","jpeg","png"])
            st.markdown('</div>', unsafe_allow_html=True)
            if uploaded:
                st.markdown('<div class="card"><div class="card-label">Input Scan</div>', unsafe_allow_html=True)
                st.image(uploaded, width=270)
                st.markdown('</div>', unsafe_allow_html=True)
        with right:
            if uploaded:
                if not os.path.exists(model_path):
                    st.markdown(f'<div class="rbox r-red">⚠ Model not found: <code>{model_path}</code></div>', unsafe_allow_html=True)
                else:
                    try:
                        from tensorflow.keras.preprocessing.image import load_img, img_to_array
                        mdl   = load_saved_model(model_path)
                        arr   = np.expand_dims(img_to_array(load_img(uploaded, target_size=(IMAGE_SIZE,IMAGE_SIZE)))/255.,0)
                        preds = mdl.predict(arr)
                        idx   = int(np.argmax(preds)); conf = float(np.max(preds)); label = CLASS_LABELS[idx]
                        st.session_state.last_scan = {"diagnosis":label,"confidence":conf,"all_probs":preds[0].tolist(),
                                                       "date":datetime.now().strftime("%Y-%m-%d %H:%M"),"operator":st.session_state.username}
                        st.markdown('<div class="card"><div class="card-label">Confidence Scores</div>', unsafe_allow_html=True)
                        fig, ax = dark_fig(5, 2.8)
                        colors = ['#00e5ff']*4; colors[idx] = '#00e676' if label=='notumor' else '#ff4d6d'
                        bars = ax.barh(CLASS_LABELS, preds[0], color=colors, height=0.5)
                        ax.set_xlim(0,1); ax.set_xlabel("Confidence",fontsize=9)
                        for bar,v in zip(bars,preds[0]):
                            ax.text(v+.01,bar.get_y()+bar.get_height()/2,f'{v*100:.1f}%',va='center',color='#6a8a9a',fontsize=8)
                        plt.tight_layout(); st.pyplot(fig); plt.close()
                        st.markdown('</div>', unsafe_allow_html=True)
                        if label=="notumor":
                            st.markdown(f'<div class="rbox r-green">✅ No Tumor Detected &nbsp;·&nbsp; Confidence: {conf*100:.1f}%</div>', unsafe_allow_html=True)
                            st.markdown('<div class="rbox r-cyan">💡 You\'re clear! Maintain healthy habits.</div>', unsafe_allow_html=True)
                        else:
                            bc = "r-amber" if conf<0.6 else "r-red"; tg = "⚠️ Low Confidence" if conf<0.6 else "🔴 High Confidence"
                            st.markdown(f'<div class="rbox {bc}">{tg} &nbsp;·&nbsp; Tumor: <strong>{label.upper()}</strong> &nbsp;·&nbsp; {conf*100:.1f}%</div>', unsafe_allow_html=True)
                            st.markdown('<div class="rbox r-amber">⚕️ Consult a neurologist immediately.</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="rbox r-red">❌ Error: <code>{e}</code></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="empty-state"><span style="font-size:2.8rem;opacity:.25">🧠</span><p style="color:#6a8a9a;font-size:.9rem;margin:0">Upload an MRI scan to begin</p></div>', unsafe_allow_html=True)

        if st.session_state.last_scan and uploaded:
            st.markdown('<div class="sec">💾 Save Scan to Patient Record</div>', unsafe_allow_html=True)
            with st.expander("📁 Link this scan to a patient record", expanded=False):
                patient_names = [p["name"] for p in patients]
                save_mode = st.radio("Patient", ["Existing Patient","New Patient"], horizontal=True)
                if save_mode == "Existing Patient":
                    if patient_names:
                        sel_p = st.selectbox("Select Patient", patient_names, key="save_sel")
                        dnote = st.text_area("Doctor Notes (optional)", placeholder="Clinical observations…", key="save_note_ex")
                        if st.button("💾  Save Scan"):
                            se = {**st.session_state.last_scan,"patient_name":sel_p,"doctor_notes":dnote,"id":f"SCN{len(scans)+1:04d}"}
                            scans.append(se); save_data("scans.json",scans)
                            for p in patients:
                                if p["name"]==sel_p: p.setdefault("scans",[]).append(se["id"])
                            save_data("patients.json",patients)
                            st.markdown('<div class="rbox r-green">✅ Scan saved!</div>', unsafe_allow_html=True)
                            st.session_state.last_scan = None
                    else:
                        st.markdown('<div class="rbox r-cyan">No patients yet. Use "New Patient".</div>', unsafe_allow_html=True)
                else:
                    n1,n2 = st.columns(2)
                    np_name = n1.text_input("Patient Name",key="np_name"); np_age = n2.number_input("Age",1,120,30,key="np_age")
                    n3,n4 = st.columns(2)
                    np_gender = n3.selectbox("Gender",["Male","Female","Other"],key="np_gender"); np_contact = n4.text_input("Contact",key="np_contact")
                    dnote = st.text_area("Doctor Notes (optional)",placeholder="Clinical observations…",key="save_note_new")
                    if st.button("💾  Save & Add Patient"):
                        if np_name:
                            pid = f"PAT{len(patients)+1:04d}"
                            se  = {**st.session_state.last_scan,"patient_name":np_name,"doctor_notes":dnote,"id":f"SCN{len(scans)+1:04d}"}
                            scans.append(se)
                            patients.append({"id":pid,"name":np_name,"age":np_age,"gender":np_gender,"contact":np_contact,
                                             "notes":"","dob":"","registered":datetime.now().strftime("%Y-%m-%d"),
                                             "diabetic":"Unknown","blood_group":"Unknown","blood_pressure":"Unknown",
                                             "blood_sugar":"Unknown","cholesterol":"Unknown","allergies":"","blood_report":"","scans":[se["id"]]})
                            save_data("patients.json",patients); save_data("scans.json",scans)
                            st.markdown('<div class="rbox r-green">✅ Patient added and scan saved!</div>', unsafe_allow_html=True)
                            st.session_state.last_scan = None
                        else:
                            st.markdown('<div class="rbox r-amber">⚠ Patient name required.</div>', unsafe_allow_html=True)

    # ══ TAB 3 — PATIENT RECORDS ═══════════════════════════════════════════════
    with t3:
        st.markdown('<div class="sec">👥 Patient Records</div>', unsafe_allow_html=True)
        pr_left, pr_right = st.columns([1.7,1], gap="large")

        with pr_left:
            search   = st.text_input("🔍 Search by name or ID", placeholder="Type to filter…")
            filtered = [p for p in patients if
                        search.lower() in p["name"].lower() or
                        search.lower() in p.get("id","").lower()] if search else patients
            st.markdown(f'<div style="color:var(--muted);font-size:.78rem;margin-bottom:.4rem">{len(filtered)} patient(s) found</div>', unsafe_allow_html=True)

            if not filtered:
                st.markdown('<div class="empty-state" style="min-height:160px"><span style="font-size:2.2rem;opacity:.2">👥</span><p style="color:#6a8a9a;font-size:.88rem;margin:0">No patients found</p></div>', unsafe_allow_html=True)
            else:
                for p in filtered:
                    pid     = p.get("id","")
                    p_scans = [s for s in scans if s.get("patient_name")==p["name"]]
                    last_d  = p_scans[-1]["diagnosis"] if p_scans else "—"
                    d_col   = "#00e676" if last_d=="notumor" else ("#ff4d6d" if last_d!="—" else "#6a8a9a")
                    diab    = p.get("diabetic","Unknown")
                    bg      = p.get("blood_group","Unknown")

                    rc_info, rc_edit, rc_del = st.columns([1, 0.07, 0.07])
                    with rc_info:
                        st.markdown(f"""
                        <div class="patient-row">
                          <div style="width:36px;height:36px;border-radius:50%;flex-shrink:0;
                               background:linear-gradient(135deg,#006680,#00e5ff);
                               display:flex;align-items:center;justify-content:center;font-weight:700;color:#000;font-size:.9rem">
                            {p['name'][0].upper()}
                          </div>
                          <div style="flex:1;min-width:0">
                            <div style="font-weight:500;color:var(--white);font-size:.9rem">{p['name']}</div>
                            <div style="font-size:.72rem;color:var(--muted);margin-top:.1rem">
                              {pid} &nbsp;·&nbsp; Age {p.get('age','')} &nbsp;·&nbsp; {p.get('gender','')} &nbsp;·&nbsp; Reg: {p.get('registered','')}
                            </div>
                            <div style="margin-top:.35rem">
                              <span class="pill {'pill-yes' if diab=='Yes' else 'pill-no' if diab=='No' else 'pill-info'}">💉 Diabetic: {diab}</span>
                              <span class="pill pill-info">🩸 {bg}</span>
                            </div>
                          </div>
                          <div style="text-align:right;flex-shrink:0;padding-right:.3rem">
                            <div style="color:{d_col};font-family:'Orbitron',sans-serif;font-size:.65rem">{last_d.upper()}</div>
                            <div style="font-size:.7rem;color:var(--muted)">{len(p_scans)} scan(s)</div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    with rc_edit:
                        # .edit-btn wrapper → cyan-bordered dark button
                        st.markdown('<div class="edit-btn">', unsafe_allow_html=True)
                        if st.button("✏️", key=f"edit_{pid}", help="Update profile"):
                            st.session_state.editing_patient = None if st.session_state.editing_patient==pid else pid
                            st.session_state.confirm_del_pat = None; st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

                    with rc_del:
                        # .del-btn wrapper → red-bordered dark button
                        st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                        if st.button("🗑️", key=f"del_{pid}", help="Delete patient"):
                            st.session_state.confirm_del_pat = None if st.session_state.confirm_del_pat==pid else pid
                            st.session_state.editing_patient = None; st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

                    if st.session_state.editing_patient == pid:
                        st.markdown('<div class="edit-panel">', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-family:Orbitron,sans-serif;font-size:.7rem;color:var(--cyan);letter-spacing:.1em;margin-bottom:.8rem">✏️ EDITING — {p["name"]}</p>', unsafe_allow_html=True)
                        with st.form(key=f"edit_form_{pid}"):
                            st.markdown("**Basic Info**")
                            ef1, ef2 = st.columns(2)
                            e_name   = ef1.text_input("Full Name", value=p.get("name",""))
                            e_age    = ef2.number_input("Age", 1, 120, int(p.get("age",30)))
                            ef3, ef4 = st.columns(2)
                            g_opts   = ["Male","Female","Other"]; cur_g = p.get("gender","Male")
                            e_gender = ef3.selectbox("Gender", g_opts, index=g_opts.index(cur_g) if cur_g in g_opts else 0)
                            e_contact= ef4.text_input("Contact", value=p.get("contact",""))
                            e_dob    = st.text_input("Date of Birth", value=p.get("dob",""), placeholder="DD/MM/YYYY")
                            st.markdown("**Medical Info**")
                            m1, m2, m3 = st.columns(3)
                            d_opts=["Unknown","Yes","No"]; cur_d=p.get("diabetic","Unknown")
                            e_diab = m1.selectbox("Diabetic", d_opts, index=d_opts.index(cur_d) if cur_d in d_opts else 0)
                            bg_opts=["Unknown","A+","A-","B+","B-","AB+","AB-","O+","O-"]; cur_bg=p.get("blood_group","Unknown")
                            e_bg   = m2.selectbox("Blood Group", bg_opts, index=bg_opts.index(cur_bg) if cur_bg in bg_opts else 0)
                            bp_opts=["Unknown","Normal","High","Low"]; cur_bp=p.get("blood_pressure","Unknown")
                            e_bp   = m3.selectbox("Blood Pressure", bp_opts, index=bp_opts.index(cur_bp) if cur_bp in bp_opts else 0)
                            m4, m5 = st.columns(2)
                            bs_opts=["Unknown","Normal","Pre-diabetic","Diabetic"]; cur_bs=p.get("blood_sugar","Unknown")
                            e_bs   = m4.selectbox("Blood Sugar", bs_opts, index=bs_opts.index(cur_bs) if cur_bs in bs_opts else 0)
                            ch_opts=["Unknown","Normal","Borderline","High"]; cur_ch=p.get("cholesterol","Unknown")
                            e_chol = m5.selectbox("Cholesterol", ch_opts, index=ch_opts.index(cur_ch) if cur_ch in ch_opts else 0)
                            e_allerg = st.text_input("Allergies", value=p.get("allergies",""), placeholder="e.g. Penicillin…")
                            e_notes  = st.text_area("Medical History / Notes", value=p.get("notes",""), placeholder="Relevant history…")
                            sc, cc = st.columns(2)
                            with sc:
                                st.markdown('<div class="save-btn">', unsafe_allow_html=True)
                                submitted = st.form_submit_button("💾  Save Changes", use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            with cc:
                                st.markdown('<div class="cancel-btn">', unsafe_allow_html=True)
                                cancelled = st.form_submit_button("✖  Cancel", use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            if submitted:
                                if not e_name.strip():
                                    st.error("Name cannot be empty.")
                                else:
                                    old_name = p["name"]
                                    for sc_item in scans:
                                        if sc_item.get("patient_name")==old_name: sc_item["patient_name"]=e_name.strip()
                                    p.update({"name":e_name.strip(),"age":int(e_age),"gender":e_gender,"contact":e_contact,
                                              "dob":e_dob,"diabetic":e_diab,"blood_group":e_bg,"blood_pressure":e_bp,
                                              "blood_sugar":e_bs,"cholesterol":e_chol,"allergies":e_allerg,"notes":e_notes})
                                    save_data("patients.json",patients); save_data("scans.json",scans)
                                    st.session_state.editing_patient = None; st.rerun()
                            if cancelled:
                                st.session_state.editing_patient = None; st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

                    if st.session_state.confirm_del_pat == pid:
                        st.markdown('<div class="del-panel">', unsafe_allow_html=True)
                        st.warning(f"⚠️ Delete **{p['name']}**? All scans will be removed permanently.")
                        dc1, dc2 = st.columns(2)
                        with dc1:
                            st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                            if st.button("✅  Yes, Delete", key=f"yes_del_{pid}", use_container_width=True):
                                delete_patient(pid, patients, scans); st.session_state.confirm_del_pat = None; st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
                        with dc2:
                            if st.button("❌  Cancel", key=f"no_del_{pid}", use_container_width=True):
                                st.session_state.confirm_del_pat = None; st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)

        with pr_right:
            with st.expander("➕ Add New Patient", expanded=True):
                a1, a2      = st.columns(2)
                add_name    = a1.text_input("Full Name", key="add_name", placeholder="Patient name")
                add_age     = a2.number_input("Age", 1, 120, 30, key="add_age")
                a3, a4      = st.columns(2)
                add_gender  = a3.selectbox("Gender", ["Male","Female","Other"], key="add_gender")
                add_contact = a4.text_input("Contact", key="add_contact", placeholder="Phone / email")
                add_dob     = st.text_input("Date of Birth", key="add_dob", placeholder="DD/MM/YYYY (optional)")
                st.markdown('<p style="color:var(--cyan);font-size:.78rem;font-family:Orbitron,sans-serif;letter-spacing:.08em;margin:.8rem 0 .4rem">🩺 MEDICAL INFO</p>', unsafe_allow_html=True)
                b1, b2, b3  = st.columns(3)
                add_diabetic= b1.selectbox("Diabetic", ["Unknown","Yes","No"], key="add_diabetic")
                add_bg      = b2.selectbox("Blood Group", ["Unknown","A+","A-","B+","B-","AB+","AB-","O+","O-"], key="add_bg")
                add_bp      = b3.selectbox("Blood Pressure", ["Unknown","Normal","High","Low"], key="add_bp")
                b4, b5      = st.columns(2)
                add_bs      = b4.selectbox("Blood Sugar", ["Unknown","Normal","Pre-diabetic","Diabetic"], key="add_bs")
                add_chol    = b5.selectbox("Cholesterol", ["Unknown","Normal","Borderline","High"], key="add_chol")
                add_allerg  = st.text_input("Allergies (optional)", key="add_allerg", placeholder="e.g. Penicillin, Pollen…")
                add_notes   = st.text_area("Medical History / Notes (optional)", key="add_notes", placeholder="Relevant history…")
                st.markdown('<p style="color:var(--cyan);font-size:.78rem;font-family:Orbitron,sans-serif;letter-spacing:.08em;margin:.8rem 0 .4rem">🧪 BLOOD TEST REPORT</p>', unsafe_allow_html=True)
                st.caption("Optional — you can skip this and add the patient first.")
                blood_file  = st.file_uploader("Upload Blood Test Report (PDF/Image)", type=["pdf","jpg","jpeg","png"], key="blood_upload")
                if st.button("➕  Add Patient"):
                    if add_name:
                        pid = f"PAT{len(patients)+1:04d}"
                        blood_report_saved = ""
                        if blood_file:
                            report_path = f"data/blood_reports/{pid}_{blood_file.name}"
                            with open(report_path, "wb") as bf: bf.write(blood_file.read())
                            blood_report_saved = report_path
                        patients.append({"id":pid,"name":add_name,"age":int(add_age),"gender":add_gender,
                                          "contact":add_contact,"dob":add_dob,"notes":add_notes,
                                          "diabetic":add_diabetic,"blood_group":add_bg,"blood_pressure":add_bp,
                                          "blood_sugar":add_bs,"cholesterol":add_chol,"allergies":add_allerg,
                                          "blood_report":blood_report_saved,"registered":datetime.now().strftime("%Y-%m-%d"),"scans":[]})
                        save_data("patients.json", patients)
                        st.markdown('<div class="rbox r-green">✅ Patient added successfully!</div>', unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.markdown('<div class="rbox r-amber">⚠ Full name is required.</div>', unsafe_allow_html=True)

    # ══ TAB 4 — PATIENT REPORTS ═══════════════════════════════════════════════
    with t4:
        st.markdown('<div class="sec">📋 Patient Reports</div>', unsafe_allow_html=True)
        if not patients:
            st.markdown('<div class="empty-state"><span style="font-size:2.5rem;opacity:.2">📋</span><p style="color:#6a8a9a;font-size:.9rem;margin:0">No patients yet.</p></div>', unsafe_allow_html=True)
        else:
            sel_name = st.selectbox("Select Patient", [p["name"] for p in patients], key="rep_sel")
            sel_pat  = next((p for p in patients if p["name"]==sel_name), None)

            if sel_pat:
                pt_scans = [s for s in scans if s.get("patient_name")==sel_name]
                diab  = sel_pat.get("diabetic","Unknown"); bg   = sel_pat.get("blood_group","Unknown")
                bp    = sel_pat.get("blood_pressure","Unknown"); bs = sel_pat.get("blood_sugar","Unknown")
                chol  = sel_pat.get("cholesterol","Unknown"); allerg = sel_pat.get("allergies","—")

                st.markdown("<br>", unsafe_allow_html=True)
                with st.container():
                    col_av, col_info = st.columns([0.08, 1])
                    with col_av:
                        st.markdown(f"""
                        <div style="width:52px;height:52px;border-radius:50%;margin-top:.3rem;
                             background:linear-gradient(135deg,#006680,#00e5ff);
                             display:flex;align-items:center;justify-content:center;
                             font-weight:900;color:#000;font-size:1.3rem">
                          {sel_pat['name'][0].upper()}
                        </div>""", unsafe_allow_html=True)
                    with col_info:
                        st.markdown(f"""
                        <div style="padding:.2rem 0">
                          <div style="font-family:'Orbitron',sans-serif;font-size:1rem;color:var(--white);font-weight:700;letter-spacing:.05em">{sel_pat['name']}</div>
                          <div style="font-size:.78rem;color:var(--muted);margin-top:.25rem">
                            {sel_pat.get('id','')} &nbsp;·&nbsp; Age {sel_pat.get('age','')} &nbsp;·&nbsp;
                            {sel_pat.get('gender','')} &nbsp;·&nbsp; 📞 {sel_pat.get('contact','—')} &nbsp;·&nbsp;
                            Registered: {sel_pat.get('registered','')}
                          </div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="sec">🩺 Medical Profile</div>', unsafe_allow_html=True)
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)

                def med_tile(col, label, value, color="#e8f4f8"):
                    col.markdown(f"""
                    <div style="background:#0d1828;border:1px solid #1a2a40;border-radius:10px;
                         padding:.8rem .9rem;text-align:center">
                      <div style="font-size:.65rem;color:var(--muted);letter-spacing:.07em;text-transform:uppercase;margin-bottom:.3rem">{label}</div>
                      <div style="font-size:.92rem;font-weight:600;color:{color}">{value}</div>
                    </div>""", unsafe_allow_html=True)

                med_tile(mc1,"Diabetic",diab,"#ff4d6d" if diab=="Yes" else "#00e676" if diab=="No" else "#6a8a9a")
                med_tile(mc2,"Blood Group",bg,"#00e5ff")
                med_tile(mc3,"Blood Pressure",bp,"#ff4d6d" if bp=="High" else "#00e676" if bp=="Normal" else "#6a8a9a")
                med_tile(mc4,"Blood Sugar",bs,"#ff4d6d" if bs in ["Diabetic","Pre-diabetic"] else "#00e676" if bs=="Normal" else "#6a8a9a")
                med_tile(mc5,"Cholesterol",chol,"#ff4d6d" if chol=="High" else "#00e676" if chol=="Normal" else "#6a8a9a")

                if allerg and allerg != "—":
                    st.markdown(f'<div class="rbox r-amber" style="margin-top:.6rem">⚠️ <strong>Allergies:</strong> {allerg}</div>', unsafe_allow_html=True)
                if sel_pat.get("notes"):
                    st.markdown(f'<div class="rbox r-cyan" style="margin-top:.5rem">📝 <strong>Notes:</strong> {sel_pat["notes"]}</div>', unsafe_allow_html=True)

                # ── Blood Report — info bar + 👁️ View button ──────────────────
                br = sel_pat.get("blood_report","")
                pat_id_key = sel_pat.get("id","")
                if br and os.path.exists(br):
                    br_left, br_right = st.columns([3, 1])
                    with br_left:
                        st.markdown(
                            f'<div class="rbox r-amber" style="margin-top:.5rem">'
                            f'🧪 <strong>Blood Report on file:</strong>&nbsp; {os.path.basename(br)}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with br_right:
                        st.markdown("<div style='margin-top:.5rem'>", unsafe_allow_html=True)
                        st.markdown('<div class="view-report-btn">', unsafe_allow_html=True)
                        is_open   = st.session_state.viewing_blood_report == pat_id_key
                        btn_label = "🙈  Hide Report" if is_open else "👁️  View Report"
                        if st.button(btn_label, key=f"view_br_{pat_id_key}", use_container_width=True):
                            st.session_state.viewing_blood_report = None if is_open else pat_id_key
                            st.rerun()
                        st.markdown('</div></div>', unsafe_allow_html=True)

                    # ── Inline viewer panel ────────────────────────────────────
                    if st.session_state.viewing_blood_report == pat_id_key:
                        st.markdown('<div class="blood-report-panel">', unsafe_allow_html=True)
                        st.markdown(
                            f'<p style="font-family:Orbitron,sans-serif;font-size:.7rem;color:#ffab40;'
                            f'letter-spacing:.1em;margin:0 0 .8rem">🧪 BLOOD TEST REPORT — {os.path.basename(br)}</p>',
                            unsafe_allow_html=True
                        )
                        ext = os.path.splitext(br)[1].lower()
                        if ext in [".jpg", ".jpeg", ".png"]:
                            # Image: render inline
                            st.image(br, use_container_width=True)
                        elif ext == ".pdf":
                            # PDF: offer download (browsers can't render PDF inside Streamlit iframes)
                            with open(br, "rb") as pdf_f:
                                pdf_bytes = pdf_f.read()
                            st.download_button(
                                label="⬇️  Download PDF Report",
                                data=pdf_bytes,
                                file_name=os.path.basename(br),
                                mime="application/pdf",
                                key=f"dl_br_{pat_id_key}"
                            )
                            st.markdown(
                                '<div class="rbox r-cyan" style="margin-top:.4rem;font-size:.82rem">'
                                '📄 PDF files can be downloaded above and viewed in your PDF reader.</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div class="rbox r-amber">⚠️ Unsupported format for preview.</div>',
                                unsafe_allow_html=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="sec">🔬 Scan History &nbsp;<span style="font-size:.75rem;color:var(--muted);font-family:\'DM Sans\',sans-serif;font-weight:400">{len(pt_scans)} scan(s)</span></div>', unsafe_allow_html=True)

                if not pt_scans:
                    st.markdown('<div class="rbox r-cyan">No scans on record. Run a Scan Analysis and save it.</div>', unsafe_allow_html=True)
                else:
                    for i, s in enumerate(reversed(pt_scans)):
                        scan_num     = len(pt_scans) - i
                        diag         = s.get("diagnosis",""); conf = s.get("confidence",0)
                        result_label = "✅ No Tumor Detected" if diag=="notumor" else f"🔴 {diag.upper()} Detected"
                        scan_id      = s.get("id",f"idx_{i}")
                        diag_color   = "#00e676" if diag=="notumor" else ("#ffab40" if conf<0.6 else "#ff4d6d")

                        with st.expander(f"🔬 Scan #{scan_num}  ·  {s.get('date','')}  ·  {diag.upper()}", expanded=(i==0)):
                            st.markdown(f"""
                            <div class="scan-card">
                              <div class="scan-card-header">SCAN RESULT SUMMARY</div>
                              <div style="font-size:1.05rem;font-weight:700;color:{diag_color};margin-bottom:.6rem">{result_label}</div>
                              <div style="background:rgba(0,229,255,.05);border-radius:8px;padding:.5rem .8rem;display:inline-block;font-size:.88rem;color:var(--white)">
                                AI Confidence: <strong style="color:{diag_color}">{conf*100:.1f}%</strong>
                              </div>
                              <div class="scan-detail-grid">
                                <div class="scan-detail-item">
                                  <span class="scan-detail-label">Scan Date</span>
                                  <span class="scan-detail-value">{s.get('date','—')}</span>
                                </div>
                                <div class="scan-detail-item">
                                  <span class="scan-detail-label">Operator</span>
                                  <span class="scan-detail-value">{s.get('operator','—')}</span>
                                </div>
                                <div class="scan-detail-item">
                                  <span class="scan-detail-label">Scan ID</span>
                                  <span class="scan-detail-value" style="color:var(--cyan);font-family:'Orbitron',monospace;font-size:.78rem">{scan_id}</span>
                                </div>
                              </div>
                            </div>""", unsafe_allow_html=True)

                            if s.get("doctor_notes"):
                                st.markdown(f'<div class="rbox r-cyan" style="margin-top:.4rem">📝 <strong>Doctor Notes:</strong> {s["doctor_notes"]}</div>', unsafe_allow_html=True)

                            with st.form(key=f"note_{scan_id}"):
                                new_note = st.text_area("Update / Add Doctor Notes", value=s.get("doctor_notes",""), placeholder="Clinical findings…")
                                if st.form_submit_button("💾  Update Notes"):
                                    for sc in scans:
                                        if sc.get("id")==scan_id: sc["doctor_notes"]=new_note
                                    save_data("scans.json",scans); st.success("Notes updated!"); st.rerun()

                            st.markdown("---")
                            st.markdown('<p style="color:var(--muted);font-size:.76rem;margin-bottom:.3rem">⚠️ Danger Zone</p>', unsafe_allow_html=True)
                            if st.session_state.confirm_delete == scan_id:
                                st.markdown('<div class="rbox r-red" style="margin-top:0">🗑️ Permanently delete this scan?</div>', unsafe_allow_html=True)
                                cy, cn = st.columns(2)
                                with cy:
                                    st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                                    if st.button("✅  Yes, Delete", key=f"yes_{scan_id}"):
                                        delete_scan(scan_id,scans,patients); st.session_state.confirm_delete=None; st.rerun()
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with cn:
                                    if st.button("❌  Cancel", key=f"no_{scan_id}"):
                                        st.session_state.confirm_delete=None; st.rerun()
                            else:
                                if st.button("🗑️  Delete This Scan", key=f"delscan_{scan_id}"):
                                    st.session_state.confirm_delete=scan_id; st.rerun()

    # ══ TAB 5 — DASHBOARD ═════════════════════════════════════════════════════
    with t5:
        st.markdown('<div class="sec">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
        total_p  = len(patients); total_s = len(scans)
        tumor_s  = sum(1 for s in scans if s.get("diagnosis")!="notumor")
        det_rate = (tumor_s/total_s*100) if total_s>0 else 0
        avg_conf = (sum(s.get("confidence",0) for s in scans)/total_s*100) if total_s>0 else 0

        sc1,sc2,sc3,sc4 = st.columns(4)
        for col,icon,num,label in [
            (sc1,"👥",total_p,"Total Patients"),(sc2,"🔬",total_s,"Scans Processed"),
            (sc3,"📈",f"{det_rate:.1f}%","Detection Rate"),(sc4,"🎯",f"{avg_conf:.1f}%","Avg Confidence"),
        ]:
            col.markdown(f'<div class="stat-card"><div class="stat-icon">{icon}</div><div class="stat-num">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if not scans:
            st.markdown('<div class="empty-state"><span style="font-size:2.5rem;opacity:.2">📊</span><p style="color:#6a8a9a;font-size:.9rem;margin:0">No scan data yet.</p></div>', unsafe_allow_html=True)
        else:
            da1,da2 = st.columns(2,gap="medium")
            with da1:
                st.markdown('<div class="sec">🧬 Diagnosis Distribution</div>', unsafe_allow_html=True)
                dc = Counter(s.get("diagnosis","unknown") for s in scans)
                pal = {'notumor':'#00e676','glioma':'#ff4d6d','meningioma':'#ffab40','pituitary':'#00e5ff'}
                fig,ax = dark_fig(5,3.8)
                _,_,ats = ax.pie(list(dc.values()),labels=list(dc.keys()),colors=[pal.get(l,'#6a8a9a') for l in dc.keys()],
                                  autopct='%1.1f%%',startangle=140,textprops={'color':'#6a8a9a','fontsize':9},
                                  wedgeprops={'linewidth':2,'edgecolor':'#04080f'})
                for at in ats: at.set_color('#e8f4f8'); at.set_fontsize(9)
                ax.set_title('Scan Results Distribution'); plt.tight_layout(); st.pyplot(fig
