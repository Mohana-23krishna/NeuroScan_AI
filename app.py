import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
import random
import json
import hashlib
from datetime import datetime
import pandas as pd
from collections import Counter, defaultdict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 128
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
TRAIN_DIR    = "MRI_Images/Training"
TEST_DIR     = "MRI_Images/Testing"

# ── Data Persistence ──────────────────────────────────────────────────────────
def ensure_dirs():
    os.makedirs("data", exist_ok=True)

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

# ── Session State Init ─────────────────────────────────────────────────────────
for k, v in [
    ("logged_in",  False),
    ("username",   ""),
    ("auth_mode",  "login"),
    ("last_scan",  None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Global CSS ────────────────────────────────────────────────────────────────
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
  border: 1.5px solid rgba(0,229,255,.7);
  animation: ring 2.6s ease-out infinite;
}
.pulse-ring-wrap::before { width: 100%; height: 100%; }
.pulse-ring-wrap::after  { width: 155%; height: 155%; animation-delay: .9s; opacity:.45; }
@keyframes ring {
  0%   { transform: scale(.8);  opacity: .9; }
  100% { transform: scale(1.7); opacity: 0;  }
}
.brain { font-size: 3rem; filter: drop-shadow(0 0 16px var(--cyan)); }
.hero-title {
  font-family: 'Orbitron', sans-serif; font-weight: 900;
  font-size: clamp(2.2rem, 5vw, 3.6rem);
  letter-spacing: .14em; line-height: 1.1; margin: 0 0 .35rem; color: var(--white);
}
.hero-title .accent {
  color: var(--cyan);
  text-shadow: 0 0 18px var(--cyan), 0 0 50px rgba(0,229,255,.28);
}
.hero-sub {
  font-weight: 300; font-size: .95rem; letter-spacing: .22em;
  text-transform: uppercase; color: var(--muted); margin: 0 0 1.8rem;
}
.badges { display: flex; gap: .55rem; flex-wrap: wrap; justify-content: center; }
.badge {
  background: rgba(0,229,255,.07); border: 1px solid rgba(0,229,255,.22);
  border-radius: 20px; padding: .26rem .9rem;
  font-size: .7rem; letter-spacing: .08em; color: var(--cyan); font-weight: 500;
}
.divider {
  width: 100%; height: 1px; margin: 0 0 1.6rem;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0; background: var(--surface);
  border-radius: 10px; padding: 4px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 7px; color: var(--muted); background: transparent; border: none;
  font-family: 'DM Sans', sans-serif; font-weight: 500;
  font-size: .87rem; letter-spacing: .04em; padding: .48rem 1.2rem;
}
.stTabs [aria-selected="true"] {
  background: rgba(0,229,255,.12) !important;
  color: var(--cyan) !important;
  box-shadow: inset 0 0 0 1px rgba(0,229,255,.3);
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem;
  position: relative; overflow: hidden;
}
.card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
}
.card-label {
  font-family: 'Orbitron', sans-serif; font-size: .62rem;
  letter-spacing: .18em; color: var(--cyan); text-transform: uppercase; margin-bottom: .7rem;
}

.stat-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.4rem 1rem; text-align: center;
  position: relative; overflow: hidden;
}
.stat-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
}
.stat-num {
  font-family: 'Orbitron', sans-serif; font-size: 2rem; font-weight: 900;
  color: var(--cyan); line-height: 1;
}
.stat-label {
  font-size: .72rem; color: var(--muted); letter-spacing: .08em;
  text-transform: uppercase; margin-top: .4rem;
}
.stat-icon { font-size: 1.4rem; margin-bottom: .4rem; }

.sec {
  font-family: 'Orbitron', sans-serif; font-size: .88rem; font-weight: 700;
  color: var(--white); letter-spacing: .1em; margin: 1.4rem 0 .8rem;
  display: flex; align-items: center; gap: .6rem;
}
.sec::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.rbox {
  border-radius: 11px; padding: 1rem 1.4rem; margin-top: .8rem;
  border-left: 4px solid; font-size: .95rem; font-weight: 500;
}
.r-green { background: rgba(0,230,118,.08);  border-color: var(--green);  color: var(--green); }
.r-red   { background: rgba(255,77,109,.08); border-color: var(--red);    color: var(--red); }
.r-amber { background: rgba(255,171,64,.08); border-color: var(--amber);  color: var(--amber); }
.r-cyan  { background: rgba(0,229,255,.07);  border-color: var(--cyan);   color: var(--cyan); }

.empty-state {
  min-height: 260px; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: .8rem;
  border: 1px dashed var(--border); border-radius: 14px; background: var(--surface);
}

.stTextInput input, .stNumberInput input, .stTextArea textarea {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important; color: var(--white) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px rgba(0,229,255,.15) !important;
}
label { color: var(--muted) !important; font-size: .82rem !important; letter-spacing: .05em; }

div.stButton > button {
  background: linear-gradient(135deg, #006680, #00e5ff);
  color: #000 !important; font-family: 'Orbitron', sans-serif;
  font-size: .74rem; font-weight: 700; letter-spacing: .1em;
  border: none; border-radius: 8px; padding: .6rem 2rem;
  text-transform: uppercase; transition: all .2s;
}
div.stButton > button:hover {
  transform: translateY(-1px); box-shadow: 0 6px 22px rgba(0,229,255,.28);
}

[data-testid="stFileUploader"] {
  background: var(--surface); border: 1px dashed var(--cdim); border-radius: 12px; padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: var(--cyan); }

.stProgress > div > div {
  background: linear-gradient(90deg, #006680, #00e5ff) !important; border-radius: 4px;
}

.stImage img { border: 1px solid var(--border); border-radius: 10px; }
.stSpinner > div { border-top-color: var(--cyan) !important; }

.patient-row {
  display: flex; align-items: center; gap: 1rem;
  padding: .7rem 1rem; border-bottom: 1px solid var(--border);
  border-radius: 8px; transition: background .15s; margin-bottom: .1rem;
}
.patient-row:hover { background: rgba(0,229,255,.04); }

.login-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 18px; padding: 2.5rem 2rem;
  position: relative; overflow: hidden;
}
.login-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
}
.login-title {
  font-family: 'Orbitron', sans-serif; font-size: 1rem;
  color: var(--cyan); letter-spacing: .12em; text-align: center; margin-bottom: .3rem;
}
.login-sub {
  font-size: .8rem; color: var(--muted); text-align: center; margin-bottom: 1.6rem;
}

.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: .6rem 1.2rem; background: var(--surface);
  border-bottom: 1px solid var(--border); margin-bottom: 0;
  position: sticky; top: 0; z-index: 999;
}
.topbar-brand {
  font-family: 'Orbitron', sans-serif; font-size: .9rem;
  color: var(--white); letter-spacing: .1em;
}
.topbar-brand span { color: var(--cyan); }
.topbar-user { font-size: .8rem; color: var(--muted); }

div[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--white) !important;
}

.stDataFrame { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
iframe[title="st.dataframe"] { background: var(--surface) !important; }

.stRadio > label { color: var(--muted) !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--white) !important; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib Dark Theme Helper ──────────────────────────────────────────────
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

# ── ML Helpers ────────────────────────────────────────────────────────────────
def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    return np.array(image) / 255.0

def open_images(paths, augment=False):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    out = []
    for p in paths:
        arr = img_to_array(load_img(p, target_size=(IMAGE_SIZE, IMAGE_SIZE)))
        out.append(augment_image(arr) if augment else arr / 255.0)
    return np.array(out)

def load_dataset_paths(data_dir):
    paths, labels = [], []
    if not os.path.exists(data_dir):
        return paths, labels
    for cls in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(d, f))
                labels.append(cls)
    return paths, labels

def encode_label(labels, c2i):
    return np.array([c2i[l] for l in labels])

@st.cache_resource
def build_model(num_classes):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.optimizers import Adam
    base = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
    for layer in base.layers: layer.trainable = False
    for layer in base.layers[-4:-1]: layer.trainable = True
    m = Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)), base,
        Flatten(), Dropout(0.3), Dense(128, activation='relu'),
        Dropout(0.2), Dense(num_classes, activation='softmax')
    ])
    m.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
    return m

@st.cache_resource
def load_saved_model(path):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    try:
        return load_model(path, compile=False)
    except Exception:
        pass
    try:
        from tensorflow.keras import layers as kl
        original_from_config = kl.InputLayer.from_config
        @classmethod
        def patched_from_config(cls, config):
            config.pop("batch_shape", None)
            config.pop("optional", None)
            if "shape" not in config:
                config["shape"] = (IMAGE_SIZE, IMAGE_SIZE, 3)
            return original_from_config.__func__(cls, config)
        kl.InputLayer.from_config = patched_from_config
        try:
            model = load_model(path, compile=False)
            return model
        finally:
            kl.InputLayer.from_config = original_from_config
    except Exception:
        pass
    try:
        model = build_model(len(CLASS_LABELS))
        model.load_weights(path)
        return model
    except Exception as final_err:
        raise RuntimeError(
            f"Could not load model after three attempts.\n\nRoot cause: {final_err}\n\n"
            f"Deployment is running TF {tf.__version__}. The most reliable fix is "
            f"to re-save your model.h5 locally using TF 2.16.1:\n  model.save('model.h5')"
        )

# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE
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
            </div>
            """, unsafe_allow_html=True)
            uname = st.text_input("Username", placeholder="Enter your username")
            pwd   = st.text_input("Password", type="password", placeholder="Enter your password")

            if st.button("🔑  Sign In", use_container_width=True):
                if uname in users and users[uname]["password"] == hash_pw(pwd):
                    st.session_state.logged_in = True
                    st.session_state.username  = uname
                    st.rerun()
                else:
                    st.markdown('<div class="rbox r-red">❌ Invalid username or password.</div>',
                                unsafe_allow_html=True)

            st.markdown('<p style="text-align:center;color:var(--muted);font-size:.83rem;margin-top:1rem">No account yet?</p>',
                        unsafe_allow_html=True)
            if st.button("✨  Create Account", use_container_width=True):
                st.session_state.auth_mode = "signup"
                st.rerun()

        else:  # signup
            st.markdown("""
            <div class="login-card">
              <div class="login-title">✨ CREATE ACCOUNT</div>
              <div class="login-sub">Join the NeuroScan AI platform</div>
            </div>
            """, unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            new_uname   = s1.text_input("Username",  placeholder="Choose username")
            new_name    = s2.text_input("Full Name",  placeholder="Your full name")
            new_role    = st.selectbox("Role", ["Radiologist", "Doctor", "Technician", "Admin"])
            new_pwd     = st.text_input("Password", type="password", placeholder="Min 6 characters")
            conf_pwd    = st.text_input("Confirm Password", type="password", placeholder="Repeat password")

            if st.button("✨  Create Account", use_container_width=True):
                if not all([new_uname, new_name, new_pwd, conf_pwd]):
                    st.markdown('<div class="rbox r-amber">⚠ All fields are required.</div>',
                                unsafe_allow_html=True)
                elif new_uname in users:
                    st.markdown('<div class="rbox r-red">❌ Username already taken.</div>',
                                unsafe_allow_html=True)
                elif new_pwd != conf_pwd:
                    st.markdown('<div class="rbox r-red">❌ Passwords do not match.</div>',
                                unsafe_allow_html=True)
                elif len(new_pwd) < 6:
                    st.markdown('<div class="rbox r-amber">⚠ Password must be at least 6 characters.</div>',
                                unsafe_allow_html=True)
                else:
                    users[new_uname] = {
                        "name": new_name, "role": new_role,
                        "password": hash_pw(new_pwd),
                        "created": datetime.now().isoformat()
                    }
                    save_data("users.json", users)
                    st.markdown('<div class="rbox r-green">✅ Account created! Please sign in.</div>',
                                unsafe_allow_html=True)
                    st.session_state.auth_mode = "login"
                    st.rerun()

            st.markdown('<p style="text-align:center;color:var(--muted);font-size:.83rem;margin-top:1rem">Already have an account?</p>',
                        unsafe_allow_html=True)
            if st.button("← Back to Sign In", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
def show_app():
    users     = load_data("users.json", {})
    user_info = users.get(st.session_state.username, {})
    patients  = load_data("patients.json", [])
    scans     = load_data("scans.json", [])

    # ── Top sticky bar ────────────────────────────────────────────────────────
    tb1, tb2 = st.columns([5, 1])
    with tb1:
        st.markdown(f"""
        <div class="topbar">
          <div class="topbar-brand">NEURO<span>SCAN</span> AI &nbsp;·&nbsp;
            <span style="color:var(--muted);font-size:.78rem;font-family:'DM Sans',sans-serif">
              Diagnostic Centre Platform
            </span>
          </div>
          <div class="topbar-user">
            👤 &nbsp;<strong style="color:var(--white)">{user_info.get('name', st.session_state.username)}</strong>
            &nbsp;·&nbsp; {user_info.get('role','User')}
          </div>
        </div>
        """, unsafe_allow_html=True)
    with tb2:
        st.markdown("<div style='margin-top:.3rem'></div>", unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.rerun()

    # ── Hero ─────────────────────────────────────────────────────────────────
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
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "🏠  Home",
        "🔍  Scan Analysis",
        "👥  Patient Records",
        "📋  Patient Reports",
        "📊  Dashboard",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 1 — HOME
    # ═════════════════════════════════════════════════════════════════════════
    with t1:
        st.markdown('<div class="sec">🏥 Centre Overview</div>', unsafe_allow_html=True)

        total_p  = len(patients)
        total_s  = len(scans)
        tumor_s  = sum(1 for s in scans if s.get("diagnosis") != "notumor")
        clear_s  = total_s - tumor_s
        det_rate = (tumor_s / total_s * 100) if total_s > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        for col, icon, num, label in [
            (c1, "👥", total_p,  "Total Patients"),
            (c2, "🔬", total_s,  "Scans Processed"),
            (c3, "🔴", tumor_s,  "Tumors Detected"),
            (c4, "✅", clear_s,  "Clear Scans"),
        ]:
            col.markdown(f"""
            <div class="stat-card">
              <div class="stat-icon">{icon}</div>
              <div class="stat-num">{num}</div>
              <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Recent Scans (full width now, System Info removed) ────────────
        st.markdown('<div class="sec">🕐 Recent Scans</div>', unsafe_allow_html=True)
        recent = sorted(scans, key=lambda x: x.get("date", ""), reverse=True)[:6]
        if recent:
            for s in recent:
                diag  = s.get("diagnosis", "")
                color = "#00e676" if diag == "notumor" else "#ff4d6d"
                st.markdown(f"""
                <div class="patient-row">
                  <div style="width:34px;height:34px;border-radius:50%;
                       background:linear-gradient(135deg,#006680,#00e5ff);
                       display:flex;align-items:center;justify-content:center;
                       font-weight:700;color:#000;font-size:.85rem;flex-shrink:0">
                    {s.get('patient_name','?')[0].upper()}
                  </div>
                  <div style="flex:1;min-width:0">
                    <div style="font-weight:500;color:var(--white);font-size:.9rem">{s.get('patient_name','Unknown')}</div>
                    <div style="font-size:.74rem;color:var(--muted)">{s.get('date','')}</div>
                  </div>
                  <div style="color:{color};font-family:'Orbitron',sans-serif;font-size:.68rem;text-align:right">
                    {diag.upper()}<br>
                    <span style="color:var(--muted);font-family:'DM Sans',sans-serif;font-size:.72rem">
                      {s.get('confidence',0)*100:.1f}%
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state" style="min-height:160px">
              <span style="font-size:2rem;opacity:.2">🔬</span>
              <p style="color:#6a8a9a;font-size:.88rem;margin:0">No scans yet. Start from Scan Analysis.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="rbox r-cyan" style="font-size:.82rem;margin-top:1rem">
          💡 Go to <strong>Scan Analysis</strong> to upload an MRI and run AI detection.
          Save results directly to patient records.
        </div>
        """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 2 — SCAN ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════
    with t2:
        st.markdown('<div class="sec">🔍 MRI Scan Analysis</div>', unsafe_allow_html=True)
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<div class="card"><div class="card-label">Configuration</div>', unsafe_allow_html=True)
            model_path = st.text_input("Model path", value="model.h5")
            uploaded   = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])
            st.markdown('</div>', unsafe_allow_html=True)
            if uploaded:
                st.markdown('<div class="card"><div class="card-label">Input Scan</div>', unsafe_allow_html=True)
                st.image(uploaded, width=270)
                st.markdown('</div>', unsafe_allow_html=True)

        with right:
            if uploaded:
                if not os.path.exists(model_path):
                    st.markdown(
                        f'<div class="rbox r-red">⚠ Model <code>{model_path}</code> not found. Train first.</div>',
                        unsafe_allow_html=True)
                else:
                    try:
                        from tensorflow.keras.preprocessing.image import load_img, img_to_array
                        mdl   = load_saved_model(model_path)
                        arr   = np.expand_dims(
                                    img_to_array(load_img(uploaded, target_size=(IMAGE_SIZE, IMAGE_SIZE))) / 255., 0)
                        preds = mdl.predict(arr)
                        idx   = int(np.argmax(preds))
                        conf  = float(np.max(preds))
                        label = CLASS_LABELS[idx]

                        st.session_state.last_scan = {
                            "diagnosis":  label,
                            "confidence": conf,
                            "all_probs":  preds[0].tolist(),
                            "date":       datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "operator":   st.session_state.username,
                        }

                        st.markdown('<div class="card"><div class="card-label">Confidence Scores</div>',
                                    unsafe_allow_html=True)
                        fig, ax = dark_fig(5, 2.8)
                        colors = ['#00e5ff'] * 4
                        colors[idx] = '#00e676' if label == 'notumor' else '#ff4d6d'
                        bars = ax.barh(CLASS_LABELS, preds[0], color=colors, height=0.5)
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Confidence", fontsize=9)
                        for bar, v in zip(bars, preds[0]):
                            ax.text(v + .01, bar.get_y() + bar.get_height() / 2,
                                    f'{v * 100:.1f}%', va='center', color='#6a8a9a', fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        st.markdown('</div>', unsafe_allow_html=True)

                        if label == "notumor":
                            st.markdown(
                                f'<div class="rbox r-green">✅ No Tumor Detected &nbsp;·&nbsp; Confidence: {conf*100:.1f}%</div>',
                                unsafe_allow_html=True)
                            st.markdown(
                                '<div class="rbox r-cyan">💡 You\'re clear! Maintain healthy habits and regular check-ups.</div>',
                                unsafe_allow_html=True)
                        else:
                            box_cls = "r-amber" if conf < 0.6 else "r-red"
                            tag     = "⚠️ Low Confidence" if conf < 0.6 else "🔴 High Confidence"
                            st.markdown(
                                f'<div class="rbox {box_cls}">{tag} &nbsp;·&nbsp; Tumor: <strong>{label.upper()}</strong> &nbsp;·&nbsp; {conf*100:.1f}%</div>',
                                unsafe_allow_html=True)
                            st.markdown(
                                '<div class="rbox r-amber">⚕️ Consult a neurologist immediately for proper diagnosis.</div>',
                                unsafe_allow_html=True)
                            st.markdown(
                                '<div class="rbox r-cyan">💡 Healthy diet &nbsp;·&nbsp; Reduce stress &nbsp;·&nbsp; Follow doctor\'s advice &nbsp;·&nbsp; No self-medication</div>',
                                unsafe_allow_html=True)

                    except Exception as e:
                        st.markdown(
                            f'<div class="rbox r-red">❌ Model load failed: <code>{str(e)}</code></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="empty-state">'
                    '<span style="font-size:2.8rem;opacity:.25">🧠</span>'
                    '<p style="color:#6a8a9a;font-size:.9rem;margin:0">Upload an MRI scan to begin analysis</p>'
                    '</div>',
                    unsafe_allow_html=True)

        # ── Save to Patient Record ─────────────────────────────────────────
        if st.session_state.last_scan and uploaded:
            st.markdown('<div class="sec">💾 Save Scan to Patient Record</div>', unsafe_allow_html=True)
            with st.expander("📁 Link this scan to a patient record", expanded=False):
                patient_names = [p["name"] for p in patients]
                save_mode = st.radio("Patient", ["Existing Patient", "New Patient"], horizontal=True)

                if save_mode == "Existing Patient":
                    if patient_names:
                        sel_patient  = st.selectbox("Select Patient", patient_names, key="save_sel")
                        doctor_note  = st.text_area("Doctor Notes (optional)",
                                                    placeholder="Add clinical observations…", key="save_note_ex")
                        if st.button("💾  Save Scan"):
                            scan_entry = {
                                **st.session_state.last_scan,
                                "patient_name": sel_patient,
                                "doctor_notes": doctor_note,
                                "id": f"SCN{len(scans)+1:04d}",
                            }
                            scans.append(scan_entry)
                            save_data("scans.json", scans)
                            for p in patients:
                                if p["name"] == sel_patient:
                                    p.setdefault("scans", []).append(scan_entry["id"])
                            save_data("patients.json", patients)
                            st.markdown('<div class="rbox r-green">✅ Scan saved to patient record!</div>',
                                        unsafe_allow_html=True)
                            st.session_state.last_scan = None
                    else:
                        st.markdown('<div class="rbox r-cyan">No existing patients. Use "New Patient" to add one.</div>',
                                    unsafe_allow_html=True)

                else:  # New Patient
                    n1, n2 = st.columns(2)
                    np_name    = n1.text_input("Patient Name",    key="np_name")
                    np_age     = n2.number_input("Age", 1, 120, 30, key="np_age")
                    n3, n4     = st.columns(2)
                    np_gender  = n3.selectbox("Gender", ["Male", "Female", "Other"], key="np_gender")
                    np_contact = n4.text_input("Contact",         key="np_contact")
                    doctor_note= st.text_area("Doctor Notes (optional)",
                                              placeholder="Add clinical observations…", key="save_note_new")
                    if st.button("💾  Save & Add Patient"):
                        if np_name:
                            pid        = f"PAT{len(patients)+1:04d}"
                            scan_entry = {
                                **st.session_state.last_scan,
                                "patient_name": np_name,
                                "doctor_notes": doctor_note,
                                "id": f"SCN{len(scans)+1:04d}",
                            }
                            scans.append(scan_entry)
                            patients.append({
                                "id": pid, "name": np_name, "age": np_age,
                                "gender": np_gender, "contact": np_contact,
                                "notes": "", "dob": "",
                                "registered": datetime.now().strftime("%Y-%m-%d"),
                                "scans": [scan_entry["id"]],
                            })
                            save_data("patients.json", patients)
                            save_data("scans.json",    scans)
                            st.markdown('<div class="rbox r-green">✅ Patient added and scan saved!</div>',
                                        unsafe_allow_html=True)
                            st.session_state.last_scan = None
                        else:
                            st.markdown('<div class="rbox r-amber">⚠ Patient name is required.</div>',
                                        unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 3 — PATIENT RECORDS
    # ═════════════════════════════════════════════════════════════════════════
    with t3:
        st.markdown('<div class="sec">👥 Patient Records</div>', unsafe_allow_html=True)
        pr_left, pr_right = st.columns([1.6, 1], gap="large")

        with pr_left:
            search = st.text_input("🔍 Search by name or ID", placeholder="Type to filter…")
            filtered = [p for p in patients if
                        search.lower() in p["name"].lower() or
                        search.lower() in p.get("id", "").lower()
                       ] if search else patients

            st.markdown(f'<div style="color:var(--muted);font-size:.78rem;margin-bottom:.5rem">'
                        f'{len(filtered)} patient(s) found</div>', unsafe_allow_html=True)

            if not filtered:
                st.markdown("""
                <div class="empty-state" style="min-height:200px">
                  <span style="font-size:2.2rem;opacity:.2">👥</span>
                  <p style="color:#6a8a9a;font-size:.88rem;margin:0">No patients found</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for p in filtered:
                    p_scans   = [s for s in scans if s.get("patient_name") == p["name"]]
                    last_diag = p_scans[-1]["diagnosis"] if p_scans else "—"
                    d_color   = ("#00e676" if last_diag == "notumor"
                                 else ("#ff4d6d" if last_diag != "—" else "#6a8a9a"))
                    st.markdown(f"""
                    <div class="patient-row">
                      <div style="width:38px;height:38px;border-radius:50%;flex-shrink:0;
                           background:linear-gradient(135deg,#006680,#00e5ff);
                           display:flex;align-items:center;justify-content:center;
                           font-weight:700;color:#000;font-size:.95rem">
                        {p['name'][0].upper()}
                      </div>
                      <div style="flex:1;min-width:0">
                        <div style="font-weight:500;color:var(--white)">{p['name']}</div>
                        <div style="font-size:.74rem;color:var(--muted)">
                          {p.get('id','')} &nbsp;·&nbsp; Age {p.get('age','')} &nbsp;·&nbsp; {p.get('gender','')}
                          &nbsp;·&nbsp; Registered: {p.get('registered','')}
                        </div>
                      </div>
                      <div style="text-align:right;flex-shrink:0">
                        <div style="color:{d_color};font-family:'Orbitron',sans-serif;font-size:.68rem">
                          {last_diag.upper()}
                        </div>
                        <div style="font-size:.72rem;color:var(--muted)">{len(p_scans)} scan(s)</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        with pr_right:
            st.markdown('<div class="card"><div class="card-label">Add New Patient</div>',
                        unsafe_allow_html=True)
            a1, a2      = st.columns(2)
            add_name    = a1.text_input("Full Name",   key="add_name",    placeholder="Patient name")
            add_age     = a2.number_input("Age", 1, 120, 30, key="add_age")
            a3, a4      = st.columns(2)
            add_gender  = a3.selectbox("Gender", ["Male", "Female", "Other"], key="add_gender")
            add_contact = a4.text_input("Contact",     key="add_contact", placeholder="Phone / email")
            add_dob     = st.text_input("Date of Birth", key="add_dob",   placeholder="DD/MM/YYYY (optional)")
            add_notes   = st.text_area("Medical History / Notes", key="add_notes",
                                       placeholder="Relevant history, allergies, prior conditions…")

            if st.button("➕  Add Patient"):
                if add_name:
                    pid = f"PAT{len(patients)+1:04d}"
                    patients.append({
                        "id": pid, "name": add_name, "age": int(add_age),
                        "gender": add_gender, "contact": add_contact,
                        "dob": add_dob, "notes": add_notes,
                        "registered": datetime.now().strftime("%Y-%m-%d"),
                        "scans": [],
                    })
                    save_data("patients.json", patients)
                    st.markdown('<div class="rbox r-green">✅ Patient added successfully!</div>',
                                unsafe_allow_html=True)
                    st.rerun()
                else:
                    st.markdown('<div class="rbox r-amber">⚠ Full name is required.</div>',
                                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 4 — PATIENT REPORTS
    # ═════════════════════════════════════════════════════════════════════════
    with t4:
        st.markdown('<div class="sec">📋 Patient Reports</div>', unsafe_allow_html=True)

        if not patients:
            st.markdown("""
            <div class="empty-state">
              <span style="font-size:2.5rem;opacity:.2">📋</span>
              <p style="color:#6a8a9a;font-size:.9rem;margin:0">
                No patients yet. Add patients and run scans first.
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            pat_names = [p["name"] for p in patients]
            sel_name  = st.selectbox("Select Patient", pat_names, key="rep_sel")
            sel_pat   = next((p for p in patients if p["name"] == sel_name), None)

            if sel_pat:
                pt_scans = [s for s in scans if s.get("patient_name") == sel_name]

                # ── Patient summary using native Streamlit (no raw HTML card) ──
                st.markdown("<br>", unsafe_allow_html=True)
                pi1, pi2, pi3, pi4 = st.columns(4)
                pi1.metric("Patient", sel_pat["name"])
                pi2.metric("Age / Gender", f"{sel_pat.get('age','')} · {sel_pat.get('gender','')}")
                pi3.metric("Contact", sel_pat.get("contact", "—"))
                pi4.metric("Total Scans", len(pt_scans))

                if sel_pat.get("notes"):
                    st.info(f"📝 Medical Notes: {sel_pat['notes']}")

                st.markdown("<br>", unsafe_allow_html=True)

                if not pt_scans:
                    st.markdown(
                        '<div class="rbox r-cyan">No scans on record for this patient. '
                        'Run a Scan Analysis and save it.</div>',
                        unsafe_allow_html=True)
                else:
                    for i, s in enumerate(reversed(pt_scans)):
                        scan_num  = len(pt_scans) - i
                        diag      = s.get("diagnosis", "")
                        conf      = s.get("confidence", 0)
                        d_color   = "#00e676" if diag == "notumor" else "#ff4d6d"
                        box_class = "r-green" if diag == "notumor" else ("r-amber" if conf < 0.6 else "r-red")
                        result_label = "✅ No Tumor Detected" if diag == "notumor" else f"🔴 {diag.upper()} Detected"

                        with st.expander(
                            f"🔬 Scan #{scan_num}  ·  {s.get('date','')}  ·  {diag.upper()}",
                            expanded=(i == 0)
                        ):
                            # Diagnosis result box
                            st.markdown(
                                f'<div class="rbox {box_class}">{result_label} &nbsp;·&nbsp; '
                                f'Confidence: <strong>{conf*100:.1f}%</strong></div>',
                                unsafe_allow_html=True)

                            # Scan details using native metrics
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Scan Date", s.get("date", "—"))
                            m2.metric("Operator",  s.get("operator", "—"))
                            m3.metric("Scan ID",   s.get("id", "—"))

                            # Doctor notes display
                            if s.get("doctor_notes"):
                                st.markdown(
                                    f'<div class="rbox r-cyan" style="margin-top:.5rem">'
                                    f'<strong>📝 Doctor Notes:</strong> {s["doctor_notes"]}</div>',
                                    unsafe_allow_html=True)

                            # Edit notes inline
                            with st.form(key=f"note_form_{s.get('id', i)}"):
                                new_note = st.text_area(
                                    "Update / Add Doctor Notes",
                                    value=s.get("doctor_notes", ""),
                                    placeholder="Enter clinical findings, recommendations…"
                                )
                                if st.form_submit_button("💾  Update Notes"):
                                    for sc in scans:
                                        if sc.get("id") == s.get("id"):
                                            sc["doctor_notes"] = new_note
                                    save_data("scans.json", scans)
                                    st.success("Notes updated successfully!")
                                    st.rerun()

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 5 — DASHBOARD / ANALYTICS
    # ═════════════════════════════════════════════════════════════════════════
    with t5:
        st.markdown('<div class="sec">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

        total_p  = len(patients)
        total_s  = len(scans)
        tumor_s  = sum(1 for s in scans if s.get("diagnosis") != "notumor")
        clear_s  = total_s - tumor_s
        det_rate = (tumor_s / total_s * 100) if total_s > 0 else 0
        avg_conf = (sum(s.get("confidence", 0) for s in scans) / total_s * 100) if total_s > 0 else 0

        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, icon, num, label in [
            (sc1, "👥", total_p,             "Total Patients"),
            (sc2, "🔬", total_s,             "Scans Processed"),
            (sc3, "📈", f"{det_rate:.1f}%",  "Detection Rate"),
            (sc4, "🎯", f"{avg_conf:.1f}%",  "Avg Confidence"),
        ]:
            col.markdown(f"""
            <div class="stat-card">
              <div class="stat-icon">{icon}</div>
              <div class="stat-num">{num}</div>
              <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if not scans:
            st.markdown("""
            <div class="empty-state">
              <span style="font-size:2.5rem;opacity:.2">📊</span>
              <p style="color:#6a8a9a;font-size:.9rem;margin:0">
                No scan data yet. Run scans and save them to see analytics.
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            da1, da2 = st.columns(2, gap="medium")

            with da1:
                st.markdown('<div class="sec">🧬 Diagnosis Distribution</div>', unsafe_allow_html=True)
                diag_counts = Counter(s.get("diagnosis", "unknown") for s in scans)
                pal_pie = {
                    'notumor':    '#00e676',
                    'glioma':     '#ff4d6d',
                    'meningioma': '#ffab40',
                    'pituitary':  '#00e5ff',
                }
                labels_d  = list(diag_counts.keys())
                vals_d    = list(diag_counts.values())
                colors_d  = [pal_pie.get(l, '#6a8a9a') for l in labels_d]
                fig, ax = dark_fig(5, 3.8)
                wedges, texts, autotexts = ax.pie(
                    vals_d, labels=labels_d, colors=colors_d,
                    autopct='%1.1f%%', startangle=140,
                    textprops={'color': '#6a8a9a', 'fontsize': 9},
                    wedgeprops={'linewidth': 2, 'edgecolor': '#04080f'}
                )
                for at in autotexts:
                    at.set_color('#e8f4f8')
                    at.set_fontsize(9)
                ax.set_title('Scan Results Distribution')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with da2:
                st.markdown('<div class="sec">📅 Daily Scan Volume</div>', unsafe_allow_html=True)
                date_counts = defaultdict(int)
                for s in scans:
                    d = s.get("date", "")[:10]
                    if d:
                        date_counts[d] += 1
                sorted_dates = sorted(date_counts.keys())
                fig, ax = dark_fig(5, 3.8)
                if len(sorted_dates) > 1:
                    y_vals = [date_counts[d] for d in sorted_dates]
                    ax.plot(sorted_dates, y_vals, '.-', color='#00e5ff', lw=2, markersize=8)
                    ax.fill_between(range(len(sorted_dates)), y_vals,
                                    alpha=0.15, color='#00e5ff')
                    ax.set_xticks(range(len(sorted_dates)))
                    ax.set_xticklabels(sorted_dates, rotation=30, ha='right', fontsize=7)
                elif sorted_dates:
                    ax.bar(sorted_dates, [date_counts[d] for d in sorted_dates],
                           color='#00e5ff', width=0.4)
                else:
                    ax.text(0.5, 0.5, 'No date data', ha='center', va='center',
                            color='#6a8a9a', transform=ax.transAxes)
                ax.set_ylabel("Scans")
                ax.set_title("Daily Scan Volume")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            db1, db2 = st.columns(2, gap="medium")

            with db1:
                st.markdown('<div class="sec">🔴 Tumor Type Breakdown</div>', unsafe_allow_html=True)
                tumor_list = [s for s in scans if s.get("diagnosis") != "notumor"]
                if tumor_list:
                    tc     = Counter(s.get("diagnosis") for s in tumor_list)
                    pal2   = ['#ff4d6d', '#ffab40', '#00e5ff']
                    fig, ax = dark_fig(5, 3.2)
                    bars   = ax.bar(list(tc.keys()), list(tc.values()),
                                    color=pal2[:len(tc)], width=0.5)
                    for bar, v in zip(bars, tc.values()):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.05, str(v),
                                ha='center', color='#e8f4f8', fontsize=9)
                    ax.set_ylabel("Count")
                    ax.set_title("Tumor Type Distribution")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.markdown('<div class="rbox r-green">✅ No tumor cases recorded yet.</div>',
                                unsafe_allow_html=True)

            with db2:
                st.markdown('<div class="sec">🎯 AI Confidence Distribution</div>', unsafe_allow_html=True)
                confs = [s.get("confidence", 0) * 100 for s in scans]
                fig, ax = dark_fig(5, 3.2)
                ax.hist(confs, bins=min(10, len(confs)), color='#00e5ff',
                        alpha=0.8, edgecolor='#04080f', linewidth=1.2)
                ax.axvline(x=np.mean(confs), color='#ffab40', linestyle='--', lw=1.8,
                           label=f'Mean: {np.mean(confs):.1f}%')
                ax.set_xlabel("Confidence (%)")
                ax.set_ylabel("Count")
                ax.set_title("AI Confidence Distribution")
                ax.legend(facecolor='#0b1220', labelcolor='#e8f4f8', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.markdown('<div class="sec">📋 Full Scan History</div>', unsafe_allow_html=True)
            df_rows = []
            for s in reversed(scans):
                df_rows.append({
                    "Scan ID":    s.get("id", ""),
                    "Patient":    s.get("patient_name", ""),
                    "Date":       s.get("date", ""),
                    "Diagnosis":  s.get("diagnosis", "").upper(),
                    "Confidence": f"{s.get('confidence', 0)*100:.1f}%",
                    "Operator":   s.get("operator", ""),
                    "Notes":      s.get("doctor_notes", "")[:40] + "…"
                                  if len(s.get("doctor_notes","")) > 40
                                  else s.get("doctor_notes",""),
                })
            df = pd.DataFrame(df_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.logged_in:
    show_app()
else:
    show_auth()