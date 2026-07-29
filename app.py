import numpy as np
import joblib
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Signal — Startup Success Predictor", page_icon="◆", layout="centered")

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    model = joblib.load("model.pkl")
    meta = joblib.load("feature_meta.pkl")
    return scaler, encoder, model, meta

scaler, encoder, model, meta = load_artifacts()

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: transparent !important;
}
html, body { background: #10131C !important; }

#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

.block-container {
    max-width: 760px;
    padding-top: 3.5rem;
    padding-bottom: 4rem;
}

* { font-family: 'Inter', sans-serif; }

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.28em;
    color: #3ED9C7;
    font-size: 0.72rem;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: clamp(2.1rem, 5vw, 3.4rem);
    color: #EDEFF2;
    margin: 0 0 0.7rem 0;
    line-height: 1.05;
}
.hero-sub {
    color: #8A93A3;
    font-size: 1.05rem;
    max-width: 38rem;
    margin-bottom: 2.2rem;
    line-height: 1.55;
}
.panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.16em;
    color: #FFC857;
    margin: 0.4rem 0 0.9rem 0;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(18, 24, 33, 0.55) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 20px !important;
}

label, .stMarkdown p { color: #C7CCD6 !important; }

[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    color: #EDEFF2 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(90deg, #FFC857, #3ED9C7);
    color: #10131C;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 999px;
    padding: 0.75rem 1.6rem;
    width: 100%;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 28px rgba(255, 200, 87, 0.28);
    color: #10131C;
}

.result-verdict {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    text-align: center;
    margin-top: 0.6rem;
}
.result-note {
    color: #8A93A3;
    text-align: center;
    font-size: 0.92rem;
    max-width: 30rem;
    margin: 0.4rem auto 0 auto;
}
.footnote {
    font-family: 'IBM Plex Mono', monospace;
    color: #565D6B;
    font-size: 0.72rem;
    text-align: center;
    margin-top: 2.5rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Interactive constellation background (injected into the parent document)
# ---------------------------------------------------------------------------
components.html("""
<script>
(function() {
  try {
    var doc = window.parent.document;
    if (doc.getElementById('signal-bg-canvas')) return;

    var canvas = document.createElement('canvas');
    canvas.id = 'signal-bg-canvas';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    canvas.style.zIndex = '-1';
    canvas.style.pointerEvents = 'none';
    doc.body.appendChild(canvas);

    var win = window.parent;
    var ctx = canvas.getContext('2d');
    var w, h, dpr;

    function resize() {
      dpr = win.devicePixelRatio || 1;
      w = win.innerWidth;
      h = win.innerHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    win.addEventListener('resize', resize);

    var reduceMotion = win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches;

    var N = Math.max(30, Math.min(90, Math.floor((w * h) / 24000)));
    var nodes = [];
    for (var i = 0; i < N; i++) {
      nodes.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.18,
        vy: (Math.random() - 0.5) * 0.18,
        r: Math.random() * 1.5 + 0.7
      });
    }

    var mouse = { x: w / 2, y: h / 2, active: false };
    win.addEventListener('mousemove', function(e) {
      mouse.x = e.clientX; mouse.y = e.clientY; mouse.active = true;
    });
    win.addEventListener('mouseleave', function() { mouse.active = false; });

    var linkDist = 130;
    var mouseDist = 150;

    function drawFrame() {
      ctx.clearRect(0, 0, w, h);

      for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > w) n.vx *= -1;
        if (n.y < 0 || n.y > h) n.vy *= -1;

        if (mouse.active) {
          var dx = n.x - mouse.x, dy = n.y - mouse.y;
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d < mouseDist && d > 0.01) {
            var f = ((mouseDist - d) / mouseDist) * 0.025;
            n.x += dx * f; n.y += dy * f;
          }
        }
      }

      for (var a = 0; a < nodes.length; a++) {
        for (var b = a + 1; b < nodes.length; b++) {
          var na = nodes[a], nb = nodes[b];
          var ddx = na.x - nb.x, ddy = na.y - nb.y;
          var dist = Math.sqrt(ddx * ddx + ddy * ddy);
          if (dist < linkDist) {
            var alpha = (1 - dist / linkDist) * 0.16;
            ctx.strokeStyle = 'rgba(62,217,199,' + alpha + ')';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(na.x, na.y);
            ctx.lineTo(nb.x, nb.y);
            ctx.stroke();
          }
        }
      }

      for (var k = 0; k < nodes.length; k++) {
        var nn = nodes[k];
        ctx.beginPath();
        ctx.arc(nn.x, nn.y, nn.r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,200,87,0.5)';
        ctx.fill();
      }

      if (!reduceMotion) win.requestAnimationFrame(drawFrame);
    }
    drawFrame();
  } catch (err) { /* background is decorative; fail silently */ }
})();
</script>
""", height=0)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="eyebrow">SIGNAL</div>
<div class="hero-title">Will it scale?</div>
<div class="hero-sub">Predict a startup's odds of staying alive past its next funding round —
operating, acquired, or public — versus shutting down. Enter the funding profile
and company details below.</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input panel
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="panel-label">FUNDING PROFILE</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        funding_total_usd = st.number_input(
            "Total funding raised (USD)", min_value=0, value=1_000_000, step=10_000
        )
        founded_year = st.number_input(
            "Year founded", min_value=1950, max_value=2017, value=2012
        )
    with c2:
        funding_rounds = st.number_input(
            "Number of funding rounds", min_value=1, max_value=20, value=1
        )
        funding_gap_years = st.number_input(
            "Years between first and last round", min_value=0.0, max_value=30.0, value=1.0, step=0.1
        )

    st.markdown('<div class="panel-label">COMPANY PROFILE</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        country_code = st.selectbox("Country", meta["country_options"])
    with c4:
        primary_category = st.selectbox("Primary category", meta["category_options"])

    st.write("")
    predict_clicked = st.button("Run the numbers →", use_container_width=True)

if predict_clicked:
    X_num = np.array([[funding_total_usd, funding_rounds, founded_year, funding_gap_years]])
    X_num_scaled = scaler.transform(X_num)
    X_cat = np.array([[country_code, primary_category]])
    X_cat_encoded = encoder.transform(X_cat)
    X = np.hstack([X_num_scaled, X_cat_encoded])

    prob = float(model.predict_proba(X)[0][1])
    pct = round(prob * 100)
    success = prob >= 0.5
    color = "#FFC857" if success else "#FF6B6B"
    verdict = "Likely to scale" if success else "At risk of shutting down"

    r = 85
    circumference = 2 * np.pi * r
    dash = circumference * prob

    gauge_svg = f"""
    <svg viewBox="0 0 200 200" width="220" height="220" style="display:block;margin:0 auto;">
      <circle cx="100" cy="100" r="{r}" stroke="rgba(255,255,255,0.08)" stroke-width="14" fill="none"/>
      <circle cx="100" cy="100" r="{r}" stroke="{color}" stroke-width="14" fill="none"
        stroke-linecap="round"
        stroke-dasharray="{dash:.2f} {circumference:.2f}"
        transform="rotate(-90 100 100)"/>
      <text x="100" y="94" text-anchor="middle" font-family="'IBM Plex Mono', monospace"
        font-size="34" font-weight="500" fill="{color}">{pct}%</text>
      <text x="100" y="118" text-anchor="middle" font-family="'Inter', sans-serif"
        font-size="12" fill="#8A93A3">probability of success</text>
    </svg>
    """

    with st.container(border=True):
        st.markdown(gauge_svg, unsafe_allow_html=True)
        st.markdown(f'<div class="result-verdict" style="color:{color};">{verdict}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="result-note">Estimated from historical funding and category patterns. '
            'One signal among many — not investment advice.</div>',
            unsafe_allow_html=True
        )

st.markdown(
    '<div class="footnote">MODEL — RANDOM FOREST · TRAINED ON HISTORICAL STARTUP FUNDING DATA<br>'
    'PREDICTIONS ARE STATISTICAL ESTIMATES, NOT GUARANTEES</div>',
    unsafe_allow_html=True
)
