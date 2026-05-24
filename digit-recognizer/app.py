"""
Handwritten Digit Recognizer — Streamlit App
Draw a digit (0-9) on the canvas and the CNN will recognize it instantly.
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="✏️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .main { background: #0f0f0f; }
    .title { text-align: center; font-size: 2.5rem; font-weight: 700;
             background: linear-gradient(135deg, #a78bfa, #60a5fa);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { text-align: center; color: #9ca3af; margin-bottom: 2rem; }
    .result-box { background: #1f2937; border-radius: 16px; padding: 1.5rem;
                  text-align: center; border: 2px solid #374151; }
    .digit-display { font-size: 5rem; font-weight: 700; color: #a78bfa; line-height: 1; }
    .confidence { font-size: 1.2rem; color: #60a5fa; margin-top: 0.5rem; }
    .stButton>button { width: 100%; background: linear-gradient(135deg, #7c3aed, #2563eb);
                       color: white; border: none; border-radius: 10px;
                       padding: 0.6rem; font-weight: 600; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/mnist_cnn.keras")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model not found! Run `python train_model.py` first.\n\n{e}")

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="title">✏️ Digit Recognizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw any digit (0–9) and watch the AI recognize it</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("#### 🖊️ Draw here")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if st.button("🗑️ Clear Canvas"):
        st.rerun()

with col2:
    st.markdown("#### 🧠 Prediction")

    if canvas_result.image_data is not None and model_loaded:
        img_array = canvas_result.image_data.astype(np.uint8)

        # Check if canvas has drawing (not all black)
        if img_array[..., :3].sum() > 1000:
            # Preprocess
            img = Image.fromarray(img_array).convert("L")
            img = ImageOps.invert(img)
            img = img.resize((28, 28), Image.LANCZOS)
            x = np.array(img, dtype=np.float32) / 255.0
            x = x[np.newaxis, ..., np.newaxis]

            # Predict
            probs = model(x, training=False).numpy()[0]
            pred  = int(np.argmax(probs))
            conf  = float(probs[pred]) * 100

            # Result display
            st.markdown(f"""
            <div class="result-box">
                <div class="digit-display">{pred}</div>
                <div class="confidence">{conf:.1f}% confident</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability bar chart
            fig = go.Figure(go.Bar(
                x=list(range(10)),
                y=probs * 100,
                marker=dict(
                    color=[("#a78bfa" if i == pred else "#374151") for i in range(10)],
                    line=dict(width=0),
                ),
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="outside",
                textfont=dict(size=9, color="white"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#111827",
                font=dict(color="white", family="Space Grotesk"),
                xaxis=dict(tickvals=list(range(10)), title="Digit", gridcolor="#1f2937"),
                yaxis=dict(title="Confidence %", range=[0, 115], gridcolor="#1f2937"),
                margin=dict(l=10, r=10, t=10, b=30),
                height=220,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="result-box">
                <div style="font-size:3rem">🤔</div>
                <div style="color:#9ca3af; margin-top:0.5rem">Draw a digit to start!</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-box">
            <div style="font-size:3rem">✏️</div>
            <div style="color:#9ca3af; margin-top:0.5rem">Draw a digit on the left</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6b7280; font-size:0.85rem'>"
    "Built with TensorFlow + Streamlit &nbsp;|&nbsp; Trained on MNIST (99%+ accuracy)"
    "</div>",
    unsafe_allow_html=True,
)
