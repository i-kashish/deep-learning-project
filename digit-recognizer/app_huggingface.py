"""
Hugging Face Spaces — Gradio Interface
Handwritten Digit Recognizer
"""

import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image, ImageOps

# ── Load Model ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model("model/mnist_cnn.keras")


def predict_digit(sketch):
    """
    Takes a sketchpad drawing dict from Gradio and returns digit predictions.
    """
    if sketch is None:
        return {str(i): 0.0 for i in range(10)}

    # Gradio sketchpad returns a dict with 'composite' key (RGBA numpy array)
    img_array = sketch["composite"] if isinstance(sketch, dict) else sketch
    img = Image.fromarray(img_array.astype(np.uint8)).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)

    x = np.array(img, dtype=np.float32) / 255.0
    x = x[np.newaxis, ..., np.newaxis]

    probs = model(x, training=False).numpy()[0]
    return {str(i): float(probs[i]) for i in range(10)}


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="✏️ Digit Recognizer",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="blue"),
    css="""
        .title { text-align: center; font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
        .subtitle { text-align: center; color: #6b7280; margin-bottom: 1.5rem; }
    """
) as demo:

    gr.HTML('<div class="title">✏️ Handwritten Digit Recognizer</div>')
    gr.HTML('<div class="subtitle">Draw a digit (0–9) — the CNN will recognize it instantly!</div>')

    with gr.Row():
        with gr.Column():
            sketch = gr.Sketchpad(
                label="Draw here",
                type="numpy",
                height=300,
                width=300,
                brush=gr.Brush(default_size=18, colors=["#ffffff"], default_color="#ffffff"),
                canvas_size=(300, 300),
            )
            with gr.Row():
                clear_btn  = gr.ClearButton(sketch, value="🗑️ Clear")
                submit_btn = gr.Button("🔍 Recognize", variant="primary")

        with gr.Column():
            label_output = gr.Label(
                label="Prediction Confidence",
                num_top_classes=10,
            )

    submit_btn.click(fn=predict_digit, inputs=sketch, outputs=label_output)
    sketch.change(fn=predict_digit, inputs=sketch, outputs=label_output)

    gr.Examples(
        examples=[],
        inputs=sketch,
    )

    gr.Markdown("""
    ---
    **Model:** Custom CNN trained on MNIST &nbsp;|&nbsp; **Accuracy:** ~99%  
    **Stack:** TensorFlow · Keras · Gradio  
    """)

if __name__ == "__main__":
    demo.launch()
