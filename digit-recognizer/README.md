# ✏️ Handwritten Digit Recognizer

A live deep learning web app — draw any digit (0–9) on a canvas and a CNN instantly recognizes it.

**Live Demo →** [Hugging Face Spaces](#) &nbsp;|&nbsp; [Streamlit Cloud](#)

---

## 🎯 What it does

| | |
|---|---|
| 🖊️ | Draw a digit on an interactive canvas |
| 🧠 | CNN model predicts the digit in real-time |
| 📊 | Confidence bar chart for all 10 digits |
| ♻️ | Clear and redraw anytime |

---

## 🏗️ Model Architecture

Custom CNN trained on **MNIST** (70,000 handwritten digit images):

```
Input (28×28×1)
  ↓
[Conv3x3 → BN → ReLU] × 2 → MaxPool → Dropout
  ↓
[Conv3x3 → BN → ReLU] × 2 → MaxPool → Dropout
  ↓
GlobalAveragePooling → Dense(128) → Dropout
  ↓
Dense(10, softmax)
```

**Test Accuracy: ~99%**

---

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (only once ~2 min)
python train_model.py

# 3. Launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deploy

### Streamlit Cloud (Free)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file:** `app.py`
5. Click **Deploy** ✅

> ⚠️ Upload `model/mnist_cnn.keras` to the repo before deploying (GitHub allows files up to 100MB; MNIST model is ~2MB)

### Hugging Face Spaces (Free)
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Gradio** as SDK
3. Upload all files + rename `app_huggingface.py` → `app.py`
4. Upload `model/mnist_cnn.keras`
5. Use `requirements_huggingface.txt` as `requirements.txt`
6. Space auto-deploys ✅

---

## 📁 Project Structure

```
digit-recognizer/
├── app.py                    # Streamlit app
├── app_huggingface.py        # Hugging Face / Gradio app
├── train_model.py            # Model training script
├── model/
│   └── mnist_cnn.keras       # Saved model (generate via train_model.py)
├── requirements.txt          # Streamlit dependencies
├── requirements_huggingface.txt  # HF dependencies
└── README.md
```

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** — CNN model
- **Streamlit** + **streamlit-drawable-canvas** — Web UI
- **Gradio** — Hugging Face interface
- **Plotly** — Confidence chart
- **MNIST** — Training dataset

---

## 📄 License

MIT License
