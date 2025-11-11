# ============================================================
# 📰 Real-Time Fake News Detector (AI Powered - No Dataset)
# Model: roberta-base-openai-detector
# ============================================================

import streamlit as st
from transformers import pipeline

# ------------------------------------------------------------
# 1. Streamlit Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Title and intro
st.title("📰 AI-Powered Fake News Detector (Real-Time)")
st.write(
    "This tool uses a **pre-trained RoBERTa model** to analyze the authenticity "
    "of any news text in real time. No dataset or local training required — "
    "the model is already trained on massive web data."
)
st.markdown("---")

# ------------------------------------------------------------
# 2. Load Pre-Trained Model (Hugging Face)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    st.info("⏳ Loading the AI model... please wait a few seconds.")
    model = pipeline(
        "text-classification",
        model="roberta-base-openai-detector",
        tokenizer="roberta-base-openai-detector"
    )
    return model

model = load_model()
st.success("✅ Model loaded successfully! Ready to analyze.")

# ------------------------------------------------------------
# 3. User Input
# ------------------------------------------------------------
st.markdown("### 🧠 Try it yourself:")
input_text = st.text_area(
    "Paste or type any news headline or article below:",
    height=180,
    placeholder="Example: 'Delhi police confirm explosion near India Gate...'"
)

# ------------------------------------------------------------
# 4. Prediction Logic
# ------------------------------------------------------------
if st.button("Analyze Now 🧾"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some news text first.")
    else:
        with st.spinner("Analyzing authenticity..."):
            result = model(input_text)[0]
            label = result["label"]
            score = result["score"]

            st.markdown("### 🧩 Result:")
            if "FAKE" in label.upper() or "GENERATED" in label.upper():
                st.error(f"🚫 This news seems **Fake or AI-Generated** "
                         f"(Confidence: {score*100:.2f}%)")
            else:
                st.success(f"✅ This news seems **Real / Human-Written** "
                           f"(Confidence: {score*100:.2f}%)")

        st.markdown("---")
        st.caption(
            "⚙️ Powered by [RoBERTa-base-OpenAI-Detector](https://huggingface.co/roberta-base-openai-detector) — "
            "a transformer model trained to distinguish real vs AI-generated text."
        )
