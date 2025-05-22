# MADGuard AI Explorer – Robust Version (Streamlit App)

import streamlit as st
import graphviz
import pandas as pd
import nltk
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np

nltk.download("punkt")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="MADGuard AI Explorer")
st.title("🧠 MADGuard AI Explorer - Robust Diagnostic Mode")

st.markdown(
    """
This tool simulates potential feedback loops in RAG pipelines **and** evaluates risk of model autophagy disorder using your uploaded model outputs and training sets.

## 🔍 Project Scope and Purpose
**MADGuard AI Explorer** is a modern, thoughtful, and feature-rich tool designed to detect and assess feedback loop risks (Model Autophagy Disorder, or MAD) in Retrieval-Augmented Generation (RAG) pipelines.

### 🎯 Core Goals
- Detect risks of synthetic data being reabsorbed into model training
- Provide clear, explainable diagnostics (TTR, embedding similarity, MAD Score)
- Educate users about pipeline hygiene with visual simulation
- Serve as an early-warning tool before retraining

### ✅ Features
- Interactive RAG pipeline simulation with branching logic
- Upload model outputs + training sets for evaluation
- Lexical diversity analysis (Type-Token Ratio)
- Semantic similarity scoring (using Sentence Transformers)
- Combined MAD Risk Score with interpretation
- Visual feedback through color-coded gauges
- Accessible UI using Streamlit
- Optional signup interest for **Pro version** (multi-file analysis)

### 👤 Ideal Users
- AI engineers managing RAG stacks
- ML researchers exploring model drift or overfitting
- Product managers auditing AI retraining workflows
- Open-source contributors building responsible GenAI systems

### 📦 Install All Dependencies
To install everything required to run this app, use:
```bash
pip install streamlit graphviz pandas nltk scikit-learn sentence-transformers matplotlib numpy
```

This will match all imports and functionality used in the code.



### ✉️ Interested in the Pro version?
The upcoming Pro version of MADGuard will allow:
- Bulk upload of `.csv` or folder of `.txt` files
- Automatic batch scoring and trend visualization
- Exportable audit reports

👉 [Click here to express interest](https://forms.gle/your-form-link) and be first in line!
"""
)

# Section 1: Pipeline Simulation
st.header("1. Pipeline Simulation")
source = st.markdown(
    "ℹ️ **Explanation**: Choose how your data is sourced. Real User Inputs reflect human questions like queries, commands, or tasks. Synthetic Generated Data simulates model-generated completions, summaries, or synthetic text being reused — which may cause feedback loop risk if retrained."
)

st.radio(
    "Select input source:",
    ["Real User Inputs", "Synthetic Generated Data"],
    help="Choose the origin of your data. Synthetic inputs may increase feedback loop risk if reused for training.",
)

# Generate pipeline diagram
dot = graphviz.Digraph()
dot.edge("User Query", "Retriever")
dot.edge("Retriever", "LLM")
dot.edge("LLM", "Response")
dot.edge(
    "Response",
    "Retraining Set" if source == "Synthetic Generated Data" else "Embedding Store",
)
st.graphviz_chart(dot)

if source == "Synthetic Generated Data":
    st.error("⚠️ High loop risk: Model may be learning from its own outputs.")
else:
    st.success("✅ Healthy loop: Using diverse real inputs.")

# Section 2: Upload files for evaluation
st.header("2. Upload Output & Training Data")
output_file = st.file_uploader(
    "Upload Model Output File (TXT)",
    type="txt",
    help="Upload the file containing AI-generated responses.",
)
training_file = st.file_uploader(
    "Upload Future Training File (TXT)",
    type="txt",
    help="Upload the dataset planned for fine-tuning or retraining.",
)

if output_file and training_file:
    # Basic size check to help users avoid breaking limits
    if len(output_file.read()) > 1000000 or len(training_file.read()) > 1000000:
        st.warning(
            "⚠️ One or both files are very large. This tool is designed for smaller samples. Consider chunking or summarizing your content."
        )
    output_file.seek(0)
    training_file.seek(0)
    output_text = output_file.read().decode("utf-8")
    train_text = training_file.read().decode("utf-8")

    # Tokenization and TTR
    tokenizer = TreebankWordTokenizer()
    output_tokens = tokenizer.tokenize(output_text)
    train_tokens = tokenizer.tokenize(train_text)

    ttr_output = len(set(output_tokens)) / len(output_tokens)
    ttr_train = len(set(train_tokens)) / len(train_tokens)

    st.write(
        f"Lexical Diversity (Output): {ttr_output:.2f}",
        help="TTR = unique words / total words. Low TTR may signal repetition or synthetic patterning.",
    )
    st.write(
        f"Lexical Diversity (Training Set): {ttr_train:.2f}",
        help="TTR of your training data. Higher TTR generally means better generalization potential.",
    )

    # Embedding Similarity
    emb_output = model.encode(output_text)
    emb_train = model.encode(train_text)
    similarity = cosine_similarity([emb_output], [emb_train])[0][0]

    st.write(
        f"Semantic Similarity (Embedding Cosine): {similarity:.2f}",
        help="Cosine similarity measures how closely the meaning of your output matches training content.",
    )

    # MAD Score
    mad_score = (1 - ttr_output) * 0.3 + similarity * 0.7
    risk_level = "High" if mad_score > 0.75 else "Medium" if mad_score > 0.5 else "Low"

    st.subheader("MAD Risk Score")
    st.write(
        f"Calculated Score: {mad_score:.2f} → **{risk_level} Risk**",
        help="MAD Score = 0.3 × (1 - TTR) + 0.7 × Semantic Similarity",
    )

    # Display gauge-style bar
    fig, ax = plt.subplots()
    ax.barh(
        ["MAD Score"],
        [mad_score],
        color=(
            "red"
            if risk_level == "High"
            else "orange" if risk_level == "Medium" else "green"
        ),
    )
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    st.info(
        "This is an experimental score based on lexical repetition and semantic overlap. Interpret with context."
    )

    with st.expander("🔍 What does this score mean?"):
        with st.expander("🔬 Score Breakdown"):
            st.write(f"TTR Component: {(1 - ttr_output) * 0.3:.2f}")
        st.write(f"Similarity Component: {similarity * 0.7:.2f}")
        st.caption("MAD Score = 0.3 × (1 - TTR) + 0.7 × Semantic Similarity")
        st.markdown(
            """
        - A **high MAD score** means your model's recent outputs are **very similar** to the data you're about to train on.
        - This indicates a **feedback loop** — where the model may reinforce its own phrasing without learning anything new.
        - It's like a student memorizing their own answers over and over instead of studying new material.

        ### 🔴 What You Can Do:
        - Add more **real user input** to training data.
        - Avoid retraining on model outputs with **high similarity** (> 0.8).
        - **Paraphrase or diversify** generated content before reuse.
        - Use filters to **remove repetitive patterns** from model-generated responses.
        """
        )

# Call-to-action footer
st.markdown("---")
st.markdown("👀 Want to try batch scoring or get early access to Pro version?")
st.markdown("[**📩 Join the waitlist**](https://forms.gle/your-form-link)")
