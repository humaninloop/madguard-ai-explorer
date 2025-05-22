import streamlit as st
import nltk
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from visuals.layout import (
    render_page_header,
    render_core_reference,
    render_pipeline,
    render_strategy_alignment,
)
from visuals.score_card import render_score_card

nltk.download("punkt")
model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="MADGuard AI Explorer", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(-45deg, #e0f7fa, #e1f5fe, #f1f8e9, #fff3e0);
        background-size: 400% 400%;
        animation: oceanWaves 20s ease infinite;
    }
    @keyframes oceanWaves {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

render_page_header()

st.header("1. Pipeline Simulation")
source = render_pipeline()

st.header("2. Upload CSV or JSON File")
file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_json(file)

    st.subheader("📄 File Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()
    model_col = st.selectbox("Select column for model output", columns)
    train_col = st.selectbox("Select column for future training data", columns)

    output_text = " ".join(df[model_col].astype(str))
    train_text = " ".join(df[train_col].astype(str))

    tokenizer = TreebankWordTokenizer()
    output_tokens = tokenizer.tokenize(output_text)
    train_tokens = tokenizer.tokenize(train_text)

    ttr_output = len(set(output_tokens)) / len(output_tokens) if output_tokens else 0
    ttr_train = len(set(train_tokens)) / len(train_tokens) if train_tokens else 0

    emb_output = model.encode(output_text)
    emb_train = model.encode(train_text)
    similarity = cosine_similarity([emb_output], [emb_train])[0][0]

    mad_score = (1 - ttr_output) * 0.3 + similarity * 0.7
    risk_level = "High" if mad_score > 0.75 else "Medium" if mad_score > 0.5 else "Low"

    render_score_card(ttr_output, ttr_train, similarity, mad_score, risk_level)

render_strategy_alignment()
render_core_reference()

st.markdown("---")
st.markdown(
    """
**The upcoming Pro version of MADGuard will allow:**
- Bulk upload of `.csv` or folder of `.txt` files  
- Automatic batch scoring and trend visualization  
- Exportable audit reports  

[**📩 Join the waitlist**](https://forms.gle/your-form-link)
    """
)
