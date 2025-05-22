# File: visuals/layout.py

import streamlit as st
import graphviz
import pandas as pd


def render_page_header():
    with st.container():
        st.markdown(
            """
            <div style="padding: 1.5rem 0; text-align: center;">
                <img src="https://raw.githubusercontent.com/humaninloop/madguard-ai-explorer/main/visuals/header.png" alt="Header Image" style="width: 100%; max-width: 800px; margin-bottom: 1rem;" />
                <h1 style="margin-bottom: 0.25rem;">MADGuard AI Explorer</h1>
                <h4 style="color: grey; font-weight: 400;">Robust Diagnostic Mode for RAG Pipeline Feedback Loops</h4>
            </div>
        """,
            unsafe_allow_html=True,
        )


def render_core_reference():
    with st.expander("📚 Research Reference: arXiv:2307.01850"):
        st.markdown(
            """
**Self-consuming LLMs: How and When Models Feed Themselves** – *Santurkar et al., 2023*  
This paper introduces and explores **Model Autophagy Disorder (MAD)** — showing that large language models trained on their own outputs tend to lose performance and accumulate error over time.

The paper proposes detection strategies that MADGuard implements, including:
- Lexical diversity analysis
- Embedding-based similarity checks
- Warnings for training loop risks

> _"MADGuard AI Explorer is inspired by key findings from this research, aligning with early warnings and pipeline hygiene practices recommended in their work."_

📎 [Read Full Paper (arXiv)](https://arxiv.org/pdf/2307.01850)
        """
        )


def render_pipeline(default="Real User Inputs"):
    source = st.radio(
        "Select input source:",
        ["Real User Inputs", "Synthetic Generated Data"],
        index=0 if default == "Real User Inputs" else 1,
        help="Choose the origin of your data. Synthetic inputs may increase feedback loop risk if reused for training.",
    )
    st.caption(
        "ℹ️ Real User Inputs reflect human queries. Synthetic Generated Data simulates model-generated text being reused for retraining."
    )

    # Transparent and centered container with backdrop blur
    st.markdown(
        """
        <div style='
            display: flex;
            justify-content: center;
            padding: 1rem;
        '>
            <div style='
                backdrop-filter: blur(6px);
                background-color: rgba(255, 255, 255, 0.3);
                padding: 1rem;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                width: fit-content;
            '>
        """,
        unsafe_allow_html=True,
    )

    dot = graphviz.Digraph(
        graph_attr={"rankdir": "LR", "bgcolor": "transparent"},
        node_attr={
            "style": "filled",
            "fillcolor": "#fefefe",
            "color": "#888888",
            "fontname": "Helvetica",
            "fontsize": "12",
        },
        edge_attr={"color": "#999999"},
    )
    dot.edge("User Query", "Retriever")
    dot.edge("Retriever", "LLM")
    dot.edge("LLM", "Response")
    dot.edge(
        "Response",
        "Retraining Set" if source == "Synthetic Generated Data" else "Embedding Store",
    )
    st.graphviz_chart(dot)

    st.markdown("</div></div>", unsafe_allow_html=True)

    if source == "Synthetic Generated Data":
        st.error("⚠️ High loop risk: Model may be learning from its own outputs.")
    else:
        st.success("✅ Healthy loop: Using diverse real inputs.")

    return source


def render_strategy_alignment():
    with st.expander("📋 Research-Based Strategy Alignment"):
        data = {
            "Strategy from Research": [
                "Lexical redundancy (e.g., n-gram overlap)",
                "Embedding-based similarity scoring",
                "Flagging high similarity for retraining risk",
                "Distinguishing real vs. synthetic data",
                "Tracking degradation over retraining iterations",
            ],
            "Status in MADGuard": [
                "✅ Implemented via TTR",
                "✅ Implemented",
                "✅ Implemented (early warning)",
                "❌ Not implemented",
                "❌ Not implemented",
            ],
            "Explanation": [
                "MADGuard uses Type-Token Ratio, a proxy for repetition.",
                "Uses SentenceTransformers + cosine similarity.",
                "Provides a risk score but doesn't block data.",
                "Does not currently track source origin.",
                "No multi-round training history/logs yet.",
            ],
        }
        df = pd.DataFrame(data)
        st.table(df)
