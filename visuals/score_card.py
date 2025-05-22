# File: visuals/score_card.py
import streamlit as st
import matplotlib.pyplot as plt


def render_score_card(ttr_output, ttr_train, similarity, mad_score, risk_level):
    st.markdown("### 🔍 Evaluation Summary")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Lexical Diversity (Output)", f"{ttr_output:.2f}")
            st.caption("TTR = unique words / total words")
            st.metric("Lexical Diversity (Training Set)", f"{ttr_train:.2f}")
            st.caption("Broader content = higher TTR")

        with col2:
            st.metric("Semantic Similarity (Cosine)", f"{similarity:.2f}")
            st.caption("Cosine similarity between embeddings")

    color = {"High": "#e57373", "Medium": "#ffb74d", "Low": "#81c784"}[risk_level]
    st.markdown(
        f"""
    <div style="padding: 1rem; background-color: #fdfdfd; border-left: 6px solid {color}; border-radius: 6px;">
        <strong>MAD Risk Score:</strong> {mad_score:.2f} → <span style='color: {color}; font-weight: bold;'>{risk_level} Risk</span>
    </div>
    <div style='margin-top: 0.5rem; width: 100%; background: #e0e0e0; border-radius: 10px; height: 16px;'>
        <div style='width: {mad_score * 100:.0f}%; background: {color}; height: 100%; border-radius: 10px;'></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("📊 Score Breakdown"):
        st.write(f"TTR Component (0.3 × (1 - TTR)): {(1 - ttr_output) * 0.3:.2f}")
        st.write(f"Similarity Component (0.7 × Cosine): {similarity * 0.7:.2f}")
        st.caption("MAD Score = 0.3 × (1 - TTR) + 0.7 × Semantic Similarity")

    with st.expander("🔍 What does this score mean?"):
        if risk_level == "High":
            st.markdown(
                """
🚨 **High Risk Detected**  
Your model outputs are **very similar** to your planned training data.

This suggests a **strong feedback loop**, meaning the model is likely to reinforce existing patterns rather than learning new behaviors.

**What You Can Do**:
- Replace synthetic data with more **diverse real user input**
- Use **paraphrasing techniques** before reuse
- Add **augmentation or filtering** before retraining
                """
            )
        elif risk_level == "Medium":
            st.markdown(
                """
🟠 **Moderate Risk Identified**  
There is some overlap between your outputs and training content.

Your model may partially reinforce existing phrasing patterns.

**Suggestions**:
- Mix synthetic and real inputs carefully
- Monitor training logs for semantic redundancy
                """
            )
        else:
            st.markdown(
                """
🟢 **Low Risk Score**  
Your model output and training data appear **diverse** and distinct.

This is a good sign that your model is learning from **new and varied sources**.

**You’re on the right track!**
                """
            )
