import gradio as gr
from typing import Tuple


def render_score_card(
    ttr_output: float,
    ttr_train: float,
    similarity: float,
    mad_score: float,
    risk_level: str,
) -> Tuple[str, str, str]:
    """Renders the evaluation summary and score details."""

    color = {"High": "#e57373", "Medium": "#ffb74d", "Low": "#81c784"}[risk_level]

    risk_explanations = {
        "High": """
🚨 **High Risk Detected** Your model outputs are **very similar** to your planned training data.  
This suggests a **strong feedback loop**, meaning the model is likely to reinforce existing patterns rather than learning new behaviors.  
**What You Can Do**:  
- Replace synthetic data with more **diverse real user input** - Use **paraphrasing techniques** before reuse  
- Add **augmentation or filtering** before retraining
""",
        "Medium": """
🟠 **Moderate Risk Identified** There is some overlap between your outputs and training content.  
Your model may partially reinforce existing phrasing patterns.  
**Suggestions**:  
- Mix synthetic and real inputs carefully  
- Monitor training logs for semantic redundancy
""",
        "Low": """
🟢 **Low Risk Score** Your model output and training data appear **diverse** and distinct.  
This is a good sign that your model is learning from **new and varied sources**.  
**You’re on the right track!**
""",
    }

    summary = f"""
### 🔍 Evaluation Summary

**Lexical Diversity (Output):** {ttr_output:.2f}  
TTR = unique words / total words

**Lexical Diversity (Training Set):** {ttr_train:.2f}  
Broader content = higher TTR

**Semantic Similarity (Cosine):** {similarity:.2f}  
Cosine similarity between embeddings

<div style="padding: 1rem; background-color: #fdfdfd; border-left: 6px solid {color}; border-radius: 6px;">
    <strong>MAD Risk Score:</strong> {mad_score:.2f} → <span style='color: {color}; font-weight: bold;'>{risk_level} Risk</span>
</div>
<div style='margin-top: 0.5rem; width: 100%; background: #e0e0e0; border-radius: 10px; height: 16px;'>
    <div style='width: {mad_score * 100:.0f}%; background: {color}; height: 100%; border-radius: 10px;'></div>
</div>
"""

    details = f"""
<details>
<summary>📊 Score Breakdown</summary>
TTR Component (0.3 × (1 - TTR)): {(1 - ttr_output) * 0.3:.2f}  
Similarity Component (0.7 × Cosine): {similarity * 0.7:.2f}  
MAD Score = 0.3 × (1 - TTR) + 0.7 × Semantic Similarity
</details>
"""

    explanation = f"""
<details>
<summary>🔍 What does this score mean?</summary>
{risk_explanations[risk_level]}
</details>
"""

    return summary, details, explanation
