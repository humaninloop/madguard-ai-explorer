import gradio as gr
import graphviz
import pandas as pd
from typing import Tuple
import tempfile
import os


def render_page_header() -> str:
    """Renders the page header."""
    return """
    <div style="text-align: center; margin-top: 1rem;">
        <h1 style="margin-bottom: 0.25rem;">MADGuard AI Explorer</h1>
        <h4 style="color: grey; font-weight: 400;">Robust Diagnostic Mode for RAG Pipeline Feedback Loops</h4>
    </div>
    """


def render_core_reference() -> str:
    """Renders the research reference section."""
    return """
    <details>
    <summary>📚 arXiv:2307.01850</summary>
    <p>
    <b>Self-consuming LLMs: How and When Models Feed Themselves</b> – <i>Santurkar et al., 2023</i><br>
    This paper introduces and explores <b>Model Autophagy Disorder (MAD)</b> — showing that large language models trained on their own outputs tend to lose performance and accumulate error over time.

    The paper proposes detection strategies that MADGuard implements, including:
    - Lexical diversity analysis
    - Embedding-based similarity checks
    - Warnings for training loop risks

    <i>"MADGuard AI Explorer is inspired by key findings from this research, aligning with early warnings and pipeline hygiene practices recommended in their work."</i>

    📎 <a href="https://arxiv.org/pdf/2307.01850" target="_blank">Read Full Paper (arXiv)</a>
    </p>
    </details>
    """


def render_pipeline(default: str = "Real User Inputs") -> Tuple[gr.Radio, str]:
    """Renders the pipeline input selection."""
    with gr.Row():
        source = gr.Radio(
            ["Real User Inputs", "Synthetic Generated Data"],
            label="Select input source:",
            value=default,
            # Removed 'help' parameter to avoid TypeError with Gradio 4.44.0
        )
    description = """<center>ℹ️ Real User Inputs reflect human queries. Synthetic Generated Data simulates model-generated text being reused for retraining.</center>"""
    return source, description


def render_pipeline_graph(source: str) -> str:
    """Generates a graph of the RAG pipeline and returns the image file path."""
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

    # Save to a temporary file and return the file path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        output_path = tmp_file.name
    dot.render(filename=output_path, format="png", cleanup=True)
    return output_path + ".png"


def render_pipeline_warning(source: str) -> str:
    """Renders a warning message based on the data source."""
    if source == "Synthetic Generated Data":
        return "<div style='color:red; font-weight:bold;'>⚠️ High loop risk: Model may be learning from its own outputs.</div>"
    else:
        return "<div style='color:green; font-weight:bold;'>✅ Healthy loop: Using diverse real inputs.</div>"


def render_strategy_alignment() -> str:
    """Renders the strategy alignment table."""
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
    html = """
    <style>
    table { width: 100%; border-collapse: collapse; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f2f2f2; }
    </style>
    <table>
        <thead>
            <tr><th>Strategy from Research</th><th>Status in MADGuard</th><th>Explanation</th></tr>
        </thead>
        <tbody>
    """
    for i in range(len(data["Strategy from Research"])):
        html += f"""
            <tr>
                <td>{data["Strategy from Research"][i]}</td>
                <td>{data["Status in MADGuard"][i]}</td>
                <td>{data["Explanation"][i]}</td>
            </tr>
        """
    html += """
        </tbody>
    </table>
    """
    return html
