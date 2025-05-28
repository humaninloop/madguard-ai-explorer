import gradio as gr
import nltk
import os
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import graphviz
from typing import Tuple, Optional
from visuals.score_card import render_score_card  # Updated import
from visuals.layout import (
    render_page_header,
    render_core_reference,
    render_pipeline,
    render_pipeline_graph,
    render_pipeline_warning,
    render_strategy_alignment,
)  # Updated import

# Ensure NLTK data is downloaded
try:
    nltk.download("punkt", quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_ttr(text: str) -> float:
    """Calculates Type-Token Ratio (TTR) for lexical diversity."""
    if not text:
        return 0.0
    words = text.split()
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0.0


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculates cosine similarity between two texts."""
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


def calculate_mad_score(ttr: float, similarity: float) -> float:
    """Calculates the MAD score."""
    return 0.3 * (1 - ttr) + 0.7 * similarity


def get_risk_level(mad_score: float) -> str:
    """Determines the risk level based on the MAD score."""
    if mad_score > 0.7:
        return "High"
    elif 0.4 <= mad_score <= 0.7:
        return "Medium"
    else:
        return "Low"


def process_data(file_obj, model_col: str, train_col: str, data_source: str) -> Tuple[
    Optional[str],
    Optional[bytes],
    Optional[str],
    Optional[str],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """Processes the uploaded file and calculates metrics."""
    try:
        if not file_obj:
            return "Error: No file uploaded.", None, None, None, None, None, None

        global uploaded_df
        df = uploaded_df.get("data")
        if df is None:
            return "Error: File not yet processed.", None, None, None, None, None, None

        if model_col not in df.columns or train_col not in df.columns:
            return (
                "Error: Selected columns not found in the file.",
                None,
                None,
                None,
                None,
                None,
                None,
            )

        output_text = " ".join(df[model_col].astype(str))
        train_text = " ".join(df[train_col].astype(str))

        ttr_output = calculate_ttr(output_text)
        ttr_train = calculate_ttr(train_text)
        similarity = calculate_similarity(output_text, train_text)
        mad_score = calculate_mad_score(ttr_output, similarity)
        risk_level = get_risk_level(mad_score)

        summary, details, explanation = render_score_card(
            ttr_output, ttr_train, similarity, mad_score, risk_level
        )
        evaluation_markdown = summary + details + explanation

        return (
            None,
            render_pipeline_graph(data_source),
            df.head().to_markdown(index=False, numalign="left", stralign="left"),
            evaluation_markdown,
            ttr_output,
            ttr_train,
            similarity,
        )
    except Exception as e:
        return f"An error occurred: {str(e)}", None, None, None, None, None, None


# Store uploaded DataFrame globally for later access
uploaded_df = {}


def update_dropdowns(file_obj) -> Tuple[gr.Dropdown, gr.Dropdown, str]:
    global uploaded_df
    if not file_obj:
        uploaded_df["data"] = None  # Clear cached file
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            "No file uploaded.",
        )

    # Read the file and extract columns
    try:
        file_name = getattr(file_obj, "name", "")
        if file_name.endswith(".csv"):
            df = pd.read_csv(file_obj)
        elif file_name.endswith(".json"):
            df = pd.read_json(file_obj)
        else:
            return (
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                "Invalid file type.",
            )

        uploaded_df["data"] = df
        columns = df.columns.tolist()
        preview = df.head().to_markdown(index=False, numalign="left", stralign="left")

        return (
            gr.update(choices=columns, value=None),
            gr.update(choices=columns, value=None),
            preview,
        )

    except Exception as e:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            f"Error reading file: {e}",
        )


def clear_all_fields():
    global uploaded_df
    uploaded_df.clear()  # Clear stored DataFrame
    return (
        None,  # file_input
        gr.update(choices=[], value=None),  # model_col_input
        gr.update(choices=[], value=None),  # train_col_input
        "",  # file_preview
        "",  # output_markdown
        "",  # warning_output
        None,  # ttr_output_metric
        None,  # ttr_train_metric
        None,  # similarity_metric
        render_pipeline_graph("Synthetic Generated Data"),  # pipeline_output default
    )


def main_interface():
    css = """
    .gradio-container {
        background: linear-gradient(-45deg, #e0f7fa, #e1f5fe, #f1f8e9, #fff3e0);
        background-size: 400% 400%;
        animation: oceanWaves 20s ease infinite;
    }
    @keyframes oceanWaves {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    """

    with gr.Blocks(css=css, title="MADGuard AI Explorer") as interface:
        gr.HTML(render_page_header())

        gr.Markdown(
            """
            > 🧠 **MADGuard AI Explorer** helps AI engineers, researchers, and MLOps teams simulate feedback loops in RAG pipelines and detect **Model Autophagy Disorder (MAD)** — where models start learning from their own outputs, leading to degraded performance.

            - Compare **real vs. synthetic input effects**
            - Visualize the data flow
            - Upload your `.csv` or `.json` data
            - Get immediate MAD risk diagnostics based on lexical diversity and semantic similarity
            """
        )

        with gr.Accordion("📚 Research Reference", open=False):
            gr.HTML(render_core_reference())
        gr.HTML(
            """
            <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 20px;">
                <h3 style="text-align: center;">📽️ How to Use MADGuard AI Explorer</h3>
                <iframe width="720" height="405"
                        src="https://www.youtube.com/embed/qjMwvaBXQeY"
                        title="MADGuard AI Tutorial" frameborder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowfullscreen></iframe>
            </div>
            """
        )

        gr.Markdown("## 1. Pipeline Simulation")
        data_source, description = render_pipeline(default="Synthetic Generated Data")

        gr.HTML(description)
        pipeline_output = gr.Image(type="filepath", label="Pipeline Graph")
        warning_output = gr.HTML()
        data_source.change(
            fn=render_pipeline_warning, inputs=data_source, outputs=warning_output
        )
        data_source.change(
            fn=render_pipeline_graph, inputs=data_source, outputs=pipeline_output
        )
        interface.load(
            fn=render_pipeline_graph,
            inputs=[data_source],
            outputs=[pipeline_output],
        )

        gr.Markdown("## 2. Upload CSV or JSON File")
        file_input = gr.File(
            file_types=[".csv", ".json"], label="Upload a CSV or JSON file"
        )
        clear_btn = gr.Button("🧹 Clear All")

        gr.Markdown(
            """
    📝 **Note:**  
    - **Model Output Column**: Select the column that contains generated responses, completions, or predictions from your model.  
    - **Training Data Column**: Select the column that may be used for future training or fine-tuning.  
    This helps MADGuard simulate feedback loops by comparing lexical and semantic overlap between current output and future inputs.
    """
        )

        with gr.Row():
            model_col_input = gr.Dropdown(
                choices=[],
                value=None,
                label="Select column for model output",
                interactive=True,
            )
            train_col_input = gr.Dropdown(
                choices=[],
                value=None,
                label="Select column for future training data",
                interactive=True,
            )
        file_preview = gr.Markdown(label="📄 File Preview")

        output_markdown = gr.Markdown(label="🔍 Evaluation Summary")

        with gr.Accordion("📋 Research-Based Strategy Alignment", open=False):
            gr.HTML(render_strategy_alignment())

        with gr.Row():
            ttr_output_metric = gr.Number(label="Lexical Diversity (Output)")
            ttr_train_metric = gr.Number(label="Lexical Diversity (Training Set)")
            similarity_metric = gr.Number(label="Semantic Similarity (Cosine)")

        def handle_file_upload(file_obj, data_source_val):
            dropdowns = update_dropdowns(file_obj)
            graph = render_pipeline_graph(data_source_val)
            return *dropdowns, graph

        file_input.change(
            fn=handle_file_upload,
            inputs=[file_input, data_source],
            outputs=[model_col_input, train_col_input, file_preview, pipeline_output],
        )

        def process_and_generate(
            file_obj, model_col_val: str, train_col_val: str, data_source_val: str
        ):
            error, graph, preview, markdown, ttr_out, ttr_tr, sim = process_data(
                file_obj, model_col_val, train_col_val, data_source_val
            )
            if error:
                return error, graph, warning_output, preview, None, None, None, None
            return (
                "",
                graph,
                render_pipeline_warning(data_source_val),
                preview,
                markdown,
                ttr_out,
                ttr_tr,
                sim,
            )

        inputs = [file_input, model_col_input, train_col_input, data_source]
        outputs = [
            gr.Markdown(label="⚠️ Error Message"),
            pipeline_output,
            warning_output,
            file_preview,
            output_markdown,
            ttr_output_metric,
            ttr_train_metric,
            similarity_metric,
        ]
        clear_btn.click(
            fn=clear_all_fields,
            inputs=[],
            outputs=[
                file_input,
                model_col_input,
                train_col_input,
                file_preview,
                output_markdown,
                warning_output,
                ttr_output_metric,
                ttr_train_metric,
                similarity_metric,
                pipeline_output,
            ],
        )

        for input_component in inputs:
            input_component.change(
                fn=process_and_generate, inputs=inputs, outputs=outputs
            )

        gr.Markdown("---")
        gr.Markdown(
            """
        **The upcoming Pro version of MADGuard will allow:**
        - 📂 Bulk upload support for `.csv` files or folders of `.txt` documents  
        - 📊 Automated batch scoring with trend visualizations over time  
        - 🧾 One-click export of audit-ready diagnostic reports

        [**📩 Join the waitlist**](https://docs.google.com/forms/d/e/1FAIpQLSfAPPC_Gm7DQElQSWGSnoB6T5hMxb_rXSu48OC8E6TNGZuKgQ/viewform?usp=sharing&ouid=118007615320536574300)
        """
        )

    return interface


# Launch the Gradio interface
if __name__ == "__main__":
    interface = main_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)
