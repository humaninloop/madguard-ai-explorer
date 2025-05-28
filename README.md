---
title: MADGuard AI Explorer
emoji: 🧠
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# 🧠 MADGuard AI Explorer

A diagnostic Gradio tool to simulate feedback loops in Retrieval-Augmented Generation (RAG) pipelines and detect **Model Autophagy Disorder (MAD)** risks.

---

## 🛠️ Tool Description

- Toggle between **real** and **synthetic** input sources
- Visualize pipeline feedback loops with **Graphviz**
- Analyze training data via:
  - Type-Token Ratio (TTR)
  - Cosine Similarity
  - Composite MAD Risk Score

---

## 🚀 Run It Locally

```bash
git clone <your-repo-url>
cd madguard
pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## 🌐 Deploy on Hugging Face Spaces

1. Create a new Space (select **Gradio** as the SDK)
2. Upload:
   - `app.py`
   - `requirements.txt`
   - All files in the `visuals/` folder
3. Hugging Face builds the app and gives you a public URL

---

<details>
<summary>📚 Research Background</summary>

### 📄 Self-consuming LLMs: How and When Models Feed Themselves – Santurkar et al., 2023

This paper introduces and explores **Model Autophagy Disorder (MAD)** — showing that large language models trained on their own outputs tend to lose performance and accumulate error over time.

**MADGuard implements several of the paper’s proposed detection strategies:**

| Research Recommendation                     | MADGuard Implementation                   |
| ------------------------------------------- | ----------------------------------------- |
| Lexical redundancy analysis                 | ✅ via Type-Token Ratio (TTR)             |
| Embedding-based similarity scoring          | ✅ via SentenceTransformers + cosine      |
| Warning system for feedback loop risk       | ✅ risk score (Low / Medium / High)       |
| Distinguishing real vs. synthetic inputs    | ❌ not implemented (user-controlled only) |
| Multi-round retraining degradation tracking | ❌ not yet supported                      |

> “MADGuard AI Explorer is inspired by key findings from this research, aligning with early warnings and pipeline hygiene practices recommended in their work.”

📎 [Read Full Paper on arXiv](https://arxiv.org/abs/2307.01850)

</details>

---

<details>
<summary>👥 Who Is It For?</summary>

- **AI/ML Engineers**: Prevent model collapse due to training on synthetic outputs
- **MLOps Professionals**: Pre-retraining diagnostics
- **AI Researchers**: Study model feedback loops
- **Responsible AI Teams**: Audit data pipelines for ethical AI

### Why Use It?

- Avoid data contamination
- Ensure model freshness
- Support data-centric decisions
- Provide audit-ready diagnostics

</details>

---

<details>
<summary>🧱 Limitations & Future Plans</summary>

### 🔸 Current Limitations

| Area                | Missing Element                           |
| ------------------- | ----------------------------------------- |
| Multi-batch Uploads | No history or comparative dataset support |
| Real/Synthetic Tag  | No auto-tagging or provenance logging     |
| Visual Analytics    | No charts, timelines, or embeddings view  |
| Custom Thresholds   | Fixed MAD score weightings                |
| Provenance Tracking | No metadata or source history logging     |

### 🔮 Future Plans

- 📊 Batch evaluations with historical trendlines
- 🧠 RAG framework integration (e.g., LangChain)
- 🧩 Live evaluation API endpoint
- 🔒 Source tracking and audit trails
- 🧾 Exportable audit/compliance reports

</details>

---

<details>
<summary>📄 More Details</summary>

### 🔍 Features Recap

- Simulates feedback loops in RAG pipelines
- Visualizes flow using Graphviz
- Accepts `.csv` or `.json` data
- Calculates TTR, cosine similarity, MAD score
- Classifies risk (Low / Medium / High)
- Offers human-readable suggestions
- Based on: [Santurkar et al., 2023 – arXiv:2307.01850](https://arxiv.org/abs/2307.01850)

### 📜 License

MIT License (see [LICENSE](LICENSE))

</details>

---
