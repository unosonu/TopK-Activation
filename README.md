# 🧠 Top-K Activation Analysis (TKAA) & Bias Auditor

[![Ollama](https://img.shields.io/badge/Ollama-Local-blue.svg)](https://ollama.ai/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)


An implementation of the **Top-K Activation Analysis (TKAA)** methodology for interpreting and auditing dense word embeddings. This toolkit allows you to "open the black box" of Large Language Model (LLM) embeddings, mapping mathematical dimensions to human-readable semantic concepts and detecting latent social biases.

Based on the research paper: *"Top-K Activation Analysis of Dense Word Embeddings"*.

---

## 🚀 Key Features

*   **🔍 High-Dimensional Interpretability**: Maps the thousands of dimensions in models like `llama3.2` (3072-dim) or `Gemini` to specific concepts (e.g., *Dimension 112 = Numbers*, *Dimension 573 = Medical Anatomy*).
*   **⚖️ Bias Audit Engine**: Automatically detects gender, racial, and religious biases by measuring geometric correlations between demographic axes and conceptual axes (STEM, Crime, Sentiment).
*   **📉 Local-First Execution**: Fully optimized for **Ollama**. Process massive dictionaries (80k+ words) on local hardware with zero API costs and no rate limits.
*   **📖 Formal Dictionary Support**: Uses the Webster 1913 Unabridged Dictionary for "pure word" context-free embeddings—satisfying the rigorous constraints of the original research.
*   **📊 Visualization Suite**: Generates Jaccard overlap heatmaps, word-level activation charts, and comprehensive bias reports.

---

## 🛠️ Tech Stack

*   **Language**: Python 3.10+
*   **Embeddings**: Ollama (Local `llama3.2`), Google Gemini API (`text-embedding-004`)
*   **Analysis**: NumPy, SciPy, Pandas
*   **Visualization**: Matplotlib, Seaborn

---

## 📦 Getting Started

### 1. Prerequisites
Install [Ollama](https://ollama.ai/) and download the reference model:
```bash
ollama pull llama3.2
```

### 2. Installation
```bash
git clone https://github.com/yourusername/topk-interpretability.git
cd topk-interpretability
pip install numpy scipy matplotlib seaborn pandas google-generativeai
```

---

## 🎮 Usage

### Part 1: Generate Embeddings & TKAA
Run the local Ollama pipeline to fetch embeddings for 80,000+ words and perform the initial dimension interpretation.
```bash
python tkaa_ollama.py
```
*Outputs: `embeddings.npy`, `vocabulary.txt`, `tkaa_results.txt`*

### Part 2: Bias Auditing
Audit the generated embeddings for social biases and conceptual conflation.
```bash
python tkaa_bias_analysis.py
```
*Outputs: `tkaa_bias_report.txt`, `tkaa_bias_scores.png`, `tkaa_bias_heatmap.png`*

---

## 📈 Understanding Results

### Semantic Dimension Mapping
The analysis identifies "Monosemantic" dimensions. For example:
- **Dim 0020**: `butyl`, `ethyl`, `methyl` (Chemistry)
- **Dim 0033**: `psychomancy`, `cartomancy`, `halomancy` (Divination)
- **Dim 0518**: `occipital`, `rectal`, `jejunal` (Anatomy)

### Bias Reporting
The auditor calculates **Differential Cosine Similarity**. A positive score in "Gender × STEM" indicates the model's geometry associates that concept more strongly with one gender over another.

| Test | Result |
|---|---|
| Gender Association | Found significant Female correlation with Care roles (+0.034) |
| Racial Association | Found Non-Western correlation with Criminal dimensions (+0.015) |

---

## 💰 Commercial Applications

1.  **AI Safety Compliance**: Audit corporate models for regulatory bias benchmarks (GDPR, EU AI Act).
2.  **Vector Search Optimization**: Weight specific semantic dimensions to improve search relevance in RAG applications.
3.  **Model De-biasing**: Identify and "nullify" specific dimensions that encode undesirable biases without retraining the model.
4.  **Brand Safety**: Detect if brand embeddings are congregating in high-negative-sentiment dimensions.



---
*Developed as an implementation of the TKAA methodology for LLM Interpretability.*
