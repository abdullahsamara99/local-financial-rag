# Local Financial RAG System 📊

An end-to-end, privacy-focused **Retrieval-Augmented Generation (RAG)** system built completely from scratch. It allows users to query dense financial PDF reports locally using an open-source Large Language Model (LLM), optimized specifically to run efficiently on **CPU**.

## 🚀 Key Features
- **100% Local & Secure:** Zero external API dependencies (No OpenAI, No Anthropic). Fully protects sensitive financial data.
- **Custom Vector Store:** Built from scratch using **NumPy** and **Cosine Similarity** instead of heavy database engines like Chroma or FAISS.
- **Page-Level Deduplication:** Implements a smart retrieval strategy that enforces page diversity, preventing a single page from monopolizing the context window.
- **Deterministic Text Generation:** Integrated with a strict system prompt chat template tailored for **Qwen2.5-1.5B-Instruct** to fully eliminate hallucinations and conversational fluff.

## 🛠️ Tech Stack
- **Language:** Python 3
- **LLM:** Qwen2.5-1.5B-Instruct (via Hugging Face Transformers)
- **Embeddings:** `all-MiniLM-L6-v2` (via Sentence-Transformers)
- **Mathematical Engine:** NumPy (Cosine Similarity matrix calculations)
- **Orchestration:** Pure Python & LangChain Core (Prompt Templates)

## 📁 Project Structure
```text
local-financial-rag/
│
├── data/
│   └── vector_store.json          # Local vector storage (Excluded via .gitignore)
│
├── src/
│   ├── __init__.py
│   ├── embedding.py               # Local embedding generation (SentenceTransformers)
│   ├── vector_store.py            # NumPy-based similarity search & page deduplication
│   └── rag_pipeline.py            # Qwen LLM Chat Template pipeline initialization
│
├── main.py                        # Main application CLI execution loop
├── .gitignore                     # Git ignore rules for Windows/Python
└── requirements.txt               # Documented project dependencies