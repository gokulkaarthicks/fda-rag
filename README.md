# FDA RAG

Ask questions over FDA digital-health guidance PDFs and get answers with page citations.

## Pipeline

- **Chunking:** recursive split, size 1200 / overlap 200
- **Embeddings:** Chroma default (ONNX MiniLM), cosine, persisted in `data/chroma/`
- **Retrieval:** dense Chroma + BM25, fused with Reciprocal Rank Fusion
- **Rerank:** `cross-encoder/ms-marco-MiniLM-L-6-v2` → top 5
- **Generation:** OpenRouter (`OPENROUTER_MODEL`), cites document + page

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # set OPENROUTER_API_KEY
```

## Run

PDFs, processed text, and Chroma are generated locally (not in git). Ingest first:

```bash
python scripts/ingest.py   # download → parse → chunk → embed
python scripts/ask.py "What is a Medical Device Data System?"
python scripts/evaluate.py
```

## Eval

Recall@5 = **78.57%** (33/42) on 42 hand-labeled questions across 21 guidance docs. Hybrid retrieval + RRF + rerank. This scores retrieval only, not answer quality.

## Limits

Live FDA scrape can change; needs network for ingest and models; answers depend on retrieved context; no UI.

## Layout

`src/fda_rag/` pipeline · `scripts/` CLIs · `data/raw/` PDFs · `data/chroma/` store · `data/evals/` questions
