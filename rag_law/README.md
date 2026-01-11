# Vietnamese Traffic Law RAG System

RAG pipeline for Vietnamese traffic regulations with hybrid search and re-ranking.

## Project Structure

```
├── rag_law/
│   ├── processing/          # Document processing
│   │   ├── clean_documents.py
│   │   ├── doc_to_markdown.py
│   │   ├── chunk_markdown.py
│   │   └── parse_legal_documents.py
│   ├── rag/                  # RAG search
│   │   ├── search.py         # Main search module
│   │   ├── qdrant_rag.py     # Qdrant setup
│   │   └── model_loader.py
│   └── structured_law/       # Output chunks
├── documents_vn/             # Raw crawled documents
├── documents_clean/          # Cleaned documents
├── documents_markdown/       # Markdown documents
├── run_pipeline.py           # Full pipeline runner
├── .env.example              # Environment template
└── requirements.txt
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your Qdrant credentials

# 5. Run full pipeline
python run_pipeline.py

# 6. Search
python -m rag_law.rag.search
```

## Pipeline Steps

| Step | Command | Description |
|------|---------|-------------|
| 1 | `python run_pipeline.py --only 1` | Clean documents |
| 2 | `python run_pipeline.py --only 2` | Convert to Markdown |
| 3 | `python run_pipeline.py --only 3` | Chunk documents |
| 4 | `python run_pipeline.py --only 4` | Upload to Qdrant |

## Search API

```python
from rag_law.rag.search import search_for_llm, preload_models

preload_models()  # Load once
context = search_for_llm("Mức phạt không đội mũ bảo hiểm", top_k=5)
```

## Environment Variables

```env
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=VN_traffic_regulation_md_hybrid
DENSE_MODEL_ID=Savoxism/vietnamese-legal-embedding-finetuned
SPARSE_MODEL_ID=Qdrant/bm25
RERANK_MODEL_ID=namdp-ptit/ViRanker
```

## Models

- **Dense**: `Savoxism/vietnamese-legal-embedding-finetuned`
- **Sparse**: `Qdrant/bm25` (FastEmbed)
- **Re-ranker**: `namdp-ptit/ViRanker`


