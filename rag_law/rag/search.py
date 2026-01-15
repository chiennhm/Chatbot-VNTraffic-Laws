# -*- coding: utf-8 -*-
"""
Hybrid RAG Search with Re-ranker for Vietnamese Traffic Law.
Features: Hybrid Search (Dense + Sparse), RRF Fusion, Cross-Encoder Re-ranking.
"""

import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger(__name__)

# Configuration from environment
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "VN_traffic_regulation_md_hybrid_v2")
DENSE_MODEL_ID = os.getenv("DENSE_MODEL_ID", "Savoxism/vietnamese-legal-embedding-finetuned")
SPARSE_MODEL_ID = os.getenv("SPARSE_MODEL_ID", "Qdrant/bm25")
RERANK_MODEL_ID = os.getenv("RERANK_MODEL_ID", "namdp-ptit/ViRanker")

# Hybrid weights
WEIGHT_DENSE = float(os.getenv("WEIGHT_DENSE", "0.8"))
WEIGHT_SPARSE = float(os.getenv("WEIGHT_SPARSE", "0.2"))
RRF_K = int(os.getenv("RRF_K", "50"))

# Global model cache
_dense_model = None
_sparse_model = None
_reranker = None
_qdrant_client = None


@dataclass
class SearchResult:
    """Single search result with metadata."""
    content: str
    full_content: str
    document_id: str
    article: str
    score: float
    rerank_score: float


def get_full_content(payload: Dict) -> str:
    """
    Combine contextual_content and content intelligently.
    Avoids duplication if content already contains context.
    """
    contextual = (payload.get("contextual_content") or "").strip()
    main_content = (payload.get("content") or "").strip()

    if not contextual:
        return main_content
    if not main_content:
        return contextual

    # Avoid duplication if context is already in main content
    if contextual in main_content:
        return main_content

    return f"{contextual}\n\n{main_content}"



def get_qdrant_client():
    """Get or create Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        log.info("Qdrant client initialized")
    return _qdrant_client


def get_dense_model():
    """Get or create dense embedding model."""
    global _dense_model
    if _dense_model is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading dense model: {DENSE_MODEL_ID}")
        _dense_model = SentenceTransformer(DENSE_MODEL_ID)
        log.info("Dense model loaded")
    return _dense_model


def get_sparse_model():
    """Get or create sparse embedding model (BM25)."""
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        log.info(f"Loading sparse model: {SPARSE_MODEL_ID}")
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_ID, threads=1)
        log.info("Sparse model loaded")
    return _sparse_model


def get_reranker():
    """Get or create cross-encoder re-ranker."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading re-ranker: {RERANK_MODEL_ID} on {device}")
        _reranker = CrossEncoder(RERANK_MODEL_ID, device=device)
        log.info("Re-ranker loaded")
    return _reranker


def preload_models():
    """Preload all models for faster first query."""
    log.info("Preloading RAG models...")
    get_qdrant_client()
    get_dense_model()
    get_sparse_model()
    get_reranker()
    log.info("All RAG models loaded")


def _compute_rrf_score(rank: int, weight: float) -> float:
    """Compute Reciprocal Rank Fusion score."""
    return weight * (1.0 / (RRF_K + rank))


def _hybrid_retrieval(
    query: str,
    retrieval_limit: int = 50
) -> Dict[str, Dict]:
    """
    Step 1: Hybrid retrieval using Dense + Sparse vectors.
    Returns fusion scores for each document.
    """
    from qdrant_client.http import models

    client = get_qdrant_client()
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()

    # Encode query
    dense_vector = dense_model.encode(query).tolist()
    sparse_output = list(sparse_model.embed([query]))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_output.indices.tolist(),
        values=sparse_output.values.tolist()
    )

    # Dense search
    dense_response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=dense_vector,
        using="dense",
        limit=retrieval_limit,
        with_payload=True
    )

    # Sparse search
    sparse_response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=sparse_vector,
        using="sparse",
        limit=retrieval_limit,
        with_payload=True
    )

    # Weighted RRF Fusion
    fusion_scores = {}

    for rank, hit in enumerate(dense_response.points):
        point_id = hit.id
        if point_id not in fusion_scores:
            fusion_scores[point_id] = {"score": 0.0, "payload": hit.payload}
        fusion_scores[point_id]["score"] += _compute_rrf_score(rank + 1, WEIGHT_DENSE)

    for rank, hit in enumerate(sparse_response.points):
        point_id = hit.id
        if point_id not in fusion_scores:
            fusion_scores[point_id] = {"score": 0.0, "payload": hit.payload}
        fusion_scores[point_id]["score"] += _compute_rrf_score(rank + 1, WEIGHT_SPARSE)

    log.debug(f"Hybrid retrieval found {len(fusion_scores)} candidates")
    return fusion_scores


def _rerank_candidates(
    query: str,
    candidates: List[Dict],
    batch_size: int = 8
) -> List[Dict]:
    """
    Step 2: Re-rank candidates using Cross-Encoder.
    """
    if not candidates:
        return []

    reranker = get_reranker()

    # Prepare inputs: [query, document_text] pairs
    rerank_inputs = [
        [query, get_full_content(item["payload"])]
        for item in candidates
    ]

    # Get re-rank scores
    rerank_scores = reranker.predict(rerank_inputs, batch_size=batch_size)

    # Attach scores to candidates
    for i, item in enumerate(candidates):
        item["rerank_score"] = float(rerank_scores[i])

    # Sort by re-rank score
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    log.debug(f"Re-ranked {len(reranked)} candidates")

    return reranked


def search(
    query: str,
    top_k: int = 5,
    retrieval_limit: int = 50,
    rerank_limit: int = 30
) -> List[SearchResult]:
    """
    Full hybrid search pipeline.

    Args:
        query: User query string
        top_k: Number of final results to return
        retrieval_limit: Number of candidates from each vector search
        rerank_limit: Number of candidates to pass to re-ranker

    Returns:
        List of SearchResult objects sorted by relevance
    """
    if not query or not query.strip():
        return []

    query = query.strip()
    log.info(f"Searching: {query[:50]}...")

    # Step 1: Hybrid retrieval
    fusion_scores = _hybrid_retrieval(query, retrieval_limit)

    # Sort by fusion score and take top candidates for re-ranking
    candidates = sorted(
        fusion_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:rerank_limit]

    if not candidates:
        log.warning("No candidates found in retrieval")
        return []

    # Step 2: Re-rank
    reranked = _rerank_candidates(query, candidates)

    # Step 3: Build results
    results = []
    for item in reranked[:top_k]:
        payload = item["payload"]
        results.append(SearchResult(
            content=payload.get("content", ""),
            full_content=get_full_content(payload),
            document_id=payload.get("document_id", ""),
            article=payload.get("article", ""),
            score=item["score"],
            rerank_score=item["rerank_score"]
        ))

    log.info(f"Found {len(results)} results")
    return results


def search_for_llm(query: str, top_k: int = 5) -> str:
    """
    Search and format results as context for LLM.

    Args:
        query: User query string
        top_k: Number of results to include

    Returns:
        Formatted context string for LLM prompt
    """
    results = search(query, top_k=top_k)

    if not results:
        return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

    context_parts = []
    for i, result in enumerate(results, 1):
        part = (
            f"[{i}] Nguồn: {result.document_id}\n"
            f"Điều khoản: {result.article}\n"
            f"{result.full_content}"
        )
        context_parts.append(part)

    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="RAG Search CLI")
    parser.add_argument("--query", "-q", type=str, required=True, help="Search query")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    print("Loading models...")
    preload_models()

    print(f"\nSearching: {args.query}\n")
    results = search(args.query, top_k=args.top_k)

    print("=" * 60)
    print(f"FOUND {len(results)} RESULTS")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r.rerank_score:.4f} | Doc: {r.document_id}")
        print(f"Article: {r.article}")
        print(f"Content: {r.full_content[:300]}...")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("CONTEXT FOR LLM:")
    print("=" * 60)
    print(search_for_llm(args.query, top_k=args.top_k))
