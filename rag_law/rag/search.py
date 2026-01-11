# -*- coding: utf-8 -*-
"""
Hybrid RAG Search with Re-ranker for Vietnamese Traffic Law.
Loads config from .env file.
"""

import os
import torch
from typing import List, Dict
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "VN_traffic_regulation_md_hybrid")
DENSE_MODEL_ID = os.getenv("DENSE_MODEL_ID", "Savoxism/vietnamese-legal-embedding-finetuned")
SPARSE_MODEL_ID = os.getenv("SPARSE_MODEL_ID", "Qdrant/bm25")
RERANK_MODEL_ID = os.getenv("RERANK_MODEL_ID", "namdp-ptit/ViRanker")
WEIGHT_DENSE = float(os.getenv("WEIGHT_DENSE", "0.8"))
WEIGHT_SPARSE = float(os.getenv("WEIGHT_SPARSE", "0.2"))
RRF_K = int(os.getenv("RRF_K", "50"))

_dense_model = None
_sparse_model = None
_reranker = None
_qdrant_client = None


@dataclass
class SearchResult:
    content: str
    full_content: str
    document_id: str
    article: str
    score: float
    rerank_score: float


def get_full_content(payload: Dict) -> str:
    c_content = payload.get('contextual_content') or ""
    m_content = payload.get('content') or ""
    c_content = c_content.strip()
    m_content = m_content.strip()

    if c_content and m_content:
        if c_content in m_content:
            return m_content
        return f"{c_content}\n\n{m_content}"
    elif c_content:
        return c_content
    return m_content


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


def get_dense_model():
    global _dense_model
    if _dense_model is None:
        from sentence_transformers import SentenceTransformer
        _dense_model = SentenceTransformer(DENSE_MODEL_ID)
    return _dense_model


def get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_ID, threads=1)
    return _sparse_model


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _reranker = CrossEncoder(RERANK_MODEL_ID, device=device)
    return _reranker


def preload_models():
    get_dense_model()
    get_sparse_model()
    get_reranker()
    get_qdrant_client()


def search(query: str, top_k: int = 5, retrieval_limit: int = 50, rerank_limit: int = 30) -> List[SearchResult]:
    from qdrant_client.http import models
    
    client = get_qdrant_client()
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()
    reranker = get_reranker()
    
    dense_vector = dense_model.encode(query).tolist()
    sparse_output = list(sparse_model.embed([query]))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_output.indices.tolist(),
        values=sparse_output.values.tolist()
    )
    
    dense_response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=dense_vector,
        using="dense",
        limit=retrieval_limit,
        with_payload=True
    )
    
    sparse_response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=sparse_vector,
        using="sparse",
        limit=retrieval_limit,
        with_payload=True
    )
    
    fusion_scores = {}
    
    def compute_rrf_score(rank: int, weight: float) -> float:
        return weight * (1 / (RRF_K + rank))
    
    for rank, hit in enumerate(dense_response.points):
        point_id = hit.id
        if point_id not in fusion_scores:
            fusion_scores[point_id] = {"score": 0, "payload": hit.payload}
        fusion_scores[point_id]["score"] += compute_rrf_score(rank + 1, WEIGHT_DENSE)
    
    for rank, hit in enumerate(sparse_response.points):
        point_id = hit.id
        if point_id not in fusion_scores:
            fusion_scores[point_id] = {"score": 0, "payload": hit.payload}
        fusion_scores[point_id]["score"] += compute_rrf_score(rank + 1, WEIGHT_SPARSE)
    
    candidates = sorted(fusion_scores.values(), key=lambda x: x["score"], reverse=True)[:rerank_limit]
    
    rerank_inputs = [[query, get_full_content(item['payload'])] for item in candidates]
    rerank_scores = reranker.predict(rerank_inputs, batch_size=4)
    
    for i, item in enumerate(candidates):
        item["rerank_score"] = float(rerank_scores[i])
    
    final_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    
    results = []
    for item in final_results:
        payload = item['payload']
        results.append(SearchResult(
            content=payload.get('content', ''),
            full_content=get_full_content(payload),
            document_id=payload.get('document_id', ''),
            article=payload.get('article', ''),
            score=item['score'],
            rerank_score=item['rerank_score']
        ))
    
    return results


def search_for_llm(query: str, top_k: int = 5) -> str:
    results = search(query, top_k=top_k)
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"[{i}] Nguồn: {result.document_id}\n"
            f"Điều khoản: {result.article}\n"
            f"{result.full_content}"
        )
    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    query = "Mức phạt tiền khi không đội mũ bảo hiểm"
    print("Loading models...")
    preload_models()
    print(f"Searching: {query}")
    context = search_for_llm(query, top_k=5)
    print("CONTEXT FOR LLM:\n")
    print(context)
