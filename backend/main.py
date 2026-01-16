# -*- coding: utf-8 -*-
"""
FastAPI Backend for Traffic Law Chatbot.
Connects Next.js frontend with LLM and RAG modules.
"""

# Import unsloth first ONLY if using local VLM (for optimizations)
import os
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
if LLM_PROVIDER == "local":
    import unsloth  # noqa: F401

import sys
import base64
import tempfile
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Configuration
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
USE_VLM = os.getenv("USE_VLM", "true").lower() == "true"

# Global state
_rag_loaded = False
_vlm_loaded = False


# --- Pydantic Models ---

class ChatMessage(BaseModel):
    id: str
    role: str
    text: str
    attachments: Optional[List[str]] = None
    createdAt: int


class ChatRequest(BaseModel):
    text: str
    attachments: Optional[List[str]] = None  # base64 encoded images
    history: Optional[List[ChatMessage]] = None
    provider: Optional[str] = "gemini"  # "gemini" or "local"


class ChatResponse(BaseModel):
    text: str
    query: Optional[str] = None  # The summarized RAG query
    sources: Optional[List[Dict[str, Any]]] = None


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    context: str


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models on startup."""
    global _rag_loaded, _vlm_loaded
    
    log.info("Starting backend server...")
    
    # Load RAG models
    try:
        from rag_law.rag.search import preload_models as preload_rag
        log.info("Loading RAG models...")
        preload_rag()
        _rag_loaded = True
        log.info("RAG models loaded successfully")
    except Exception as e:
        log.warning(f"Failed to load RAG models: {e}")
        _rag_loaded = False
    
    # Load LLM (Gemini or Local VLM)
    if USE_VLM:
        try:
            from llm.llm_rag import preload_models as preload_llm
            log.info("Loading LLM models...")
            preload_llm()
            _vlm_loaded = True
            log.info("LLM models loaded successfully")
        except Exception as e:
            log.warning(f"Failed to load LLM models: {e}")
            _vlm_loaded = False
    
    yield
    
    log.info("Shutting down backend server...")


# --- FastAPI App ---

app = FastAPI(
    title="Traffic Law Chatbot API",
    description="API for Vietnamese Traffic Law RAG + VLM Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions ---

def decode_base64_image(base64_str: str) -> str:
    """Decode base64 image and save to temp file. Returns file path."""
    # Handle data URL format
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    
    image_data = base64.b64decode(base64_str)
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(image_data)
        return f.name


def cleanup_temp_file(path: str):
    """Clean up temporary file."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception:
        pass


# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_loaded": _rag_loaded,
        "vlm_loaded": _vlm_loaded,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Flow:
    1. If image attached + VLM available: VLM summarizes → RAG search → VLM answers
    2. If text only or no VLM: Direct RAG search → Format response
    """
    if not _rag_loaded:
        raise HTTPException(status_code=503, detail="RAG models not loaded")
    
    text = request.text.strip()
    attachments = request.attachments or []
    
    temp_image_path = None
    
    try:
        # Decode first image if present
        if attachments:
            try:
                temp_image_path = decode_base64_image(attachments[0])
                log.info(f"Decoded image to: {temp_image_path}")
            except Exception as e:
                log.warning(f"Failed to decode image: {e}")
        
        # --- LLM + RAG Pipeline ---
        provider = request.provider or "gemini"
        log.info(f"Using LLM provider: {provider}")
        
        if _rag_loaded and (temp_image_path or text):
            from llm.llm_rag import full_pipeline
            
            # Convert history to simple dict format for LLM
            history_for_llm = None
            if history:
                history_for_llm = [
                    {"role": msg.role, "text": msg.text}
                    for msg in history[-3:]  # Only last 6 messages
                ]
            
            result = full_pipeline(
                image_path=temp_image_path,
                user_text=text,
                top_k=5,
                provider=provider,
                chat_history=history_for_llm,
            )
            
            if result.get("success"):
                return ChatResponse(
                    text=result.get("answer", "Không thể tạo câu trả lời."),
                    query=result.get("query"),
                    sources=[
                        {"document_id": r["document_id"], "article": r["article"]}
                        for r in result.get("rag_results", [])
                    ],
                )
        
        # --- RAG Only Fallback ---
        from rag_law.rag.search import search_for_llm, search
        
        query = text
        results = search(query, top_k=5)
        context = search_for_llm(query, top_k=5)
        
        # Format response
        if results:
            response_text = f"**Kết quả tìm kiếm cho:** {query}\n\n{context}"
        else:
            response_text = "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu luật giao thông."
        
        return ChatResponse(
            text=response_text,
            query=query,
            sources=[
                {"document_id": r.document_id, "article": r.article, "score": r.rerank_score}
                for r in results
            ],
        )
    
    except Exception as e:
        log.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_image_path:
            cleanup_temp_file(temp_image_path)


@app.post("/search", response_model=SearchResponse)
async def search_rag(request: SearchRequest):
    """
    Direct RAG search endpoint.
    Returns raw search results without VLM processing.
    """
    if not _rag_loaded:
        raise HTTPException(status_code=503, detail="RAG models not loaded")
    
    try:
        from rag_law.rag.search import search, search_for_llm
        
        results = search(request.query, top_k=request.top_k)
        context = search_for_llm(request.query, top_k=request.top_k)
        
        return SearchResponse(
            query=request.query,
            results=[
                {
                    "content": r.content,
                    "full_content": r.full_content,
                    "document_id": r.document_id,
                    "article": r.article,
                    "score": r.rerank_score,
                }
                for r in results
            ],
            context=context,
        )
    
    except Exception as e:
        log.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Main ---

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=True,
    )
