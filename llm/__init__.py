# -*- coding: utf-8 -*-
"""
LLM module for Gemini API and Local VLM integration with RAG.
"""

from .llm_rag import (
    # Core functions
    generate,
    generate_answer,
    summarize_for_rag,
    full_pipeline,
    
    # Provider-specific
    generate_with_gemini,
    generate_with_local_vlm,
    get_gemini_client,
    get_local_model,
    
    # Utilities
    load_prompts,
    preload_models,
    
    # Types
    LLMResponse,
)

__all__ = [
    "generate",
    "generate_answer",
    "summarize_for_rag",
    "full_pipeline",
    "generate_with_gemini",
    "generate_with_local_vlm",
    "get_gemini_client",
    "get_local_model",
    "load_prompts",
    "preload_models",
    "LLMResponse",
]
