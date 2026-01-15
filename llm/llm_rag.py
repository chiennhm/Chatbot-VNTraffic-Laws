# -*- coding: utf-8 -*-
"""
LLM Integration for Vietnamese Traffic Law RAG.
Supports: Google Gemini API (default) and Local VLM (Qwen3-VL-8B).
"""

import os
import json
import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "local"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
VLM_LORA_PATH = os.getenv("VLM_LORA_PATH", None)
VLM_LOAD_4BIT = os.getenv("VLM_LOAD_4BIT", "true").lower() == "true"
PROMPTS_FILE = Path(__file__).parent / "prompts.json"

# Global cache
_gemini_client = None
_local_model = None
_local_tokenizer = None
_prompts = None


@dataclass
class LLMResponse:
    """Response from LLM inference."""
    text: str
    success: bool
    error: Optional[str] = None


def load_prompts() -> Dict[str, Any]:
    """Load prompt templates from JSON file."""
    global _prompts
    if _prompts is None:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            _prompts = json.load(f)
    return _prompts


# =============================================================================
# Gemini API
# =============================================================================

def get_gemini_client():
    """Get or create Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        log.info(f"Gemini client initialized with model: {GEMINI_MODEL}")
    return _gemini_client


def _image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_image_mime_type(image_path: str) -> str:
    """Get MIME type from image file extension."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


def generate_with_gemini(
    prompt: str,
    system_prompt: Optional[str] = None,
    image_path: Optional[str] = None,
) -> LLMResponse:
    """
    Generate response using Gemini API.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system instruction
        image_path: Optional path to image file
    
    Returns:
        LLMResponse with generated text
    """
    try:
        from google.genai import types
        
        client = get_gemini_client()
        prompts = load_prompts()
        gen_config = prompts.get("generation_config", {})
        
        # Build content parts (similar to local VLM)
        content_parts = []
        
        # Add image if provided
        if image_path and os.path.exists(image_path):
            image_data = _image_to_base64(image_path)
            mime_type = _get_image_mime_type(image_path)
            content_parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(image_data),
                    mime_type=mime_type
                )
            )
        
        # Add text prompt
        content_parts.append(types.Part.from_text(prompt))
        
        # Build user message
        user_message = types.Content(
            role="user",
            parts=content_parts
        )
        
        # Generate config
        config = types.GenerateContentConfig(
            temperature=gen_config.get("temperature", 0.1),
            top_p=gen_config.get("top_p", 0.9),
            max_output_tokens=gen_config.get("max_new_tokens", 1500),
        )
        
        # Add system instruction if provided
        if system_prompt:
            config.system_instruction = system_prompt
        
        # Generate response
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[user_message],
            config=config,
        )
        
        answer = response.text
        
        # Clean up thinking tags (same as local VLM)
        if "<think>" in answer and "</think>" in answer:
            think_end = answer.find("</think>") + len("</think>")
            answer = answer[think_end:].strip()
        
        return LLMResponse(
            text=answer,
            success=True
        )
        
    except Exception as e:
        log.error(f"Gemini generation failed: {e}")
        return LLMResponse(
            text="",
            success=False,
            error=str(e)
        )


# =============================================================================
# Local VLM (Qwen3-VL-8B with Unsloth)
# =============================================================================

def get_local_model():
    """Load and cache the local VLM model."""
    global _local_model, _local_tokenizer
    
    if _local_model is not None:
        return _local_model, _local_tokenizer
    
    from unsloth import FastVisionModel
    import torch
    
    log.info(f"Loading local model: {VLM_MODEL_ID}")
    
    if VLM_LORA_PATH and os.path.exists(VLM_LORA_PATH):
        log.info(f"Loading LoRA adapter from: {VLM_LORA_PATH}")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=VLM_LORA_PATH,
            load_in_4bit=VLM_LOAD_4BIT,
        )
    else:
        model, tokenizer = FastVisionModel.from_pretrained(
            VLM_MODEL_ID,
            load_in_4bit=VLM_LOAD_4BIT,
            use_gradient_checkpointing="unsloth",
        )
    
    FastVisionModel.for_inference(model)
    
    _local_model = model
    _local_tokenizer = tokenizer
    
    log.info("Local model loaded successfully")
    return model, tokenizer


def generate_with_local_vlm(
    prompt: str,
    system_prompt: Optional[str] = None,
    image_path: Optional[str] = None,
) -> LLMResponse:
    """
    Generate response using local VLM.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system instruction
        image_path: Optional path to image file
    
    Returns:
        LLMResponse with generated text
    """
    try:
        import torch
        
        model, tokenizer = get_local_model()
        prompts = load_prompts()
        
        # Build message content
        content = []
        if image_path and os.path.exists(image_path):
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        
        # Prepare inputs
        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if image_path and os.path.exists(image_path):
            inputs = tokenizer(
                image_path,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(device)
        else:
            inputs = tokenizer(
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(device)
        
        # Generate
        gen_config = prompts.get("generation_config", {})
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config.get("max_new_tokens", 1500),
                temperature=gen_config.get("temperature", 0.1),
                top_p=gen_config.get("top_p", 0.9),
                do_sample=gen_config.get("do_sample", True),
                repetition_penalty=gen_config.get("repetition_penalty", 1.2),
                no_repeat_ngram_size=gen_config.get("no_repeat_ngram_size", 4),
                use_cache=True,
            )
        
        # Decode output
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        answer = raw_output[len(prompt_text):].strip()
        
        # Clean up thinking tags
        if "<think>" in answer and "</think>" in answer:
            think_end = answer.find("</think>") + len("</think>")
            answer = answer[think_end:].strip()
        
        return LLMResponse(
            text=answer,
            success=True
        )
        
    except Exception as e:
        log.error(f"Local VLM generation failed: {e}")
        return LLMResponse(
            text="",
            success=False,
            error=str(e)
        )


def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    image_path: Optional[str] = None,
    provider: Optional[str] = None,
) -> LLMResponse:
    """
    Generate response using configured LLM provider.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system instruction
        image_path: Optional path to image file
        provider: Override LLM_PROVIDER ("gemini" or "local")
    
    Returns:
        LLMResponse with generated text
    """
    selected_provider = provider or LLM_PROVIDER
    log.info(f"[LLM] Generating with provider: {selected_provider}")
    log.info(f"[LLM] Prompt length: {len(prompt)} chars")
    
    if selected_provider == "gemini":
        log.info("[LLM] Calling Gemini API...")
        response = generate_with_gemini(prompt, system_prompt, image_path)
    elif selected_provider == "local":
        log.info("[LLM] Calling Local VLM...")
        response = generate_with_local_vlm(prompt, system_prompt, image_path)
    else:
        response = LLMResponse(
            text="",
            success=False,
            error=f"Unknown provider: {selected_provider}"
        )
    
    if response.success:
        log.info(f"[LLM] Response received: {len(response.text)} chars")
    else:
        log.error(f"[LLM] Generation failed: {response.error}")
    
    return response


def generate_answer(
    question: str,
    context: str,
    image_path: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """
    Generate answer for user question with RAG context.
    
    Args:
        question: User's question
        context: Retrieved context from RAG
        image_path: Optional image for reference
        provider: LLM provider ("gemini" or "local")
    
    Returns:
        Generated answer string
    """
    prompts = load_prompts()
    system_prompt = prompts.get("system_prompt", "")
    
    # Build prompt with context
    full_prompt = f"""### THÔNG TIN PHÁP LÝ LIÊN QUAN:
{context}

### CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

Hãy trả lời câu hỏi dựa trên thông tin pháp lý ở trên. Trích dẫn Điều, Khoản cụ thể."""

    response = generate(
        prompt=full_prompt,
        system_prompt=system_prompt,
        image_path=image_path,
        provider=provider,
    )
    
    if response.success:
        return response.text
    else:
        return f"Lỗi: {response.error}"


def summarize_for_rag(
    user_text: str = "",
    image_path: Optional[str] = None,
    provider: Optional[str] = None,
) -> LLMResponse:
    """
    Summarize user input into a search query for RAG.
    
    Args:
        user_text: User's text input
        image_path: Optional path to image file
        provider: LLM provider ("gemini" or "local")
    
    Returns:
        LLMResponse with summarized query
    """
    prompts = load_prompts()
    instruction = prompts["summarize_for_rag"]["instruction"]
    
    if user_text:
        prompt = f"{instruction}\n\nCâu hỏi của người dùng: {user_text}"
    else:
        prompt = instruction
    
    return generate(
        prompt=prompt,
        image_path=image_path,
        provider=provider,
    )


def full_pipeline(
    user_text: str = "",
    image_path: Optional[str] = None,
    top_k: int = 5,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete end-to-end pipeline.
    
    1. Summarize user input into query (if image provided)
    2. Search RAG with query
    3. Generate final answer with context
    
    Args:
        user_text: User's text question
        image_path: Optional path to image file
        top_k: Number of RAG results
        provider: LLM provider ("gemini" or "local")
    
    Returns:
        Dict with query, context, rag_results, and answer
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "rag_law"))
    
    from rag_law.rag.search import search_for_llm, search
    
    # Step 1: Determine query
    if image_path:
        # Use VLM to summarize image + text
        summary = summarize_for_rag(user_text=user_text, image_path=image_path, provider=provider)
        if not summary.success:
            return {
                "success": False,
                "error": summary.error,
                "query": None,
                "rag_results": [],
                "context": "",
                "answer": ""
            }
        query = summary.text
    else:
        # Text-only: use directly
        query = user_text
    
    log.info(f"RAG Query: {query[:100]}...")
    
    # Step 2: Search RAG
    results = search(query, top_k=top_k)
    context = search_for_llm(query, top_k=top_k)
    
    # Step 3: Generate answer
    answer = generate_answer(
        question=user_text or query,
        context=context,
        image_path=image_path,
        provider=provider,
    )
    
    return {
        "success": True,
        "query": query,
        "rag_results": [
            {
                "content": r.content,
                "full_content": r.full_content,
                "document_id": r.document_id,
                "article": r.article,
                "score": r.rerank_score
            }
            for r in results
        ],
        "context": context,
        "answer": answer
    }


def preload_models():
    """Preload LLM models based on provider."""
    if LLM_PROVIDER == "gemini":
        get_gemini_client()
    elif LLM_PROVIDER == "local":
        get_local_model()
    log.info(f"LLM provider '{LLM_PROVIDER}' initialized")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM + RAG for Traffic Law")
    parser.add_argument("--text", "-t", type=str, required=True, help="User question")
    parser.add_argument("--image", "-i", type=str, help="Path to image file")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of RAG results")
    parser.add_argument("--provider", "-p", choices=["gemini", "local"], help="LLM provider")
    
    args = parser.parse_args()
    
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    
    print(f"Using provider: {LLM_PROVIDER}")
    print("Running pipeline...")
    
    result = full_pipeline(
        user_text=args.text,
        image_path=args.image,
        top_k=args.top_k
    )
    
    print(f"\nQuery: {result.get('query', 'N/A')}")
    print(f"\nAnswer:\n{result.get('answer', 'N/A')}")