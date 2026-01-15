# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B Vision-Language Model for RAG Integration.
Summarizes user input (image + text) for traffic law search.
"""
from unsloth import FastVisionModel
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Configuration
MODEL_ID = os.getenv("VLM_MODEL_ID", "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
LORA_PATH = os.getenv("VLM_LORA_PATH", None)  # Optional: path to fine-tuned LoRA
LOAD_IN_4BIT = os.getenv("VLM_LOAD_4BIT", "true").lower() == "true"
PROMPTS_FILE = Path(__file__).parent / "prompts.json"

# Global model cache
_model = None
_tokenizer = None
_prompts = None


@dataclass
class VLMResponse:
    """Response from VLM inference."""
    query: str  # Summarized query for RAG
    raw_output: str  # Full model output
    success: bool
    error: Optional[str] = None


def load_prompts() -> Dict[str, Any]:
    """Load prompt templates from JSON file."""
    global _prompts
    if _prompts is None:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            _prompts = json.load(f)
    return _prompts


def get_model():
    """Load and cache the VLM model."""
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    log.info(f"Loading model: {MODEL_ID}")
    
    if LORA_PATH and os.path.exists(LORA_PATH):
        log.info(f"Loading LoRA adapter from: {LORA_PATH}")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=LORA_PATH,
            load_in_4bit=LOAD_IN_4BIT,
        )
    else:
        model, tokenizer = FastVisionModel.from_pretrained(
            MODEL_ID,
            load_in_4bit=LOAD_IN_4BIT,
            use_gradient_checkpointing="unsloth",
        )
    
    FastVisionModel.for_inference(model)
    
    _model = model
    _tokenizer = tokenizer
    
    log.info("Model loaded successfully")
    return model, tokenizer


def summarize_for_rag(
    image_path: Optional[str] = None,
    user_text: str = "",
    image_url: Optional[str] = None,
) -> VLMResponse:
    """
    Analyze image and text, generate a summarized query for RAG search.
    
    Args:
        image_path: Local path to image file
        user_text: User's text input/question
        image_url: URL to image (alternative to image_path)
    
    Returns:
        VLMResponse with summarized query for RAG
    """
    try:
        model, tokenizer = get_model()
        prompts = load_prompts()
        
        # Build instruction
        instruction = prompts["summarize_for_rag"]["instruction"]
        if user_text:
            instruction = f"{instruction}\n\nCâu hỏi của người dùng: {user_text}"
        
        # Build message content
        content = []
        
        # Add image if provided
        if image_path or image_url:
            content.append({"type": "image"})
        
        content.append({"type": "text", "text": instruction})
        
        messages = [{"role": "user", "content": content}]
        
        # Prepare inputs
        input_text = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True
        )
        
        # Handle image input
        image_input = image_path or image_url
        if image_input:
            inputs = tokenizer(
                image_input,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            inputs = tokenizer(
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate
        gen_config = prompts.get("generation_config", {})
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config.get("max_new_tokens", 256),
                temperature=gen_config.get("temperature", 0.7),
                min_p=gen_config.get("min_p", 0.1),
                do_sample=gen_config.get("do_sample", True),
                use_cache=True,
            )
        
        # Decode output
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        query = raw_output[len(prompt_text):].strip()
        
        # Clean up query - remove any thinking tags if present
        if "<think>" in query and "</think>" in query:
            think_end = query.find("</think>") + len("</think>")
            query = query[think_end:].strip()
        
        return VLMResponse(
            query=query,
            raw_output=raw_output,
            success=True
        )
        
    except Exception as e:
        log.error(f"VLM inference failed: {e}")
        return VLMResponse(
            query="",
            raw_output="",
            success=False,
            error=str(e)
        )


def process_user_input(
    image_path: Optional[str] = None,
    user_text: str = "",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Full pipeline: Summarize input with VLM, then search RAG.
    
    Args:
        image_path: Path to user's image
        user_text: User's text question
        top_k: Number of RAG results to return
    
    Returns:
        Dict with query, rag_results, and context
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "rag_law"))
    
    from rag_law.rag.search import search_for_llm, search
    
    # Step 1: Summarize with VLM
    vlm_response = summarize_for_rag(
        image_path=image_path,
        user_text=user_text
    )
    
    if not vlm_response.success:
        return {
            "success": False,
            "error": vlm_response.error,
            "query": None,
            "rag_results": [],
            "context": ""
        }
    
    query = vlm_response.query
    log.info(f"Generated RAG query: {query}")
    
    # Step 2: Search RAG
    results = search(query, top_k=top_k)
    context = search_for_llm(query, top_k=top_k)
    
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
        "context": context
    }


def generate_answer(
    question: str,
    context: str,
    image_path: Optional[str] = None,
) -> str:
    """
    Generate final answer using VLM with RAG context.
    
    Args:
        question: User's original question
        context: Retrieved context from RAG
        image_path: Optional image for reference
    
    Returns:
        Generated answer string
    """
    try:
        model, tokenizer = get_model()
        prompts = load_prompts()
        
        # Build prompt with context using answer_theory_logic instruction
        answer_instruction = prompts["answer_theory_logic"]["instruction"]
        full_prompt = f"""{answer_instruction}

### THÔNG TIN PHÁP LÝ LIÊN QUAN:
{context}

### CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

Hãy trả lời câu hỏi dựa trên thông tin pháp lý ở trên. Trích dẫn Điều, Khoản cụ thể khi cần thiết."""
        
        content = []
        if image_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": full_prompt})
        
        messages = [
            {"role": "system", "content": prompts["system_prompt"]},
            {"role": "user", "content": content}
        ]
        
        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        if image_path:
            inputs = tokenizer(
                image_path,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            inputs = tokenizer(
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        gen_config = prompts.get("generation_config", {})
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config.get("max_new_tokens", 512),
                temperature=gen_config.get("temperature", 0.7),
                min_p=gen_config.get("min_p", 0.1),
                do_sample=gen_config.get("do_sample", True),
                use_cache=True,
            )
        
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        answer = raw_output[len(prompt_text):].strip()
        
        # Clean up thinking tags
        if "<think>" in answer and "</think>" in answer:
            think_end = answer.find("</think>") + len("</think>")
            answer = answer[think_end:].strip()
        
        return answer
        
    except Exception as e:
        log.error(f"Answer generation failed: {e}")
        return f"Lỗi: {e}"


def full_pipeline(
    image_path: Optional[str] = None,
    user_text: str = "",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Complete end-to-end pipeline.
    
    1. Summarize user input (image + text) with VLM
    2. Search RAG with summarized query
    3. Generate final answer with VLM + RAG context
    
    Args:
        image_path: Path to user's image
        user_text: User's text question  
        top_k: Number of RAG results
    
    Returns:
        Dict with query, context, and answer
    """
    # Step 1 & 2: Get RAG results
    rag_result = process_user_input(
        image_path=image_path,
        user_text=user_text,
        top_k=top_k
    )
    
    if not rag_result["success"]:
        return rag_result
    
    # Step 3: Generate answer
    answer = generate_answer(
        question=user_text or rag_result["query"],
        context=rag_result["context"],
        image_path=image_path
    )
    
    return {
        **rag_result,
        "answer": answer
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLM + RAG for Traffic Law")
    parser.add_argument("--image", "-i", type=str, help="Path to image file")
    parser.add_argument("--text", "-t", type=str, default="", help="User question")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of RAG results")
    parser.add_argument("--mode", "-m", choices=["summarize", "search", "full"], 
                       default="full", help="Pipeline mode")
    
    args = parser.parse_args()
    
    if not args.image and not args.text:
        parser.error("At least --image or --text must be provided")
    
    print("Loading models...")
    
    if args.mode == "summarize":
        result = summarize_for_rag(image_path=args.image, user_text=args.text)
        print(f"\nGenerated Query: {result.query}")
        
    elif args.mode == "search":
        result = process_user_input(
            image_path=args.image,
            user_text=args.text,
            top_k=args.top_k
        )
        print(f"\nQuery: {result['query']}")
        print(f"\nRAG Context:\n{result['context']}")
        
    else:  # full
        result = full_pipeline(
            image_path=args.image,
            user_text=args.text,
            top_k=args.top_k
        )
        print(f"\nQuery: {result['query']}")
        print(f"\nAnswer:\n{result['answer']}")