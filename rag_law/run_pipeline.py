# -*- coding: utf-8 -*-
"""
Full RAG Pipeline - Run all processing steps from start to finish.

Steps:
1. Clean documents (documents_vn -> documents_clean)
2. Convert to markdown (documents_doc -> documents_markdown) 
3. Chunk markdown files (documents_markdown -> structured_law/rag_chunks_md.jsonl)
4. Upload to Qdrant

Usage:
    python run_pipeline.py              # Run all steps
    python run_pipeline.py --step 3     # Run from step 3
    python run_pipeline.py --only 2     # Run only step 2
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAG_LAW = os.path.join(PROJECT_ROOT, "rag_law")


def run_step(name: str, module: str, args: list = None):
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {name}")
        return False
    
    print(f"[OK] {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run RAG processing pipeline")
    parser.add_argument("--step", type=int, help="Start from step N")
    parser.add_argument("--only", type=int, help="Run only step N")
    parser.add_argument("--skip-upload", action="store_true", help="Skip Qdrant upload")
    args = parser.parse_args()
    
    steps = [
        ("1. Clean documents", "rag_law.processing.clean_documents", [
            "-i", os.path.join(PROJECT_ROOT, "documents_vn"),
            "-o", os.path.join(PROJECT_ROOT, "documents_clean")
        ]),
        ("2. Convert DOC to Markdown", "rag_law.processing.doc_to_markdown", [
            "-i", os.path.join(PROJECT_ROOT, "documents_doc"),
            "-o", os.path.join(PROJECT_ROOT, "documents_markdown")
        ]),
        ("3. Chunk Markdown", "rag_law.processing.chunk_markdown", [
            "-i", os.path.join(PROJECT_ROOT, "documents_markdown"),
            "-o", os.path.join(RAG_LAW, "structured_law", "rag_chunks_md.jsonl")
        ]),
        ("4. Upload to Qdrant", "rag_law.rag.qdrant_rag", ["setup"]),
    ]
    
    if args.skip_upload:
        steps = steps[:-1]
    
    print("="*60)
    print("RAG PIPELINE")
    print("="*60)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Steps: {len(steps)}")
    
    start = (args.step or 1) - 1
    
    for i, (name, module, step_args) in enumerate(steps):
        if args.only and i + 1 != args.only:
            continue
        if i < start:
            print(f"[SKIP] {name}")
            continue
        
        if not run_step(name, module, step_args):
            print("\nPipeline stopped due to error.")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
