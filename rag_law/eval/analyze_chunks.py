# -*- coding: utf-8 -*-
"""
Analyze RAG chunks quality and statistics
"""

import json
import os
from collections import Counter
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

def analyze_chunks():
    chunks_file = os.path.join(DATA_DIR, 'rag_chunks_md.jsonl')
    
    print("=" * 60)
    print("RAG CHUNKS ANALYSIS")
    print("=" * 60)
    
    # Load chunks
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"\nTotal chunks: {len(chunks):,}")
    
    # Analyze by document
    print("\n" + "-" * 60)
    print("CHUNKS BY DOCUMENT")
    print("-" * 60)
    doc_counts = Counter(c['document_id'] for c in chunks)
    for doc_id, count in sorted(doc_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {doc_id}: {count} chunks")
    print(f"  ... and {len(doc_counts) - 15} more documents")
    
    # Analyze chunk lengths
    print("\n" + "-" * 60)
    print("CHUNK LENGTH DISTRIBUTION")
    print("-" * 60)
    lengths = [len(c['content']) for c in chunks]
    
    print(f"  Min length: {min(lengths):,} chars")
    print(f"  Max length: {max(lengths):,} chars")
    print(f"  Avg length: {sum(lengths)//len(lengths):,} chars")
    
    # Length buckets
    buckets = {
        "< 100": 0,
        "100-500": 0,
        "500-1000": 0,
        "1000-2000": 0,
        "2000-4000": 0,
        "> 4000": 0
    }
    for l in lengths:
        if l < 100: buckets["< 100"] += 1
        elif l < 500: buckets["100-500"] += 1
        elif l < 1000: buckets["500-1000"] += 1
        elif l < 2000: buckets["1000-2000"] += 1
        elif l < 4000: buckets["2000-4000"] += 1
        else: buckets["> 4000"] += 1
    
    print("\n  Length distribution:")
    for bucket, count in buckets.items():
        pct = count * 100 / len(chunks)
        bar = "█" * int(pct / 2)
        print(f"    {bucket:>10}: {count:>4} ({pct:5.1f}%) {bar}")
    
    # Sample chunks
    print("\n" + "-" * 60)
    print("SAMPLE CHUNKS")
    print("-" * 60)
    
    # Sample from different documents
    sample_docs = ['Luat_Duong_Bo_2024', 'QCVN_41_2024_BGTVT_Bao_hieu_duong_bo', 'ND_168_2024_Xu_phat_vi_pham_hanh_chinh_ATGT']
    
    for doc_id in sample_docs:
        doc_chunks = [c for c in chunks if c['document_id'] == doc_id]
        if doc_chunks:
            sample = doc_chunks[len(doc_chunks)//2]  # Middle chunk
            print(f"\n[{doc_id}]")
            print(f"  ID: {sample['chunk_id']}")
            print(f"  Context: {sample['context'][:150]}...")
            print(f"  Content: {sample['content'][:200]}...")
    
    # Check for potential issues
    print("\n" + "-" * 60)
    print("QUALITY CHECKS")
    print("-" * 60)
    
    empty_content = [c for c in chunks if len(c['content'].strip()) < 10]
    print(f"  Empty/very short content (< 10 chars): {len(empty_content)}")
    
    missing_context = [c for c in chunks if not c.get('context')]
    print(f"  Missing context: {len(missing_context)}")
    
    unknown_titles = [c for c in chunks if 'Unknown' in c.get('document_title', '')]
    print(f"  Unknown titles: {len(unknown_titles)}")
    
    oversized = [c for c in chunks if len(c['content']) > 4000]
    print(f"  Oversized chunks (> 4000 chars): {len(oversized)}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    analyze_chunks()
