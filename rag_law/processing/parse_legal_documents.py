# -*- coding: utf-8 -*-
"""
Script to parse Vietnamese legal documents into structured data for RAG training.
Parses: Chương (Chapter) → Mục (Section) → Điều (Article) → Khoản (Clause) → Điểm (Point)
Output formats: JSON (hierarchical), JSONL (flat chunks), CSV

VERSION 3.0 - CONTEXTUAL CHUNKING:
- Removed URL, crawled_date fields
- Each chunk now contains COMPLETE Điều or Khoản content
- Added CONTEXTUAL PREFIX to preserve context hierarchy
"""

import os
import re
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Point:
    point_id: str
    point_letter: str
    content: str

@dataclass
class Clause:
    clause_id: str
    clause_number: int
    content: str
    raw_content: str  # Full raw content including points
    points: List[Point] = field(default_factory=list)

@dataclass
class Article:
    article_id: str
    article_number: int
    article_title: str
    full_content: str
    raw_content: str  # Full raw content including all clauses
    clauses: List[Clause] = field(default_factory=list)

@dataclass
class Section:
    section_id: str
    section_number: int
    section_title: str
    articles: List[Article] = field(default_factory=list)

@dataclass
class Chapter:
    chapter_id: str
    chapter_number: str
    chapter_title: str
    sections: List[Section] = field(default_factory=list)
    articles: List[Article] = field(default_factory=list)

@dataclass
class LegalDocument:
    document_id: str
    document_type: str
    document_number: str
    title: str
    chapters: List[Chapter] = field(default_factory=list)
    articles: List[Article] = field(default_factory=list)

@dataclass
class RAGChunk:
    """RAG chunk with CONTEXTUAL CHUNKING - includes context prefix for better retrieval."""
    chunk_id: str
    document_id: str
    document_type: str
    document_title: str
    document_number: str
    chapter: str
    article: str
    article_number: int
    clause_number: Optional[int]
    content: str  # Raw content of Điều or Khoản
    context: str  # Contextual prefix describing hierarchy
    contextual_content: str  # Full content WITH context prefix (use this for RAG)
    char_count: int

# ============================================================================
# PATTERNS
# ============================================================================

ROMAN_MAP = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
    'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20
}

def roman_to_int(roman: str) -> int:
    return ROMAN_MAP.get(roman.upper(), 0)

# ============================================================================
# DOCUMENT TYPE DETECTION
# ============================================================================

def detect_document_type(filename: str, content: str) -> Tuple[str, str]:
    """Detect document type and number from filename and content."""
    filename_lower = filename.lower()
    
    if filename_lower.startswith('luat_'):
        doc_type = "Luật"
        match = re.search(r'Luật\s+số[:\s]*(\d+/\d+/QH\d+)', content)
        if not match:
            match = re.search(r'(\d+/\d+/QH\d+)', content)
        doc_number = match.group(1) if match else ""
    elif filename_lower.startswith('nd_'):
        doc_type = "Nghị định"
        match = re.search(r'(\d+/\d+/NĐ-CP)', filename)
        if not match:
            match = re.search(r'ND_(\d+)_(\d+)', filename)
            doc_number = f"{match.group(1)}/{match.group(2)}/NĐ-CP" if match else ""
        else:
            doc_number = match.group(1)
    elif filename_lower.startswith('tt_'):
        doc_type = "Thông tư"
        match = re.search(r'TT_(\d+)_(\d+)_([A-Z]+)', filename)
        if match:
            doc_number = f"{match.group(1)}/{match.group(2)}/TT-{match.group(3)}"
        else:
            doc_number = ""
    else:
        doc_type = "Văn bản"
        doc_number = ""
    
    return doc_type, doc_number

def extract_title(content: str, filename: str = "") -> str:
    """Extract document title from content or filename."""
    # Try to find title from first line starting with #
    match = re.search(r'^#\s*(.+?)(?:\r?\n|$)', content, re.MULTILINE)
    if match:
        title = match.group(1).strip()
        # Clean up - remove metadata like "CHÍNH PHỦ CỘNG HÒA..."
        title = re.split(r'\s+(?:CHÍNH PHỦ|BỘ CÔNG AN|BỘ GIAO THÔNG|BỘ QUỐC PHÒNG|BỘ Y TẾ|CỘNG HÒA)', title)[0].strip()
        if title:
            return title
    
    # Fallback: generate title from filename
    if filename:
        # ND_119_2024_Thanh_toan_dien_tu_giao_thong.txt -> Nghị định 119/2024
        name = filename.replace('.txt', '').replace('_', ' ')
        return name
    
    return "Unknown Title"

# ============================================================================
# IMPROVED PARSING FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Clean up text by normalizing whitespace but preserve structure."""
    # Normalize multiple spaces to single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize multiple newlines to single newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def normalize_for_chunk(text: str) -> str:
    """Normalize text for chunk content - single line."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_chapters(content: str) -> List[Tuple[str, str, str]]:
    """Split content into chapters with COMPLETE content."""
    chapters = []
    
    # Pattern for chapter headers - more robust
    chapter_pattern = re.compile(
        r'(?:^|\n)(Chương\s+([IVXLCDM]+|\d+)[:\.\s]*([^\n]+?))\s*(?=\n)',
        re.IGNORECASE | re.UNICODE | re.MULTILINE
    )
    
    matches = list(chapter_pattern.finditer(content))
    
    if not matches:
        return [("0", "Nội dung chính", content)]
    
    for i, match in enumerate(matches):
        chapter_num = match.group(2).strip()
        chapter_title = clean_text(match.group(3))
        
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        chapter_content = content[start_pos:end_pos]
        
        chapters.append((chapter_num, chapter_title, chapter_content))
    
    return chapters

def split_into_articles(content: str) -> List[Tuple[int, str, str]]:
    """Split content into articles with COMPLETE raw content."""
    articles = []
    
    # Pattern for article headers - capture article number and remaining text
    article_pattern = re.compile(
        r'(?:^|\n)(Điều\s+(\d+)\.?\s*(.+?)(?=\s+\d+\.\s|\s*$))',
        re.UNICODE | re.MULTILINE | re.DOTALL
    )
    
    # Simpler pattern to find article starts
    article_start_pattern = re.compile(
        r'(?:^|\n)(Điều\s+(\d+)\.?\s*)',
        re.UNICODE | re.MULTILINE
    )
    
    matches = list(article_start_pattern.finditer(content))
    
    if not matches:
        return []
    
    for i, match in enumerate(matches):
        article_num = int(match.group(2))
        
        # Get COMPLETE article content from this match to next article
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        article_raw = content[start_pos:end_pos].strip()
        
        # Extract article title - text after "Điều X." before first clause "1." or limited length
        title_match = re.match(r'Điều\s+\d+\.?\s*([^0-9].+?)(?=\s+1\.\s|\s*$)', article_raw)
        if title_match:
            article_title = title_match.group(1).strip()
            # Truncate if too long (title shouldn't be more than 100 chars typically)
            if len(article_title) > 100:
                # Find a good break point
                cut = article_title[:100].rfind(' ')
                if cut > 50:
                    article_title = article_title[:cut] + "..."
                else:
                    article_title = article_title[:100] + "..."
        else:
            # Fallback: take first 80 chars after Điều number
            rest = article_raw[match.end() - match.start():]
            article_title = rest[:80].strip()
            if len(rest) > 80:
                article_title += "..."
        
        article_title = clean_text(article_title)
        articles.append((article_num, article_title, article_raw))
    
    return articles


def split_into_clauses(article_content: str) -> List[Tuple[int, str]]:
    """Split article content into clauses - return COMPLETE clause content.
    
    Works with both:
    - Multi-line format (clauses on separate lines)
    - Single-line merged format (all clauses inline with article)
    """
    clauses = []
    
    # First try: Look for clause patterns that work with inline text
    # Match " 1. ", " 2. " etc - number preceded by space and followed by dot+space
    # This pattern works for inline text where clauses are merged
    clause_pattern = re.compile(
        r'(?:^|\s)(\d+)\.\s+',
        re.UNICODE
    )
    
    matches = list(clause_pattern.finditer(article_content))
    
    # Filter to only get sequential clause numbers starting from 1
    # This avoids matching dates like "2024." or other numbers
    valid_matches = []
    expected_num = 1
    for match in matches:
        num = int(match.group(1))
        # Accept if it's the expected number (sequential) or close to it
        if num == expected_num or (num > 0 and num <= expected_num + 2):
            valid_matches.append(match)
            expected_num = num + 1
    
    if not valid_matches:
        return []
    
    for i, match in enumerate(valid_matches):
        clause_num = int(match.group(1))
        
        # Get content from after this clause number to next clause or end
        start_pos = match.end()
        if i + 1 < len(valid_matches):
            end_pos = valid_matches[i + 1].start()
        else:
            end_pos = len(article_content)
        
        clause_content = article_content[start_pos:end_pos].strip()
        
        if clause_content:  # Only add if has content
            clauses.append((clause_num, clause_content))
    
    return clauses

def parse_document(filepath: str) -> Optional[LegalDocument]:
    """Parse a single legal document file with improved completeness."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    filename = os.path.basename(filepath)
    doc_id = os.path.splitext(filename)[0]
    
    # Extract metadata (no URL/crawled_date)
    doc_type, doc_number = detect_document_type(filename, content)
    title = extract_title(content, filename)
    
    document = LegalDocument(
        document_id=doc_id,
        document_type=doc_type,
        document_number=doc_number,
        title=title
    )
    
    # Extract main content
    content_match = re.search(r'=== NỘI DUNG VĂN BẢN ===\s*\r?\n(.+)', content, re.DOTALL)
    if content_match:
        main_content = content_match.group(1)
    else:
        main_content = content
    
    # Clean up content
    main_content = clean_text(main_content)
    
    # Parse chapters
    chapters_data = split_into_chapters(main_content)
    
    for ch_num, ch_title, ch_content in chapters_data:
        chapter = Chapter(
            chapter_id=f"chuong_{ch_num}",
            chapter_number=ch_num,
            chapter_title=ch_title
        )
        
        # Parse articles with COMPLETE content
        articles_data = split_into_articles(ch_content)
        
        for art_num, art_title, art_raw in articles_data:
            article = Article(
                article_id=f"dieu_{art_num}",
                article_number=art_num,
                article_title=art_title,
                full_content=normalize_for_chunk(art_raw),
                raw_content=art_raw
            )
            
            # Parse clauses with COMPLETE content
            clauses_data = split_into_clauses(art_raw)
            
            for cl_num, cl_raw in clauses_data:
                clause = Clause(
                    clause_id=f"khoan_{cl_num}",
                    clause_number=cl_num,
                    content=normalize_for_chunk(cl_raw),
                    raw_content=cl_raw
                )
                article.clauses.append(clause)
            
            chapter.articles.append(article)
        
        document.chapters.append(chapter)
    
    return document

# ============================================================================
# CHUNK GENERATION - CONTEXTUAL CHUNKING v3.1
# ============================================================================

# Maximum chunk size in characters (for RAG, typically 500-4000 chars is optimal)
MAX_CHUNK_SIZE = 4000

def split_large_content(content: str, max_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """Split large content into smaller chunks at sentence boundaries."""
    if len(content) <= max_size:
        return [content]
    
    parts = []
    current = ""
    
    # Split by sentences (period followed by space, or Vietnamese patterns)
    sentences = re.split(r'(?<=[.;:])\s+', content)
    
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_size:
            current = current + " " + sentence if current else sentence
        else:
            if current:
                parts.append(current.strip())
            # If single sentence is too long, split by comma or just truncate
            if len(sentence) > max_size:
                # Split by comma
                sub_parts = re.split(r'(?<=,)\s*', sentence)
                temp = ""
                for sub in sub_parts:
                    if len(temp) + len(sub) + 1 <= max_size:
                        temp = temp + " " + sub if temp else sub
                    else:
                        if temp:
                            parts.append(temp.strip())
                        temp = sub[:max_size] if len(sub) > max_size else sub
                if temp:
                    current = temp
            else:
                current = sentence
    
    if current:
        parts.append(current.strip())
    
    return parts if parts else [content[:max_size]]

def generate_chunks(document: LegalDocument) -> List[RAGChunk]:
    """
    Generate RAG chunks with CONTEXTUAL CHUNKING.
    
    Each chunk includes:
    - context: A prefix describing the document hierarchy
    - content: The raw content of Điều/Khoản  
    - contextual_content: context + content (USE THIS FOR RAG EMBEDDING)
    
    Large chunks are automatically split into smaller parts.
    """
    chunks = []
    
    def create_chunk(chunk_id, context, content, article_number, clause_number, chapter_str, article_header, part_num=None):
        """Helper to create a single chunk."""
        if part_num:
            chunk_id = f"{chunk_id}_part{part_num}"
            context = context + f" (phần {part_num})"
        
        contextual_content = f"{context}\n\n{content}"
        
        return RAGChunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            document_type=document.document_type,
            document_title=document.title,
            document_number=document.document_number,
            chapter=chapter_str,
            article=article_header,
            article_number=article_number,
            clause_number=clause_number,
            content=content,
            context=context,
            contextual_content=contextual_content,
            char_count=len(contextual_content)
        )
    
    for chapter in document.chapters:
        chapter_str = f"Chương {chapter.chapter_number}: {chapter.chapter_title}"
        
        for article in chapter.articles:
            article_header = f"Điều {article.article_number}. {article.article_title}"
            
            if article.clauses:
                # Create chunk for each clause with COMPLETE context
                for clause in article.clauses:
                    base_chunk_id = f"{document.document_id}_dieu_{article.article_number}_khoan_{clause.clause_number}"
                    
                    # Build CONTEXTUAL PREFIX
                    base_context = f"""Văn bản: {document.title} ({document.document_number})
{chapter_str}
{article_header} - Khoản {clause.clause_number}"""
                    
                    # Split large content if needed
                    content_parts = split_large_content(clause.content)
                    
                    if len(content_parts) == 1:
                        chunks.append(create_chunk(
                            base_chunk_id, base_context, content_parts[0],
                            article.article_number, clause.clause_number,
                            chapter_str, article_header
                        ))
                    else:
                        for i, part in enumerate(content_parts, 1):
                            chunks.append(create_chunk(
                                base_chunk_id, base_context, part,
                                article.article_number, clause.clause_number,
                                chapter_str, article_header, part_num=i
                            ))
            else:
                # No clauses - create chunk for entire article
                base_chunk_id = f"{document.document_id}_dieu_{article.article_number}"
                
                # Build CONTEXTUAL PREFIX
                base_context = f"""Văn bản: {document.title} ({document.document_number})
{chapter_str}
{article_header}"""
                
                # Split large content if needed
                content_parts = split_large_content(article.full_content)
                
                if len(content_parts) == 1:
                    chunks.append(create_chunk(
                        base_chunk_id, base_context, content_parts[0],
                        article.article_number, None,
                        chapter_str, article_header
                    ))
                else:
                    for i, part in enumerate(content_parts, 1):
                        chunks.append(create_chunk(
                            base_chunk_id, base_context, part,
                            article.article_number, None,
                            chapter_str, article_header, part_num=i
                        ))
    
    return chunks


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_hierarchical_json(documents: List[LegalDocument], output_path: str):
    """Save documents as hierarchical JSON (without URL/time)."""
    
    def dataclass_to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for k, v in asdict(obj).items():
                # Skip raw_content to reduce size
                if k == 'raw_content':
                    continue
                result[k] = dataclass_to_dict(v)
            return result
        elif isinstance(obj, list):
            return [dataclass_to_dict(i) for i in obj]
        else:
            return obj
    
    output = {
        "metadata": {
            "total_documents": len(documents),
            "created_at": datetime.now().isoformat(),
            "description": "Vietnamese Traffic Law 2025"
        },
        "documents": [dataclass_to_dict(doc) for doc in documents]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Saved hierarchical JSON: {output_path}")

def save_chunks_jsonl(chunks: List[RAGChunk], output_path: str):
    """Save chunks as JSONL."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + '\n')
    
    print(f"[OK] Saved JSONL chunks: {output_path} ({len(chunks)} chunks)")

def save_chunks_csv(chunks: List[RAGChunk], output_path: str):
    """Save chunks as CSV."""
    if not chunks:
        print("[WARN] No chunks to save as CSV")
        return
    
    fieldnames = list(asdict(chunks[0]).keys())
    
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for chunk in chunks:
            writer.writerow(asdict(chunk))
    
    print(f"[OK] Saved CSV chunks: {output_path} ({len(chunks)} chunks)")

# ============================================================================
# MAIN
# ============================================================================

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration from environment
RAG_LAW_DIR = Path(__file__).parent.parent
PROJECT_ROOT = RAG_LAW_DIR.parent
CLEAN_DIR = os.getenv("CLEAN_DIR", str(PROJECT_ROOT / "documents_clean"))
DATA_DIR = os.getenv("DATA_DIR", str(RAG_LAW_DIR / "structured_law"))

def main() -> int:
    """Main function to parse all documents and generate output."""
    parser = argparse.ArgumentParser(description="Parse Vietnamese legal documents for RAG")
    parser.add_argument("--input", "-i", default=CLEAN_DIR, help="Input directory with .txt files")
    parser.add_argument("--output", "-o", default=DATA_DIR, help="Output directory for structured data")
    args = parser.parse_args()
    
    documents_dir = args.input
    output_dir = args.output
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Input:  {documents_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    if not os.path.exists(documents_dir):
        print(f"Error: Input directory not found: {documents_dir}")
        return 1
    
    txt_files = [f for f in os.listdir(documents_dir) if f.endswith('.txt')]
    print(f"\nFound {len(txt_files)} document files\n")
    
    if not txt_files:
        print("No .txt files found")
        return 0
    
    all_documents = []
    all_chunks = []
    
    for i, filename in enumerate(sorted(txt_files), 1):
        filepath = os.path.join(documents_dir, filename)
        print(f"[{i}/{len(txt_files)}] Parsing: {filename}")
        
        document = parse_document(filepath)
        if document:
            all_documents.append(document)
            chunks = generate_chunks(document)
            all_chunks.extend(chunks)
            
            num_chapters = len(document.chapters)
            num_articles = sum(len(ch.articles) for ch in document.chapters)
            print(f"    -> {num_chapters} chapters, {num_articles} articles, {len(chunks)} chunks")
    
    json_path = os.path.join(output_dir, "structured_data.json")
    jsonl_path = os.path.join(output_dir, "rag_chunks.jsonl")
    csv_path = os.path.join(output_dir, "rag_chunks.csv")
    
    save_hierarchical_json(all_documents, json_path)
    save_chunks_jsonl(all_chunks, jsonl_path)
    save_chunks_csv(all_chunks, csv_path)
    
    # Stats
    print(f"Documents parsed: {len(all_documents)}")
    print(f"RAG chunks:       {len(all_chunks)}")
    
    # Sample chunk info
    if all_chunks:
        avg_len = sum(c.char_count for c in all_chunks) // len(all_chunks)
        min_len = min(c.char_count for c in all_chunks)
        max_len = max(c.char_count for c in all_chunks)
        print(f"\nChunk stats:")
        print(f"  Avg length: {avg_len} chars")
        print(f"  Min length: {min_len} chars")
        print(f"  Max length: {max_len} chars")
    
    print(f"\nOutput files:")
    print(f"  - {json_path}")
    print(f"  - {jsonl_path}")
    print(f"  - {csv_path}")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())

