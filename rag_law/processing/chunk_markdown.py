"""Hierarchical chunking for markdown legal documents."""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration from environment
RAG_LAW_DIR = Path(__file__).parent.parent
PROJECT_ROOT = RAG_LAW_DIR.parent
MARKDOWN_DIR = os.getenv("MARKDOWN_DIR", str(PROJECT_ROOT / "documents_markdown"))
DATA_DIR = os.getenv("DATA_DIR", str(RAG_LAW_DIR / "structured_law"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1500"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_title: str
    chapter: str
    article: str
    article_number: int
    clause_number: Optional[int]
    point_letter: Optional[str]
    content: str
    context: str
    contextual_content: str
    char_count: int


def extract_doc_info(content: str, filename: str) -> Tuple[str, str]:
    """Extract document ID and title from content."""
    doc_id = os.path.splitext(filename)[0]
    match = re.search(r'^#\s+(.+?)$', content, re.MULTILINE)
    title = match.group(1).strip() if match else doc_id.replace('_', ' ')
    return doc_id, title


def parse_chapters(content: str) -> List[Tuple[str, str, str]]:
    """Parse chapters from markdown content."""
    pattern = re.compile(
        r'^##\s+Chương\s+([IVXLCDM]+|\d+)\s*\n+(.+?)(?=\n##\s+Chương|\Z)',
        re.MULTILINE | re.DOTALL
    )
    matches = list(pattern.finditer(content))
    if not matches:
        return [("0", "", content)]
    
    chapters = []
    for m in matches:
        ch_num, ch_content = m.group(1), m.group(2).strip()
        title_match = re.match(r'^([A-ZĐÀÁẢÃẠ\s,]+)\s*\n', ch_content)
        ch_title = title_match.group(1).strip() if title_match else ""
        chapters.append((ch_num, ch_title, ch_content))
    return chapters


def parse_articles(content: str) -> List[Tuple[int, str, str]]:
    """Parse articles from chapter content."""
    pattern = re.compile(
        r'^####\s+Điều\s+(\d+)\.\s*(.+?)(?=\n####\s+Điều|\Z)',
        re.MULTILINE | re.DOTALL
    )
    articles = []
    for m in pattern.finditer(content):
        art_num = int(m.group(1))
        art_content = m.group(2).strip()
        lines = art_content.split('\n')
        title = lines[0].strip()[:120] if lines else ""
        articles.append((art_num, title, art_content))
    return articles


def parse_clauses(content: str) -> List[Tuple[int, str]]:
    """Parse clauses from article content."""
    pattern = re.compile(
        r'(?:^|\n)\s*\*\*(\d+)\.\*\*\s*(.+?)(?=\n\s*\*\*\d+\.\*\*|\Z)',
        re.DOTALL
    )
    return [(int(m.group(1)), m.group(2).strip()) for m in pattern.finditer(content)]


def parse_points(content: str) -> List[Tuple[str, str]]:
    """Parse points from clause content."""
    pattern = re.compile(
        r'(?:^|\n)\s*-?\s*\*\*([a-zđ])\)\*\*\s*(.+?)(?=\n\s*-?\s*\*\*[a-zđ]\)\*\*|\Z)',
        re.DOTALL
    )
    return [(m.group(1), m.group(2).strip()) for m in pattern.finditer(content)]


def extract_clause_intro(clause_content: str) -> str:
    """Extract the full intro text from clause content before points.
    
    This captures the complete intro including penalty and subject, e.g.:
    'Phạt tiền từ 200.000 đồng đến 400.000 đồng đối với người điều khiển xe 
     thực hiện một trong các hành vi vi phạm sau đây:'
    """
    # Find the first point marker: **a)**, **b)**, etc.
    point_pattern = re.compile(r'-?\s*\*\*[a-zđ]\)\*\*')
    point_match = point_pattern.search(clause_content)
    
    if point_match and point_match.start() > 5:
        # Get everything before the first point
        intro = clause_content[:point_match.start()].strip()
        # Clean up any trailing bullet markers
        intro = re.sub(r'\s*-\s*$', '', intro)
        if intro:
            return intro
    
    return ""


def split_text(text: str, max_chars: int) -> List[str]:
    """Split text at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for sentence in re.split(r'(?<=[.;:])\s+', text):
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip() if current else sentence
        else:
            if current:
                parts.append(current)
            current = sentence[:max_chars] if len(sentence) > max_chars else sentence
    if current:
        parts.append(current)
    return parts or [text[:max_chars]]


def build_context(doc_title: str, chapter: str, article: str,
                  clause_num: Optional[int] = None,
                  point_letter: Optional[str] = None) -> str:
    """Build hierarchical context string."""
    parts = [f"Văn bản: {doc_title}"]
    if chapter:
        parts.append(chapter)
    if article:
        parts.append(article)
    if clause_num:
        ctx = f"Khoản {clause_num}"
        if point_letter:
            ctx += f" - Điểm {point_letter}"
        parts.append(ctx)
    return "\n".join(parts)


def create_chunk(chunk_id: str, doc_id: str, doc_title: str,
                 chapter: str, article: str, article_num: int,
                 content: str, clause_num: Optional[int] = None,
                 point_letter: Optional[str] = None,
                 part_num: Optional[int] = None,
                 clause_intro: Optional[str] = None) -> Chunk:
    """Create a single chunk with context.
    
    Args:
        clause_intro: Penalty/intro text from clause level to include in point chunks.
        
    Note:
        For points (điểm), contextual_content only contains context + clause_intro (mức phạt),
        NOT the point content itself (which is already in 'content' field).
    """
    if part_num:
        chunk_id = f"{chunk_id}_part{part_num}"
    context = build_context(doc_title, chapter, article, clause_num, point_letter)
    
    # Build contextual_content
    if clause_intro and point_letter:
        # For points: contextual = context + clause intro (penalty) ONLY
        # Point content is in 'content' field, not duplicated here
        contextual = f"{context}\n\n {clause_intro}"
    else:
        contextual = f"{context}\n\n{content}"
    
    return Chunk(
        chunk_id=chunk_id, document_id=doc_id, document_title=doc_title,
        chapter=chapter, article=article, article_number=article_num,
        clause_number=clause_num, point_letter=point_letter,
        content=content, context=context, contextual_content=contextual,
        char_count=len(contextual) + len(content)  # Total size for limit check
    )


def generate_chunks(filepath: str, max_chars: int) -> List[Chunk]:
    """Generate chunks from a markdown file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    filename = os.path.basename(filepath)
    doc_id, doc_title = extract_doc_info(content, filename)
    chunks = []
    
    for ch_num, ch_title, ch_content in parse_chapters(content):
        chapter_str = f"Chương {ch_num}: {ch_title}" if ch_title else f"Chương {ch_num}"
        
        for art_num, art_title, art_content in parse_articles(ch_content):
            article_str = f"Điều {art_num}. {art_title}"
            clauses = parse_clauses(art_content)
            
            if clauses:
                for cl_num, cl_content in clauses:
                    points = parse_points(cl_content)
                    clause_intro = extract_clause_intro(cl_content) if points else ""
                    if points:
                        for pt_letter, pt_content in points:
                            base_id = f"{doc_id}_dieu_{art_num}_khoan_{cl_num}_diem_{pt_letter}"
                            context = build_context(doc_title, chapter_str, article_str, cl_num, pt_letter)
                            # Account for clause intro in available space
                            intro_len = len(clause_intro) + 15 if clause_intro else 0  # +15 for "[Mức phạt] \n\n"
                            available = max_chars - len(context) - intro_len - 2
                            parts = split_text(pt_content, max(available, 200))
                            for i, part in enumerate(parts, 1):
                                chunks.append(create_chunk(
                                    base_id, doc_id, doc_title, chapter_str, article_str,
                                    art_num, part, cl_num, pt_letter,
                                    part_num=i if len(parts) > 1 else None,
                                    clause_intro=clause_intro
                                ))
                    else:
                        base_id = f"{doc_id}_dieu_{art_num}_khoan_{cl_num}"
                        context = build_context(doc_title, chapter_str, article_str, cl_num)
                        available = max_chars - len(context) - 2
                        parts = split_text(cl_content, max(available, 200))
                        for i, part in enumerate(parts, 1):
                            chunks.append(create_chunk(
                                base_id, doc_id, doc_title, chapter_str, article_str,
                                art_num, part, cl_num,
                                part_num=i if len(parts) > 1 else None
                            ))
            else:
                base_id = f"{doc_id}_dieu_{art_num}"
                context = build_context(doc_title, chapter_str, article_str)
                available = max_chars - len(context) - 2
                parts = split_text(art_content, max(available, 200))
                for i, part in enumerate(parts, 1):
                    chunks.append(create_chunk(
                        base_id, doc_id, doc_title, chapter_str, article_str,
                        art_num, part, part_num=i if len(parts) > 1 else None
                    ))
    return chunks


def process(input_dir: str, output_path: str, max_chars: int) -> int:
    """Process all markdown files."""
    if not os.path.exists(input_dir):
        log.error(f"Input not found: {input_dir}")
        return 1
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    md_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.md')])
    log.info(f"Processing {len(md_files)} files")
    
    all_chunks = []
    for filename in md_files:
        filepath = os.path.join(input_dir, filename)
        chunks = generate_chunks(filepath, max_chars)
        all_chunks.extend(chunks)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + '\n')
    
    over_limit = sum(1 for c in all_chunks if c.char_count > max_chars)
    log.info(f"Total: {len(all_chunks)} chunks, over_limit: {over_limit}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk markdown documents")
    parser.add_argument("--input", "-i", default=MARKDOWN_DIR, help="Input directory")
    parser.add_argument("--output", "-o", default=os.path.join(DATA_DIR, "rag_chunks_md.jsonl"))
    parser.add_argument("--max-chars", "-m", type=int, default=MAX_CHUNK_CHARS)
    args = parser.parse_args()
    return process(args.input, args.output, args.max_chars)


if __name__ == "__main__":
    sys.exit(main())
