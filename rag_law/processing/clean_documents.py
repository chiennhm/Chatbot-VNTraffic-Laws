"""Clean legal documents by merging fragmented lines into complete paragraphs."""

import argparse
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLEAN_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def is_section_start(line: str) -> bool:
    """Check if line starts a new legal structure section."""
    s = line.strip()
    if not s:
        return False
    if s.startswith('#'):
        return True
    if re.match(r'^Chương\s+[IVXLCDM\d]+', s, re.IGNORECASE):
        return True
    if re.match(r'^Mục\s+\d+', s, re.IGNORECASE):
        return True
    if re.match(r'^Điều\s+\d+[\.:]+', s, re.IGNORECASE):
        return True
    if re.match(r'^\d+\.\s', s):
        return True
    if re.match(r'^[a-zđ]\)\s', s, re.IGNORECASE):
        return True
    return False


def clean_document(content: str) -> str:
    """Merge all text into complete paragraphs, split only at structural markers."""
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    lines = []
    for line in content.split('\n'):
        stripped = line.strip()
        if re.match(r'^[-=_*~\.]{3,}$', stripped):
            continue
        lines.append(stripped)
    
    result = []
    current = []
    
    for line in lines:
        if not line:
            continue
        if is_section_start(line) and current:
            merged = re.sub(r'\s+', ' ', ' '.join(current)).strip()
            if merged:
                result.append(merged)
            current = []
        current.append(line)
    
    if current:
        merged = re.sub(r'\s+', ' ', ' '.join(current)).strip()
        if merged:
            result.append(merged)
    
    return '\n'.join(result)


def process(input_dir: str, output_dir: str) -> int:
    """Process all documents in input directory."""
    if not os.path.exists(input_dir):
        log.error(f"Input not found: {input_dir}")
        return 1
    
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])
    log.info(f"Processing {len(files)} files")
    
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned = clean_document(content)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
    
    log.info(f"Output: {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean legal documents")
    parser.add_argument("--input", "-i", required=True, help="Input directory")
    parser.add_argument("--output", "-o", default=CLEAN_DIR, help="Output directory")
    args = parser.parse_args()
    return process(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())
