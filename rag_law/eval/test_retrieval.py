# -*- coding: utf-8 -*-
"""Test set for retrieval evaluation."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration from environment
RAG_LAW_DIR = Path(__file__).parent.parent
DATA_DIR = os.getenv("DATA_DIR", str(RAG_LAW_DIR / "structured_law"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

TEST_CASES = [
    {"id": "test_001", "query": "Mức phạt khi vượt đèn đỏ là bao nhiêu?", "category": "penalty", "expected_docs": ["168_2024_ND-CP_619502"]},
    {"id": "test_002", "query": "Phạt bao nhiêu tiền nếu không đội mũ bảo hiểm?", "category": "penalty", "expected_docs": ["168_2024_ND-CP_619502"]},
    {"id": "test_003", "query": "Nồng độ cồn bao nhiêu thì bị phạt?", "category": "penalty", "expected_docs": ["168_2024_ND-CP_619502"]},
    {"id": "test_004", "query": "Lái xe quá tốc độ 20km/h bị phạt thế nào?", "category": "penalty", "expected_docs": ["168_2024_ND-CP_619502"]},
    {"id": "test_005", "query": "Bao nhiêu tuổi được lái xe máy?", "category": "license", "expected_docs": ["36_2024_QH15_613129"]},
    {"id": "test_006", "query": "Điều kiện cấp giấy phép lái xe hạng B1 là gì?", "category": "license", "expected_docs": ["161_2024_ND-CP_623394"]},
    {"id": "test_007", "query": "Giấy phép lái xe bị trừ điểm khi nào?", "category": "license", "expected_docs": ["168_2024_ND-CP_619502"]},
    {"id": "test_008", "query": "Cách phục hồi điểm giấy phép lái xe?", "category": "license", "expected_docs": ["168_2024_ND-CP_619502"]},
    {"id": "test_009", "query": "Thủ tục đăng ký xe ô tô mới mua?", "category": "registration", "expected_docs": ["151_2024_ND-CP_619564"]},
    {"id": "test_010", "query": "Xe ô tô phải đăng kiểm bao lâu một lần?", "category": "registration", "expected_docs": ["166_2024_ND-CP_623277"]},
    {"id": "test_011", "query": "Khoảng cách an toàn khi lái xe trên cao tốc?", "category": "rules", "expected_docs": ["36_2024_QH15_613129"]},
    {"id": "test_012", "query": "Xe nào được ưu tiên đi trước?", "category": "rules", "expected_docs": ["36_2024_QH15_613129"]},
    {"id": "test_013", "query": "Tốc độ tối đa trong khu dân cư là bao nhiêu?", "category": "rules", "expected_docs": ["36_2024_QH15_613129"]},
    {"id": "test_014", "query": "Nội dung giáo dục an toàn giao thông cho học sinh tiểu học?", "category": "education", "expected_docs": ["151_2024_ND-CP_619564"]},
    {"id": "test_015", "query": "Học sinh THPT học lái xe gắn máy ở đâu?", "category": "education", "expected_docs": ["151_2024_ND-CP_619564"]},
    {"id": "test_016", "query": "Xe kinh doanh vận tải phải lắp thiết bị gì?", "category": "vehicle", "expected_docs": ["151_2024_ND-CP_619564"]},
    {"id": "test_017", "query": "Màu sơn xe chở học sinh mầm non?", "category": "vehicle", "expected_docs": ["151_2024_ND-CP_619564"]},
    {"id": "test_018", "query": "Trình tự cấp giấy phép sử dụng còi ưu tiên?", "category": "procedure", "expected_docs": ["151_2024_ND-CP_619564"]},
    {"id": "test_019", "query": "Thời hiệu xử phạt vi phạm giao thông là bao lâu?", "category": "procedure", "expected_docs": ["168_2024_ND-CP_619502"]},
    {"id": "test_020", "query": "Hình thức xử phạt vi phạm hành chính giao thông?", "category": "procedure", "expected_docs": ["168_2024_ND-CP_619502"]},
]


def save_test_set(output_path: str) -> None:
    """Save test set to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "version": "1.0",
            "total_tests": len(TEST_CASES),
            "categories": list(set(t["category"] for t in TEST_CASES)),
            "test_cases": TEST_CASES
        }, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(TEST_CASES)} test cases")


def load_test_set(input_path: str) -> List[Dict]:
    """Load test set from JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f).get("test_cases", [])


def evaluate_retrieval(retrieved: List[Dict], test: Dict, top_k: int = 5) -> Dict:
    """Evaluate retrieval results against expected documents."""
    expected = set(test["expected_docs"])
    retrieved_docs = set(c.get("document_id", "") for c in retrieved[:top_k])
    hits = expected & retrieved_docs
    
    precision = len(hits) / len(retrieved_docs) if retrieved_docs else 0
    recall = len(hits) / len(expected) if expected else 0
    
    return {
        "test_id": test["id"],
        "query": test["query"],
        "precision": precision,
        "recall": recall,
        "hit": 1 if hits else 0
    }


def run_evaluation(retrieval_func: Callable, top_k: int = 5) -> Dict:
    """Run evaluation on all test cases."""
    results = []
    for test in TEST_CASES:
        retrieved = retrieval_func(test["query"], top_k=top_k)
        results.append(evaluate_retrieval(retrieved, test, top_k))
    
    n = len(results)
    return {
        "total": n,
        "avg_precision": round(sum(r["precision"] for r in results) / n, 3),
        "avg_recall": round(sum(r["recall"] for r in results) / n, 3),
        "hit_rate": round(sum(r["hit"] for r in results) / n, 3),
        "results": results
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval test set")
    parser.add_argument("--output", "-o", default=os.path.join(DATA_DIR, "test_retrieval.json"))
    parser.add_argument("--save", action="store_true", help="Save test set to file")
    args = parser.parse_args()
    
    if args.save:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_test_set(args.output)
    else:
        log.info(f"Test cases: {len(TEST_CASES)}")
        categories = {}
        for t in TEST_CASES:
            categories[t["category"]] = categories.get(t["category"], 0) + 1
        for cat, count in sorted(categories.items()):
            log.info(f"  {cat}: {count}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
