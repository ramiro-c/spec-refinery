"""Evalúa Recall@5 del retriever híbrido contra golden_set.json.

Requiere ingesta local previa (`python ingest.py`). Exit 0 si ≥4/5 aciertos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from config import TOP_K
from retriever import retrieve

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"
HIT_THRESHOLD = 4


def evaluate_golden() -> bool:
    """Corre el golden set y devuelve True si Recall@5 ≥ 4/5."""
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    hits = 0
    print(f"{'Pregunta':<55} {'Esperado':<28} {'Hit':>5}")
    print("-" * 92)
    for case in data["cases"]:
        question = case["question"]
        expected = case["expected_document_id"]
        ids = [c.document_id for c in retrieve(question)[:TOP_K]]
        hit = expected in ids
        hits += int(hit)
        mark = "✓" if hit else "✗"
        print(f"{question[:54]:<55} {expected:<28} {mark:>5}")
    print("-" * 92)
    print(f"Recall@{TOP_K}: {hits}/{len(data['cases'])}")
    return hits >= HIT_THRESHOLD


def main() -> None:
    ok = evaluate_golden()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
