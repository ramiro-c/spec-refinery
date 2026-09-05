"""Evalúa Recall@5 del retriever híbrido contra golden_set.json.

Requiere ingesta local previa (`python ingest.py`). Exit 0 si ≥4/5 aciertos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from retriever import retrieve

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"
TOP_K = 5
UMBRAL_ACIERTOS = 4


def evaluate_golden() -> bool:
    """Corre el golden set y devuelve True si Recall@5 ≥ 4/5."""
    datos = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    aciertos = 0
    print(f"{'Pregunta':<55} {'Esperado':<28} {'Hit':>5}")
    print("-" * 92)
    for caso in datos["casos"]:
        pregunta = caso["pregunta"]
        esperado = caso["documento_id_esperado"]
        ids = [c.document_id for c in retrieve(pregunta)[:TOP_K]]
        hit = esperado in ids
        aciertos += int(hit)
        marca = "✓" if hit else "✗"
        print(f"{pregunta[:54]:<55} {esperado:<28} {marca:>5}")
    print("-" * 92)
    print(f"Recall@{TOP_K}: {aciertos}/{len(datos['casos'])}")
    return aciertos >= UMBRAL_ACIERTOS


def main() -> None:
    ok = evaluate_golden()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
