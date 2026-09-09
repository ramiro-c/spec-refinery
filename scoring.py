from __future__ import annotations
import re
from schemas import Citation
from catalog import SERVICE_IDS

CYBER_TICKET = (
    "Para el Cyber Monday queremos un checkout más rápido, tipo Amazon: "
    "que el comprar ahora no pase por el carrito."
)

_VAGUE = (
    "más rápido",
    "mas rapido",
    "tipo amazon",
    "mejorar",
    "más fácil",
    "mas facil",
    "etc",
)

_SLOT_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("actor", re.compile(r"\b(buyer|seller|squad|pm|usuario|actor)\b", re.I)),
    ("criterio_medible", re.compile(r"\b(\d+|%|segundos|clicks|conversi[oó]n|p95)\b", re.I)),
    ("alcance", re.compile(r"\b(todos|solo|excepto|sku|categor[ií]a|1p|marketplace)\b", re.I)),
    ("fuera_de_alcance", re.compile(r"\b(fuera de alcance|no incluye|no tocar)\b", re.I)),
    ("dependencias", re.compile(r"\b(depende|bloqueado por|requiere)\b", re.I)),
)


def find_vague_hits(text: str) -> list[str]:
    low = text.lower()
    return [p for p in _VAGUE if p in low]


def empty_slots(text: str) -> list[str]:
    return [name for name, pat in _SLOT_HINTS if not pat.search(text)]


def fanout(text: str) -> list[str]:
    low = text.lower()
    return [sid for sid in SERVICE_IDS if sid in low]


def vaguedad_score(text: str) -> int:
    return len(find_vague_hits(text)) + len(empty_slots(text))


def rank_questions(text: str, citations: list[Citation]) -> list[str]:
    ranked: list[str] = []
    for c in citations:
        ranked.append(
            f"El pedido se pisa con «{c.title}» ({c.document_id}). "
            "¿«Comprar ahora» saltea el carrito de verdad, o solo acorta "
            "la pantalla y el precio se calcula igual detrás?"
        )
        break
    slots = empty_slots(text)
    if "alcance" in slots:
        ranked.append("¿Vale para todos los productos o solo algunos?")
    if "criterio_medible" in slots or "más rápido" in find_vague_hits(text):
        ranked.append(
            "¿Qué es «más rápido»? ¿Menos clicks, menos segundos, más conversión?"
        )
    if "actor" in slots:
        ranked.append("¿Quién pide esto (squad, PM, buyer) y quién lo implementa?")
    if "fuera_de_alcance" in slots:
        ranked.append("¿Qué queda explícitamente fuera de este pedido?")
    # unique preserve order, max 3
    seen: set[str] = set()
    out: list[str] = []
    for q in ranked:
        if q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) == 3:
            break
    return out
