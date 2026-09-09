# Spec Refinery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un PM pega un ticket vago, el sistema busca reglas de Marketplace Andes, hace 3 preguntas y reescribe una spec en cada turno; “Cerrar spec” entrega aunque falten huecos.

**Architecture:** Supervisor estilo P6 rutea `retriever` → `intake` → `writer` por turno. Intake no inventa ranking: `scoring.py` calcula vaguedad, huecos, fanout y top 3. FastAPI es el sistema; Streamlit solo llama. Checkpointer SQLite por `thread_id`. Sin Tavily, sin `interrupt()`.

**Tech Stack:** Python 3.12+, LangGraph ≥ 1.2, LangChain 1.x, FastAPI, Streamlit, Chroma (default) / Pinecone (env), SqliteSaver, Pydantic, pytest, Arize Phoenix o LangSmith (trazas a mano).

**Spec:** `docs/superpowers/specs/2026-09-05-spec-refinery-design.md`

## Global Constraints

- Python 3.12+.
- Sin Tavily / sin búsqueda web.
- Nada interno de MELI en el corpus.
- Vector store: Chroma local por default; Pinecone solo si `VECTOR_BACKEND=pinecone`.
- Errores: `RetryPolicy` (3 intentos) / loop una vez al supervisor / pause = fin de turno HTTP / `error_handler` → writer / bugs bubble up. Sin `except Exception` genérico en nodos.
- Comentarios en español. Identificadores en inglés.
- Cerrar es del humano: el sistema no veta.
- Duplicados de specs viejas: fuera de alcance.
- Portar, no reescribir de memoria: factory desde `ai-engineering-coderhouse-course/pre-entrega-6/clients/factory.py`; retry desde `pre-entrega-6/agents/retry.py`; híbrido desde `pre-entrega-4/rag_system.py` (EnsembleRetriever RRF c=60, pesos 0.5/0.5).
- LLM: solo Vertex o OpenRouter. `ProviderName = Literal["gemini", "openrouter"]`. `gemini` = `ChatGoogleGenerativeAI` + `GOOGLE_GENAI_USE_VERTEXAI=true` + ADC (mismo wiring P5/P6). `openrouter` = `ChatOpenRouter` + un modelo por rol. `build_chat_model` / `build_role_models` se copian; se borran las ramas `openai` y `anthropic`. Roles: `supervisor`, `writer`. Retriever e intake no llaman LLM.
- Ticket de demo (constante en tests): `Para el Cyber queremos un checkout más rápido, tipo Amazon: que el comprar ahora no pase por el carrito.`

---

## File map

| Path | Responsabilidad |
|------|-----------------|
| `schemas.py` | `Citation`, `AcceptanceCriterion`, `SpecStatus`, `SpecDocument` |
| `scoring.py` | Cuentas: vaguedad, huecos, fanout, `rank_questions` |
| `catalog.py` | IDs de servicios Andes |
| `state.py` | `RefineryState` + `initial_fields` |
| `config.py` | env, paths, `MAX_STEPS`, `RECURSION_LIMIT` |
| `corpus/**` | 11 markdowns Andes |
| `retriever.py` | RAG híbrido + `document_id` |
| `ingest.py` | Indexa `corpus/` a Chroma |
| `golden_set.json` + `evaluate.py` | Recall del retriever |
| `agents/retry.py` | `NODE_RETRY` + `node_error_handler` → writer |
| `agents/supervisor.py` | Rúbrica + structured output |
| `agents/retriever_node.py` | Nodo que llama `retriever.retrieve` |
| `agents/intake.py` | Corre `scoring` + arma top 3 |
| `agents/writer.py` | Reescribe `SpecDocument` |
| `graph.py` | `StateGraph` + `route_from_supervisor` |
| `checkpoint.py` | SqliteSaver async-wrap (P5) |
| `app.py` | FastAPI: empezar / seguir / cerrar |
| `ui.py` | Streamlit |
| `clients/factory.py` | Copia P6, sin Tavily |
| `run.sh` | API + UI + ingest si hace falta |
| `tests/test_*.py` | TDD por task |

---

### Task 1: Scaffold + `SpecDocument`

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `.gitignore`, `.env.example`, `config.py`, `schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: nada
- Produces: `SpecDocument` y tipos anidados que el resto importa de `schemas.py`

- [ ] **Step 1: Init git y archivos muertos**

```bash
cd /Users/ramiro/Desktop/personal/spec-refinery
git init -b main
```

`requirements.txt`:

```
langchain==1.3.14
langgraph==1.2.11
langgraph-checkpoint-sqlite==3.1.1
langchain-community==0.4.2
langchain-classic>=1.0.7
langchain-chroma>=0.2.0
langchain-huggingface>=0.1.0
sentence-transformers>=3.0.0
tiktoken>=0.8.0
rank-bm25>=0.2.2
fastapi>=0.115.0
uvicorn>=0.32.0
streamlit>=1.40.0
pydantic>=2.0
python-dotenv>=1.0.0
httpx>=0.27.0
pytest>=8.0.0
langchain-google-genai==4.3.3
langchain-openrouter>=0.1.0
```

`pytest.ini`:

```ini
[pytest]
testpaths = tests
pythonpath = .
```

`.gitignore`: `.env`, `.venv/`, `__pycache__/`, `*.sqlite`, `.chroma/`, `.secrets/`

`.env.example`:

```
# gemini (Vertex) | openrouter
LLM_PROVIDER=gemini
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_APPLICATION_CREDENTIALS=
GOOGLE_CLOUD_PROJECT=
GOOGLE_CLOUD_LOCATION=
OPENROUTER_API_KEY=
VECTOR_BACKEND=chroma
PINECONE_API_KEY=
CHECKPOINT_PATH=checkpoints.sqlite
```

- [ ] **Step 2: Failing test**

```python
# tests/test_schemas.py
from schemas import AcceptanceCriterion, Citation, SpecDocument, SpecStatus


def test_spec_document_roundtrip_empty_boxes():
    spec = SpecDocument(
        pedido="ticket",
        que_entendimos="",
        choques=[],
        servicios=[],
        criterios=[],
        preguntas=[],
        estado=SpecStatus(se_puede_cerrar=False, vaguedad=0, razon="inicio"),
    )
    dumped = spec.model_dump()
    assert dumped["pedido"] == "ticket"
    assert dumped["preguntas"] == []
    again = SpecDocument.model_validate(dumped)
    assert again.estado.se_puede_cerrar is False
```

- [ ] **Step 3: Run test — must fail**

```bash
cd /Users/ramiro/Desktop/personal/spec-refinery
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/test_schemas.py -v
```

Expected: `ModuleNotFoundError: schemas`

- [ ] **Step 4: Minimal implementation**

```python
# config.py
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()

LLM_PROVIDER = _env_str("LLM_PROVIDER", "gemini")
VECTOR_BACKEND = _env_str("VECTOR_BACKEND", "chroma")
CHECKPOINT_PATH = _env_str("CHECKPOINT_PATH", str(BASE_DIR / "checkpoints.sqlite"))
CORPUS_DIR = BASE_DIR / "corpus"
CHROMA_DIR = BASE_DIR / ".chroma"
MAX_STEPS = 8
RECURSION_LIMIT = 20
TOP_K = 5
```

```python
# schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field

class Citation(BaseModel):
    document_id: str
    title: str = ""
    excerpt: str = ""

class AcceptanceCriterion(BaseModel):
    dado: str = ""
    cuando: str = ""
    entonces: str = ""

class SpecStatus(BaseModel):
    se_puede_cerrar: bool
    vaguedad: int = Field(ge=0)
    razon: str

class SpecDocument(BaseModel):
    pedido: str
    que_entendimos: str = ""
    choques: list[Citation] = Field(default_factory=list)
    servicios: list[str] = Field(default_factory=list)
    criterios: list[AcceptanceCriterion] = Field(default_factory=list)
    preguntas: list[str] = Field(default_factory=list)
    estado: SpecStatus
```

- [ ] **Step 5: Tests pass + commit**

```bash
pytest tests/test_schemas.py -v
git add requirements.txt pytest.ini .gitignore .env.example config.py schemas.py tests/test_schemas.py
git commit -m "feat: scaffold SpecDocument and project config"
```

---

### Task 2: Cuentas deterministas (`scoring.py`)

**Files:**
- Create: `catalog.py`, `scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `Citation` from `schemas.py`
- Produces:
  - `CYBER_TICKET: str`
  - `SERVICE_IDS: tuple[str, ...]`
  - `find_vague_hits(text: str) -> list[str]`
  - `empty_slots(text: str) -> list[str]`
  - `fanout(text: str) -> list[str]`
  - `rank_questions(text: str, citations: list[Citation]) -> list[str]` (máx 3)
  - `vaguedad_score(text: str) -> int` (= `len(hits) + len(empty_slots)`)

- [ ] **Step 1: Failing tests**

```python
# tests/test_scoring.py
from schemas import Citation
from scoring import (
    CYBER_TICKET,
    empty_slots,
    fanout,
    find_vague_hits,
    rank_questions,
    vaguedad_score,
)

def test_cyber_ticket_vague_hits():
    hits = find_vague_hits(CYBER_TICKET)
    assert "más rápido" in hits
    assert "tipo amazon" in hits

def test_cyber_ticket_has_empty_scope_and_metric_slots():
    slots = empty_slots(CYBER_TICKET)
    assert "criterio_medible" in slots
    assert "alcance" in slots

def test_fanout_finds_catalog_ids_only_when_named():
    assert fanout("tocamos cart-service y checkout-api") == [
        "cart-service",
        "checkout-api",
    ]
    assert fanout(CYBER_TICKET) == []

def test_rank_questions_collision_first_then_slot_then_vague():
    citations = [
        Citation(
            document_id="adr-cart-price.md",
            title="El precio se cierra en el carrito",
            excerpt="No se saltea.",
        )
    ]
    qs = rank_questions(CYBER_TICKET, citations)
    assert len(qs) == 3
    assert "adr-cart-price.md" in qs[0]
    assert "saltea" in qs[0].lower() or "carrito" in qs[0].lower()
    assert any("producto" in q.lower() for q in qs)
    assert any("más rápido" in q.lower() or "rapido" in q.lower() for q in qs)

def test_rank_questions_without_citations_has_no_collision():
    qs = rank_questions(CYBER_TICKET, [])
    assert all("adr-cart-price.md" not in q for q in qs)
    assert len(qs) <= 3

def test_vaguedad_score_is_hits_plus_empty_slots():
    text = CYBER_TICKET
    assert vaguedad_score(text) == len(find_vague_hits(text)) + len(empty_slots(text))
```

- [ ] **Step 2: Run — must fail**

```bash
pytest tests/test_scoring.py -v
```

Expected: `ModuleNotFoundError: scoring`

- [ ] **Step 3: Implementation**

```python
# catalog.py
SERVICE_IDS: tuple[str, ...] = (
    "cart-service",
    "checkout-api",
    "pricing-engine",
    "promotions-service",
    "payments-vault",
    "shipping-service",
    "inventory-service",
    "returns-service",
)
```

```python
# scoring.py
from __future__ import annotations
import re
from schemas import Citation
from catalog import SERVICE_IDS

CYBER_TICKET = (
    "Para el Cyber queremos un checkout más rápido, tipo Amazon: "
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
```

- [ ] **Step 4: Tests pass + commit**

```bash
pytest tests/test_scoring.py -v
git add catalog.py scoring.py tests/test_scoring.py
git commit -m "feat: deterministic vagueness, slots, fanout, top-3 ranking"
```

---

### Task 3: Corpus Andes (11 docs)

**Files:**
- Create: `corpus/rules/*.md`, `corpus/glossary.md`, `corpus/catalog.md`, `corpus/specs/*.md`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `catalog.SERVICE_IDS`
- Produces: archivos con front-matter `document_id:` estable. La regla de la demo es `adr-cart-price.md`.

- [ ] **Step 1: Failing test**

```python
# tests/test_corpus.py
from pathlib import Path
from catalog import SERVICE_IDS

CORPUS = Path(__file__).resolve().parents[1] / "corpus"

def test_eleven_markdown_docs_exist():
    files = list(CORPUS.rglob("*.md"))
    assert len(files) == 11

def test_cart_price_rule_has_stable_id_and_no_skip():
    text = (CORPUS / "rules" / "adr-cart-price.md").read_text()
    assert "document_id: adr-cart-price.md" in text
    assert "no se saltea" in text.lower() or "No se saltea" in text

def test_catalog_lists_every_service_id():
    text = (CORPUS / "catalog.md").read_text()
    for sid in SERVICE_IDS:
        assert sid in text
```

- [ ] **Step 2: Run — must fail** (faltan archivos)

- [ ] **Step 3: Write the 11 docs**

Cada archivo empieza con:

```markdown
---
document_id: <filename>
title: <título>
---
```

Contenido mínimo (una página, prosa, IDs de servicio donde aplique):

1. `corpus/rules/adr-cart-price.md` — El precio final (promos, envío, cantidad) se cierra en el `cart-service`. No se saltea. `checkout-api` lee el total ya cerrado.
2. `corpus/rules/adr-promo-at-cart.md` — Una promo de `promotions-service` solo vale si se aplicó en el carrito.
3. `corpus/rules/adr-cyber-flags.md` — En Cyber, cambios de checkout van por flag; no quedan prendidos.
4. `corpus/rules/adr-card-challenge.md` — `payments-vault` no reusa tarjeta sin un paso extra.
5. `corpus/rules/adr-shipping-step.md` — Costo y dirección se confirman en `shipping-service` después del carrito. Comprar ahora no inventa envío default.
6. `corpus/rules/adr-stock-reserve.md` — `inventory-service` reserva al entrar al carrito. Sin carrito no hay reserva.
7. `corpus/rules/adr-returns-window.md` — `returns-service`: el plazo arranca al entregar, no al click.
8. `corpus/glossary.md` — *comprar ahora*, *carrito*, *Cyber*, *buyer*, *orden*.
9. `corpus/catalog.md` — una línea por cada `SERVICE_IDS`.
10. `corpus/specs/spec-checkout-current.md` — carrito → envío → pago → confirmación.
11. `corpus/specs/spec-cyber-banner.md` — banner Cyber del año pasado, por flag.

Ningún texto nombra Mercado Libre, MELI, ni equipos reales.

- [ ] **Step 4: Tests pass + commit**

```bash
pytest tests/test_corpus.py -v
git add corpus tests/test_corpus.py
git commit -m "feat: Marketplace Andes corpus (11 docs)"
```

---

### Task 4: Retriever híbrido + golden set

**Files:**
- Create: `retriever.py`, `ingest.py`, `golden_set.json`, `evaluate.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `CORPUS_DIR`, `CHROMA_DIR`, `TOP_K`, `VECTOR_BACKEND`
- Produces:
  - `load_corpus_documents() -> list[Document]` (metadata `document_id`)
  - `retrieve(query: str, *, bm25=..., vector=...) -> list[Citation]`
  - `ingest_chroma() -> None`
  - `evaluate_golden() -> bool` (Recall@5: 4/5)

Portar fusión de P4: `EnsembleRetriever` RRF c=60, pesos 0.5/0.5. En tests, **stubs** de BM25 y vector (como `pre-entrega-4/tests/test_rag_system.py`). No bajar el modelo de embeddings en CI.

- [ ] **Step 1: Failing test (stubs, sin red)**

```python
# tests/test_retriever.py
from langchain_core.documents import Document
from retriever import retrieve

class _Stub:
    def __init__(self, docs):
        self._docs = docs
    def invoke(self, query: str):
        return self._docs

def test_retrieve_dedupes_by_document_id_and_keeps_cart_rule():
    cart = Document(
        page_content="El precio se cierra en el carrito. No se saltea.",
        metadata={"document_id": "adr-cart-price.md", "title": "Precio en carrito"},
    )
    other = Document(
        page_content="Flags de Cyber.",
        metadata={"document_id": "adr-cyber-flags.md", "title": "Flags"},
    )
    citations = retrieve(
        "¿Se puede saltear el carrito?",
        lexical=_Stub([cart, other]),
        semantic=_Stub([cart]),
    )
    ids = [c.document_id for c in citations]
    assert "adr-cart-price.md" in ids
    assert ids.count("adr-cart-price.md") == 1
```

- [ ] **Step 2: Run — must fail**

- [ ] **Step 3: `retriever.py` mínimo**

```python
from __future__ import annotations
from langchain_core.documents import Document
from schemas import Citation

def _as_citation(doc: Document) -> Citation:
    meta = doc.metadata or {}
    return Citation(
        document_id=str(meta.get("document_id") or ""),
        title=str(meta.get("title") or ""),
        excerpt=doc.page_content[:240],
    )

def retrieve(
    query: str,
    *,
    lexical,
    semantic,
    top_k: int = 5,
) -> list[Citation]:
    # Tests inyectan stubs. Producción usa EnsembleRetriever.
    seen: set[str] = set()
    out: list[Citation] = []
    for block in (lexical.invoke(query), semantic.invoke(query)):
        for doc in block:
            cit = _as_citation(doc)
            if not cit.document_id or cit.document_id in seen:
                continue
            seen.add(cit.document_id)
            out.append(cit)
            if len(out) >= top_k:
                return out
    return out
```

Después, ampliar `build_hybrid(lexical, semantic)` con `EnsembleRetriever(retrievers=[...], weights=[0.5, 0.5])` para producción. `ingest.py` lee markdown, pone `document_id` del front-matter, persiste Chroma en `CHROMA_DIR` y arma BM25 sobre los mismos `Document`. Si `VECTOR_BACKEND=pinecone`, rama copiada de P4; default Chroma.

`golden_set.json`:

```json
{
  "casos": [
    {"pregunta": "¿Se puede saltear el carrito?", "documento_id_esperado": "adr-cart-price.md"},
    {"pregunta": "¿Cuándo se reserva el stock?", "documento_id_esperado": "adr-stock-reserve.md"},
    {"pregunta": "¿La promo vale después del carrito?", "documento_id_esperado": "adr-promo-at-cart.md"},
    {"pregunta": "¿Los cambios de checkout en Cyber quedan prendidos?", "documento_id_esperado": "adr-cyber-flags.md"},
    {"pregunta": "¿Cuándo arranca el plazo de devolución?", "documento_id_esperado": "adr-returns-window.md"}
  ]
}
```

`evaluate.py`: carga golden, llama retrieve real (requiere ingest local), imprime tabla, exit 0 si ≥4/5.

- [ ] **Step 4: Tests de stub pasan + commit**

```bash
pytest tests/test_retriever.py -v
git add retriever.py ingest.py golden_set.json evaluate.py tests/test_retriever.py
git commit -m "feat: hybrid retrieve with document_id citations"
```

Correr `python ingest.py && python evaluate.py` a mano una vez (baja embeddings). No es gate de pytest.

---

### Task 5: Supervisor + rúbrica

**Files:**
- Create: `state.py`, `agents/supervisor.py`
- Test: `tests/test_supervisor.py`

**Interfaces:**
- Consumes: `SpecDocument`, `MAX_STEPS`
- Produces:
  - `NextAgent = Literal["retriever", "intake", "FINISH"]`
  - `RefineryState` (MessagesState + slots)
  - `apply_rubric(...) -> NextAgent`
  - `supervisor_turn(state, llm) -> dict`

Reglas duras (el grafo nunca ve un `next_agent` ilegal):

1. Si `last_error` no vacío → `FINISH` (error_handler ya apuntó al writer; el supervisor no reintenta el mismo nodo).
2. Si `step_count >= MAX_STEPS` → `FINISH`.
3. Si `close_requested` → `FINISH`.
4. Si no hay `citations` y no `close_requested` → `retriever` (aunque el LLM pida FINISH).
5. Si hay citations, `questions` de este turno vacío, y no close → `intake` si el LLM pide FINISH.
6. Si no, se respeta el `proposed` si es `retriever` | `intake` | `FINISH`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_supervisor.py
from agents.supervisor import apply_rubric

def test_rubric_blocks_finish_without_citations():
    assert apply_rubric(
        citations_empty=True,
        questions_empty=True,
        step_count=1,
        proposed="FINISH",
        close_requested=False,
        last_error="",
    ) == "retriever"

def test_rubric_allows_finish_on_close_without_citations():
    assert apply_rubric(
        citations_empty=True,
        questions_empty=True,
        step_count=1,
        proposed="intake",
        close_requested=True,
        last_error="",
    ) == "FINISH"

def test_rubric_blocks_finish_without_questions_when_open():
    assert apply_rubric(
        citations_empty=False,
        questions_empty=True,
        step_count=2,
        proposed="FINISH",
        close_requested=False,
        last_error="",
    ) == "intake"

def test_rubric_caps_steps():
    assert apply_rubric(
        citations_empty=False,
        questions_empty=False,
        step_count=8,
        proposed="retriever",
        close_requested=False,
        last_error="",
    ) == "FINISH"
```

`MAX_STEPS` es 8: `step_count >= 8` corta.

- [ ] **Step 2: Run — must fail**

- [ ] **Step 3: `state.py` + `apply_rubric` + `supervisor_turn`**

```python
# state.py
from __future__ import annotations
from typing import Literal
from langgraph.graph import MessagesState
from schemas import Citation, SpecDocument, SpecStatus

NextAgent = Literal["retriever", "intake", "FINISH"]

class RefineryState(MessagesState):
    next_agent: NextAgent
    citations: list[Citation]
    questions: list[str]
    spec: SpecDocument | None
    close_requested: bool
    last_agent: str
    step_count: int
    last_error: str
    ticket: str

def empty_spec(ticket: str) -> SpecDocument:
    return SpecDocument(
        pedido=ticket,
        estado=SpecStatus(se_puede_cerrar=False, vaguedad=0, razon="inicio"),
    )

def initial_fields(ticket: str, *, close_requested: bool = False) -> dict:
    return {
        "next_agent": "FINISH",
        "citations": [],
        "questions": [],
        "spec": empty_spec(ticket),
        "close_requested": close_requested,
        "last_agent": "",
        "step_count": 0,
        "last_error": "",
        "ticket": ticket,
    }
```

`agents/supervisor.py`: copiar la forma de `pre-entrega-6/agents/supervisor.py` (`SupervisorDecision`, `_snapshot`, `with_structured_output`). `apply_rubric` implementa las 6 reglas de arriba. Prompt: no buscás ni escribís; elegís retriever / intake / FINISH.

- [ ] **Step 4: Tests pass + commit**

```bash
pytest tests/test_supervisor.py -v
git add state.py agents/supervisor.py tests/test_supervisor.py
git commit -m "feat: supervisor rubric for retriever/intake/FINISH"
```

---

### Task 6: Grafo + nodos inyectables

**Files:**
- Create: `agents/retry.py`, `agents/retriever_node.py`, `agents/intake.py`, `agents/writer.py`, `graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `retrieve`, `rank_questions`, `fanout`, `vaguedad_score`, `apply_rubric`
- Produces:
  - `build_graph(*, supervisor, retriever, intake, writer, checkpointer=None) -> CompiledStateGraph`
  - `route_from_supervisor(state) -> Literal["retriever", "intake", "writer"]`
  - `run_turn(graph, ticket, messages, *, thread_id, close_requested) -> RefineryState`

Portar `agents/retry.py` **verbatim** de P6, cambiando el `Command(goto="writer")` (ya apunta a writer; los nombres de nodo nuevos no cambian eso).

Nodos (tests inyectan fakes; producción usa factories):

- `retriever_node`: `citations = retrieve(state["ticket"] + last human)`; pisa `citations`.
- `intake_node`: `questions = rank_questions(ticket, citations)`; `servicios = fanout(ticket + excerpts)`.
- `writer_node`: reescribe `SpecDocument`. En tests, un writer dummy que copia ticket + questions + citations. En prod, LLM structured output a `SpecDocument` (cajas `que_entendimos` y `criterios`). Si `close_requested`, `preguntas=[]` y `estado.se_puede_cerrar = vaguedad==0`. `pedido` nunca se reescribe: es `state["ticket"]`.

`set_node_defaults(retry_policy=NODE_RETRY, error_handler=node_error_handler)` en supervisor/retriever/intake. Writer: `retry_policy=None` (como P6) para no reescribir la spec a medias.

- [ ] **Step 1: Failing test — happy path del Cyber**

```python
# tests/test_graph.py
from langchain_core.messages import HumanMessage
from schemas import Citation
from scoring import CYBER_TICKET
from state import initial_fields
from graph import build_graph, run_turn

def _supervisor(state):
    from agents.supervisor import apply_rubric
    step = int(state.get("step_count") or 0) + 1
    nxt = apply_rubric(
        citations_empty=not state.get("citations"),
        questions_empty=not state.get("questions"),
        step_count=step,
        proposed="FINISH",
        close_requested=bool(state.get("close_requested")),
        last_error=state.get("last_error") or "",
    )
    return {"next_agent": nxt, "step_count": step, "last_agent": "supervisor"}

def _retriever(state):
    return {
        "citations": [
            Citation(
                document_id="adr-cart-price.md",
                title="El precio se cierra en el carrito",
                excerpt="No se saltea.",
            )
        ],
        "last_agent": "retriever",
    }

def _intake(state):
    from scoring import rank_questions
    qs = rank_questions(state["ticket"], state["citations"])
    return {"questions": qs, "last_agent": "intake"}

def _writer(state):
    spec = state["spec"]
    spec.preguntas = [] if state.get("close_requested") else list(state.get("questions") or [])
    spec.choques = list(state.get("citations") or [])
    spec.estado.razon = "dummy"
    spec.estado.se_puede_cerrar = bool(state.get("close_requested")) and spec.estado.vaguedad == 0
    return {"spec": spec, "last_agent": "writer"}

def test_open_turn_visits_retriever_intake_writer():
    g = build_graph(
        supervisor=_supervisor,
        retriever=_retriever,
        intake=_intake,
        writer=_writer,
    )
    hops, final = run_turn(g, CYBER_TICKET, [HumanMessage(content=CYBER_TICKET)])
    assert hops == ["supervisor", "retriever", "supervisor", "intake", "supervisor", "writer"]
    assert final["spec"].choques[0].document_id == "adr-cart-price.md"
    assert len(final["spec"].preguntas) == 3

def test_close_skips_to_writer():
    g = build_graph(
        supervisor=_supervisor,
        retriever=_retriever,
        intake=_intake,
        writer=_writer,
    )
    hops, final = run_turn(
        g,
        CYBER_TICKET,
        [HumanMessage(content="cerrá")],
        close_requested=True,
    )
    assert "retriever" not in hops
    assert hops[-1] == "writer"
    assert final["spec"].preguntas == []
```

`run_turn` usa `graph.stream(..., stream_mode=["updates", "values"])` como `stream_query` de P6 y devuelve `(hops, final)`.

- [ ] **Step 2: Run — must fail**

- [ ] **Step 3: Implement `graph.py` + nodos**

`route_from_supervisor`: `retriever`/`intake` o else `writer`.

```python
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_from_supervisor, {...})
builder.add_edge("retriever", "supervisor")
builder.add_edge("intake", "supervisor")
builder.add_edge("writer", END)
```

- [ ] **Step 4: Tests pass + commit**

```bash
pytest tests/test_graph.py tests/test_supervisor.py -v
git add agents graph.py tests/test_graph.py
git commit -m "feat: hierarchical graph retriever-intake-writer"
```

---

### Task 7: Checkpointer + `thread_id`

**Files:**
- Create: `checkpoint.py`
- Modify: `graph.py` (`build_graph(..., checkpointer=)`)
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `CHECKPOINT_PATH`
- Produces: `open_checkpointer()` — copia el wrap async de `pre-entrega-5/graph.py` (`SqliteSaver` con `aget_tuple`/`aput`). `build_graph` hace `builder.compile(checkpointer=cp)` si se pasa.

- [ ] **Step 1: Failing test**

Dos `run_turn` con el mismo `thread_id`: el segundo ve `citations` del primero (o `ticket` persistido). Usar los fakes del task 6. Config: `{"configurable": {"thread_id": "t1"}, "recursion_limit": 20}`.

- [ ] **Step 2–4: Implement wrap, pass, commit**

```bash
git commit -m "feat: sqlite checkpointer by thread_id"
```

---

### Task 8: FastAPI (empezar / seguir / cerrar)

**Files:**
- Create: `app.py`
- Test: `tests/test_api.py` (httpx `TestClient`, grafo fake inyectado)

**Interfaces:**
- Consumes: `build_graph`, `open_checkpointer`, `initial_fields`, `SpecDocument`
- Produces:
  - `POST /threads` body `{"ticket": str}` → `{"thread_id": str, "spec": SpecDocument}`
  - `POST /threads/{thread_id}/messages` body `{"content": str}` → `{"spec": SpecDocument}`
  - `POST /threads/{thread_id}/close` → `{"spec": SpecDocument}`
  - `404` `{"detail": "thread no existe"}` si `thread_id` no tiene checkpoint

Generar `thread_id` con `uuid4()`. “Seguir” y “cerrar” primero `get_tuple`; si no hay, 404. No tragar bugs: excepción inesperada → 500. Swagger sale solo (FastAPI).

Dependencia `get_graph` overrideable en tests.

- [ ] **Step 1: Tests**

```python
from fastapi.testclient import TestClient
from scoring import CYBER_TICKET

def test_start_continue_close_and_unknown_thread(client: TestClient):
    r = client.post("/threads", json={"ticket": CYBER_TICKET})
    assert r.status_code == 200
    thread_id = r.json()["thread_id"]
    spec = r.json()["spec"]
    assert spec["pedido"] == CYBER_TICKET
    assert len(spec["preguntas"]) == 3

    r2 = client.post(f"/threads/{thread_id}/messages", json={"content": "para todo, no medimos"})
    assert r2.status_code == 200

    r3 = client.post(f"/threads/{thread_id}/close")
    assert r3.status_code == 200
    assert r3.json()["spec"]["preguntas"] == []

    missing = client.post("/threads/does-not-exist/messages", json={"content": "hola"})
    assert missing.status_code == 404
```

Fixture: app con el grafo dummy del task 6 + checkpointer tempdir.

- [ ] **Step 2–4: Implement, pass, commit**

```bash
git commit -m "feat: FastAPI start/continue/close threads"
```

---

### Task 9: Factory LLM + nodos reales

**Files:**
- Create: `clients/factory.py` (copia de `ai-engineering-coderhouse-course/pre-entrega-6/clients/factory.py`)
- Modify: `schemas.py` (`ProviderName`, `RoleName`), `agents/writer.py`, `graph.py`

**Interfaces:**
- Consumes: `LLM_PROVIDER`, vars Vertex (`GOOGLE_*`), `OPENROUTER_API_KEY`
- Produces:
  - `ProviderName = Literal["gemini", "openrouter"]`
  - `RoleName = Literal["supervisor", "writer"]`
  - `build_chat_model(provider=None, *, role=None, model=None) -> BaseChatModel`
  - `build_role_models(provider=None) -> dict[RoleName, BaseChatModel]`

Copiar la factory P6 y recortar así:

- Borrar ramas `openai` y `anthropic`. `_normalize_provider` solo acepta `gemini` | `openrouter`; otro valor → `ValueError("Unsupported LLM_PROVIDER: ... use gemini (Vertex) or openrouter")`.
- `gemini`: dejar el `_GeminiChat` / `AutomaticFunctionCallingConfig(disable=True)` y `ChatGoogleGenerativeAI`. Vertex se prende con `GOOGLE_GENAI_USE_VERTEXAI=true` + ADC (no hace falta `GEMINI_API_KEY` en el happy path).
- `openrouter`: `ChatOpenRouter`. SKUs:
  - `supervisor`: `nvidia/nemotron-3-ultra-550b-a55b:free` (structured output, ruteo).
  - `writer`: `minimax/minimax-m3:free` (el que era `analyst` en la P6).
- `build_role_models`: solo `supervisor` y `writer`. En `gemini`, el mismo modelo para los dos. En `openrouter`, uno por rol.
- Retriever e intake no reciben LLM.
- Writer: `llm.with_structured_output(SpecDocument)`. `pedido=state["ticket"]`. Si `close_requested`, `preguntas=[]`.
- Tests: `build_chat_model(provider="openai")` debe explotar. Writer con `RunnableLambda` fake, como P4. No prender Vertex en pytest.

```bash
git commit -m "feat: wire production nodes and LLM writer"
```

---

### Task 10: Streamlit + `run.sh` + README

**Files:**
- Create: `ui.py`, `run.sh`, `README.md`
- No test de browser. Test manual: `run.sh` levanta API en `:8000` y Streamlit en `:8501`.

**UI (contrato de la spec):**
- Columna izquierda: `st.chat_input` + historial. Primer submit → `POST /threads`. Siguientes → `POST /threads/{id}/messages`. Botón `Cerrar spec` → `POST /threads/{id}/close`.
- Columna derecha: renderizar las 7 cajas de `SpecDocument`.
- Guardar `thread_id` en `st.session_state`.
- URL API: `SPEC_REFINERY_API` default `http://127.0.0.1:8000`.
- Phoenix no se embebe. README: cómo exportar 5 trazas del ticket Cyber.

`run.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# ingest si falta .chroma
# uvicorn app:app --port 8000 &
# streamlit run ui.py --server.port 8501
```

README: quick path de la spec, tabla del grafo, link al design doc, “diagramas: usar Archify del repo (`.agents/skills/archify`)”. Generar HTML del grafo con Archify y dejarlo en `docs/grafo.html` (un architecture: supervisor, retriever, intake, writer, API, Streamlit, Chroma).

```bash
git add ui.py run.sh README.md docs/grafo.html
git commit -m "feat: Streamlit UI, run.sh, and architecture diagram"
```

---

## Self-review

**Spec coverage**

| Spec | Task |
|------|------|
| Demo Cyber + regla carrito | 2, 3, 6 |
| Cierre humano, no veta | 5, 6, 8 |
| Spec viva + top 3 | 2, 6 |
| Sin web | Global + factory |
| Grafo 2 especialistas + writer | 6 |
| Streamlit piel / API sistema | 8, 10 |
| 11 docs Andes | 3 |
| Cuentas fuera del LLM | 2 |
| Retry / error_handler / bubble | 6 (retry.py P6) |
| Checkpointer | 7 |
| Golden set | 4 |
| Phoenix 5 corridas | 10 README |
| Duplicados fuera | no task |
| Archify local | 10 |

**Placeholders:** ninguno. Factory se copia de un path concreto, no “similar a P6” sin ruta.

**Tipos:** `NextAgent`, `SpecDocument`, `Citation`, `CYBER_TICKET`, `apply_rubric(...)` se reusan con la misma firma.

---

## Execution Handoff

Plan en `spec-refinery/docs/superpowers/plans/2026-09-05-spec-refinery.md`.

Dos formas de ejecutarlo:

**1. Subagent-Driven (recomendado)** — un subagente por task, review entre medio.

**2. Inline** — tasks en esta sesión, con checkpoints.

¿Cuál?
