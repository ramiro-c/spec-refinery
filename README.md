# Spec Refinery

Refinador de requerimientos para PMs: pegás un ticket vago, el sistema busca reglas de la empresa, hace hasta 3 preguntas y reescribe una spec en cada turno. El humano cierra cuando quiere.

Diseño completo: [docs/superpowers/specs/2026-09-05-spec-refinery-design.md](docs/superpowers/specs/2026-09-05-spec-refinery-design.md)

Diagrama interactivo del grafo: [docs/grafo.html](docs/grafo.html) (generado con [Archify](.agents/skills/archify)).

## Quick path (demo Cyber Monday)

1. Pegá el ticket: *“Para el Cyber Monday queremos un checkout más rápido, tipo Amazon: que el comprar ahora no pase por el carrito.”*
2. El sistema encuentra la regla del carrito y devuelve la spec con 3 preguntas.
3. Respondés en el chat; la spec de la derecha se reescribe.
4. Apretás **Cerrar spec**. Sale el documento final (con “no cerraría” si aún faltan huecos).

## Levantar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

- API FastAPI: http://127.0.0.1:8000 (Swagger en `/docs`)
- UI Streamlit: http://127.0.0.1:8501
- `run.sh` corre `ingest.py` si falta `.chroma`, levanta la API en background y Streamlit en primer plano.

QA sin credenciales LLM (grafo dummy + Phoenix):

```bash
docker compose up --build
```

- API: http://127.0.0.1:8000/health
- UI: http://127.0.0.1:8501
- Phoenix: http://127.0.0.1:6010
- `SPEC_REFINERY_GRAPH=fake` (default en compose). Para Vertex/OpenRouter: `SPEC_REFINERY_GRAPH=live` y un `.env` con las keys.

Variables útiles:

| Variable | Default | Uso |
|----------|---------|-----|
| `SPEC_REFINERY_API` | `http://127.0.0.1:8000` | URL de la API para Streamlit |
| `LLM_PROVIDER` | `gemini` | `gemini` (Vertex/ADC) u `openrouter` |
| `SUPERVISOR_MODEL` | default del proveedor | Modelo del supervisor (`openrouter` o override por rol) |
| `WRITER_MODEL` | default del proveedor | Modelo del writer (`openrouter` o override por rol) |
| `VECTOR_BACKEND` | `chroma` | `chroma` local (único backend soportado) |

## Grafo (un turno)

```
mensaje del PM
    → supervisor
         ├─ no hay docs de este turno     → retriever → supervisor
         ├─ hay docs, no hay preguntas    → intake    → supervisor
         └─ hay ambos, o dijo cerrar      → writer    → fin del turno
```

| Nodo | Rol |
|------|-----|
| **Supervisor** | Rutea según rúbrica dura; no busca ni escribe. |
| **Retriever** | RAG híbrido (BM25 + embeddings + RRF) sobre Chroma. |
| **Intake** | Cuentas de vaguedad, huecos y fanout; ranking top 3 preguntas. |
| **Writer** | Reescribe las 7 cajas de `SpecDocument` en cada turno. |

La API (`app.py`) es el sistema; Streamlit (`ui.py`) es la piel.

## API

| Acción | Endpoint |
|--------|----------|
| Empezar | `POST /threads` → `{ticket}` → `thread_id` + spec |
| Seguir | `POST /threads/{id}/messages` → `{content}` → spec |
| Cerrar | `POST /threads/{id}/close` → spec final |

## Trazas con Phoenix (5 corridas Cyber Monday)

Phoenix **no** va embebido en Streamlit. Para observar 5 corridas del ticket de Cyber Monday con modelo real:

```bash
pip install arize-phoenix openinference-instrumentation-langchain
phoenix serve   # UI en http://localhost:6006
```

En otra terminal, antes de levantar la API:

```bash
export PHOENIX_COLLECTOR_ENDPOINT=http://127.0.0.1:6006/v1/traces
python - <<'PY'
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
register(project_name="spec-refinery")
LangChainInstrumentor().instrument()
print("Phoenix instrumentado — ahora ./run.sh o uvicorn app:app")
PY
```

Corré 5 veces el quick path (ticket Cyber Monday → responder → cerrar). En Phoenix, filtrá por `spec-refinery` y exportá las trazas (menú **Export** → JSON/CSV según versión).

Alternativa: [LangSmith](https://smith.langchain.com/) con `LANGCHAIN_TRACING_V2=true` y `LANGCHAIN_API_KEY`.

## Diagramas

Para regenerar el HTML del grafo:

```bash
node .agents/skills/archify/bin/archify.mjs deliver architecture \
  docs/spec-refinery.architecture.json docs/grafo.html --quality showcase
```

Diagramas: usar Archify del repo (`.agents/skills/archify`).

## Tests

```bash
.venv/bin/pytest tests/ -v
```

## System evidence tests

`qa_system.py` runs 6 end-to-end scenarios against the live API and verifies
that each one produced traces in Arize Phoenix (rubric evidence: 5+ system
tests with traces visible in Phoenix or LangSmith).

```bash
docker compose up -d          # or: ./run.sh (API on :8000)
.venv/bin/python qa_system.py
```

What it proves:

- **S1** `GET /health` + `POST /threads` → 200 with a valid Pydantic response.
- **S2** Golden question #1 (from `golden_set.json`) in a new thread → 200 and
  `spec.choques` populated with citations (`document_id`).
- **S3** Golden question #2 in a **different** new thread → same retrieval path.
- **S4** Follow-up message on the **same** thread → 200 with `spec.pedido`
  unchanged → the checkpointer kept state (multi-turn continuity).
- **S5** Malformed payload (`POST /threads` without `ticket`) → 422 with a
  Pydantic error detail.
- **S6** (bonus) `POST /threads/{id}/close` → 200; the spec reflects closure.

Trace verification: after the scenarios, the script queries Phoenix
(`QA_PHOENIX_URL`, default `http://localhost:6010` — the compose mapping) for
spans inside each scenario's execution window and prints per-scenario span
counts/names. It exits 0 only if all 5 mandatory scenarios pass **and** the
trace check passes. Fallbacks (never fabricated as a pass): Phoenix
unreachable or missing → `SKIP`; LangSmith tracing active
(`LANGCHAIN_TRACING_V2`/`LANGSMITH_TRACING`) → Phoenix check `N/A`, traces
land in LangSmith. Useful flags: `--phoenix-url http://localhost:6006`
(standalone `phoenix serve`), `--skip-traces`, and `--selftest` (runs all
scenarios in-process against the fake graph, no server or credentials).


Sin modelo en CI; golden set y API con grafo dummy.
