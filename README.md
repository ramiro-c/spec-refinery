# Spec Refinery

Refinador de requerimientos para PMs: pegás un ticket vago, el sistema busca reglas de la empresa, hace hasta 3 preguntas y reescribe una spec en cada turno. El humano cierra cuando quiere.

Diseño completo: [docs/superpowers/specs/2026-09-05-spec-refinery-design.md](docs/superpowers/specs/2026-09-05-spec-refinery-design.md)

Diagrama interactivo del grafo: [docs/grafo.html](docs/grafo.html) (generado con [Archify](.agents/skills/archify)).

## Quick path (demo Cyber)

1. Pegá el ticket: *“Para el Cyber queremos un checkout más rápido, tipo Amazon: que el comprar ahora no pase por el carrito.”*
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

Variables útiles:

| Variable | Default | Uso |
|----------|---------|-----|
| `SPEC_REFINERY_API` | `http://127.0.0.1:8000` | URL de la API para Streamlit |
| `LLM_PROVIDER` | `gemini` | `gemini` (Vertex/ADC) u `openrouter` |
| `VECTOR_BACKEND` | `chroma` | `chroma` local o `pinecone` |

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

## Trazas con Phoenix (5 corridas Cyber)

Phoenix **no** va embebido en Streamlit. Para observar 5 corridas del ticket Cyber con modelo real:

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

Corré 5 veces el quick path (ticket Cyber → responder → cerrar). En Phoenix, filtrá por `spec-refinery` y exportá las trazas (menú **Export** → JSON/CSV según versión).

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

Sin modelo en CI; golden set y API con grafo dummy.
