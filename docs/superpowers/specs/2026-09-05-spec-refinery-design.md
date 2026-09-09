# Spec Refinery — refinador de requerimientos

**Fecha:** 2026-09-05
**Estado:** listo para review
**Dónde:** `spec-refinery/` (proyecto nuevo; ensambla P4 + P5 + P6)

Un PM pega un ticket vago. El sistema busca reglas de la empresa, hace hasta 3 preguntas, y reescribe una spec en cada turno. El humano cierra cuando quiere; si faltan huecos, la spec sale igual y lo dice. La API es el sistema. Streamlit es la pantalla.

Encuadre: fase Inception de AI-DLC. El resto del ciclo no se construye.

## Quick path (demo)

1. El PM pega: *“Para el Cyber Monday queremos un checkout más rápido, tipo Amazon: que el comprar ahora no pase por el carrito.”*
2. El sistema encuentra la regla: el precio se cierra en el carrito; no se saltea.
3. Devuelve la spec (con huecos) y 3 preguntas: ¿saltea de verdad?, ¿todos los productos?, ¿qué es “más rápido”?
4. El PM responde. La spec de la derecha se reescribe y cita el choque.
5. El PM aprieta **Cerrar spec**. Sale el documento, con “no cerraría” si aún faltan huecos.

## Decisiones

| Tema | Qué quedó |
|------|-----------|
| Oficio | Convergencia. Preguntar hasta que el ticket se pueda commitear. |
| Analogía | `grill-with-docs` (Matt Pocock) contra docs de la org, no contra un repo. No es un wrap de `grill-me`. |
| Cierre | El humano manda. “Cerrar” entrega aunque falte. El sistema recomienda, no veta. |
| Cada turno | Spec viva + 3 preguntas. El writer corre siempre. |
| Preguntas | Top 3 (choque con regla > hueco > texto vago). El resto va en Estado, no a la cara. |
| Web | No. Sin Tavily. |
| Grafo | Supervisor + retriever + intake + writer. El impacto es una cuenta, no un agente. |
| Orden de un turno | Sin docs → retriever. Con docs y sin preguntas de este turno → intake. Con ambos, o “cerrar” → writer. |
| UI | Streamlit: chat izquierda, spec derecha, botón Cerrar spec. |
| Org | Marketplace Andes. Nada interno de MELI. |
| Vector store | Chroma local por default. Pinecone opcional por env. |
| LLM | Factory de la P6. Solo `gemini` (Vertex/ADC) u `openrouter`. Nada de OpenAI/Anthropic directo. |
| Errores | Taxonomía de LangGraph: retry / loop al supervisor / pause en la API / `error_handler` / bubble up. Sin `interrupt()` en este corte. |

## Fuera de alcance

- Búsqueda web.
- Agente de impacto.
- Listar conversaciones viejas.
- CI/CD, cloud.
- Las otras fases de AI-DLC (construir, deploy, incidente).
- Corpus real de MELI, aunque esté anonimizado.

## Grafo

Misma forma que la P6. El supervisor no busca ni escribe: elige el próximo. Una rúbrica dura le corrige si quiere ir al writer sin docs, o si se pasa del tope de visitas del turno. “Cerrar” es una marca en el mensaje, no un agente.

```
mensaje del PM
    → supervisor
         ├─ no hay docs de este turno     → retriever → supervisor
         ├─ hay docs, no hay preguntas    → intake    → supervisor
         └─ hay ambos, o dijo cerrar      → writer    → fin del turno
```

**Retriever.** RAG híbrido (BM25 + embeddings + RRF), patrón P4. Devuelve citas con `document_id`.

**Intake.** Corre las cuentas (texto vago, huecos, fanout de servicios). Arma la lista larga de preguntas y deja 3 arriba. No inventa el ranking: lo calcula.

**Writer.** Reescribe todas las cajas de la spec. Si la marca es cerrar, no agrega preguntas nuevas a la cara; deja las abiertas en Estado.

Slots por especialista (como la P6): cada uno pisa lo suyo, no el del otro.

## La spec (lo que ve el PM)

| Caja | Qué va |
|------|--------|
| Pedido | El texto que pegó, sin reescribir. |
| Qué entendimos | 4–5 líneas. |
| Choques | Reglas que se pisan, con `document_id`. |
| Servicios que tocaría | Lista corta, o “no se pudo armar”. |
| Criterios | Dado / cuando / entonces. Vacío al principio. |
| Preguntas (3) | Las de este turno. Vacío si cerró. |
| Estado | “se puede cerrar” / “no cerraría”, un entero (hits de texto vago + huecos vacíos), y por qué. |

Si no entra en una caja, no va.

## UI

Streamlit habla con la API. El grafo no vive solo adentro de Streamlit.

- Izquierda: chat. Primer mensaje = ticket. Después = respuestas. Botón **Cerrar spec**.
- Derecha: la hoja, las cajas de arriba. Se refresca en cada turno.
- Un id de conversación. Si volvés con el mismo id, sigue (Checkpointer SQLite, patrón P5).
- Phoenix aparte, no embebido.

## API

Tres llamadas. Pydantic en entradas y salidas.

| Acción | Contrato |
|--------|----------|
| Empezar | Ticket → `thread_id` + spec |
| Seguir | `thread_id` + mensaje → spec |
| Cerrar | `thread_id` + marca cerrar → spec final |

Id inexistente → error claro, no 500. Swagger documenta estas tres. Nada más en el primer corte.

## Cuentas (fuera del LLM)

- **Vaguedad:** palabras tipo “más rápido”, “tipo Amazon”, “mejorar”; cuantificadores sin número; pasiva sin actor.
- **Huecos:** actor, acción, resultado visible, criterio medible, fuera de alcance, dependencias.
- **Fanout:** si el texto o las docs nombran un servicio del catálogo, se anota. No lo adivina el modelo.
- **Ranking de preguntas:** choque con regla > hueco > texto vago.

Detección de duplicados contra specs viejas: queda para un v2. En este corte no.

## Corpus — Marketplace Andes (11 docs)

### Reglas

1. El precio final se cierra en el carrito. No se saltea. ← choque de la demo
2. Una promo solo vale si se aplicó en el carrito, no después.
3. En fechas pico (Cyber Monday) los cambios de checkout van por flag, no quedan prendidos para siempre.
4. La tarjeta guardada no se usa sin un paso extra de confirmación.
5. El costo y la dirección de envío se confirman después del carrito. “Comprar ahora” no inventa un envío default.
6. La unidad se reserva cuando entra al carrito. Sin carrito no hay reserva.
7. El plazo de devolución arranca cuando se entrega el paquete, no cuando se hace click.

### Cómo se habla

8. Glosario: *comprar ahora*, *carrito*, *Cyber Monday*, *buyer*, *orden*.

### Sistemas

9. Catálogo: `cart-service`, `checkout-api`, `pricing-engine`, `promotions-service`, `payments-vault`, `shipping-service`, `inventory-service`, `returns-service`.

### Specs viejas

10. Checkout actual: carrito → envío → pago → confirmación.
11. Banner de Cyber Monday del año pasado: se prendió con flag, no fue un cambio permanente.

Nada de estos textos es de MELI.

## Errores

Según [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph#handle-errors-appropriately): cada tipo lo arregla alguien distinto. No se mezclan.

| Tipo | Quién lo arregla | En este sistema |
|------|------------------|-----------------|
| Transitorio (red, rate limit) | El runtime | `RetryPolicy` en retriever y en los nodos que llaman al modelo (3 intentos). El PM no lo ve. |
| El modelo puede corregir (parseo, tool) | El modelo | Se guarda el error en el estado y el supervisor elige de nuevo. Una sola vuelta. |
| Falta dato del humano | El PM | Las 3 preguntas. No es un crash: el turno termina, la API espera el próximo mensaje. Eso es el pause; no usamos `interrupt()` adentro del grafo (Streamlit + un request por turno). |
| Falló después de los reintentos | El grafo | `error_handler` (LangGraph ≥ 1.2, ya en la P6): anota el paso, manda al writer, la spec queda como en el turno anterior. |
| Inesperado (bug) | Nosotros | No se traga. Sube. La API responde error; Phoenix muestra la traza. |
| `thread_id` inventado | La API | “no existe”, el grafo no corre. |
| Cerrar con huecos | Nadie: no es error | Estado = “no cerraría”. |
| Demasiadas visitas en un turno | La rúbrica | Corta y el writer entrega con lo que haya. |

Se porta de la P6: `RetryPolicy` + `error_handler` + rúbrica que impide el loop. No se cachea un `except Exception` genérico en los nodos.

## Tests

Sin prender el modelo:

- Texto de Cyber Monday → las 3 preguntas de la demo arriba.
- Supervisor: sin docs no va al writer; “cerrar” sí.
- API: empezar / seguir / cerrar, e id inexistente.
- Búsqueda, golden set chico (4–5 preguntas). “¿Se puede saltear el carrito?” → regla 1.

Con modelo, a mano: 5 corridas del ticket de Cyber Monday, trazas visibles en Phoenix o LangSmith.

## Stack que se porta

| De | Qué |
|----|-----|
| P4 | Híbrido BM25 + embeddings + RRF, golden set, Chroma default / Pinecone por env |
| P5 | `thread_id`, SqliteSaver |
| P6 | Supervisor, rúbrica dura, tope de visitas, slots por especialista, `clients/factory.py` (`build_chat_model` / `build_role_models`) |
| Nuevo | Intake + cuentas, writer por turno, FastAPI, Streamlit, corpus Andes |

LLM: `LLM_PROVIDER=gemini` con `GOOGLE_GENAI_USE_VERTEXAI=true` (default, mismo wiring Vertex que P5/P6) **o** `LLM_PROVIDER=openrouter`. La factory no acepta `openai` ni `anthropic`. Roles que llaman al modelo: `supervisor` y `writer`. Retriever e intake no. En OpenRouter, un SKU por rol (supervisor: Nemotron structured output; writer: el de razonamiento, el que era `analyst` en la P6).

Python 3.12+. Un comando para levantar (Compose o `run.sh`) en la implementación, no en este doc.

Diagramas del README (grafo, flujo de un turno): skill **Archify** en este repo (`spec-refinery/.agents/skills/archify`), no global.

## Checklist de review

- [ ] El oficio es preguntar y reescribir spec, no un brief one-shot.
- [ ] Cerrar es del humano; el sistema no veta.
- [ ] Cada turno muestra spec + 3 preguntas.
- [ ] No hay web.
- [ ] El grafo es supervisor + 2 especialistas + writer.
- [ ] Streamlit es piel; la API es el sistema.
- [ ] La demo es el ticket de Cyber Monday vs la regla del carrito.
- [ ] El corpus son 11 docs inventados, incluidas envío / stock / devoluciones.
- [ ] Duplicados quedan afuera de este corte.

## Next step

Review de este archivo. Si está, plan de implementación (todavía no hay código).
