# Spec Refinery — caso de demo (Jira Cyber Monday)

**Fecha:** 2026-09-09
**Estado:** listo para review
**Dónde:** `spec-refinery/` (texto de demo + SKUs OpenRouter; no cambia el grafo)

Un PM pega un ticket tipo Jira, vago a propósito. El sistema choca con la regla del carrito, hace 3 preguntas, el PM pega una respuesta ensayada, y se cierra la spec. Encuadre: el mismo arco de [2026-09-05-spec-refinery-design.md](2026-09-05-spec-refinery-design.md). Acá cambia el texto de la demo, cómo se muestra, y qué modelo escribe la spec.

## Oficio de la demo

Misma colisión: *comprar ahora* vs “el precio se cierra en el carrito”. El ticket suena a Jira (título, tipo, prioridad, labels, contexto, descripción, AC vacío). No es un comentario de Slack.

Enfoque: solo el texto. Un botón pega el ticket. La respuesta del segundo turno está para copiar, no tiene botón.

## Ticket (turno 1)

El id no lleva números. `empty_slots` trata `\d+` como criterio medible; `ANDES-1842` llenaría ese hueco y el primer turno cambiaría.

```
ANDES-CHECKOUT  Checkout más rápido para Cyber Monday

Tipo: Story
Prioridad: High
Labels: cyber-monday, checkout

Contexto
Se viene Cyber Monday y el checkout se siente lento. Pidieron algo tipo Amazon.

Descripción
Queremos un checkout más rápido: que el comprar ahora no pase por el carrito.

Acceptance Criteria
—
```

Huecos que tiene que dejar vacíos (si alguno se llena, la demo se desarma):

| Slot / vaguedad | Palabras que no van en este texto |
|-----------------|-----------------------------------|
| actor | buyer, seller, squad, pm, usuario, actor |
| criterio medible | dígitos, %, segundos, clicks, conversión, p95 |
| alcance | todos, solo, excepto, sku, categoría, 1p, marketplace |
| fuera de alcance | fuera de alcance, no incluye, no tocar |
| dependencias | depende, bloqueado por, requiere |

Sí tiene que decir “más rápido”, “tipo Amazon” y “no pase por el carrito”.

## Respuesta (turno 2)

Se copia de una, después de las 3 preguntas:

```
No: comprar ahora solo acorta la pantalla. El precio se cierra en el carrito igual que ahora.
Vale solo durante la ventana de Cyber Monday, no el resto del año.
Más rápido = menos segundos hasta la confirmación de la orden.
```

Eso responde: no saltea el carrito; solo la ventana del evento; métrica en segundos. El choque queda resuelto (la regla se respeta). Después: **Cerrar spec**.

## Qué tiene que pasar

1. Botón **Usar el ticket de demo** → el Jira entra al chat.
2. Primer turno: cita `adr-cart-price.md` (precio cerrado en el carrito). Tres preguntas: ¿saltea de verdad?, ¿todos los productos o solo algunos?, ¿qué es “más rápido”?
3. Se pega la respuesta de demo.
4. La spec se reescribe. Estado cerrable. Choque citado y resuelto: la pantalla se acorta, el total sigue saliendo del carrito, el cambio vive en la ventana Cyber Monday.
5. **Cerrar spec** entrega el documento.

## Dónde vive

| Pieza | Dónde |
|-------|--------|
| Ticket | `CYBER_TICKET` en `scoring.py`. El botón de la UI pega esa constante. |
| Respuesta | `DEMO_REPLY` en `scoring.py`, al lado del ticket. Sidebar: expander **Respuesta de demo**, para copiar. Sin botón que la envíe. |
| Caption del caso | Sidebar: Cyber Monday, no saltea, solo la ventana, métrica en segundos. |
| Quick path | `README.md` y el design del 2026-09-05, mismo Jira y misma respuesta. |

Los tests que ya usan `CYBER_TICKET` siguen. El texto nuevo tiene que seguir disparando las mismas 3 preguntas.

## LLM (live)

`LLM_PROVIDER=openrouter`. La clave va en `.env` (`OPENROUTER_API_KEY`); no se commitea.

| Rol | Modelo | Por qué |
|-----|--------|---------|
| supervisor | `nvidia/nemotron-3-ultra-550b-a55b:free` | Ruteo corto + `with_structured_output`. Laguna no declara `response_format`. |
| writer | `poolside/laguna-s-2.1:free` | Redacta la spec. |
| writer fallback | el mismo Nemotron del supervisor | Una sola vez, si Laguna falla al parsear `SpecDocument` (400, JSON, validación). Si Nemotron también falla, el error sube: no hay segunda red. |

El supervisor no usa Laguna ni fallback: ya está en Nemotron.

`.env.example` documenta `LLM_PROVIDER=openrouter` y los dos SKUs. Defaults en `OPENROUTER_ROLE_MODELS`. El endpoint `:free` tiene cupo diario; un turno pega varias llamadas (supervisor + writer).

## Fuera de alcance

- Segundo botón que envíe la respuesta.
- Cambiar scoring, ranking, corpus, retriever o la API.
- Cambiar la forma del grafo.
- Test de oro del writer con LLM (el wording varía).
- Otro escenario (promos, stock, envío).
- Fallback en el supervisor.

## Checklist de review

- [ ] El ticket es Jira, no Slack, y no trae dígitos ni las palabras de la tabla.
- [ ] Turno 1: choque con el carrito + las 3 preguntas de siempre.
- [ ] Turno 2: no saltea, solo Cyber Monday, segundos hasta confirmar.
- [ ] Un botón pega el ticket; la respuesta se copia a mano.
- [ ] Writer = Laguna; supervisor = Nemotron; una reintentada del writer a Nemotron si el schema falla.
- [ ] Grafo y corpus no se tocan.
