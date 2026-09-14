"""Dummy nodes for QA / Docker without an LLM or embeddings.

These are deliberately fixed: they exist so the API can be exercised without
credentials. The real graph decides everything with the LLM.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from schemas import Citation, DocumentAssessment, SpecStatus
from state import RefineryState

FAKE_QUESTIONS = [
    "¿«Comprar ahora» saltea el carrito de verdad o solo acorta la pantalla?",
    "¿Vale para todos los productos o solo algunos?",
    "¿Qué es «más rápido»: menos clicks, menos segundos, más conversión?",
]


def fake_supervisor(state: RefineryState) -> dict:
    from agents.supervisor import apply_rubric

    step = int(state.get("step_count") or 0) + 1
    nxt = apply_rubric(
        citations_empty=not state.get("citations"),
        grilled=bool(state.get("grilled")),
        step_count=step,
        proposed="FINISH",
        close_requested=bool(state.get("close_requested")),
        last_error=state.get("last_error") or "",
    )
    return {"next_agent": nxt, "step_count": step, "last_agent": "supervisor"}


def fake_retriever(state: RefineryState) -> dict:
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


def fake_intake(state: RefineryState) -> dict:
    return {
        "questions": list(FAKE_QUESTIONS),
        "assessment": SpecStatus(
            se_puede_cerrar=False,
            vaguedad=5,
            razon="grafo fake (QA, sin LLM)",
        ),
        "grilled": True,
        "clasificaciones": [
            DocumentAssessment(document_id="adr-cart-price.md", tipo="choque")
        ],
        "messages": [AIMessage(content="\n".join(FAKE_QUESTIONS), name="intake")],
        "last_agent": "intake",
    }


def fake_writer(state: RefineryState) -> dict:
    from agents.writer import _is_frozen, resolve_choques_y_contexto

    spec = state["spec"]
    ticket = state.get("ticket") or ""
    close_requested = bool(state.get("close_requested"))
    spec.pedido = ticket
    spec.que_entendimos = (
        "Checkout «comprar ahora» que no pasa por el carrito (demo QA)."
        if ticket
        else spec.que_entendimos
    )
    spec.preguntas = [] if close_requested else list(state.get("questions") or [])
    spec.choques, spec.contexto = resolve_choques_y_contexto(state)
    assessment = state.get("assessment")
    if assessment is not None:
        spec.estado = assessment.model_copy()
    spec.estado.razon = "grafo fake (QA, sin LLM)"
    if close_requested:
        # QA closes the thread and expects a clean, closable document.
        spec.estado.se_puede_cerrar = True
        spec.estado.vaguedad = 0
    spec.cerrada = _is_frozen(state)
    return {"spec": spec, "last_agent": "writer"}
