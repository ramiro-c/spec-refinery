"""Nodos dummy para QA / Docker sin LLM ni embeddings."""

from __future__ import annotations

from schemas import Citation
from scoring import rank_questions
from state import RefineryState


def fake_supervisor(state: RefineryState) -> dict:
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
    qs = rank_questions(state["ticket"], state.get("citations") or [])
    return {"questions": qs, "last_agent": "intake"}


def fake_writer(state: RefineryState) -> dict:
    spec = state["spec"]
    ticket = state.get("ticket") or ""
    questions = list(state.get("questions") or [])
    citations = list(state.get("citations") or [])
    spec.pedido = ticket
    spec.que_entendimos = (
        "Checkout «comprar ahora» que no pasa por el carrito (demo QA)."
        if ticket
        else spec.que_entendimos
    )
    spec.preguntas = [] if state.get("close_requested") else questions
    spec.choques = citations
    spec.estado.razon = "grafo fake (QA, sin LLM)"
    spec.estado.se_puede_cerrar = bool(state.get("close_requested")) and spec.estado.vaguedad == 0
    return {"spec": spec, "last_agent": "writer"}
