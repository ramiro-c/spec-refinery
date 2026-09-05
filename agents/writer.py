"""Nodo writer: reescribe SpecDocument al cierre de cada turno."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from schemas import SpecDocument, SpecStatus
from scoring import vaguedad_score
from state import RefineryState, empty_spec

WRITER_PROMPT = """Sos el redactor de specs de la refinería.
Reescribí que_entendimos y criterios según el ticket, citations y preguntas.
No inventes reglas que no estén en las citations. Pedido y preguntas los fija el sistema.
"""


def writer_turn(state: RefineryState, llm: BaseChatModel | None = None) -> dict:
    ticket = state.get("ticket") or ""
    close_requested = bool(state.get("close_requested"))
    citations = state.get("citations") or []
    questions = state.get("questions") or []
    spec = state.get("spec") or empty_spec(ticket)
    vaguedad = vaguedad_score(ticket)

    if llm is None:
        spec.pedido = ticket
        spec.choques = list(citations)
        spec.preguntas = [] if close_requested else list(questions)
        spec.estado = SpecStatus(
            se_puede_cerrar=close_requested and vaguedad == 0,
            vaguedad=vaguedad,
            razon="sin LLM",
        )
    else:
        draft = llm.with_structured_output(SpecDocument).invoke(
            [
                SystemMessage(content=WRITER_PROMPT),
                HumanMessage(
                    content=(
                        f"Ticket: {ticket}\n"
                        f"Citations: {[c.document_id for c in citations]}\n"
                        f"Preguntas: {questions}"
                    )
                ),
            ]
        )
        spec = draft
        spec.pedido = ticket
        spec.choques = list(citations)
        spec.preguntas = [] if close_requested else list(questions)
        spec.estado = SpecStatus(
            se_puede_cerrar=close_requested and vaguedad == 0,
            vaguedad=vaguedad,
            razon="writer",
        )

    return {
        "spec": spec,
        "messages": [AIMessage(content="Spec actualizada", name="writer")],
        "last_agent": "writer",
    }


def make_writer_node(llm: BaseChatModel | None = None):
    def writer_node(state: RefineryState) -> dict:
        return writer_turn(state, llm)

    return writer_node
