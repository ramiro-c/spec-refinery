"""Nodo writer: reescribe SpecDocument al cierre de cada turno."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from schemas import SpecDocument, SpecStatus
from scoring import vaguedad_score
from state import RefineryState, empty_spec

WRITER_PROMPT = """Sos el redactor de specs de la refinería.
Reescribí que_entendimos y criterios según el ticket, mensajes del PM, spec anterior,
citations y preguntas. No inventes reglas que no estén en las citations.
Pedido y preguntas los fija el sistema.
"""


def _message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _human_texts(messages: list) -> list[str]:
    """Todos los mensajes humanos acumulados en el hilo."""
    return [
        _message_text(message)
        for message in messages or []
        if getattr(message, "type", None) == "human"
    ]


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
        human_lines = _human_texts(state.get("messages") or [])
        prior = state.get("spec")
        prior_text = prior.model_dump_json(indent=2) if prior is not None else "(ninguna)"
        pm_block = "\n".join(f"- {line}" for line in human_lines) if human_lines else "(ninguno)"
        draft = llm.with_structured_output(SpecDocument).invoke(
            [
                SystemMessage(content=WRITER_PROMPT),
                HumanMessage(
                    content=(
                        f"Ticket: {ticket}\n"
                        f"Mensajes del PM:\n{pm_block}\n"
                        f"Spec anterior:\n{prior_text}\n"
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
