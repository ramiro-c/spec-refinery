"""Writer node: rewrites the SpecDocument at the end of every turn."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.text import transcript_lines
from catalog import SERVICE_IDS
from schemas import Decision, SpecDocument, SpecStatus
from state import RefineryState, empty_spec

WRITER_PROMPT = """You are the spec writer of the refinery.
Rewrite que_entendimos, criterios and servicios from the ticket, the
conversation, the previous spec and the citations. Do not invent rules that
are not in the citations, and pick servicios only from the catalog you are
given. Reflect the decisions the PM already took, including rules they chose
to change. Pedido, choques, decisiones, preguntas and estado are fixed by the
system.
"""

_UNSCORED = SpecStatus(
    se_puede_cerrar=False,
    vaguedad=10,
    razon="Todavía no evalué el pedido.",
)


def _merge_decisions(state: RefineryState) -> list[Decision]:
    """Decisions accumulate over the thread; the last word on a topic wins."""
    prior = state.get("spec")
    merged: list[Decision] = list(prior.decisiones) if prior is not None else []
    for fresh in state.get("decisiones") or []:
        key = fresh.tema.strip().lower()
        for index, existing in enumerate(merged):
            if existing.tema.strip().lower() == key:
                merged[index] = fresh
                break
        else:
            merged.append(fresh)
    return merged


def _estado(state: RefineryState, open_questions: int) -> SpecStatus:
    """The interrogator owns the verdict; the writer only carries it over.

    On an explicit close the interrogator does not run, so the last verdict
    stored on the thread is the honest one.
    """
    assessment = state.get("assessment")
    if assessment is None:
        prior = state.get("spec")
        if prior is not None and prior.estado is not None:
            assessment = prior.estado
    estado = (assessment or _UNSCORED).model_copy()
    if state.get("close_requested"):
        # Closing is the human's call. The spec closes; the status stays
        # honest about what was left open instead of refusing.
        estado.razon = (
            "Cerrada a pedido tuyo, sin puntos abiertos."
            if estado.se_puede_cerrar
            else (
                f"Cerrada a pedido tuyo con {open_questions} punto(s) sin "
                f"resolver · vaguedad {estado.vaguedad}."
            )
        )
    return estado


async def writer_turn(state: RefineryState, llm: BaseChatModel) -> dict:
    ticket = state.get("ticket") or ""
    close_requested = bool(state.get("close_requested"))
    citations = state.get("citations") or []
    questions = state.get("questions") or []
    spec = state.get("spec") or empty_spec(ticket)

    prior = state.get("spec")
    prior_text = prior.model_dump_json(indent=2) if prior is not None else "(none)"
    lines = transcript_lines(state.get("messages") or [])
    conversation = "\n".join(lines) if lines else "(none)"
    draft = await llm.with_structured_output(SpecDocument).ainvoke(
        [
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(
                content=(
                    f"Ticket: {ticket}\n"
                    f"Conversation:\n{conversation}\n"
                    f"Previous spec:\n{prior_text}\n"
                    f"Citations: {[c.document_id for c in citations]}\n"
                    f"Decisions already taken: "
                    f"{[f'{d.tema}: {d.decision}' for d in _merge_decisions(state)]}\n"
                    f"Open questions: {questions}\n"
                    f"Service catalog: {', '.join(SERVICE_IDS)}"
                )
            ),
        ]
    )
    spec = draft
    spec.pedido = ticket
    spec.choques = list(citations)
    spec.preguntas = [] if close_requested else list(questions)
    spec.decisiones = _merge_decisions(state)
    spec.servicios = [s for s in spec.servicios if s in SERVICE_IDS]
    spec.estado = _estado(state, len(questions))

    return {
        "spec": spec,
        "messages": [AIMessage(content="Spec updated", name="writer")],
        "last_agent": "writer",
    }


def make_writer_node(llm: BaseChatModel):
    async def writer_node(state: RefineryState) -> dict:
        return await writer_turn(state, llm)

    return writer_node
