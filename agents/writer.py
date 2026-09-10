"""Writer node: rewrites the SpecDocument at the end of every turn."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.text import transcript_lines
from catalog import SERVICE_IDS
from schemas import SpecDocument, SpecStatus
from state import RefineryState, empty_spec

WRITER_PROMPT = """You are the spec writer of the refinery.
Rewrite que_entendimos, criterios and servicios from the ticket, the
conversation, the previous spec and the citations. Do not invent rules that
are not in the citations, and pick servicios only from the catalog you are
given. Pedido, choques, preguntas and estado are fixed by the system.
"""

_UNSCORED = SpecStatus(
    se_puede_cerrar=False,
    vaguedad=10,
    razon="Todavía no evalué el pedido.",
)


def _estado(state: RefineryState, spec: SpecDocument) -> SpecStatus:
    """The interrogator owns the verdict; the writer only carries it over.

    On an explicit close the interrogator does not run, so the last verdict
    stored on the thread is the honest one.
    """
    assessment = state.get("assessment")
    if assessment is not None:
        return assessment.model_copy()
    prior = state.get("spec")
    if prior is not None and prior.estado is not None:
        return prior.estado.model_copy()
    return _UNSCORED.model_copy()


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
    spec.servicios = [s for s in spec.servicios if s in SERVICE_IDS]
    spec.estado = _estado(state, spec)

    return {
        "spec": spec,
        "messages": [AIMessage(content="Spec updated", name="writer")],
        "last_agent": "writer",
    }


def make_writer_node(llm: BaseChatModel):
    async def writer_node(state: RefineryState) -> dict:
        return await writer_turn(state, llm)

    return writer_node
