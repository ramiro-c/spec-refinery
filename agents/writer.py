"""Writer node: rewrites the SpecDocument at the end of every turn."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.text import human_texts
from schemas import SpecDocument, SpecStatus
from scoring import vaguedad_score
from state import RefineryState, empty_spec

WRITER_PROMPT = """You are the spec writer of the refinery.
Rewrite que_entendimos and criterios according to the ticket, PM messages,
previous spec, citations and questions. Do not invent rules that are not in
the citations. Pedido and questions are fixed by the system.
"""


async def writer_turn(state: RefineryState, llm: BaseChatModel) -> dict:
    ticket = state.get("ticket") or ""
    close_requested = bool(state.get("close_requested"))
    citations = state.get("citations") or []
    questions = state.get("questions") or []
    spec = state.get("spec") or empty_spec(ticket)
    vaguedad = vaguedad_score(ticket)

    human_lines = human_texts(state.get("messages") or [])
    prior = state.get("spec")
    prior_text = prior.model_dump_json(indent=2) if prior is not None else "(none)"
    pm_block = "\n".join(f"- {line}" for line in human_lines) if human_lines else "(none)"
    draft = await llm.with_structured_output(SpecDocument).ainvoke(
        [
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(
                content=(
                    f"Ticket: {ticket}\n"
                    f"PM messages:\n{pm_block}\n"
                    f"Previous spec:\n{prior_text}\n"
                    f"Citations: {[c.document_id for c in citations]}\n"
                    f"Questions: {questions}"
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
        "messages": [AIMessage(content="Spec updated", name="writer")],
        "last_agent": "writer",
    }


def make_writer_node(llm: BaseChatModel):
    async def writer_node(state: RefineryState) -> dict:
        return await writer_turn(state, llm)

    return writer_node
