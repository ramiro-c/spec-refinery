"""Interrogator node: grills the PM until the request is implementable.

There is no question catalog and no keyword scoring. The LLM reads the ticket,
the whole conversation and the corpus rules the retriever brought back, then
decides what still has to be settled and how vague the request still is. A
notification ticket and a checkout ticket get different questions because
nothing here is written in advance.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.text import transcript_lines
from catalog import SERVICE_IDS
from schemas import Interrogation, SpecStatus
from state import RefineryState

INTERROGATOR_PROMPT = """You are the technical interrogator of a spec refinery:
the senior engineer who refuses to let a vague requirement reach a sprint.

Your job is to grill the PM until the request is unambiguous, testable, and
does not silently contradict the company rules retrieved from the corpus.

How you work:
- Ground every challenge in the citations you were given. If the request
  contradicts a rule, name the document and make the PM resolve the conflict.
  Do not resolve it for them and do not invent rules that are not cited.
- Ask as many questions as this specific request needs. Do not ration them to
  look polite and do not pad the list to look thorough. If four things are
  unresolved, ask four.
- Never re-ask something the PM already answered; read the whole transcript.
  If an answer was evasive or contradicts an earlier one, say so and push back
  instead of asking the same question again.
- There is no checklist. Attack what is actually missing for THIS request.
- You are negotiating an agreement, not filling a form. When the PM's answers
  make the request implementable, stop asking and say so.

se_puede_cerrar is true only when an engineer could build this and a tester
could write the acceptance criteria without asking you anything else.

Write every question in Rioplatense Spanish (voseo), short and pointed, one
idea per question.
"""


def _snapshot(state: RefineryState) -> str:
    citations = state.get("citations") or []
    if citations:
        rules = "\n".join(
            f"- {c.document_id} — {c.title}: {c.excerpt}".strip() for c in citations
        )
    else:
        rules = "(the retriever found nothing)"
    lines = transcript_lines(state.get("messages") or [])
    conversation = "\n".join(lines) if lines else "(empty)"
    spec = state.get("spec")
    prior = spec.model_dump_json(indent=2) if spec is not None else "(none)"
    return (
        f"Original ticket:\n{state.get('ticket') or '(none)'}\n\n"
        f"Company rules retrieved for this request:\n{rules}\n\n"
        f"Conversation so far:\n{conversation}\n\n"
        f"Spec as it stands:\n{prior}\n\n"
        f"Known service ids (use only these):\n{', '.join(SERVICE_IDS)}"
    )


async def interrogate(state: RefineryState, llm: BaseChatModel) -> dict:
    verdict = await llm.with_structured_output(Interrogation).ainvoke(
        [
            SystemMessage(content=INTERROGATOR_PROMPT),
            HumanMessage(content=_snapshot(state)),
        ]
    )
    preguntas = [q.strip() for q in verdict.preguntas if q.strip()]
    update: dict = {
        "questions": preguntas,
        "assessment": SpecStatus(
            se_puede_cerrar=verdict.se_puede_cerrar,
            vaguedad=verdict.vaguedad,
            razon=verdict.razon,
        ),
        "grilled": True,
        "last_agent": "intake",
        # The questions belong in the transcript: that is how the next round
        # knows what it already asked.
        "messages": [
            AIMessage(
                content="\n".join(preguntas) or verdict.razon,
                name="intake",
            )
        ],
    }
    if verdict.human_wants_close:
        update["close_requested"] = True
    return update


def make_intake_node(llm: BaseChatModel):
    async def intake_node(state: RefineryState) -> dict:
        return await interrogate(state, llm)

    return intake_node
