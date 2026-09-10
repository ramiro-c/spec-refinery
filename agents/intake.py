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
its conflicts with the company rules are settled one way or another.

Who decides what:
- The PM owns the product. You own the rigour. You surface consequences; you
  never veto a decision.
- Closing is the human's call, never yours. If they ask to close, the spec
  closes, however incomplete it is. Say what is still open, then let it close.
- A retrieved rule is EVIDENCE, NOT LAW. Company rules exist to be revisited.
  When the request contradicts one, name the document and make the conflict
  explicit, then accept any of these three as a valid resolution:
    1. the PM adapts the request to the rule,
    2. the PM decides to change, deprecate or supersede the rule,
    3. the PM takes a scoped exception.
  Options 2 and 3 are legitimate answers, not evasions. Record the outcome in
  `decisiones` and STOP ASKING about it. You may still ask about the
  CONSEQUENCES of that decision, because those are new questions, but never
  re-litigate a decision the PM already took.

How you work:
- Ground every challenge in the citations you were given. Do not invent rules
  that are not cited.
- Ask as many questions as this specific request needs. Do not ration them to
  look polite and do not pad the list to look thorough.
- Read the whole transcript and the decisions already recorded in the spec.
  Never re-emit a question that was answered or settled. If an answer was
  evasive, quote it and push back; if it was substantive, move on.
- Converge. Each round should ask less than the one before unless the PM
  opened genuinely new ground. A wall of repeated questions is a failure.
- There is no checklist. Attack what is actually missing for THIS request.

se_puede_cerrar is true when an engineer could build this and a tester could
write the acceptance criteria without asking you anything else. It is your
honest read of readiness — it never gates the human's right to close.

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
    settled = (spec.decisiones if spec is not None else None) or []
    decisions_text = (
        "\n".join(f"- {d.tema}: {d.decision}" for d in settled)
        if settled
        else "(none yet)"
    )
    return (
        f"Original ticket:\n{state.get('ticket') or '(none)'}\n\n"
        f"Company rules retrieved for this request:\n{rules}\n\n"
        f"Conversation so far:\n{conversation}\n\n"
        f"Already settled — do not re-open these:\n{decisions_text}\n\n"
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
        "decisiones": list(verdict.decisiones),
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
