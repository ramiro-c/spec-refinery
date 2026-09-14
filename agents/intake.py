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
from config import MAX_QUESTIONS_PER_ROUND, MAX_ROUNDS
from schemas import Interrogation, SpecStatus
from state import RefineryState

INTERROGATOR_PROMPT_TEMPLATE = """You are the technical interrogator of a spec refinery:
the senior engineer who refuses to let a vague requirement reach a sprint.

Your job is to grill the PM until the request is unambiguous, testable, and
its conflicts with the company rules are settled one way or another.

Who decides what:
- The PM owns the product. You own the rigour. You surface consequences; you
  never veto a decision.
- Closing is the human's call, never yours. If they ask to close, the spec
  closes, however incomplete it is. Say what is still open, then let it close.
- A retrieved rule is EVIDENCE, NOT LAW. Company rules exist to be revisited.
  When the request APPEARS to contradict one, interpret the request's terms
  against the retrieved definitions FIRST (see "How you work"). If the clash is
  still real, name the document and make the conflict explicit, then accept any
  of these three as a valid resolution:
    1. the PM adapts the request to the rule,
    2. the PM decides to change, deprecate or supersede the rule,
    3. the PM takes a scoped exception.
  A fourth valid outcome exists: "no real conflict once interpreted per the glossary".
  Take it ONLY when a retrieved definition covers the term and removes the
  contradiction; state the benign reading in plain words and ask the question
  that disambiguates it. If the definition is silent, partial, or the intent
  still skips a mandatory step, the clash is REAL: name the document and
  resolve it as 1, 2 or 3. Never let the benign reading drop a real clash.
  These are legitimate answers, not evasions. Record the outcome in `decisions`
  and STOP ASKING about it. You may still ask about the CONSEQUENCES.

Your budget (the PM sees it too, so respect it):
- At most {max_questions} questions per round. Pick the ones that actually
  block implementation; the rest can wait for a later round or be dropped.
- At most {max_rounds} rounds. On the final round ask nothing: give your
  verdict and let the spec close with whatever is still open.

How you work:
- Before treating a retrieved rule as contradicted, interpret the request's
  terms against the retrieved definitions (glossary and citations). If a
  definition gives a term a benign meaning, state it and ask what the PM meant.
  If the definition does not settle the intent, ask instead of assuming a benign
  reading or conceding a conflict.
- Declare a type for EVERY retrieved document, one entry each in
  `classifications`: "clash" when the request genuinely contradicts that rule,
  "context" when the document informs the request without being contradicted.
  Use the exact `document_id` you were given; never invent one.
- Ground every challenge in the citations you were given. Do not invent rules
  that are not cited.
- Read the whole transcript and the decisions already recorded in the spec.
  Never re-emit a question that was answered or settled. If an answer was
  evasive, quote it and push back; if it was substantive, move on.
- Converge. Each round should ask less than the one before unless the PM
  opened genuinely new ground. A wall of repeated questions is a failure.
- There is no checklist. Attack what is actually missing for THIS request.

can_close is true when an engineer could build this and a tester could
write the acceptance criteria without asking you anything else. It is your
honest read of readiness — it never gates the human's right to close.

Write every question in Rioplatense Spanish (voseo), short and pointed, one
idea per question.
"""

INTERROGATOR_PROMPT = INTERROGATOR_PROMPT_TEMPLATE.format(
    max_questions=MAX_QUESTIONS_PER_ROUND,
    max_rounds=MAX_ROUNDS,
)


def _snapshot(state: RefineryState, round_number: int) -> str:
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
    settled = (spec.decisions if spec is not None else None) or []
    decisions_text = (
        "\n".join(f"- {d.topic}: {d.decision}" for d in settled)
        if settled
        else "(none yet)"
    )
    budget = f"Round {round_number} of {MAX_ROUNDS}. " + (
        "FINAL ROUND: ask nothing, give your verdict."
        if round_number >= MAX_ROUNDS
        else f"You may ask up to {MAX_QUESTIONS_PER_ROUND} questions."
    )
    return (
        f"{budget}\n\n"
        f"Original ticket:\n{state.get('ticket') or '(none)'}\n\n"
        f"Company rules retrieved for this request:\n{rules}\n\n"
        f"Conversation so far:\n{conversation}\n\n"
        f"Already settled — do not re-open these:\n{decisions_text}\n\n"
        f"Spec as it stands:\n{prior}\n\n"
        f"Known service ids (use only these):\n{', '.join(SERVICE_IDS)}"
    )


async def interrogate(state: RefineryState, llm: BaseChatModel) -> dict:
    round_number = int(state.get("round_count") or 0) + 1
    final_round = round_number >= MAX_ROUNDS
    verdict = await llm.with_structured_output(Interrogation).ainvoke(
        [
            SystemMessage(content=INTERROGATOR_PROMPT),
            HumanMessage(content=_snapshot(state, round_number)),
        ]
    )
    questions = [q.strip() for q in verdict.questions if q.strip()]
    # The budget is a contract with the PM, not a suggestion to the model.
    questions = [] if final_round else questions[:MAX_QUESTIONS_PER_ROUND]
    update: dict = {
        "questions": questions,
        "decisions": list(verdict.decisions),
        "classifications": list(verdict.classifications),
        "assessment": SpecStatus(
            can_close=verdict.can_close,
            vagueness=verdict.vagueness,
            reason=verdict.reason,
        ),
        "grilled": True,
        "round_count": round_number,
        "last_agent": "intake",
        # The questions belong in the transcript: that is how the next round
        # knows what it already asked.
        "messages": [
            AIMessage(
                content="\n".join(questions) or verdict.reason,
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
