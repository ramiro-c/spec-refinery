"""Supervisor: router + rúbrica. No busca ni escribe; solo decide el próximo nodo.

La decisión del LLM es un ``Literal`` (nombres de nodo). ``apply_rubric`` la
corrige si viola las reglas duras: sin citations no hay FINISH, y ``MAX_STEPS``
corta el bucle.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import MAX_STEPS
from state import NextAgent, RefineryState

SUPERVISOR_PROMPT = """Sos el supervisor de una refinería de specs.
No busques en el corpus ni redactes la spec. Elegí el próximo agente.

Rúbrica (en orden, la primera que aplique gana):
1. Si citations está vacío → retriever.
2. Si hay citations y questions de este turno vacío → intake antes de FINISH.
3. Si el humano pidió cerrar (close_requested) → FINISH.
4. Nunca elijas un agente que no exista. Opciones: retriever, intake, FINISH.

Respondé solo con next_agent y una rationale corta (una frase).
"""


class SupervisorDecision(BaseModel):
    next_agent: NextAgent = Field(
        description="Nodo siguiente: retriever, intake o FINISH."
    )
    rationale: str = Field(description="Por qué esa elección, una frase.")


def apply_rubric(
    *,
    citations_empty: bool,
    questions_empty: bool,
    step_count: int,
    proposed: NextAgent,
    close_requested: bool,
    last_error: str = "",
) -> NextAgent:
    """Reglas duras encima del LLM. El grafo nunca ve un next_agent ilegal."""
    if last_error.strip():
        return "FINISH"
    if step_count >= MAX_STEPS:
        return "FINISH"
    if close_requested:
        return "FINISH"
    if citations_empty:
        return "retriever"
    if questions_empty and proposed == "FINISH":
        return "intake"
    if proposed in ("retriever", "intake", "FINISH"):
        return proposed
    return "retriever"


def _message_text(message: Any) -> str:
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


def _user_query(messages: list) -> str:
    """Primera pregunta humana: la consulta original, no el ruido del grafo."""
    for message in messages:
        if getattr(message, "type", None) == "human":
            return _message_text(message)
    if messages:
        return _message_text(messages[0])
    return ""


def _snapshot(state: RefineryState) -> str:
    citations = state.get("citations") or []
    questions = state.get("questions") or []
    citations_text = (
        "(vacío)"
        if not citations
        else "\n".join(f"- {c.document_id}: {c.title or c.excerpt[:80]}" for c in citations)
    )
    questions_text = (
        "(vacío)" if not questions else "\n".join(f"- {q}" for q in questions)
    )
    last_agent = state.get("last_agent") or "(nadie)"
    step_count = int(state.get("step_count") or 0)
    last_error = (state.get("last_error") or "").strip() or "(ninguno)"
    close_requested = bool(state.get("close_requested"))
    ticket = state.get("ticket") or "(sin ticket)"
    return (
        f"Ticket: {ticket}\n"
        f"Consulta: {_user_query(state.get('messages') or [])}\n"
        f"last_agent: {last_agent}\n"
        f"step_count: {step_count}/{MAX_STEPS}\n"
        f"last_error: {last_error}\n"
        f"close_requested: {close_requested}\n\n"
        f"citations:\n{citations_text}\n\n"
        f"questions:\n{questions_text}"
    )


def supervisor_turn(state: RefineryState, llm: BaseChatModel | None = None) -> dict:
    step_count = int(state.get("step_count") or 0) + 1
    last_error = (state.get("last_error") or "").strip()
    citations = state.get("citations") or []
    questions = state.get("questions") or []
    close_requested = bool(state.get("close_requested"))

    if last_error:
        rationale = f"Hay un last_error: no reintento el mismo nodo. {last_error}"
        next_agent: NextAgent = "FINISH"
    elif step_count >= MAX_STEPS:
        rationale = f"Tope de {MAX_STEPS} pasos: cierro para no loopear."
        next_agent = "FINISH"
    elif llm is None:
        rationale = "Sin LLM: rúbrica determinística."
        next_agent = apply_rubric(
            citations_empty=not citations,
            questions_empty=not questions,
            step_count=step_count,
            proposed="retriever",
            close_requested=close_requested,
            last_error=last_error,
        )
    else:
        decision = llm.with_structured_output(SupervisorDecision).invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=_snapshot({**state, "step_count": step_count})),
            ]
        )
        next_agent = apply_rubric(
            citations_empty=not citations,
            questions_empty=not questions,
            step_count=step_count,
            proposed=decision.next_agent,
            close_requested=close_requested,
            last_error=last_error,
        )
        rationale = decision.rationale
        if next_agent != decision.next_agent:
            rationale = f"{rationale} [rúbrica: {decision.next_agent} → {next_agent}]"

    return {
        "messages": [AIMessage(content=rationale, name="supervisor")],
        "next_agent": next_agent,
        "step_count": step_count,
        "last_agent": "supervisor",
    }


def make_supervisor_node(llm: BaseChatModel):
    def supervisor_node(state: RefineryState) -> dict:
        return supervisor_turn(state, llm)

    return supervisor_node
