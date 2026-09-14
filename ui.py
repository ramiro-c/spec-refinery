"""UI Streamlit: chat a la izquierda, spec viva a la derecha."""

from __future__ import annotations

import os

import httpx
import streamlit as st

from demo import CYBER_TICKET

API_URL = os.getenv("SPEC_REFINERY_API", "http://127.0.0.1:8000").rstrip("/")
HTTP_TIMEOUT = 120.0
PANEL_HEIGHT = 640


def _collision_asked_about(spec: dict, questions: list[str]) -> dict | None:
    """La cita que alguna pregunta de este turno realmente menciona."""
    for citation in spec.get("clashes") or []:
        doc_id = str(citation.get("document_id") or "")
        if doc_id and any(doc_id in question for question in questions):
            return citation
    return None


def _assistant_reply(spec: dict) -> str:
    """Arma el turno del sistema: choque (si hay) + las preguntas de este turno."""
    questions = [
        str(question).strip()
        for question in (spec.get("questions") or [])
        if str(question).strip()
    ]
    if not questions:
        status = spec.get("status") or {}
        if status.get("can_close"):
            return "Listo: no me quedan preguntas. Si te cierra, podés cerrar la spec."
        reason = str(status.get("reason") or "").strip()
        return reason or "No tengo preguntas nuevas. Seguí o cerrá la spec."

    if len(questions) == 1:
        header = "Me queda una pregunta."
    else:
        header = f"Me quedan {len(questions)} preguntas."

    # Only announce the collision when this turn actually asks about it;
    # otherwise the lead advertises a document none of the questions mention.
    collision = _collision_asked_about(spec, questions)
    if collision is not None:
        label = (
            collision.get("title")
            or collision.get("document_id")
            or "una regla de Andes"
        )
        lead = f"Hay un choque con {label}. {header}"
    else:
        lead = f"Necesito aclarar el pedido. {header}"
    numbered = "\n".join(
        f"{idx}. {question}" for idx, question in enumerate(questions, start=1)
    )
    return f"{lead}\n\n{numbered}"


def _closing_reply(spec: dict) -> str:
    """Turno de cierre: la spec queda congelada, con o sin huecos."""
    status = spec.get("status") or {}
    vagueness = status.get("vagueness", 0)
    if status.get("can_close"):
        return "Spec cerrada. No quedaron preguntas abiertas ni huecos."
    return (
        f"Spec cerrada a pedido tuyo, pero queda vaguedad {vagueness}. "
        "Cerrar lo decidís vos; mirá el panel de la derecha antes de pasarla "
        "a desarrollo."
    )


def _spec_is_open(spec: dict | None) -> bool:
    """La spec sigue abierta mientras no esté marcada como cerrada."""
    return not bool((spec or {}).get("closed"))


def _absorb(payload: dict) -> None:
    """Guarda spec y presupuesto de rondas que devolvió el API."""
    st.session_state.spec = payload["spec"]
    st.session_state.round = payload.get("round", st.session_state.round)
    st.session_state.max_rounds = payload.get(
        "max_rounds", st.session_state.max_rounds
    )
    st.session_state.max_questions = payload.get(
        "max_questions_per_round", st.session_state.max_questions
    )


def _queue_prompt(prompt: str) -> None:
    """Guarda el texto y rerun. El API se pega en el próximo ciclo, antes de pintar."""
    st.session_state.pending_prompt = prompt
    st.rerun()


def _execute_prompt(prompt: str) -> None:
    """Manda el texto al API y deja user + assistant en el chat."""
    st.session_state.messages.append(("user", prompt))
    try:
        if st.session_state.thread_id is None:
            payload = _post_start(prompt)
            st.session_state.thread_id = payload["thread_id"]
        else:
            payload = _post_message(st.session_state.thread_id, prompt)
        _absorb(payload)
        st.session_state.messages.append(
            ("assistant", _assistant_reply(st.session_state.spec))
        )
    except httpx.HTTPError as error:
        st.session_state.api_error = f"Error de API: {error}"


def _drain_pending() -> None:
    """Consume prompt o cierre encolados. Corre antes de renderizar el chat."""
    prompt = st.session_state.pending_prompt
    if prompt:
        st.session_state.pending_prompt = None
        with st.spinner("Refinando spec..."):
            _execute_prompt(prompt)
    if st.session_state.pending_close:
        st.session_state.pending_close = False
        st.session_state.messages.append(("user", "Cerrá la spec."))
        try:
            with st.spinner("Cerrando spec..."):
                payload = _post_close(st.session_state.thread_id)
            _absorb(payload)
            st.session_state.messages.append(
                ("assistant", _closing_reply(st.session_state.spec))
            )
        except httpx.HTTPError as error:
            st.session_state.api_error = f"Error al cerrar: {error}"


def _init_session() -> None:
    """Inicializa el estado de la sesión."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "spec" not in st.session_state:
        st.session_state.spec = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "pending_close" not in st.session_state:
        st.session_state.pending_close = False
    if "api_error" not in st.session_state:
        st.session_state.api_error = None
    if "round" not in st.session_state:
        st.session_state.round = 0
    if "max_rounds" not in st.session_state:
        st.session_state.max_rounds = 5
    if "max_questions" not in st.session_state:
        st.session_state.max_questions = 3


def _post_start(ticket: str) -> dict:
    """Primer turno: POST /threads."""
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        response = client.post(f"{API_URL}/threads", json={"ticket": ticket})
        response.raise_for_status()
        return response.json()


def _post_message(thread_id: str, content: str) -> dict:
    """Turnos siguientes: POST /threads/{id}/messages."""
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        response = client.post(
            f"{API_URL}/threads/{thread_id}/messages",
            json={"content": content},
        )
        response.raise_for_status()
        return response.json()


def _post_close(thread_id: str) -> dict:
    """Cierre humano: POST /threads/{id}/close."""
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        response = client.post(f"{API_URL}/threads/{thread_id}/close")
        response.raise_for_status()
        return response.json()


def _render_citation(citation: dict) -> None:
    """Muestra una cita con document_id y extracto."""
    doc_id = citation.get("document_id", "")
    title = citation.get("title", "")
    excerpt = citation.get("excerpt", "")
    header = f"**{doc_id}**"
    if title:
        header = f"{header} — {title}"
    st.markdown(header)
    if excerpt:
        st.caption(excerpt)


def _render_spec(spec: dict) -> None:
    """Renderiza las 8 cajas del SpecDocument."""
    st.subheader("Pedido")
    st.write(spec.get("request") or "—")

    st.subheader("Qué entendimos")
    st.write(spec.get("understanding") or "—")

    st.subheader("Choques reales")
    clashes = spec.get("clashes") or []
    if clashes:
        for citation in clashes:
            _render_citation(citation)
    else:
        st.write("—")

    st.subheader("Contexto recuperado")
    context = spec.get("context") or []
    if context:
        for citation in context:
            _render_citation(citation)
    else:
        st.write("—")

    st.subheader("Decisiones")
    decisions = spec.get("decisions") or []
    if decisions:
        for decision in decisions:
            st.markdown(f"**{decision.get('topic', '—')}** — {decision.get('decision', '')}")
            impact = str(decision.get("impact") or "").strip()
            if impact:
                st.caption(f"Impacto: {impact}")
    else:
        st.write("—")

    st.subheader("Servicios que tocaría")
    services = spec.get("services") or []
    if services:
        for service in services:
            st.markdown(f"- {service}")
    else:
        st.write("—")

    st.subheader("Criterios")
    criteria = spec.get("criteria") or []
    if criteria:
        for idx, criterion in enumerate(criteria, start=1):
            st.markdown(f"**{idx}.** Dado {criterion.get('given', '—')}")
            st.markdown(f"Cuando {criterion.get('when', '—')}")
            st.markdown(f"Entonces {criterion.get('then', '—')}")
    else:
        st.write("—")

    st.subheader("Preguntas abiertas")
    questions = spec.get("questions") or []
    if questions:
        for idx, question in enumerate(questions, start=1):
            st.markdown(f"{idx}. {question}")
    else:
        st.write("—")

    st.subheader("Estado")
    status = spec.get("status") or {}
    closing = "se puede cerrar" if status.get("can_close") else "no cerraría"
    vagueness = status.get("vagueness", 0)
    reason = status.get("reason") or "—"
    st.markdown(f"**{closing}** · vaguedad: {vagueness}")
    st.caption(reason)


def _render_intro() -> None:
    """Contexto para alguien que abre esto por primera vez."""
    st.markdown(
        "**Sos el PM.** Escribí abajo un pedido vago, como el que mandarías "
        "un lunes a la mañana."
    )
    st.markdown(
        "El sistema busca las reglas internas de **Marketplace Andes** con RAG "
        "y te interroga hasta que la spec sea implementable. Si tu pedido "
        "choca con una regla, te lo planta con el documento en la mano — y "
        "podés ganarle: si decidís cambiar esa regla, lo anota como decisión "
        "en vez de seguir preguntando."
    )
    st.markdown(
        "En el panel, **Choques reales** junta sólo las reglas que tu pedido "
        "contradice de verdad; **Contexto recuperado** muestra lo que el RAG "
        "trajo para entender el pedido, sin tratarlo como contradicción."
    )
    st.markdown("**Reglas del juego**")
    st.markdown(
        f"- Hasta **{st.session_state.max_questions} preguntas por ronda**\n"
        f"- Máximo **{st.session_state.max_rounds} rondas**; después la spec "
        "se congela con lo que haya\n"
        "- **Cerrás vos** cuando quieras: el botón o pedíselo al chat"
    )
    st.caption("La spec se va reescribiendo sola en el panel de la derecha.")
    st.markdown(
        "**Si no se te ocurre nada**, el ticket de demo pide *comprar ahora* "
        "sin pasar por el carrito para Cyber Monday — y en Andes el precio se "
        "cierra justamente en el carrito."
    )


def _render_round_meter() -> None:
    """Cuánto presupuesto de interrogatorio queda."""
    round = int(st.session_state.round or 0)
    total = int(st.session_state.max_rounds or 5)
    remaining = max(total - round, 0)
    if remaining:
        detail = f"quedan {remaining}"
    else:
        detail = "spec congelada"
    st.caption(
        f"Ronda **{min(round, total)} de {total}** · {detail} · "
        f"hasta {st.session_state.max_questions} preguntas por ronda"
    )
    st.progress(min(round / total, 1.0) if total else 0.0)


def main() -> None:
    """Pantalla principal: dos columnas según la spec."""
    st.set_page_config(page_title="Spec Refinery", layout="wide")
    _init_session()
    _drain_pending()

    st.title("Spec Refinery")
    if st.session_state.api_error:
        st.error(st.session_state.api_error)
        st.session_state.api_error = None

    col_chat, col_spec = st.columns([1, 1], gap="large")

    with col_chat:
        st.subheader("Conversación")
        if not st.session_state.messages:
            _render_intro()
            if st.button("Usar el ticket de demo", type="secondary"):
                _queue_prompt(CYBER_TICKET)
        else:
            _render_round_meter()
            with st.container(height=PANEL_HEIGHT, border=False):
                for role, text in st.session_state.messages:
                    with st.chat_message(role):
                        st.write(text)

        if st.session_state.thread_id and _spec_is_open(st.session_state.spec):
            st.caption(
                "Para cerrar: tocá el botón o escribilo en el chat "
                "(«cerrá la spec»)."
            )
            if st.button("Cerrar spec", type="primary"):
                st.session_state.pending_close = True
                st.rerun()

    with col_spec:
        st.subheader("Spec viva")
        if st.session_state.spec:
            with st.container(height=PANEL_HEIGHT, border=False):
                _render_spec(st.session_state.spec)
        else:
            st.info(
                "Todavía no hay spec. Mandá un ticket en el chat y se arma acá."
            )

    # Top-level on purpose: inside a column Streamlit renders it inline and it
    # falls below the fold. At top level it stays pinned to the viewport.
    if prompt := st.chat_input("Pegá el ticket o respondé una pregunta"):
        _queue_prompt(prompt)

    st.caption(f"La UI es sólo la piel: el sistema es la API en `{API_URL}`.")


if __name__ == "__main__":
    main()
