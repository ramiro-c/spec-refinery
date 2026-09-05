"""UI Streamlit: chat a la izquierda, spec viva a la derecha."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("SPEC_REFINERY_API", "http://127.0.0.1:8000").rstrip("/")
HTTP_TIMEOUT = 120.0


def _init_session() -> None:
    """Inicializa el estado de la sesión."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "spec" not in st.session_state:
        st.session_state.spec = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


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
    """Renderiza las 7 cajas del SpecDocument."""
    st.subheader("Pedido")
    st.write(spec.get("pedido") or "—")

    st.subheader("Qué entendimos")
    st.write(spec.get("que_entendimos") or "—")

    st.subheader("Choques")
    choques = spec.get("choques") or []
    if choques:
        for citation in choques:
            _render_citation(citation)
    else:
        st.write("—")

    st.subheader("Servicios que tocaría")
    servicios = spec.get("servicios") or []
    if servicios:
        for servicio in servicios:
            st.markdown(f"- {servicio}")
    else:
        st.write("—")

    st.subheader("Criterios")
    criterios = spec.get("criterios") or []
    if criterios:
        for idx, criterio in enumerate(criterios, start=1):
            st.markdown(f"**{idx}.** Dado {criterio.get('dado', '—')}")
            st.markdown(f"Cuando {criterio.get('cuando', '—')}")
            st.markdown(f"Entonces {criterio.get('entonces', '—')}")
    else:
        st.write("—")

    st.subheader("Preguntas (3)")
    preguntas = spec.get("preguntas") or []
    if preguntas:
        for idx, pregunta in enumerate(preguntas, start=1):
            st.markdown(f"{idx}. {pregunta}")
    else:
        st.write("—")

    st.subheader("Estado")
    estado = spec.get("estado") or {}
    cierre = "se puede cerrar" if estado.get("se_puede_cerrar") else "no cerraría"
    vaguedad = estado.get("vaguedad", 0)
    razon = estado.get("razon") or "—"
    st.markdown(f"**{cierre}** · vaguedad: {vaguedad}")
    st.caption(razon)


def main() -> None:
    """Pantalla principal: dos columnas según la spec."""
    st.set_page_config(page_title="Spec Refinery", layout="wide")
    _init_session()

    st.title("Spec Refinery")
    st.caption(f"API: `{API_URL}`")

    col_chat, col_spec = st.columns([1, 1], gap="large")

    with col_chat:
        st.subheader("Conversación")
        for role, text in st.session_state.messages:
            with st.chat_message(role):
                st.write(text)

        if prompt := st.chat_input("Pegá el ticket o respondé una pregunta"):
            st.session_state.messages.append(("user", prompt))
            try:
                if st.session_state.thread_id is None:
                    payload = _post_start(prompt)
                    st.session_state.thread_id = payload["thread_id"]
                    st.session_state.spec = payload["spec"]
                else:
                    payload = _post_message(st.session_state.thread_id, prompt)
                    st.session_state.spec = payload["spec"]
            except httpx.HTTPError as error:
                st.error(f"Error de API: {error}")
            else:
                st.rerun()

        if st.session_state.thread_id:
            if st.button("Cerrar spec", type="primary"):
                try:
                    payload = _post_close(st.session_state.thread_id)
                    st.session_state.spec = payload["spec"]
                    st.session_state.messages.append(
                        ("assistant", "Spec cerrada. Revisá el panel de la derecha.")
                    )
                except httpx.HTTPError as error:
                    st.error(f"Error al cerrar: {error}")
                else:
                    st.rerun()

    with col_spec:
        st.subheader("Spec viva")
        if st.session_state.spec:
            _render_spec(st.session_state.spec)
        else:
            st.info("Pegá un ticket en el chat para empezar.")


if __name__ == "__main__":
    main()
