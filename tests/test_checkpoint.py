"""Persistencia por thread_id con SqliteSaver (sin red)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from checkpoint import create_checkpointer
from graph import build_graph, invoke_config, run_turn
from schemas import Citation
from scoring import CYBER_TICKET


def _supervisor(state):
    from agents.supervisor import apply_rubric

    step = int(state.get("step_count") or 0) + 1
    nxt = apply_rubric(
        citations_empty=not state.get("citations"),
        questions_empty=not state.get("questions"),
        step_count=step,
        proposed="FINISH",
        close_requested=bool(state.get("close_requested")),
        last_error=state.get("last_error") or "",
    )
    return {"next_agent": nxt, "step_count": step, "last_agent": "supervisor"}


def _retriever(state):
    return {
        "citations": [
            Citation(
                document_id="adr-cart-price.md",
                title="El precio se cierra en el carrito",
                excerpt="No se saltea.",
            )
        ],
        "last_agent": "retriever",
    }


def _intake(state):
    from scoring import rank_questions

    qs = rank_questions(state["ticket"], state["citations"])
    return {"questions": qs, "last_agent": "intake"}


def _writer(state):
    spec = state["spec"]
    spec.preguntas = [] if state.get("close_requested") else list(state.get("questions") or [])
    spec.choques = list(state.get("citations") or [])
    spec.estado.razon = "dummy"
    spec.estado.se_puede_cerrar = bool(state.get("close_requested")) and spec.estado.vaguedad == 0
    return {"spec": spec, "last_agent": "writer"}


def test_same_thread_id_persists_checkpoint_state(tmp_path):
    """Dos turnos con el mismo thread_id: el segundo acumula mensajes y ve estado del primero."""
    cp = create_checkpointer(tmp_path / "test-checkpoints.sqlite")
    graph = build_graph(
        supervisor=_supervisor,
        retriever=_retriever,
        intake=_intake,
        writer=_writer,
        checkpointer=cp,
    )
    config = invoke_config("t1")
    assert config == {"configurable": {"thread_id": "t1"}, "recursion_limit": 20}

    _, final1 = run_turn(
        graph,
        CYBER_TICKET,
        [HumanMessage(content="primer turno")],
        thread_id="t1",
    )
    assert final1["citations"][0].document_id == "adr-cart-price.md"
    assert final1["ticket"] == CYBER_TICKET

    snapshot = graph.get_state(config)
    assert snapshot.values["citations"][0].document_id == "adr-cart-price.md"
    assert snapshot.values["ticket"] == CYBER_TICKET
    assert cp.get_tuple(config) is not None

    _, final2 = run_turn(
        graph,
        CYBER_TICKET,
        [HumanMessage(content="segundo turno")],
        thread_id="t1",
    )
    assert len(final2["messages"]) == 2
    assert final2["citations"][0].document_id == "adr-cart-price.md"
    assert final2["ticket"] == CYBER_TICKET
