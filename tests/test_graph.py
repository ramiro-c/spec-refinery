from langchain_core.messages import HumanMessage
from schemas import Citation
from scoring import CYBER_TICKET
from state import initial_fields
from graph import build_graph, run_turn


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


def test_open_turn_visits_retriever_intake_writer():
    g = build_graph(
        supervisor=_supervisor,
        retriever=_retriever,
        intake=_intake,
        writer=_writer,
    )
    hops, final = run_turn(g, CYBER_TICKET, [HumanMessage(content=CYBER_TICKET)])
    assert hops == ["supervisor", "retriever", "supervisor", "intake", "supervisor", "writer"]
    assert final["spec"].choques[0].document_id == "adr-cart-price.md"
    assert len(final["spec"].preguntas) == 3


def test_close_skips_to_writer():
    g = build_graph(
        supervisor=_supervisor,
        retriever=_retriever,
        intake=_intake,
        writer=_writer,
    )
    hops, final = run_turn(
        g,
        CYBER_TICKET,
        [HumanMessage(content="cerrá")],
        close_requested=True,
    )
    assert "retriever" not in hops
    assert hops[-1] == "writer"
    assert final["spec"].preguntas == []
