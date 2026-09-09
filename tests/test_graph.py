"""Turn flow with the dummy nodes from agents.fakes."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from scoring import CYBER_TICKET
from agents.fakes import fake_intake, fake_retriever, fake_supervisor, fake_writer
from graph import build_graph, run_turn


def _graph():
    return build_graph(
        supervisor=fake_supervisor,
        retriever=fake_retriever,
        intake=fake_intake,
        writer=fake_writer,
    )


async def test_open_turn_visits_retriever_intake_writer():
    g = _graph()
    hops, final = await run_turn(g, CYBER_TICKET, [HumanMessage(content=CYBER_TICKET)])
    assert hops == ["supervisor", "retriever", "supervisor", "intake", "supervisor", "writer"]
    assert final["spec"].choques[0].document_id == "adr-cart-price.md"
    assert len(final["spec"].preguntas) == 3


async def test_close_skips_to_writer():
    g = _graph()
    hops, final = await run_turn(
        g,
        CYBER_TICKET,
        [HumanMessage(content="cerrá")],
        close_requested=True,
    )
    assert "retriever" not in hops
    assert hops[-1] == "writer"
    assert final["spec"].preguntas == []
