"""Turn flow with the dummy nodes from agents.fakes."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from demo import CYBER_TICKET
from agents.fakes import fake_intake, fake_retriever, fake_supervisor, fake_writer
from graph import build_graph, run_turn


def _graph():
    return build_graph(
        supervisor=fake_supervisor,
        retriever=fake_retriever,
        intake=fake_intake,
        writer=fake_writer,
    )


def _persistent_graph():
    """A thread persists across turns only with a checkpointer attached."""
    return build_graph(
        supervisor=fake_supervisor,
        retriever=fake_retriever,
        intake=fake_intake,
        writer=fake_writer,
        checkpointer=InMemorySaver(),
    )


async def test_open_turn_visits_retriever_intake_writer():
    g = _graph()
    hops, final = await run_turn(g, CYBER_TICKET, [HumanMessage(content=CYBER_TICKET)])
    assert hops == ["supervisor", "retriever", "supervisor", "intake", "supervisor", "writer"]
    assert final["spec"].clashes[0].document_id == "adr-cart-price.md"
    assert len(final["spec"].questions) == 3


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
    assert final["spec"].questions == []


async def test_turn_visits_each_node_at_most_once():
    """A turn is retriever -> intake -> writer; no node repeats."""
    g = _graph()
    hops, _ = await run_turn(g, CYBER_TICKET, [HumanMessage(content=CYBER_TICKET)])
    assert hops.count("retriever") == 1
    assert hops.count("intake") == 1
    assert hops.count("writer") == 1


async def test_closed_thread_stays_closed_after_a_later_message():
    """Closing is sticky: a later continuation turn must not revive the spec."""
    g = _persistent_graph()
    thread = "sticky-close"

    _, opened = await run_turn(
        g, CYBER_TICKET, [HumanMessage(content=CYBER_TICKET)], thread_id=thread
    )
    assert opened["spec"].closed is False

    _, closed = await run_turn(
        g,
        CYBER_TICKET,
        [HumanMessage(content="cerrá")],
        thread_id=thread,
        close_requested=True,
    )
    assert closed["spec"].closed is True

    _, after = await run_turn(
        g, CYBER_TICKET, [HumanMessage(content="otra respuesta")], thread_id=thread
    )
    assert after["spec"].closed is True


async def test_open_thread_stays_open_after_a_later_message():
    """The sticky-close fix must not freeze a thread that was never closed."""
    g = _persistent_graph()
    thread = "open-continuation"

    _, opened = await run_turn(
        g, CYBER_TICKET, [HumanMessage(content=CYBER_TICKET)], thread_id=thread
    )
    assert opened["spec"].closed is False

    _, after = await run_turn(
        g, CYBER_TICKET, [HumanMessage(content="para todo, no medimos")], thread_id=thread
    )
    assert after["spec"].closed is False


async def test_close_keeps_accumulated_clashes_and_context():
    """Closing retrieves nothing, yet the panel keeps real clashes and context."""
    g = _persistent_graph()
    thread = "close-keeps-clashes"

    _, opened = await run_turn(
        g, CYBER_TICKET, [HumanMessage(content=CYBER_TICKET)], thread_id=thread
    )
    assert [c.document_id for c in opened["spec"].clashes] == ["adr-cart-price.md"]
    assert [c.document_id for c in opened["spec"].context] == ["adr-cart-price.md"]

    _, closed = await run_turn(
        g,
        CYBER_TICKET,
        [HumanMessage(content="cerrá")],
        thread_id=thread,
        close_requested=True,
    )
    assert closed["spec"].closed is True
    assert [c.document_id for c in closed["spec"].clashes] == ["adr-cart-price.md"]
    assert [c.document_id for c in closed["spec"].context] == ["adr-cart-price.md"]
