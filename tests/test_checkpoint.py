"""Persistence by thread_id with the async SQLite checkpointer (no network)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agents.fakes import fake_intake, fake_retriever, fake_supervisor, fake_writer
from checkpoint import close_checkpointer, create_checkpointer
from graph import build_graph, invoke_config, run_turn
from scoring import CYBER_TICKET


async def test_same_thread_id_persists_checkpoint_state(tmp_path):
    """Two turns with the same thread_id: the second accumulates messages and sees first-turn state."""
    cp = create_checkpointer(tmp_path / "test-checkpoints.sqlite")
    graph = build_graph(
        supervisor=fake_supervisor,
        retriever=fake_retriever,
        intake=fake_intake,
        writer=fake_writer,
        checkpointer=cp,
    )
    config = invoke_config("t1")
    assert config == {"configurable": {"thread_id": "t1"}, "recursion_limit": 20}

    _, final1 = await run_turn(
        graph,
        CYBER_TICKET,
        [HumanMessage(content="primer turno")],
        thread_id="t1",
    )
    assert final1["citations"][0].document_id == "adr-cart-price.md"
    assert final1["ticket"] == CYBER_TICKET

    snapshot = await graph.aget_state(config)
    assert snapshot.values["citations"][0].document_id == "adr-cart-price.md"
    assert snapshot.values["ticket"] == CYBER_TICKET
    assert await cp.aget_tuple(config) is not None

    _, final2 = await run_turn(
        graph,
        CYBER_TICKET,
        [HumanMessage(content="segundo turno")],
        thread_id="t1",
    )
    assert len(final2["messages"]) == 2
    assert final2["citations"][0].document_id == "adr-cart-price.md"
    assert final2["ticket"] == CYBER_TICKET

    # The aiosqlite worker thread is non-daemon: close it or pytest never exits.
    await close_checkpointer(cp)
