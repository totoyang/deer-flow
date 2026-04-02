"""Tests for tracing binding behavior in the runtime worker."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from deerflow.runtime.runs import worker as worker_module


class _FakeAgent:
    def __init__(self):
        self.checkpointer = None
        self.store = None
        self.interrupt_before_nodes = None
        self.interrupt_after_nodes = None

    async def astream(self, *args, **kwargs):
        if False:  # pragma: no cover
            yield None


def test_run_agent_binds_tracing_after_runtime_attachments(monkeypatch):
    bridge = AsyncMock()
    run_manager = AsyncMock()
    checkpointer = AsyncMock()
    checkpointer.aget_tuple.return_value = None

    fake_agent = _FakeAgent()

    monkeypatch.setattr(
        worker_module,
        "bind_runnable_tracing",
        lambda agent, metadata: _assert_agent_is_fully_attached(
            agent,
            metadata,
            checkpointer,
            "fake-store",
            ["node-a"],
            ["node-b"],
        ),
    )

    def _sync_agent_factory(config):
        return fake_agent

    record = type(
        "Record",
        (),
        {
            "run_id": "run-1",
            "thread_id": "thread-1",
            "abort_event": type("AbortEvent", (), {"is_set": lambda self: False})(),
            "abort_action": None,
        },
    )()

    asyncio.run(
        worker_module.run_agent(
            bridge=bridge,
            run_manager=run_manager,
            record=record,
            checkpointer=checkpointer,
            store="fake-store",
            agent_factory=_sync_agent_factory,
            graph_input={"messages": []},
            config={},
            interrupt_before=["node-a"],
            interrupt_after=["node-b"],
        )
    )

    assert run_manager.set_status.await_args_list[-1].args == ("run-1", worker_module.RunStatus.success)


def _assert_agent_is_fully_attached(agent, metadata, checkpointer, store, interrupt_before, interrupt_after):
    assert agent.checkpointer is checkpointer
    assert agent.store == store
    assert agent.interrupt_before_nodes == interrupt_before
    assert agent.interrupt_after_nodes == interrupt_after
    assert metadata == {"run_id": "run-1", "thread_id": "thread-1"}
    return agent
