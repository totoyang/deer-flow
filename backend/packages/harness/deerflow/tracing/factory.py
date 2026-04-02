from __future__ import annotations

from typing import Any

from deerflow.config import (
    get_enabled_tracing_providers,
    get_tracing_config,
    validate_enabled_tracing_providers,
)


def _create_langsmith_tracer(config) -> Any:
    from langchain_core.tracers.langchain import LangChainTracer

    return LangChainTracer(project_name=config.project)


def _create_langfuse_handler(config) -> Any:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

    # langfuse>=4 initializes project-specific credentials through the client
    # singleton; the LangChain callback then attaches to that configured client.
    Langfuse(
        secret_key=config.secret_key,
        public_key=config.public_key,
        host=config.host,
    )
    return LangfuseCallbackHandler(public_key=config.public_key)


def build_tracing_callbacks() -> list[Any]:
    """Build callbacks for all explicitly enabled tracing providers."""
    validate_enabled_tracing_providers()
    enabled_providers = get_enabled_tracing_providers()
    if not enabled_providers:
        return []

    tracing_config = get_tracing_config()
    callbacks: list[Any] = []

    for provider in enabled_providers:
        if provider == "langsmith":
            try:
                callbacks.append(_create_langsmith_tracer(tracing_config.langsmith))
            except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
                raise RuntimeError(f"LangSmith tracing initialization failed: {exc}") from exc
        elif provider == "langfuse":
            try:
                callbacks.append(_create_langfuse_handler(tracing_config.langfuse))
            except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
                raise RuntimeError(f"Langfuse tracing initialization failed: {exc}") from exc

    return callbacks


def configure_runnable_tracing(
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    *,
    attach_callbacks: bool = True,
) -> dict[str, Any]:
    """Attach tracing metadata and, optionally, callbacks to runnable config."""
    if attach_callbacks:
        callbacks = build_tracing_callbacks()
        if callbacks:
            existing_callbacks = list(config.get("callbacks") or [])
            config["callbacks"] = [*existing_callbacks, *callbacks]

    if metadata:
        existing_metadata = dict(config.get("metadata") or {})
        config["metadata"] = {**existing_metadata, **metadata}

    return config


def bind_runnable_tracing(runnable: Any, metadata: dict[str, Any] | None = None) -> Any:
    """Bind tracing callbacks to a runnable before stream/astream invocation.

    Langfuse's LangChain handler does not reliably emit traces when callbacks
    are only supplied via ``astream(..., config=...)``. Binding callbacks on the
    runnable itself via ``with_config`` preserves top-level tracing for both
    sync and async execution paths.
    """
    if getattr(runnable, "_deerflow_tracing_bound", False):
        return runnable

    callbacks = build_tracing_callbacks()
    if not callbacks:
        return runnable

    config: dict[str, Any] = {"callbacks": callbacks}
    if metadata:
        config["metadata"] = metadata

    bound_runnable = runnable.with_config(config)
    setattr(bound_runnable, "_deerflow_tracing_bound", True)
    return bound_runnable
