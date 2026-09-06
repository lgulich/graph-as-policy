"""Lifecycle manager for out-of-process tool bundle servers.

Parallel to :mod:`gap.runtime.policy_manager` — the policy manager owns
long-running learned-policy websocket servers; this one owns long-running
``stdio-msgpack`` tool servers (one subprocess per bundle that declares
``gap.serving.protocol: stdio-msgpack`` in its SKILL.md). The launcher
calls :meth:`boot_all` after policies are up; on workflow exit the caller
MUST invoke :meth:`shutdown_all` from a ``finally``.

A booted bundle's tools register with the gap-runtime ToolRegistry via
:meth:`gap_core.tools.ToolRegistry.register_rpc` so the executor sees the
RPC-routed tools side-by-side with the in-process @tool catalog. The LLM
catalog and the workflow validator are unaware of the transport — they
just see another ``runtime`` tool.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ToolBundleStartupError(RuntimeError):
    """A bundle's tool server failed to come up (or its catalog handshake
    failed). Surfaces the bundle name + the underlying error."""


@dataclass
class _Managed:
    bundle: str
    client: Any  # ToolClient
    tool_names: list[str]


def _schema_from_catalog(entry: Any) -> Any:
    from gap_core.tools.schema import FieldInfo, UnitSchema

    return UnitSchema(
        name=entry.name,
        description=entry.summary,
        inputs={
            field_name: FieldInfo(
                name=str(raw.get("name", field_name)),
                python_type=Any,
                type_str=str(raw.get("type_str", "Any")),
                required=bool(raw.get("required", False)),
                default=raw.get("default"),
            )
            for field_name, raw in entry.schema_inputs.items()
        },
        outputs={
            field_name: FieldInfo(
                name=str(raw.get("name", field_name)),
                python_type=Any,
                type_str=str(raw.get("type_str", "Any")),
                required=True,
            )
            for field_name, raw in entry.schema_outputs.items()
        },
    )


class ToolBundleManager:
    """Boot every RPC-protocol tool bundle the workflow references; shut down on exit.

    Typical use (from the launcher)::

        manager = ToolBundleManager(skill_registry=skill_registry,
                                    tool_registry=tool_registry)
        manager.boot_all(["sam3", "curobo", "geometry"])  # noqa
        try:
            executor.execute()
        finally:
            manager.shutdown_all()
    """

    def __init__(
        self,
        skill_registry: Any,
        tool_registry: Any,
        *,
        startup_timeout_s: float = 60.0,
        evict_grace_s: float = 5.0,
        call_timeout_s: float | None = None,
        shared_socket_dir: str | Path | None = None,
        execution_mode: str | None = None,
        execution_modes: Mapping[str, str] | None = None,
        session_id: str | None = None,
        episode_id: str | None = None,
    ) -> None:
        self._skill_registry = skill_registry
        self._tool_registry = tool_registry
        self._timeout = float(startup_timeout_s)
        self._grace = float(evict_grace_s)
        # Per-call reply timeout passed to each ToolClient. None lets the
        # client resolve GAP_TOOL_CALL_TIMEOUT_S / its built-in default.
        self._call_timeout_s = call_timeout_s
        configured_dir = (
            str(shared_socket_dir)
            if shared_socket_dir is not None
            else os.environ.get("GAP_SHARED_TOOL_SOCKET_DIR", "").strip()
        )
        self._shared_socket_dir = Path(configured_dir) if configured_dir else None
        self._execution_mode = execution_mode or os.environ.get(
            "GAP_TOOL_EXECUTION_MODE", "serialized"
        )
        self._session_id = (
            session_id
            or os.environ.get("GAP_SESSION_ID", "").strip()
            or f"session-{os.getpid()}-{uuid.uuid4().hex}"
        )
        self._episode_id = episode_id or os.environ.get("GAP_EPISODE_ID", "").strip() or "episode-0"
        if self._execution_mode not in ("serialized", "batched"):
            raise ValueError(
                "tool execution mode must be 'serialized' or 'batched', got "
                f"{self._execution_mode!r}"
            )
        configured_modes = dict(execution_modes or {})
        for item in os.environ.get("GAP_TOOL_EXECUTION_MODES", "").split(","):
            if not item.strip():
                continue
            bundle, separator, mode = item.partition("=")
            if not separator or not bundle.strip() or not mode.strip():
                raise ValueError(
                    "GAP_TOOL_EXECUTION_MODES must contain comma-separated "
                    f"bundle=mode entries, got {item!r}"
                )
            configured_modes.setdefault(bundle.strip(), mode.strip())
        invalid_modes = {
            bundle: mode
            for bundle, mode in configured_modes.items()
            if mode not in ("serialized", "batched")
        }
        if invalid_modes:
            raise ValueError(
                "per-bundle tool execution modes must be 'serialized' or 'batched', "
                f"got {invalid_modes!r}"
            )
        self._execution_modes = configured_modes
        self._lock = threading.Lock()
        self._managed: dict[str, _Managed] = {}

    def boot_all(self, bundle_names: Iterable[str]) -> None:
        """Spawn one server per bundle, register every catalog entry.

        Bundles whose SKILL.md does NOT declare ``serving.protocol ==
        "stdio-msgpack"`` are silently skipped — they run in-process.
        """
        names = sorted(set(bundle_names))
        if not names:
            return
        started: list[_Managed] = []
        staged: list[tuple[Any, Any, Any]] = []
        staged_names: set[str] = set()
        try:
            for name in names:
                info = self._skill_registry.get(name)
                serving = info.meta.serving
                if serving is None or serving.protocol != "stdio-msgpack":
                    continue
                logger.info("[tool-bundle:%s] booting", name)
                try:
                    if self._shared_socket_dir is not None:
                        from gap_core.rpc.shared import SharedToolClient

                        socket_path = self._shared_socket_dir / f"{name}.sock"
                        expected_mode = self.execution_mode_for(name)
                        logger.info(
                            "[tool-bundle:%s] connecting to shared %s owner at %s",
                            name,
                            expected_mode,
                            socket_path,
                        )
                        client = SharedToolClient.wait_until_ready(
                            name,
                            socket_path,
                            startup_timeout_s=self._timeout,
                            call_timeout_s=self._call_timeout_s,
                            expected_mode=expected_mode,
                            session_id=self._session_id,
                            episode_id=self._episode_id,
                        )
                    else:
                        from gap_core.rpc.client import ToolClient

                        client = ToolClient(
                            bundle_name=name,
                            bundle_dir=info.meta.bundle_dir,
                            env=serving.env,
                            evict_grace_s=self._grace,
                            call_timeout_s=self._call_timeout_s,
                        )
                except BaseException as exc:
                    raise ToolBundleStartupError(
                        f"tool bundle {name!r}: failed to start its server "
                        f"({type(exc).__name__}: {exc})"
                    ) from exc

                tool_names = [entry.name for entry in client.catalog]
                started.append(_Managed(bundle=name, client=client, tool_names=tool_names))
                for entry in client.catalog:
                    if entry.name in self._tool_registry or entry.name in staged_names:
                        raise ToolBundleStartupError(
                            f"tool bundle {name!r}: cannot register {entry.name!r} — "
                            "tool name collision"
                        )
                    staged_names.add(entry.name)
                    staged.append((entry, client, _schema_from_catalog(entry)))
                logger.info("[tool-bundle:%s] %d tools prepared", name, len(tool_names))

            # Registration is the commit point: all bundle handshakes,
            # catalogs, schemas, and cross-bundle names have been validated.
            # No partially registered tools are visible on startup failure.
            for entry, client, schema in staged:
                self._tool_registry.register_rpc(
                    entry.name,
                    client,
                    summary=entry.summary,
                    schema=schema,
                    tags=tuple(entry.tags),
                    metadata=entry.metadata,
                )

            with self._lock:
                for m in started:
                    self._managed[m.bundle] = m
        except BaseException:
            # Tear down anything we started before re-raising.
            for m in started:
                try:
                    m.client.close()
                except BaseException:
                    logger.warning(
                        "[tool-bundle:%s] close during boot rollback failed",
                        m.bundle,
                        exc_info=True,
                    )
            raise

    def shutdown_all(self) -> None:
        """Terminate every tool bundle subprocess we started. Idempotent."""
        with self._lock:
            managed = list(self._managed.values())
            self._managed.clear()

        if not managed:
            return
        logger.info(
            "ToolBundleManager: shutting down %d tool bundle%s",
            len(managed),
            "" if len(managed) == 1 else "s",
        )
        for m in managed:
            try:
                m.client.close()
            except BaseException:
                logger.warning("[tool-bundle:%s] close failed", m.bundle, exc_info=True)

    def loaded_bundles(self) -> list[str]:
        return sorted(self._managed.keys())

    @property
    def execution_mode(self) -> str:
        return self._execution_mode

    def execution_mode_for(self, bundle: str) -> str:
        return self._execution_modes.get(bundle, self._execution_mode)

    def shared_metrics(self) -> dict[str, dict[str, Any]]:
        """Return metrics for shared owners; dedicated stdio clients are omitted."""
        with self._lock:
            managed = list(self._managed.values())
        return {
            item.bundle: item.client.metrics()
            for item in managed
            if hasattr(item.client, "metrics")
        }

    def set_episode_id(self, episode_id: str) -> None:
        """Rebind shared clients after this graph session starts a new episode."""
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        with self._lock:
            managed = list(self._managed.values())
            self._episode_id = episode_id
        for item in managed:
            setter = getattr(item.client, "set_episode_id", None)
            if setter is not None:
                setter(episode_id)


__all__ = ["ToolBundleManager", "ToolBundleStartupError"]
