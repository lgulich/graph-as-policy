"""Lifecycle + registration paths for ToolBundleManager.

The real ToolClient spawns a subprocess via `uv run --project`; here we
use a fake ToolClient class that records calls without spawning, so the
manager's discovery + registration + teardown paths are exercised
end-to-end without paying the uv-sync cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from gap_core.tools import ToolRegistry
from gap_core.tools._registry import RpcAdapter

from gap.runtime.shared_tool_metrics import _mode_map
from gap.runtime.shared_tool_owner import _parser as owner_parser
from gap.runtime.shared_tool_owner import _server_command
from gap.runtime.tool_bundle_boot import (
    boot_tool_bundles,
    required_rpc_tool_bundles,
)
from gap.runtime.tool_bundle_manager import (
    ToolBundleManager,
    ToolBundleStartupError,
)


class _FakeCatalogEntry:
    def __init__(self, name, summary="", tags=(), schema_inputs=None, schema_outputs=None):
        self.name = name
        self.summary = summary
        self.tags = list(tags)
        self.metadata = {}
        self.schema_inputs = schema_inputs or {}
        self.schema_outputs = schema_outputs or {}


class _FakeToolClient:
    """Stand-in for gap_core.rpc.client.ToolClient — no subprocess, no msgpack.

    Records every call/close so the manager's behavior can be asserted. The
    constructor's signature mirrors ToolClient so the manager swap is
    transparent.
    """

    def __init__(
        self,
        bundle_name,
        bundle_dir,
        env=None,
        evict_grace_s=5.0,
        call_timeout_s=None,
        catalog=None,
        on_call=None,
        raise_on_init: bool = False,
    ):
        if raise_on_init:
            raise RuntimeError(f"bundle {bundle_name}: synthetic boot failure")
        self.bundle_name = bundle_name
        self.bundle_dir = bundle_dir
        self.call_timeout_s = call_timeout_s
        self.catalog = catalog or [_FakeCatalogEntry(name=f"{bundle_name}.echo", summary="echo")]
        self.calls = []
        self.closed = False
        self._on_call = on_call

    def call(self, tool, /, **kwargs):
        self.calls.append((tool, kwargs))
        if self._on_call is not None:
            return self._on_call(tool, **kwargs)
        return {"echoed": kwargs}

    def close(self):
        self.closed = True

    def set_episode_id(self, episode_id):
        self.episode_id = episode_id


class _FakeSharedToolClient(_FakeToolClient):
    ready_calls = []

    @classmethod
    def wait_until_ready(
        cls,
        bundle_name,
        socket_path,
        *,
        startup_timeout_s,
        call_timeout_s,
        expected_mode,
        session_id,
        episode_id,
    ):
        cls.ready_calls.append(
            {
                "bundle_name": bundle_name,
                "socket_path": socket_path,
                "startup_timeout_s": startup_timeout_s,
                "call_timeout_s": call_timeout_s,
                "expected_mode": expected_mode,
                "session_id": session_id,
                "episode_id": episode_id,
            }
        )
        return cls(bundle_name, socket_path)

    def metrics(self):
        return {"mode": "batched", "requests_completed": 3}


class _FakeSkillInfo:
    def __init__(
        self,
        *,
        kind="tool",
        protocol="stdio-msgpack",
        bundle_dir="/tmp/fake",
        env=None,
        command=("python", "x"),
    ):
        if protocol is None:
            serving = None
        else:
            serving = SimpleNamespace(
                command=list(command),
                protocol=protocol,
                env=dict(env or {}),
                requires_gpu=False,
                weights_uri="",
            )
        self.kind = kind
        self.meta = SimpleNamespace(
            serving=serving,
            bundle_dir=Path(bundle_dir),
            tags=[],
        )


class _FakeSkillRegistry:
    def __init__(self, infos):
        self._infos = infos

    def get(self, name):
        if name not in self._infos:
            raise KeyError(name)
        return self._infos[name]


def test_metrics_mode_map():
    assert _mode_map(["sam3=batched", "curobo=serialized"]) == {
        "sam3": "batched",
        "curobo": "serialized",
    }
    with pytest.raises(Exception, match="BUNDLE"):
        _mode_map(["sam3=parallel"])


def test_shared_owner_command_carries_bundle_mode_and_limits(tmp_path):
    bundle_dir = tmp_path / "sam3"
    socket_dir = tmp_path / "sockets"
    args = owner_parser().parse_args(
        [
            "--bundle",
            "sam3",
            "--bundle-dir",
            str(bundle_dir),
            "--socket-dir",
            str(socket_dir),
            "--mode",
            "batched",
            "--max-queue-size",
            "32",
            "--max-batch-size",
            "8",
            "--max-wait-ms",
            "10",
        ]
    )
    command = _server_command(args, socket_dir / "sam3.sock")
    assert command[:4] == ["uv", "run", "--project", str(bundle_dir)]
    assert command[-10:] == [
        "--socket",
        str(socket_dir / "sam3.sock"),
        "--mode",
        "batched",
        "--max-queue-size",
        "32",
        "--max-batch-size",
        "8",
        "--max-wait-ms",
        "10.0",
    ]


@pytest.fixture
def fake_client_cls(monkeypatch):
    """Patch ToolClient at the manager's import site so boot_all uses
    _FakeToolClient instead of spawning real subprocesses."""
    instances: list[_FakeToolClient] = []

    def _factory(*args, **kwargs):
        c = _FakeToolClient(*args, **kwargs)
        instances.append(c)
        return c

    monkeypatch.setattr("gap_core.rpc.client.ToolClient", _factory)
    return instances


# ---------------------------------------------------------------------------
# ToolBundleManager
# ---------------------------------------------------------------------------


def test_boot_registers_catalog_under_rpc_transport(fake_client_cls, tmp_path):
    skill_registry = _FakeSkillRegistry(
        {
            "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3"),
        }
    )
    tool_registry = ToolRegistry()

    mgr = ToolBundleManager(skill_registry, tool_registry)
    mgr.boot_all(["sam3"])
    try:
        assert mgr.loaded_bundles() == ["sam3"]
        descriptor = tool_registry.get("sam3.echo")
        assert descriptor.transport == "rpc"
        assert descriptor.metadata["bundle"] == "sam3"
        # Invoking through the registry must hit the fake client.
        result = tool_registry.invoke("sam3.echo", None, x=1)
        assert result == {"echoed": {"x": 1}}
        assert fake_client_cls[0].calls == [("sam3.echo", {"x": 1})]
    finally:
        mgr.shutdown_all()
    assert fake_client_cls[0].closed is True
    assert mgr.loaded_bundles() == []


def test_boot_connects_to_configured_shared_owner(monkeypatch, tmp_path):
    _FakeSharedToolClient.ready_calls.clear()
    monkeypatch.setattr("gap_core.rpc.shared.SharedToolClient", _FakeSharedToolClient)
    skill_registry = _FakeSkillRegistry(
        {
            "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3"),
        }
    )
    tool_registry = ToolRegistry()
    manager = ToolBundleManager(
        skill_registry,
        tool_registry,
        shared_socket_dir=tmp_path / "sockets",
        execution_mode="batched",
        session_id="arena-env-0",
        episode_id="episode-4",
        startup_timeout_s=2.5,
        call_timeout_s=4.0,
    )
    manager.boot_all(["sam3"])
    try:
        assert _FakeSharedToolClient.ready_calls == [
            {
                "bundle_name": "sam3",
                "socket_path": tmp_path / "sockets" / "sam3.sock",
                "startup_timeout_s": 2.5,
                "call_timeout_s": 4.0,
                "expected_mode": "batched",
                "session_id": "arena-env-0",
                "episode_id": "episode-4",
            }
        ]
        assert tool_registry.invoke("sam3.echo", None, value=1) == {"echoed": {"value": 1}}
        assert manager.shared_metrics() == {"sam3": {"mode": "batched", "requests_completed": 3}}
    finally:
        manager.shutdown_all()
    assert manager.loaded_bundles() == []


def test_shared_owners_validate_per_bundle_execution_modes(monkeypatch, tmp_path):
    _FakeSharedToolClient.ready_calls.clear()
    monkeypatch.setattr("gap_core.rpc.shared.SharedToolClient", _FakeSharedToolClient)
    skill_registry = _FakeSkillRegistry(
        {
            "curobo": _FakeSkillInfo(bundle_dir=tmp_path / "curobo"),
            "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3"),
        }
    )
    manager = ToolBundleManager(
        skill_registry,
        ToolRegistry(),
        shared_socket_dir=tmp_path / "sockets",
        execution_mode="serialized",
        execution_modes={"sam3": "batched", "curobo": "serialized"},
        session_id="arena-env-9",
        episode_id="episode-2",
    )
    manager.boot_all(["sam3", "curobo"])
    try:
        calls = {item["bundle_name"]: item for item in _FakeSharedToolClient.ready_calls}
        assert calls["sam3"]["expected_mode"] == "batched"
        assert calls["curobo"]["expected_mode"] == "serialized"
        assert manager.execution_mode_for("sam3") == "batched"
        assert manager.execution_mode_for("curobo") == "serialized"
        manager.set_episode_id("episode-3")
        assert all(
            item.episode_id == "episode-3"
            for item in (manager._managed["sam3"].client, manager._managed["curobo"].client)
        )
    finally:
        manager.shutdown_all()


def test_explicit_shared_bundle_modes_leave_other_rpc_bundles_private(
    monkeypatch, fake_client_cls, tmp_path
):
    _FakeSharedToolClient.ready_calls.clear()
    monkeypatch.setattr("gap_core.rpc.shared.SharedToolClient", _FakeSharedToolClient)
    skill_registry = _FakeSkillRegistry(
        {
            "geometry": _FakeSkillInfo(bundle_dir=tmp_path / "geometry"),
            "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3"),
        }
    )
    manager = ToolBundleManager(
        skill_registry,
        ToolRegistry(),
        shared_socket_dir=tmp_path / "sockets",
        execution_modes={"sam3": "serialized"},
    )

    manager.boot_all(["geometry", "sam3"])
    try:
        assert [call["bundle_name"] for call in _FakeSharedToolClient.ready_calls] == [
            "sam3"
        ]
        assert [client.bundle_name for client in fake_client_cls] == ["geometry"]
    finally:
        manager.shutdown_all()


def test_boot_skips_non_rpc_bundles(fake_client_cls, tmp_path):
    """Bundles whose serving is None or in-process must NOT be booted —
    they keep using the in-process @tool path."""
    skill_registry = _FakeSkillRegistry(
        {
            "geometry": _FakeSkillInfo(protocol=None, bundle_dir=tmp_path / "geom"),
            "vlm": _FakeSkillInfo(protocol="in-process", bundle_dir=tmp_path / "vlm"),
        }
    )
    tool_registry = ToolRegistry()
    mgr = ToolBundleManager(skill_registry, tool_registry)
    mgr.boot_all(["geometry", "vlm"])
    try:
        assert mgr.loaded_bundles() == []
        assert fake_client_cls == []
    finally:
        mgr.shutdown_all()


def test_boot_unknown_bundle_raises(fake_client_cls):
    skill_registry = _FakeSkillRegistry({})  # empty
    tool_registry = ToolRegistry()
    mgr = ToolBundleManager(skill_registry, tool_registry)
    with pytest.raises(KeyError):
        mgr.boot_all(["nope"])


def test_boot_client_init_failure_propagates(monkeypatch, tmp_path):
    """If ToolClient raises during catalog handshake, boot must surface
    ToolBundleStartupError and not leak any half-started subprocesses."""
    skill_registry = _FakeSkillRegistry(
        {
            "bad": _FakeSkillInfo(bundle_dir=tmp_path / "bad"),
        }
    )

    def _failing(*args, **kwargs):
        return _FakeToolClient(*args, raise_on_init=True, **kwargs)

    monkeypatch.setattr("gap_core.rpc.client.ToolClient", _failing)
    tool_registry = ToolRegistry()
    mgr = ToolBundleManager(skill_registry, tool_registry)
    with pytest.raises(ToolBundleStartupError, match="failed to start"):
        mgr.boot_all(["bad"])


def test_boot_name_collision_tears_down(fake_client_cls, tmp_path):
    """A second bundle exporting the same tool name as the first must
    surface a startup error AND tear down both clients (not leak the
    first)."""
    overlap_catalog = [_FakeCatalogEntry(name="shared.tool", summary="x")]

    def _on_factory_side_effect(name):
        infos = {
            "first": _FakeSkillInfo(bundle_dir=tmp_path / "first"),
            "second": _FakeSkillInfo(bundle_dir=tmp_path / "second"),
        }
        return infos[name]

    # Monkey-patch fake client to always return the overlapping catalog.
    instances: list[_FakeToolClient] = []

    def _factory(*args, **kwargs):
        c = _FakeToolClient(*args, **{**kwargs, "catalog": overlap_catalog})
        instances.append(c)
        return c

    import gap_core.rpc.client as client_mod

    with patch.object(client_mod, "ToolClient", _factory):
        skill_registry = _FakeSkillRegistry(
            {
                "first": _FakeSkillInfo(bundle_dir=tmp_path / "first"),
                "second": _FakeSkillInfo(bundle_dir=tmp_path / "second"),
            }
        )
        tool_registry = ToolRegistry()
        mgr = ToolBundleManager(skill_registry, tool_registry)
        with pytest.raises(ToolBundleStartupError, match="cannot register"):
            mgr.boot_all(["first", "second"])

    # The second client got booted but its name collided; both must be closed.
    assert all(c.closed for c in instances), [c.closed for c in instances]
    assert "shared.tool" not in tool_registry


def test_shutdown_is_idempotent(fake_client_cls, tmp_path):
    skill_registry = _FakeSkillRegistry(
        {
            "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3"),
        }
    )
    mgr = ToolBundleManager(skill_registry, ToolRegistry())
    mgr.boot_all(["sam3"])
    mgr.shutdown_all()
    mgr.shutdown_all()  # second call is a no-op
    assert fake_client_cls[0].closed is True


# ---------------------------------------------------------------------------
# Discovery (required_rpc_tool_bundles + boot_tool_bundles)
# ---------------------------------------------------------------------------


def _write_workflow(tmp_path: Path, tools: list[str]) -> Path:
    wf = {
        "version": 3,
        "meta": {"name": "t"},
        "nodes": {
            "m": {"type": "subgraph", "ref": "m"},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "m"]],
        "conditional_edges": {"m": {"router_field": "exit", "mapping": {"ok": "done"}}},
        "subgraphs": {
            "m": {
                "skill": "g",
                "inputs": {},
                "outputs": {},
                "nodes": {
                    **{
                        f"n{i}": {"type": "tool", "tool": t, "inputs": {}}
                        for i, t in enumerate(tools)
                    },
                    "ok": {"type": "noop"},
                },
                "edges": [["START", "n0"]]
                + [[f"n{i}", f"n{i + 1}"] for i in range(len(tools) - 1)]
                + [[f"n{len(tools) - 1}", "ok"], ["ok", "END"]],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
            },
        },
    }
    wfd = tmp_path / "wf"
    wfd.mkdir()
    (wfd / "workflow.json").write_text(json.dumps(wf))
    return wfd


def test_required_rpc_tool_bundles_filters_by_protocol(tmp_path):
    wf_dir = _write_workflow(
        tmp_path, ["sam3.segment", "geometry.iou", "vlm.query", "sim.check_success"]
    )
    registry = _FakeSkillRegistry(
        {
            "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3", protocol="stdio-msgpack"),
            "geometry": _FakeSkillInfo(bundle_dir=tmp_path / "geom", protocol="in-process"),
            "vlm": _FakeSkillInfo(bundle_dir=tmp_path / "vlm", protocol=None),
            # `sim.*` is a connector tool — never in the registry.
        }
    )
    assert required_rpc_tool_bundles(wf_dir, registry) == {"sam3"}


def test_boot_tool_bundles_returns_none_when_no_rpc_needed(tmp_path):
    wf_dir = _write_workflow(tmp_path, ["geometry.iou"])
    registry = _FakeSkillRegistry(
        {
            "geometry": _FakeSkillInfo(bundle_dir=tmp_path / "geom", protocol=None),
        }
    )
    result = boot_tool_bundles(wf_dir, registry, ToolRegistry())
    assert result is None


def test_boot_tool_bundles_returns_manager_when_rpc_needed(fake_client_cls, tmp_path):
    wf_dir = _write_workflow(tmp_path, ["sam3.segment"])
    registry = _FakeSkillRegistry(
        {
            "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3", protocol="stdio-msgpack"),
        }
    )
    tool_registry = ToolRegistry()
    mgr = boot_tool_bundles(wf_dir, registry, tool_registry)
    try:
        assert mgr is not None
        assert mgr.loaded_bundles() == ["sam3"]
    finally:
        mgr.shutdown_all()


# ---------------------------------------------------------------------------
# RpcAdapter direct path
# ---------------------------------------------------------------------------


def test_rpc_adapter_invoke_drops_ctx():
    """The RPC path is for ctx-free tools only — ctx must not leak into
    the wire even when the gap-runtime caller passes one."""
    adapter = RpcAdapter()
    client = _FakeToolClient("b", "/tmp/b")
    adapter.register("b.tool", client)
    adapter.invoke("b.tool", ctx=object(), x=1, y=2)
    assert client.calls == [("b.tool", {"x": 1, "y": 2})]


def test_rpc_adapter_unknown_tool_raises():
    adapter = RpcAdapter()
    with pytest.raises(KeyError, match="not registered"):
        adapter.invoke("nope", None)
