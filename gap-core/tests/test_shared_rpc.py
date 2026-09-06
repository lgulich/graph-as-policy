from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import pytest

from gap_core.rpc import server as rpc_server
from gap_core.rpc.client import ToolCallTimeout, ToolClientError, ToolRemoteError
from gap_core.rpc.shared import SharedToolClient, SharedToolServer, _Metrics
from gap_core.tools import ToolRegistry, tool, tool_batch

_BATCH_SIZES: list[int] = []
_SCALAR_CALLS = 0
_ACTIVE_SCALAR_CALLS = 0
_MAX_ACTIVE_SCALAR_CALLS = 0
_SCALAR_LOCK = threading.Lock()


def test_signal_shutdown_closes_server_before_hard_exit(monkeypatch):
    events: list[object] = []

    class _Server:
        def close(self) -> None:
            events.append("closed")

    monkeypatch.setattr(rpc_server.os, "_exit", lambda code: events.append(code))
    rpc_server._shutdown_shared_server(_Server())

    assert events == ["closed", 0]


def test_queue_wait_metrics_use_observed_denominator_and_bounded_p95_sample():
    metrics = _Metrics(mode="serialized")
    metrics.add("requests_accepted", 10, tool="shared.echo")
    waits = [float(index) for index in range(2100)]
    metrics.execution(
        tool="shared.echo",
        batch_size=len(waits),
        queue_waits=waits,
        batched=False,
    )

    snapshot = metrics.snapshot()
    expected_mean = sum(waits) / len(waits)
    retained = waits[-2048:]
    expected_p95 = retained[1944] * 0.35 + retained[1945] * 0.65
    assert snapshot["queue_wait_observations"] == len(waits)
    assert snapshot["queue_wait_mean_s"] == pytest.approx(expected_mean)
    assert snapshot["queue_wait_p95_s"] == pytest.approx(expected_p95)
    assert len(metrics._queue_wait_samples) == 2048
    assert snapshot["tools"]["shared.echo"]["queue_wait_mean_s"] == pytest.approx(expected_mean)


class EchoResult(TypedDict):
    value: int


def _group_key(request: dict) -> str:
    group = request.get("group", "default")
    if group == "bad":
        raise ValueError("invalid compatibility key")
    return str(group)


@tool(name="shared.echo", summary="Scalar echo", tags=("test",))
def _echo(value: int, fail: bool = False, group: str = "default") -> EchoResult:
    global _SCALAR_CALLS
    _SCALAR_CALLS += 1
    if fail:
        raise ValueError(f"bad value {value}")
    return {"value": value}


@tool_batch(
    name="shared.echo",
    max_batch_size=8,
    max_wait_ms=100,
    compatibility_key=_group_key,
)
def _echo_batch(requests: list[dict]) -> list[dict | BaseException]:
    _BATCH_SIZES.append(len(requests))
    results: list[dict | BaseException] = []
    for request in requests:
        if request.get("fail"):
            results.append(ValueError(f"bad value {request['value']}"))
        else:
            results.append({"value": request["value"]})
    return results


@tool(name="shared.serial", summary="Unbatched scalar tool", tags=("test",))
def _serial(value: int) -> EchoResult:
    global _ACTIVE_SCALAR_CALLS, _MAX_ACTIVE_SCALAR_CALLS
    with _SCALAR_LOCK:
        _ACTIVE_SCALAR_CALLS += 1
        _MAX_ACTIVE_SCALAR_CALLS = max(_MAX_ACTIVE_SCALAR_CALLS, _ACTIVE_SCALAR_CALLS)
    time.sleep(0.02)
    with _SCALAR_LOCK:
        _ACTIVE_SCALAR_CALLS -= 1
    return {"value": value}


@tool(name="shared.slow", summary="Slow unbatched tool", tags=("test",))
def _slow(value: int, seconds: float) -> EchoResult:
    global _ACTIVE_SCALAR_CALLS, _MAX_ACTIVE_SCALAR_CALLS
    with _SCALAR_LOCK:
        _ACTIVE_SCALAR_CALLS += 1
        _MAX_ACTIVE_SCALAR_CALLS = max(_MAX_ACTIVE_SCALAR_CALLS, _ACTIVE_SCALAR_CALLS)
    time.sleep(seconds)
    with _SCALAR_LOCK:
        _ACTIVE_SCALAR_CALLS -= 1
    return {"value": value}


@tool(name="shared.unserializable", summary="Returns an invalid wire value")
def _unserializable() -> dict:
    return {"value": object()}


@pytest.fixture
def shared_server(tmp_path: Path):
    global _ACTIVE_SCALAR_CALLS, _MAX_ACTIVE_SCALAR_CALLS, _SCALAR_CALLS
    _BATCH_SIZES.clear()
    _SCALAR_CALLS = 0
    _ACTIVE_SCALAR_CALLS = 0
    _MAX_ACTIVE_SCALAR_CALLS = 0
    registry = ToolRegistry()
    registry.discover_pending()
    socket_path = tmp_path / "tools.sock"
    server = SharedToolServer(registry, socket_path, mode="batched", bundle_name="shared")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists()
    yield server, socket_path
    server.close()
    thread.join(timeout=5)


def test_independent_scalar_clients_share_one_batched_execution(shared_server):
    _, socket_path = shared_server
    clients = [SharedToolClient("shared", socket_path) for _ in range(8)]
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(client.call, "shared.echo", value=index)
                for index, client in enumerate(clients)
            ]
        assert [future.result() for future in futures] == [{"value": index} for index in range(8)]
        assert _BATCH_SIZES == [8]
        assert _SCALAR_CALLS == 0
        metrics = clients[0].metrics()
        assert metrics["mode"] == "batched"
        assert metrics["batch_executions"] == 1
        assert metrics["batch_items"] == 8
        assert metrics["batch_size_histogram"] == {"8": 1}
        tool_metrics = metrics["tools"]["shared.echo"]
        assert tool_metrics["requests_completed"] == 8
        assert tool_metrics["batch_executions"] == 1
        assert tool_metrics["batch_items"] == 8
        assert tool_metrics["execution_latency_mean_s"] >= 0
        assert tool_metrics["request_bytes"] > 0
        assert tool_metrics["response_bytes"] > 0
    finally:
        for client in clients:
            client.close()


def test_batch_item_error_does_not_fail_siblings(shared_server):
    _, socket_path = shared_server
    clients = [SharedToolClient("shared", socket_path) for _ in range(3)]
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            good_a = pool.submit(clients[0].call, "shared.echo", value=1)
            bad = pool.submit(clients[1].call, "shared.echo", value=2, fail=True)
            good_b = pool.submit(clients[2].call, "shared.echo", value=3)
        assert good_a.result() == {"value": 1}
        with pytest.raises(ToolRemoteError, match="bad value 2"):
            bad.result()
        assert good_b.result() == {"value": 3}
        assert _BATCH_SIZES == [3]
    finally:
        for client in clients:
            client.close()


def test_incompatible_requests_are_never_co_batched(shared_server):
    _, socket_path = shared_server
    clients = [SharedToolClient("shared", socket_path) for _ in range(4)]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(
                    client.call,
                    "shared.echo",
                    value=index,
                    group="even" if index % 2 == 0 else "odd",
                )
                for index, client in enumerate(clients)
            ]
        assert [future.result() for future in futures] == [{"value": index} for index in range(4)]
        assert sorted(_BATCH_SIZES) == [2, 2]
    finally:
        for client in clients:
            client.close()


def test_unbatched_tool_defaults_to_serial_execution(shared_server):
    _, socket_path = shared_server
    clients = [SharedToolClient("shared", socket_path) for _ in range(4)]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(client.call, "shared.serial", value=index)
                for index, client in enumerate(clients)
            ]
        assert [future.result() for future in futures] == [{"value": index} for index in range(4)]
        assert _MAX_ACTIVE_SCALAR_CALLS == 1
    finally:
        for client in clients:
            client.close()


def test_execution_is_serialized_across_tools_sharing_one_bundle(shared_server):
    _, socket_path = shared_server
    clients = [SharedToolClient("shared", socket_path) for _ in range(2)]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(clients[0].call, "shared.serial", value=1)
            second = pool.submit(clients[1].call, "shared.slow", value=2, seconds=0.02)
        assert first.result() == {"value": 1}
        assert second.result() == {"value": 2}
        assert _MAX_ACTIVE_SCALAR_CALLS == 1
    finally:
        for client in clients:
            client.close()


def test_bad_compatibility_key_fails_one_request_and_worker_survives(shared_server):
    _, socket_path = shared_server
    first = SharedToolClient("shared", socket_path)
    bad = SharedToolClient("shared", socket_path)
    good = SharedToolClient("shared", socket_path)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            first_result = pool.submit(first.call, "shared.echo", value=0, group="healthy")
            time.sleep(0.02)  # ensure the worker is collecting this batch
            bad_result = pool.submit(bad.call, "shared.echo", value=1, group="bad")
            good_result = pool.submit(good.call, "shared.echo", value=2, group="healthy")
        assert first_result.result() == {"value": 0}
        with pytest.raises(ToolRemoteError, match="invalid compatibility key"):
            bad_result.result()
        assert good_result.result() == {"value": 2}
        assert good.call("shared.echo", value=3, group="healthy") == {"value": 3}
    finally:
        first.close()
        bad.close()
        good.close()


def test_shared_catalog_preserves_scalar_schema(shared_server):
    _, socket_path = shared_server
    client = SharedToolClient("shared", socket_path)
    try:
        entry = next(item for item in client.catalog if item.name == "shared.echo")
        assert set(entry.schema_inputs) == {"value", "fail", "group"}
        assert "value" in entry.schema_outputs
    finally:
        client.close()


def test_call_deadline_expires_only_queued_request(shared_server):
    _, socket_path = shared_server
    slow_client = SharedToolClient("shared", socket_path)
    expiring_client = SharedToolClient("shared", socket_path, call_timeout_s=0.05)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(slow_client.call, "shared.slow", value=1, seconds=0.15)
            time.sleep(0.02)
            expiring = pool.submit(expiring_client.call, "shared.slow", value=2, seconds=0.01)
        assert slow.result() == {"value": 1}
        with pytest.raises(ToolCallTimeout, match="exceeded"):
            expiring.result()
        assert slow_client.call("shared.slow", value=3, seconds=0) == {"value": 3}
        assert slow_client.metrics()["requests_expired"] == 1
    finally:
        slow_client.close()
        expiring_client.close()


def test_bounded_queue_rejects_excess_load_without_killing_worker(tmp_path):
    registry = ToolRegistry()
    registry.discover_pending()
    socket_path = tmp_path / "bounded.sock"
    server = SharedToolServer(registry, socket_path, mode="serialized", max_queue_size=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    first = SharedToolClient("shared", socket_path)
    rejected = SharedToolClient("shared", socket_path)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            running = pool.submit(first.call, "shared.slow", value=1, seconds=0.15)
            time.sleep(0.02)
            overflow = pool.submit(rejected.call, "shared.slow", value=2, seconds=0)
        assert running.result() == {"value": 1}
        with pytest.raises(ToolRemoteError) as exc_info:
            overflow.result()
        assert exc_info.value.remote_type == "ToolQueueFull"
        assert first.call("shared.slow", value=3, seconds=0) == {"value": 3}
        assert first.metrics()["requests_rejected"] == 1
    finally:
        first.close()
        rejected.close()
        server.close()
        thread.join(timeout=5)


def test_shutdown_reports_stopping_to_queued_client_before_closing_writer(
    shared_server,
):
    server, socket_path = shared_server
    running_client = SharedToolClient("shared", socket_path)
    queued_client = SharedToolClient("shared", socket_path)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            running = pool.submit(running_client.call, "shared.slow", value=1, seconds=0.15)
            time.sleep(0.02)
            queued = pool.submit(queued_client.call, "shared.slow", value=2, seconds=0)
            time.sleep(0.02)
            server.close()
        assert running.result() == {"value": 1}
        with pytest.raises(ToolRemoteError) as exc_info:
            queued.result()
        assert exc_info.value.remote_type == "ToolServerStopping"
    finally:
        running_client.close()
        queued_client.close()


def test_unserializable_result_is_an_isolated_error_and_worker_survives(
    shared_server,
):
    _, socket_path = shared_server
    client = SharedToolClient("shared", socket_path)
    try:
        for _ in range(2):
            with pytest.raises(ToolRemoteError) as exc_info:
                client.call("shared.unserializable")
            assert exc_info.value.remote_type == "TypeError"
    finally:
        client.close()


def test_second_owner_cannot_unlink_live_socket(shared_server):
    first_server, socket_path = shared_server
    contender = SharedToolServer(ToolRegistry(), socket_path, mode="serialized")
    with pytest.raises(RuntimeError, match="already holds"):
        contender.serve_forever()
    client = SharedToolClient("shared", socket_path)
    try:
        assert client.call("shared.serial", value=4) == {"value": 4}
        assert first_server.metrics.snapshot()["requests_completed"] >= 1
    finally:
        client.close()


def test_client_rejects_wrong_bundle_and_mode(shared_server):
    _, socket_path = shared_server
    with pytest.raises(ToolClientError, match="bundle mismatch"):
        SharedToolClient("wrong", socket_path)
    with pytest.raises(ToolClientError, match="mode mismatch"):
        SharedToolClient("shared", socket_path, expected_mode="serialized")


def test_session_and_episode_identity_roundtrip(shared_server):
    _, socket_path = shared_server
    client = SharedToolClient(
        "shared",
        socket_path,
        session_id="arena-env-3",
        episode_id="episode-17",
    )
    try:
        assert client.call("shared.echo", value=5) == {"value": 5}
        client.set_episode_id("episode-18")
        assert client.call("shared.echo", value=6) == {"value": 6}
        assert client.metrics()["requests_completed"] >= 1
    finally:
        client.close()


@pytest.mark.parametrize(
    ("identity_field", "stale_value", "error_match"),
    [
        ("session_id", "arena-env-old", "session mismatch"),
        ("episode_id", "episode-old", "episode mismatch"),
    ],
)
def test_client_rejects_stale_identity_reply(
    shared_server,
    monkeypatch,
    identity_field,
    stale_value,
    error_match,
):
    _, socket_path = shared_server
    client = SharedToolClient(
        "shared",
        socket_path,
        session_id="arena-env-3",
        episode_id="episode-new",
    )
    original_read = client._read

    def _stale_read():
        frame = original_read()
        if frame.get("kind") in {"result", "error"}:
            frame[identity_field] = stale_value
        return frame

    monkeypatch.setattr(client, "_read", _stale_read)
    try:
        with pytest.raises(ToolClientError, match=error_match):
            client.call("shared.echo", value=5)
    finally:
        client.close()
