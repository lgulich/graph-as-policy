"""Multi-client local tool RPC with optional dynamic microbatching.

The wire payload and scalar ``call(tool, **kwargs)`` interface match the
stdio tool client.  A Unix-domain socket lets independent GaP processes share
one bundle process and therefore one model instance.
"""

from __future__ import annotations

import fcntl
import logging
import os
import queue
import socket
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .client import (
    CatalogEntry,
    ToolCallTimeout,
    ToolClientError,
    ToolRemoteError,
    _resolve_call_timeout,
)
from .codec import FrameError, decode_frame, write_frame

logger = logging.getLogger(__name__)

ExecutionMode = Literal["serialized", "batched"]
SHARED_TOOL_PROTOCOL_VERSION = 2
_QUEUE_WAIT_SAMPLE_LIMIT = 2048


class ToolQueueFull(RuntimeError):
    """A tool's bounded shared-execution queue has no capacity."""


class ToolRequestExpired(TimeoutError):
    """A queued request reached its caller-provided deadline."""


class ToolServerStopping(RuntimeError):
    """A queued request was cancelled during owner shutdown."""


@dataclass
class _Request:
    request_id: str
    session_id: str
    episode_id: str
    tool: str
    args: dict[str, Any]
    writer: _ConnectionWriter
    enqueued_at: float
    request_bytes: int
    deadline_unix_s: float | None = None

    def expired(self) -> bool:
        return self.deadline_unix_s is not None and time.time() >= self.deadline_unix_s


class _Metrics:
    def __init__(self, *, mode: ExecutionMode) -> None:
        self.mode = mode
        self.started_unix_s = time.time()
        self._lock = threading.Lock()
        self._values: dict[str, int | float] = {
            "requests_accepted": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "requests_expired": 0,
            "requests_rejected": 0,
            "scalar_executions": 0,
            "batch_executions": 0,
            "batch_items": 0,
            "queue_wait_total_s": 0.0,
            "queue_wait_max_s": 0.0,
            "queue_wait_observations": 0,
            "request_bytes": 0,
            "response_bytes": 0,
            "execution_latency_total_s": 0.0,
            "execution_latency_max_s": 0.0,
        }
        self._batch_histogram: dict[int, int] = {}
        self._tool_values: dict[str, dict[str, int | float]] = {}
        self._tool_batch_histograms: dict[str, dict[int, int]] = {}
        self._queue_wait_samples: deque[float] = deque(maxlen=_QUEUE_WAIT_SAMPLE_LIMIT)
        self._tool_queue_wait_samples: dict[str, deque[float]] = {}

    @staticmethod
    def _empty_values() -> dict[str, int | float]:
        return {
            "requests_accepted": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "requests_expired": 0,
            "requests_rejected": 0,
            "scalar_executions": 0,
            "batch_executions": 0,
            "batch_items": 0,
            "queue_wait_total_s": 0.0,
            "queue_wait_max_s": 0.0,
            "queue_wait_observations": 0,
            "request_bytes": 0,
            "response_bytes": 0,
            "execution_latency_total_s": 0.0,
            "execution_latency_max_s": 0.0,
        }

    def add(self, name: str, amount: int | float = 1, *, tool: str | None = None) -> None:
        with self._lock:
            self._values[name] += amount
            if tool is not None:
                values = self._tool_values.setdefault(tool, self._empty_values())
                values[name] += amount

    def execution(
        self,
        *,
        tool: str,
        batch_size: int,
        queue_waits: list[float],
        batched: bool,
    ) -> None:
        with self._lock:
            key = "batch_executions" if batched else "scalar_executions"
            self._values[key] += 1
            tool_values = self._tool_values.setdefault(tool, self._empty_values())
            tool_values[key] += 1
            if batched:
                self._values["batch_items"] += batch_size
                self._batch_histogram[batch_size] = self._batch_histogram.get(batch_size, 0) + 1
                tool_values["batch_items"] += batch_size
                histogram = self._tool_batch_histograms.setdefault(tool, {})
                histogram[batch_size] = histogram.get(batch_size, 0) + 1
            if queue_waits:
                total_wait = sum(queue_waits)
                max_wait = max(queue_waits)
                self._values["queue_wait_total_s"] += total_wait
                self._values["queue_wait_observations"] += len(queue_waits)
                self._values["queue_wait_max_s"] = max(
                    float(self._values["queue_wait_max_s"]), max_wait
                )
                self._queue_wait_samples.extend(queue_waits)
                tool_values["queue_wait_total_s"] += total_wait
                tool_values["queue_wait_observations"] += len(queue_waits)
                tool_values["queue_wait_max_s"] = max(
                    float(tool_values["queue_wait_max_s"]), max_wait
                )
                self._tool_queue_wait_samples.setdefault(
                    tool, deque(maxlen=_QUEUE_WAIT_SAMPLE_LIMIT)
                ).extend(queue_waits)

    def execution_latency(self, tool: str, elapsed_s: float) -> None:
        with self._lock:
            self._values["execution_latency_total_s"] = (
                float(self._values.get("execution_latency_total_s", 0.0)) + elapsed_s
            )
            self._values["execution_latency_max_s"] = max(
                float(self._values.get("execution_latency_max_s", 0.0)), elapsed_s
            )
            values = self._tool_values.setdefault(tool, self._empty_values())
            values["execution_latency_total_s"] += elapsed_s
            values["execution_latency_max_s"] = max(
                float(values["execution_latency_max_s"]), elapsed_s
            )

    @staticmethod
    def _finalize(
        values: dict[str, int | float], queue_wait_samples: list[float]
    ) -> dict[str, int | float]:
        out = dict(values)
        observed_waits = int(out["queue_wait_observations"])
        executions = int(out["scalar_executions"]) + int(out["batch_executions"])
        out["queue_wait_mean_s"] = (
            float(out["queue_wait_total_s"]) / observed_waits if observed_waits else 0.0
        )
        out["queue_wait_p95_s"] = _percentile(queue_wait_samples, 0.95)
        out["execution_latency_mean_s"] = (
            float(out["execution_latency_total_s"]) / executions if executions else 0.0
        )
        return out

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            values = self._finalize(
                {**self._empty_values(), **self._values},
                list(self._queue_wait_samples),
            )
            return {
                "mode": self.mode,
                "pid": os.getpid(),
                "started_unix_s": self.started_unix_s,
                **values,
                "batch_size_histogram": {
                    str(size): count for size, count in sorted(self._batch_histogram.items())
                },
                "tools": {
                    tool: {
                        **self._finalize(
                            tool_values,
                            list(self._tool_queue_wait_samples.get(tool, ())),
                        ),
                        "batch_size_histogram": {
                            str(size): count
                            for size, count in sorted(
                                self._tool_batch_histograms.get(tool, {}).items()
                            )
                        },
                    }
                    for tool, tool_values in sorted(self._tool_values.items())
                },
            }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class _ConnectionWriter:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.lock = threading.Lock()

    def send(self, frame: dict[str, Any]) -> int:
        with self.lock:
            return write_frame(self.stream, frame)


class _CountingReader:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.bytes_read += len(data)
        return data


def _error_frame(request: _Request, exc: BaseException) -> dict[str, Any]:
    return {
        "id": request.request_id,
        "session_id": request.session_id,
        "episode_id": request.episode_id,
        "kind": "error",
        "error": {
            "tool": request.tool,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        },
    }


class _ToolWorker:
    def __init__(
        self,
        registry: Any,
        tool: str,
        *,
        mode: ExecutionMode,
        max_queue_size: int,
        max_batch_size: int | None,
        max_wait_ms: float | None,
        metrics: _Metrics,
        execution_lock: threading.Lock,
    ) -> None:
        self.registry = registry
        self.tool = tool
        self.batching = registry.batching(tool) if mode == "batched" else None
        self.max_batch_size = (
            min(self.batching.max_batch_size, max_batch_size)
            if self.batching is not None and max_batch_size is not None
            else self.batching.max_batch_size
            if self.batching is not None
            else 1
        )
        self.max_wait_ms = (
            min(self.batching.max_wait_ms, max_wait_ms)
            if self.batching is not None and max_wait_ms is not None
            else self.batching.max_wait_ms
            if self.batching is not None
            else 0.0
        )
        self.requests: queue.Queue[_Request | None] = queue.Queue()
        self.pending: deque[_Request] = deque()
        self._capacity = threading.BoundedSemaphore(max_queue_size)
        self.metrics = metrics
        self.execution_lock = execution_lock
        self._stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"gap-tool-{tool}", daemon=True)
        self.thread.start()

    def submit(self, request: _Request) -> None:
        self.metrics.add("request_bytes", request.request_bytes, tool=self.tool)
        if not self._capacity.acquire(blocking=False):
            self.metrics.add("requests_rejected", tool=self.tool)
            raise ToolQueueFull(f"queue for {self.tool!r} reached max size; retry after load falls")
        self.requests.put(request)
        self.metrics.add("requests_accepted", tool=self.tool)

    def close(self) -> None:
        self.request_stop()
        self.wait_closed()

    def request_stop(self) -> None:
        self._stopping.set()
        self.requests.put(None)

    def wait_closed(self) -> None:
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            logger.warning("tool worker %r did not stop within shutdown grace", self.tool)

    def _run(self) -> None:
        while True:
            if self._stopping.is_set():
                self._cancel_queued()
                return
            first = self.pending.popleft() if self.pending else self.requests.get()
            if first is None:
                return
            if self._stopping.is_set():
                self.pending.appendleft(first)
                self._cancel_queued()
                return
            if first.expired():
                self._finish_error(first, ToolRequestExpired("request expired in tool queue"))
                continue
            batch = [first]
            if self.batching is not None:
                compatibility_key = self.batching.compatibility_key
                try:
                    first_key = compatibility_key(first.args) if compatibility_key else None
                except BaseException as exc:
                    self._finish_error(first, exc)
                    continue
                deadline = time.monotonic() + self.max_wait_ms / 1000.0
                while len(batch) < self.max_batch_size:
                    matching_index = None
                    for index, pending in enumerate(self.pending):
                        try:
                            pending_key = (
                                compatibility_key(pending.args) if compatibility_key else None
                            )
                        except BaseException as exc:
                            del self.pending[index]
                            self._finish_error(pending, exc)
                            break
                        if pending.expired():
                            del self.pending[index]
                            self._finish_error(
                                pending, ToolRequestExpired("request expired in tool queue")
                            )
                            break
                        if pending_key == first_key:
                            matching_index = index
                            break
                    else:
                        index = -1
                    if index != -1 and matching_index is None:
                        continue
                    if matching_index is not None:
                        batch.append(self.pending[matching_index])
                        del self.pending[matching_index]
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self.requests.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if item is None:
                        self.requests.put(None)
                        break
                    if item.expired():
                        self._finish_error(
                            item, ToolRequestExpired("request expired in tool queue")
                        )
                        continue
                    try:
                        item_key = compatibility_key(item.args) if compatibility_key else None
                    except BaseException as exc:
                        self._finish_error(item, exc)
                        continue
                    if item_key != first_key:
                        self.pending.append(item)
                        continue
                    batch.append(item)
            self._execute(batch)

    def _cancel_queued(self) -> None:
        while self.pending:
            self._finish_error(self.pending.popleft(), ToolServerStopping("server stopping"))
        while True:
            try:
                request = self.requests.get_nowait()
            except queue.Empty:
                return
            if request is not None:
                self._finish_error(request, ToolServerStopping("server stopping"))

    def _finish_error(self, request: _Request, exc: BaseException) -> None:
        self.metrics.add(
            "requests_expired" if isinstance(exc, ToolRequestExpired) else "requests_failed",
            tool=self.tool,
        )
        try:
            written = request.writer.send(_error_frame(request, exc))
            self.metrics.add("response_bytes", written, tool=self.tool)
        except BaseException:
            logger.debug("client disconnected before error reply", exc_info=True)
        finally:
            self._capacity.release()

    def _execute(self, batch: list[_Request]) -> None:
        live: list[_Request] = []
        for request in batch:
            if request.expired():
                self._finish_error(request, ToolRequestExpired("request expired before execution"))
            else:
                live.append(request)
        if not live:
            return
        batch = live
        self.metrics.execution(
            tool=self.tool,
            batch_size=len(batch),
            queue_waits=[max(0.0, time.monotonic() - item.enqueued_at) for item in batch],
            batched=self.batching is not None,
        )
        started = time.monotonic()
        try:
            with self.execution_lock:
                if self.batching is None:
                    results = [
                        self.registry.invoke(self.tool, None, **request.args) for request in batch
                    ]
                else:
                    results = self.registry.invoke_batch(
                        self.tool, [request.args for request in batch]
                    )
        except BaseException as exc:
            results = [exc] * len(batch)
        finally:
            self.metrics.execution_latency(self.tool, time.monotonic() - started)

        for request, result in zip(batch, results, strict=True):
            frame = (
                _error_frame(request, result)
                if isinstance(result, BaseException)
                else {
                    "id": request.request_id,
                    "session_id": request.session_id,
                    "episode_id": request.episode_id,
                    "kind": "result",
                    "result": result,
                }
            )
            try:
                written = request.writer.send(frame)
            except (BrokenPipeError, OSError):
                logger.debug("client disconnected before reply", exc_info=True)
                self.metrics.add("requests_failed", tool=self.tool)
            except BaseException as exc:
                self.metrics.add("requests_failed", tool=self.tool)
                try:
                    fallback_written = request.writer.send(_error_frame(request, exc))
                    self.metrics.add("response_bytes", fallback_written, tool=self.tool)
                except BaseException:
                    logger.debug("could not serialize fallback error reply", exc_info=True)
            else:
                self.metrics.add("response_bytes", written, tool=self.tool)
                self.metrics.add(
                    "requests_failed"
                    if isinstance(result, BaseException)
                    else "requests_completed",
                    tool=self.tool,
                )
            finally:
                self._capacity.release()


class SharedToolServer:
    """Serve one tool registry to many local clients over a Unix socket."""

    def __init__(
        self,
        registry: Any,
        socket_path: str | Path,
        *,
        mode: ExecutionMode = "serialized",
        max_queue_size: int = 64,
        max_batch_size: int | None = None,
        max_wait_ms: float | None = None,
        bundle_name: str = "",
    ) -> None:
        if mode not in ("serialized", "batched"):
            raise ValueError(f"unsupported tool execution mode {mode!r}")
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms is not None and max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        self.registry = registry
        self.socket_path = Path(socket_path)
        self.mode = mode
        self.max_queue_size = max_queue_size
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.bundle_name = bundle_name
        self.metrics = _Metrics(mode=mode)
        self._execution_lock = threading.Lock()
        self._owner_lock_file: Any | None = None
        self._listener: socket.socket | None = None
        self._workers: dict[str, _ToolWorker] = {}
        self._workers_lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._connections_lock = threading.Lock()
        self._stopped = threading.Event()
        self._writers_may_close = threading.Event()

    def serve_forever(self) -> None:
        from .server import _build_catalog

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.socket_path.with_name(f"{self.socket_path.name}.lock")
        self._owner_lock_file = lock_path.open("a+")
        try:
            fcntl.flock(self._owner_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._owner_lock_file.close()
            self._owner_lock_file = None
            raise RuntimeError(f"shared tool owner already holds {lock_path}") from exc
        if self.socket_path.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.1)
            try:
                probe.connect(str(self.socket_path))
            except OSError:
                self.socket_path.unlink()
            else:
                self._release_owner_lock()
                raise RuntimeError(f"shared tool owner is already listening at {self.socket_path}")
            finally:
                probe.close()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener = listener
        try:
            listener.bind(str(self.socket_path))
        except BaseException:
            listener.close()
            self._release_owner_lock()
            raise
        listener.listen()
        listener.settimeout(0.1)
        catalog = _build_catalog(self.registry)
        try:
            while not self._stopped.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    if self._stopped.is_set():
                        break
                    raise
                with self._connections_lock:
                    self._connections.add(connection)
                threading.Thread(
                    target=self._serve_connection,
                    args=(connection, catalog),
                    daemon=True,
                ).start()
        finally:
            self.close()

    def close(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        if self._listener is not None:
            self._listener.close()
        # Workers must drain/cancel queued requests while connection writers
        # are still live, otherwise clients observe EOF instead of the
        # request-scoped ToolServerStopping error.
        with self._workers_lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.request_stop()
        for worker in workers:
            worker.wait_closed()
        self._writers_may_close.set()
        with self._connections_lock:
            connections = list(self._connections)
            self._connections.clear()
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        self._release_owner_lock()

    def _release_owner_lock(self) -> None:
        if self._owner_lock_file is None:
            return
        fcntl.flock(self._owner_lock_file, fcntl.LOCK_UN)
        self._owner_lock_file.close()
        self._owner_lock_file = None

    def _worker(self, tool: str) -> _ToolWorker:
        with self._workers_lock:
            if self._stopped.is_set():
                raise ToolServerStopping("server stopping")
            worker = self._workers.get(tool)
            if worker is None:
                self.registry.get(tool)  # fail before accepting the request
                worker = _ToolWorker(
                    self.registry,
                    tool,
                    mode=self.mode,
                    max_queue_size=self.max_queue_size,
                    max_batch_size=self.max_batch_size,
                    max_wait_ms=self.max_wait_ms,
                    metrics=self.metrics,
                    execution_lock=self._execution_lock,
                )
                self._workers[tool] = worker
            return worker

    def _serve_connection(self, connection: socket.socket, catalog: list[dict[str, Any]]) -> None:
        reader_stream = connection.makefile("rb")
        reader = _CountingReader(reader_stream)
        writer_stream = connection.makefile("wb")
        writer = _ConnectionWriter(writer_stream)
        try:
            writer.send(
                {
                    "id": "hs-0",
                    "kind": "catalog",
                    "tools": catalog,
                    "server": {
                        "mode": self.mode,
                        "bundle": self.bundle_name,
                        "protocol_version": SHARED_TOOL_PROTOCOL_VERSION,
                        "pid": os.getpid(),
                        "max_queue_size": self.max_queue_size,
                        "max_batch_size": self.max_batch_size,
                        "max_wait_ms": self.max_wait_ms,
                    },
                }
            )
            while not self._stopped.is_set():
                try:
                    frame_start = reader.bytes_read
                    frame = decode_frame(reader)
                except FrameError:
                    return
                if frame is None:
                    return
                if frame.get("kind") == "metrics":
                    writer.send(
                        {
                            "id": str(frame.get("id", "")),
                            "session_id": str(frame.get("session_id", "")),
                            "episode_id": str(frame.get("episode_id", "")),
                            "kind": "metrics",
                            "metrics": self.metrics.snapshot(),
                        }
                    )
                    continue
                request = _Request(
                    request_id=str(frame.get("id", "")),
                    session_id=str(frame.get("session_id", "")),
                    episode_id=str(frame.get("episode_id", "")),
                    tool=str(frame.get("tool", "")),
                    args=frame.get("args") or {},
                    writer=writer,
                    enqueued_at=time.monotonic(),
                    request_bytes=reader.bytes_read - frame_start,
                    deadline_unix_s=(
                        float(frame["deadline_unix_s"])
                        if frame.get("deadline_unix_s") is not None
                        else None
                    ),
                )
                if (
                    frame.get("kind") != "call"
                    or not isinstance(request.args, dict)
                    or not request.session_id
                    or not request.episode_id
                ):
                    writer.send(_error_frame(request, ValueError("invalid call frame")))
                    continue
                try:
                    self._worker(request.tool).submit(request)
                except BaseException as exc:
                    written = writer.send(_error_frame(request, exc))
                    self.metrics.add("response_bytes", written, tool=request.tool)
        finally:
            if self._stopped.is_set():
                self._writers_may_close.wait(timeout=5.0)
            reader_stream.close()
            writer_stream.close()
            with self._connections_lock:
                self._connections.discard(connection)
            connection.close()


class SharedToolClient:
    """Scalar client for a shared local bundle server."""

    def __init__(
        self,
        bundle_name: str,
        socket_path: str | Path,
        *,
        call_timeout_s: float | None = None,
        expected_mode: ExecutionMode | None = None,
        connect_timeout_s: float | None = None,
        session_id: str | None = None,
        episode_id: str | None = None,
    ) -> None:
        self.bundle_name = bundle_name
        self.socket_path = Path(socket_path)
        self.session_id = (
            session_id
            or os.environ.get("GAP_SESSION_ID", "").strip()
            or f"session-{os.getpid()}-{uuid.uuid4().hex}"
        )
        self.episode_id = episode_id or os.environ.get("GAP_EPISODE_ID", "").strip() or "episode-0"
        self._call_timeout_s = _resolve_call_timeout(call_timeout_s)
        self._socket: socket.socket | None = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._reader: Any | None = None
        self._writer: Any | None = None
        self._lock = threading.Lock()
        self._counter = 0
        try:
            initial_timeout = connect_timeout_s or self._call_timeout_s
            if initial_timeout:
                self._socket.settimeout(initial_timeout)
            self._socket.connect(str(self.socket_path))
            self._reader = self._socket.makefile("rb")
            self._writer = self._socket.makefile("wb")
            frame = self._read()
            if frame.get("kind") != "catalog":
                raise ToolClientError("shared tool server did not send a catalog")
            self.server_info = dict(frame.get("server") or {})
            if self.server_info.get("protocol_version") != SHARED_TOOL_PROTOCOL_VERSION:
                raise ToolClientError(
                    "shared tool protocol mismatch: expected "
                    f"{SHARED_TOOL_PROTOCOL_VERSION}, got "
                    f"{self.server_info.get('protocol_version')!r}"
                )
            server_bundle = self.server_info.get("bundle")
            if server_bundle and server_bundle != self.bundle_name:
                raise ToolClientError(
                    f"shared tool server bundle mismatch: expected {self.bundle_name!r}, "
                    f"got {server_bundle!r}"
                )
            if expected_mode is not None and self.server_info.get("mode") != expected_mode:
                raise ToolClientError(
                    f"shared tool server mode mismatch: expected {expected_mode!r}, "
                    f"got {self.server_info.get('mode')!r}"
                )
            self._catalog = _parse_catalog(frame)
            self._socket.settimeout(self._call_timeout_s)
        except BaseException:
            self.close()
            raise

    @classmethod
    def wait_until_ready(
        cls,
        bundle_name: str,
        socket_path: str | Path,
        *,
        startup_timeout_s: float,
        call_timeout_s: float | None = None,
        expected_mode: ExecutionMode | None = None,
        session_id: str | None = None,
        episode_id: str | None = None,
    ) -> SharedToolClient:
        """Connect after the externally supervised shared owner is ready."""
        deadline = time.monotonic() + startup_timeout_s
        last_error: OSError | ToolClientError | None = None
        while time.monotonic() < deadline:
            remaining = max(0.001, deadline - time.monotonic())
            try:
                return cls(
                    bundle_name,
                    socket_path,
                    call_timeout_s=call_timeout_s,
                    expected_mode=expected_mode,
                    connect_timeout_s=remaining,
                    session_id=session_id,
                    episode_id=episode_id,
                )
            except (OSError, ToolClientError) as exc:
                last_error = exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        raise ToolClientError(
            f"shared tool server {bundle_name!r} was not ready at {socket_path} "
            f"within {startup_timeout_s:.1f}s: {last_error}"
        )

    @property
    def catalog(self) -> list[CatalogEntry]:
        return list(self._catalog)

    def call(self, tool: str, /, **kwargs: Any) -> Any:
        with self._lock:
            self._counter += 1
            request_id = f"{os.getpid()}-{id(self)}-{self._counter}"
            write_frame(
                self._writer,
                {
                    "id": request_id,
                    "session_id": self.session_id,
                    "episode_id": self.episode_id,
                    "kind": "call",
                    "tool": tool,
                    "args": kwargs,
                    "deadline_unix_s": (
                        time.time() + self._call_timeout_s if self._call_timeout_s else None
                    ),
                },
            )
            try:
                frame = self._read()
            except TimeoutError:
                self.close()
                raise ToolCallTimeout(
                    f"shared tool call {tool!r} exceeded "
                    f"{self._call_timeout_s:.1f}s; connection closed"
                ) from None
            if frame.get("id") != request_id:
                raise ToolClientError("shared tool reply id mismatch")
            self._validate_identity(frame)
            if frame.get("kind") == "result":
                return frame.get("result")
            if frame.get("kind") == "error":
                error = frame.get("error") or {}
                raise ToolRemoteError(
                    tool=str(error.get("tool", tool)),
                    remote_type=str(error.get("type", "Exception")),
                    message=str(error.get("message", "")),
                    remote_tb=str(error.get("traceback", "")),
                )
            raise ToolClientError(f"unexpected shared tool reply {frame.get('kind')!r}")

    def metrics(self) -> dict[str, Any]:
        """Return a point-in-time server metrics snapshot."""
        with self._lock:
            self._counter += 1
            request_id = f"metrics-{os.getpid()}-{id(self)}-{self._counter}"
            write_frame(
                self._writer,
                {
                    "id": request_id,
                    "session_id": self.session_id,
                    "episode_id": self.episode_id,
                    "kind": "metrics",
                },
            )
            frame = self._read()
            if frame.get("id") != request_id or frame.get("kind") != "metrics":
                raise ToolClientError("invalid shared tool metrics reply")
            self._validate_identity(frame)
            return dict(frame.get("metrics") or {})

    def _validate_identity(self, frame: dict[str, Any]) -> None:
        if frame.get("session_id") != self.session_id:
            self.close()
            raise ToolClientError(
                "shared tool reply session mismatch: expected "
                f"{self.session_id!r}, got {frame.get('session_id')!r}"
            )
        if frame.get("episode_id") != self.episode_id:
            self.close()
            raise ToolClientError(
                "shared tool reply episode mismatch: expected "
                f"{self.episode_id!r}, got {frame.get('episode_id')!r}"
            )

    def set_episode_id(self, episode_id: str) -> None:
        """Rebind future calls after an Arena partial reset."""
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        with self._lock:
            self.episode_id = episode_id

    def close(self) -> None:
        socket_object = self._socket
        if socket_object is None:
            return
        self._socket = None
        try:
            try:
                socket_object.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            socket_object.close()
            if self._writer is not None:
                self._writer.close()
            if self._reader is not None:
                self._reader.close()
        except (OSError, ValueError):
            pass
        finally:
            self._writer = None
            self._reader = None

    def _read(self) -> dict[str, Any]:
        try:
            frame = decode_frame(self._reader)
        except TimeoutError:
            raise
        except (FrameError, OSError) as exc:
            raise ToolClientError(f"invalid shared tool frame: {exc}") from exc
        if frame is None:
            raise ToolClientError("shared tool server closed the connection")
        return frame


def _parse_catalog(frame: dict[str, Any]) -> list[CatalogEntry]:
    entries: list[CatalogEntry] = []
    for raw in frame.get("tools") or []:
        schema = raw.get("schema") or {}
        entries.append(
            CatalogEntry(
                name=str(raw.get("name", "")),
                summary=str(raw.get("summary", "")),
                tags=list(raw.get("tags") or []),
                metadata=dict(raw.get("metadata") or {}),
                schema_inputs=dict(schema.get("inputs") or {}),
                schema_outputs=dict(schema.get("outputs") or {}),
            )
        )
    return entries


__all__ = [
    "ExecutionMode",
    "SHARED_TOOL_PROTOCOL_VERSION",
    "SharedToolClient",
    "SharedToolServer",
    "ToolQueueFull",
    "ToolRequestExpired",
    "ToolServerStopping",
]
