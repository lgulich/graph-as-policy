"""``gap_tool_server`` — runs inside a bundle's own venv, dispatches tool calls.

Spawned by ``gap.runtime.tool_bundle_manager`` as::

    uv run --project <bundle_dir> -- python -m gap_core.rpc.server --bundle <name>

with stdin/stdout pipes connected to gap. The server:

1. Imports the bundle's ``tools.py`` (against the bundle's own deps), which
   drains the bundle's ``@tool`` decorators into a local ``ToolRegistry``.
2. Emits a one-shot ``catalog`` frame on stdout listing every tool name,
   tags, summary, and inferred schema so the gap-side ``ToolClient`` can
   register them with the runtime's flat tool catalog.
3. Loops on stdin reading ``call`` frames, dispatches by tool name, and
   writes back ``result`` or ``error`` frames.

The protocol is deliberately small: no streaming, no callbacks, no ctx
remoting. Tools that need ``NodeContext`` aren't candidates for the RPC
path — they declare ``protocol: in-process`` in SKILL.md and stay in
gap-runtime's venv. The current open-robot-skills surface (sam3, curobo,
geometry, …) takes pure inputs and returns pure outputs.
"""

from __future__ import annotations

import argparse
import importlib
import io
import logging
import os
import signal
import sys
import traceback
from typing import Any

from .codec import FrameError, decode_frame, write_frame

logger = logging.getLogger(__name__)


def _shutdown_shared_server(shared_server) -> None:
    """Stop socket workers and exit without running model finalizers.

    Several GPU tool stacks install native communication libraries whose
    interpreter-shutdown handlers are not safe after CUDA model teardown.
    The shared owner has already snapshotted metrics and closed its clients
    before signalling this process, so finish the RPC shutdown and bypass
    Python/C-extension finalization.  This is deliberately limited to the
    signal-driven shared-server process; normal in-process server tests still
    return from :meth:`SharedToolServer.serve_forever` normally.
    """
    shared_server.close()
    os._exit(0)


def _isolate_framing_stdout() -> io.BufferedWriter:
    """Move the msgpack frame channel off the shared stdout fd.

    The RPC protocol frames travel over the server's *original* stdout
    (fd 1, piped to gap). Anything else that writes to fd 1 — a stray
    ``print()`` in a tool, or (the common offender) a C-extension ``printf``
    from CUDA / Warp / CuRobo — injects bytes mid-frame and desyncs the
    length-prefixed protocol: the next ``decode_frame`` reads a bogus length
    and blocks forever in ``read(length)`` while the client blocks waiting
    for the reply (a both-sides-idle RPC deadlock).

    Defuse it once at startup: dup the real stdout to a private fd used
    solely for framing, then point fd 1 (and ``sys.stdout``) at stderr so
    any pollution — Python- or C-level — becomes harmless log output. The
    returned writer is buffered so ``write_frame``'s ``write``/``flush``
    never short-writes a frame.
    """
    framing_fd = os.dup(1)
    os.dup2(2, 1)  # fd 1 now aliases stderr; stray writes go there
    sys.stdout = sys.stderr  # Python-level prints follow
    return os.fdopen(framing_fd, "wb")  # buffered: write() consumes all bytes


def _build_catalog(registry) -> list[dict[str, Any]]:
    """Snapshot the registry's runtime tools as serializable catalog entries.

    The schema is converted to plain dicts (TypedDict-flavored) so the
    gap-side ToolClient can register them with the runtime's ToolRegistry
    without dragging gap-core schema classes across processes.
    """
    out: list[dict[str, Any]] = []
    for name, desc in registry.runtime_tools().items():
        schema = desc.schema
        out.append(
            {
                "name": name,
                "summary": desc.summary,
                "tags": list(desc.tags),
                "metadata": dict(desc.metadata),
                # Schema is a UnitSchema(inputs={name: FieldInfo}, outputs={...})
                # — flatten each FieldInfo to a plain dict so msgpack can carry it.
                "schema": {
                    "inputs": {
                        fname: {
                            "name": f.name,
                            "type_str": f.type_str,
                            "required": f.required,
                            "default": _to_plain(f.default),
                        }
                        for fname, f in schema.inputs.items()
                    },
                    "outputs": {
                        fname: {"name": f.name, "type_str": f.type_str}
                        for fname, f in schema.outputs.items()
                    },
                },
            }
        )
    return out


def _to_plain(v: Any) -> Any:
    """Sentinels and non-trivial defaults aren't worth round-tripping —
    only carry primitive defaults across the wire."""
    if v is None or isinstance(v, (str, int, float, bool, list, tuple, dict)):
        return v
    return None


def _import_bundle_tools(bundle_name: str) -> None:
    """Import the bundle's ``tools.py`` against the bundle's own venv.

    The bundle directory is the spawn cwd (set by the gap-side launcher).
    Importing the module drains its ``@tool`` decorators into
    :data:`gap_core.tools._registry._PENDING_TOOLS` for the registry to
    pick up.

    Side-by-side files (``_streaming.py``, ``_impl.py``, etc.) are
    importable under both ``import tools`` (when gap_tool_server runs the
    bundle standalone in its own venv) AND under the synthetic
    ``gap_skills.tools.<bundle>`` namespace that gap-runtime uses when it
    loads the bundle in-process. Bundles authored against the gap-runtime
    convention often write ``from gap_skills.tools.<bundle> import
    _streaming``, so we synthesize that namespace here too.
    """
    import os
    from types import ModuleType

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    # Synthesize the gap_skills.tools.<bundle> namespace package pointing
    # at the bundle dir so `from gap_skills.tools.<bundle> import <sibling>`
    # resolves. Mirror gap.skills._registry._ensure_synthetic_package.
    def _synth(dotted: str, path: str | None) -> ModuleType:
        if dotted in sys.modules:
            return sys.modules[dotted]
        parent, _, leaf = dotted.rpartition(".")
        if parent:
            _synth(parent, None)
        mod = ModuleType(dotted)
        mod.__package__ = dotted
        if path is not None:
            mod.__path__ = [path]  # type: ignore[attr-defined]
        sys.modules[dotted] = mod
        if parent:
            setattr(sys.modules[parent], leaf, mod)
        return mod

    # Bundles live under either tools/<name>/ or policies/<name>/ — match
    # gap.skills._registry's _KIND_DIRS so the synthetic path matches what
    # the bundle's source code expects.
    bundle_basename = os.path.basename(cwd)
    parent_basename = os.path.basename(os.path.dirname(cwd))
    namespace_seg = (
        parent_basename if parent_basename in ("tools", "skills", "policies") else "tools"
    )
    _synth(f"gap_skills.{namespace_seg}.{bundle_basename}", cwd)

    try:
        importlib.import_module("tools")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"gap_tool_server[{bundle_name}]: bundle has no `tools.py` in cwd={cwd!r} ({exc})"
        ) from exc


def _drain_registry():
    """Build a fresh ToolRegistry that owns the just-imported @tool entries."""
    from gap_core.tools import ToolRegistry  # local import — keep startup lean
    from gap_core.tools._registry import _PENDING_TOOLS  # type: ignore

    if not _PENDING_TOOLS:
        raise SystemExit("gap_tool_server: importing tools.py registered no @tool functions")
    reg = ToolRegistry()
    reg.discover_pending()
    return reg


def _serve(registry, stdin, stdout) -> int:
    catalog = _build_catalog(registry)
    write_frame(stdout, {"id": "hs-0", "kind": "catalog", "tools": catalog})

    while True:
        try:
            frame = decode_frame(stdin)
        except FrameError as exc:
            logger.error("frame decode failed: %s", exc)
            return 2
        if frame is None:
            # Clean EOF — gap-side closed our stdin (workflow ended).
            return 0
        fid = str(frame.get("id", ""))
        kind = frame.get("kind")
        if kind != "call":
            write_frame(
                stdout,
                {
                    "id": fid,
                    "kind": "error",
                    "error": {
                        "type": "ProtocolError",
                        "message": f"unexpected frame kind {kind!r}",
                    },
                },
            )
            continue
        name = str(frame.get("tool", ""))
        args = frame.get("args") or {}
        if not isinstance(args, dict):
            write_frame(
                stdout,
                {
                    "id": fid,
                    "kind": "error",
                    "error": {
                        "type": "ProtocolError",
                        "message": f"args must be a mapping, got {type(args).__name__}",
                    },
                },
            )
            continue
        try:
            result = registry.invoke(name, None, **{str(k): v for k, v in args.items()})
        except BaseException as exc:
            write_frame(
                stdout,
                {
                    "id": fid,
                    "kind": "error",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
            )
            continue
        write_frame(stdout, {"id": fid, "kind": "result", "result": result})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gap_tool_server",
        description=(
            "Out-of-process tool server. Imports the bundle's tools.py "
            "(in cwd) and dispatches @tool calls over stdio msgpack frames."
        ),
    )
    parser.add_argument("--bundle", required=True, help="Bundle name (for log labels)")
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--socket",
        help="Serve multiple local clients over this Unix socket instead of stdio",
    )
    parser.add_argument(
        "--mode",
        choices=("serialized", "batched"),
        default=os.environ.get("GAP_TOOL_EXECUTION_MODE", "serialized"),
        help="Shared-server execution mode (default: serialized)",
    )
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=int(os.environ.get("GAP_TOOL_MAX_QUEUE_SIZE", "64")),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=(
            int(os.environ["GAP_TOOL_MAX_BATCH_SIZE"])
            if os.environ.get("GAP_TOOL_MAX_BATCH_SIZE")
            else None
        ),
    )
    parser.add_argument(
        "--max-wait-ms",
        type=float,
        default=(
            float(os.environ["GAP_TOOL_MAX_WAIT_MS"])
            if os.environ.get("GAP_TOOL_MAX_WAIT_MS")
            else None
        ),
    )
    args = parser.parse_args(argv)

    framing_out = None
    if not args.socket:
        # Isolate the frame channel BEFORE importing the bundle — bundle import
        # is exactly when CUDA / Warp / CuRobo print their init banners to stdout.
        framing_out = _isolate_framing_stdout()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f"%(asctime)s [%(levelname)s] gap_tool_server[{args.bundle}]: %(message)s",
    )

    _import_bundle_tools(args.bundle)
    registry = _drain_registry()
    if args.socket:
        from .shared import SharedToolServer

        shared_server = SharedToolServer(
            registry,
            args.socket,
            mode=args.mode,
            max_queue_size=args.max_queue_size,
            max_batch_size=args.max_batch_size,
            max_wait_ms=args.max_wait_ms,
            bundle_name=args.bundle,
        )

        def _stop_shared_server(_signum, _frame) -> None:
            _shutdown_shared_server(shared_server)

        signal.signal(signal.SIGTERM, _stop_shared_server)
        signal.signal(signal.SIGINT, _stop_shared_server)
        shared_server.serve_forever()
        return 0
    # stdin stays raw binary for framing; stdout frames go over the isolated fd.
    return _serve(registry, sys.stdin.buffer, framing_out)


if __name__ == "__main__":
    raise SystemExit(main())
