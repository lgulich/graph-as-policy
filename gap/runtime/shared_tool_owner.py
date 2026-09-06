"""Supervisor for one shared, multi-client tool-bundle owner process."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from gap_core.rpc.client import ToolClientError
from gap_core.rpc.shared import SharedToolClient

logger = logging.getLogger(__name__)


def _kill_server_group(process: subprocess.Popen) -> None:
    """Unconditionally kill a shared server group without native handlers.

    Some heavyweight GPU bundles load native libraries that replace SIGTERM
    handlers and fault while tearing down worker threads.  Shared owners are
    stopped only after clients close and metrics are captured, so a direct
    SIGKILL is the reliable process-lifecycle boundary here.
    """
    try:
        process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    if process_group <= 1 or process_group == os.getpgrp():
        raise RuntimeError(f"refusing to kill unsafe process group {process_group}")
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        logger.warning(
            "shared tool server pid=%d did not exit after SIGKILL", process.pid
        )


def _server_command(args: argparse.Namespace, socket_path: Path) -> list[str]:
    command = [
        "python",
        "-m",
        "gap_core.rpc.server",
        "--bundle",
        args.bundle,
        "--socket",
        str(socket_path),
        "--mode",
        args.mode,
        "--max-queue-size",
        str(args.max_queue_size),
    ]
    if args.max_batch_size is not None:
        command.extend(("--max-batch-size", str(args.max_batch_size)))
    if args.max_wait_ms is not None:
        command.extend(("--max-wait-ms", str(args.max_wait_ms)))
    if args.no_uv:
        command[0] = sys.executable
        return command
    return ["uv", "run", "--project", str(args.bundle_dir), "--", *command]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--socket-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("serialized", "batched"),
        default=os.environ.get("GAP_TOOL_EXECUTION_MODE", "serialized"),
    )
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=int(os.environ.get("GAP_TOOL_MAX_QUEUE_SIZE", "64")),
    )
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--max-wait-ms", type=float)
    parser.add_argument("--startup-timeout-s", type=float, default=60.0)
    parser.add_argument("--shutdown-grace-s", type=float, default=5.0)
    parser.add_argument("--no-uv", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.bundle_dir.is_dir():
        raise SystemExit(f"bundle directory does not exist: {args.bundle_dir}")
    args.socket_dir.mkdir(parents=True, exist_ok=True)
    socket_path = args.socket_dir / f"{args.bundle}.sock"
    command = _server_command(args, socket_path)
    logger.info("starting shared tool owner: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(args.bundle_dir),
        env=os.environ.copy(),
        start_new_session=True,
    )
    stopping = False

    def _stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True
        _kill_server_group(process)

    previous_term = signal.signal(signal.SIGTERM, _stop)
    previous_int = signal.signal(signal.SIGINT, _stop)
    try:
        deadline = time.monotonic() + args.startup_timeout_s
        last_error: BaseException | None = None
        while time.monotonic() < deadline and process.poll() is None:
            remaining = max(0.001, deadline - time.monotonic())
            try:
                client = SharedToolClient(
                    args.bundle,
                    socket_path,
                    expected_mode=args.mode,
                    connect_timeout_s=min(0.2, remaining),
                )
            except (OSError, ToolClientError) as exc:
                last_error = exc
                time.sleep(min(0.05, remaining))
                continue
            client.close()
            logger.info(
                "shared tool owner ready: bundle=%s mode=%s socket=%s pid=%d",
                args.bundle,
                args.mode,
                socket_path,
                process.pid,
            )
            break
        else:
            if process.poll() is not None:
                raise RuntimeError(
                    f"shared tool owner exited during startup with code {process.returncode}"
                )
            raise TimeoutError(
                f"shared tool owner was not ready within {args.startup_timeout_s:.1f}s: "
                f"{last_error}"
            )
        return_code = process.wait()
        return 0 if stopping else return_code
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        if process.poll() is None and not stopping:
            _kill_server_group(process)


if __name__ == "__main__":
    raise SystemExit(main())
