"""Snapshot metrics from shared tool owners before evaluator shutdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gap_core.rpc.shared import SharedToolClient


def snapshot_metrics(
    socket_dir: str | Path,
    bundles: list[str],
    *,
    expected_modes: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    socket_dir = Path(socket_dir)
    modes = expected_modes or {}
    snapshots = {}
    for bundle in bundles:
        client = SharedToolClient(
            bundle,
            socket_dir / f"{bundle}.sock",
            expected_mode=modes.get(bundle),
            session_id="metrics-collector",
            episode_id="shutdown",
        )
        try:
            snapshots[bundle] = client.metrics()
        finally:
            client.close()
    return snapshots


def _mode_map(values: list[str]) -> dict[str, str]:
    modes = {}
    for value in values:
        bundle, separator, mode = value.partition("=")
        if not separator or not bundle or mode not in ("serialized", "batched"):
            raise argparse.ArgumentTypeError(
                f"mode must be BUNDLE=serialized|batched, got {value!r}"
            )
        modes[bundle] = mode
    return modes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket-dir", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--mode", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = snapshot_metrics(
        args.socket_dir,
        args.bundle,
        expected_modes=_mode_map(args.mode),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
