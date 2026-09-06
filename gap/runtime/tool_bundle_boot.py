"""Discover which RPC-protocol tool bundles a workflow needs and boot them.

Parallel to :mod:`gap.runtime.policy_boot`. We discover the bundle set
from two sources:

1. Direct ``tool:`` nodes in the workflow (e.g. a node with
   ``tool: "sam3.segment"`` → the ``sam3`` bundle).
2. The ``allowed_tools`` list of every skill subgraph the workflow uses.
   Script nodes call tools via ``ctx.tool_call(...)``, not as workflow
   nodes — so a workflow's `target_sg` subgraph referencing a
   ``perceiving-objects-multiview`` skill must pre-boot every bundle that
   skill's ``allowed_tools`` declares (sam3, grounding-dino, vlm, …).
3. The workflow's own ``requires.tools``. Neither source above sees inside
   a script, and a subgraph labelled ``generic`` names no skill to ask —
   so a graph whose perception layer is its own scripts had no way to say
   what it calls, and booted nothing. See ``Workflow.requires_tools``.

A bundle whose SKILL.md declares ``gap.serving.protocol == "stdio-msgpack"``
is booted out-of-process via :class:`ToolBundleManager`. Bundles without a
``gap.serving:`` block (or with ``in-process`` protocol) are unaffected:
their @tool registrations land in the registry via the in-process path
exactly as today.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def required_rpc_tool_bundles(
    workflow_dir: str | Path,
    skill_registry: Any,
) -> set[str]:
    """Bundle names a workflow references whose serving.protocol is
    ``stdio-msgpack``."""
    out: set[str] = set()
    if skill_registry is None:
        return out
    from gap.runtime.workflow import load_workflow

    try:
        wf = load_workflow(Path(workflow_dir) / "workflow.json")
    except Exception:
        return out

    candidate_tool_names: set[str] = set(getattr(wf, "requires_tools", ()) or ())
    candidate_skills: set[str] = set()
    for sg in wf.subgraphs.values():
        # Skill that owns this subgraph contributes its allowed_tools — script
        # nodes call those via ctx.tool_call(...) and aren't workflow nodes.
        skill_name = getattr(sg, "skill", None)
        if skill_name:
            candidate_skills.add(skill_name)
        for node in sg.nodes.values():
            if node.type == "tool" and node.tool:
                candidate_tool_names.add(node.tool)

    for skill_name in candidate_skills:
        try:
            info = skill_registry.get(skill_name)
        except Exception:
            continue
        for tool_name in getattr(info.meta, "allowed_tools", []) or []:
            candidate_tool_names.add(tool_name)

    for tool_name in candidate_tool_names:
        bundle = tool_name.split(".", 1)[0]
        if bundle in out:
            continue
        try:
            info = skill_registry.get(bundle)
        except Exception:
            continue
        serving = getattr(info.meta, "serving", None)
        if serving is None:
            continue
        if getattr(serving, "protocol", None) == "stdio-msgpack":
            out.add(bundle)
    return out


def boot_tool_bundles(
    workflow_dir: str | Path,
    skill_registry: Any,
    tool_registry: Any,
    *,
    startup_timeout_s: float = 60.0,
    evict_grace_s: float = 5.0,
    shared_socket_dir: str | Path | None = None,
    execution_mode: str | None = None,
    execution_modes: Mapping[str, str] | None = None,
    session_id: str | None = None,
    episode_id: str | None = None,
) -> Any | None:
    """Boot every RPC tool bundle the workflow needs.

    Returns the :class:`ToolBundleManager` or ``None`` when no RPC bundle
    is referenced (the common case today — all current tool bundles run
    in-process). The caller MUST call ``manager.shutdown_all()`` in a
    ``finally`` to terminate the subprocesses.
    """
    required = required_rpc_tool_bundles(workflow_dir, skill_registry)
    if not required:
        return None

    from gap.runtime.tool_bundle_manager import ToolBundleManager

    manager = ToolBundleManager(
        skill_registry=skill_registry,
        tool_registry=tool_registry,
        startup_timeout_s=startup_timeout_s,
        evict_grace_s=evict_grace_s,
        shared_socket_dir=shared_socket_dir,
        execution_mode=execution_mode,
        execution_modes=execution_modes,
        session_id=session_id,
        episode_id=episode_id,
    )
    manager.boot_all(required)
    return manager


__all__ = ["boot_tool_bundles", "required_rpc_tool_bundles"]
