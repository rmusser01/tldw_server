"""Launcher contract helpers for remote vLLM process control."""

from __future__ import annotations

import json
from typing import Any


def build_ssh_launcher_command(
    action: str,
    *,
    launcher_path: str,
    launch_spec: dict[str, Any] | None = None,
    instance_id: str | None = None,
    handle: dict[str, Any] | None = None,
) -> list[str]:
    command = [str(launcher_path), str(action)]
    if instance_id:
        command.extend(["--instance-id", str(instance_id)])
    if launch_spec is not None:
        command.extend(["--json-spec", json.dumps(launch_spec, sort_keys=True)])
    remote_pid = (handle or {}).get("remote_pid")
    if remote_pid is not None:
        command.extend(["--remote-pid", str(remote_pid)])
    return command
