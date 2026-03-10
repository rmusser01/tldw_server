"""SSH executor for managed vLLM instances."""

from __future__ import annotations

from typing import Any, Callable

from tldw_Server_API.app.core.VLLM_Management.executors.base import (
    LifecycleResult,
    ProbeResult,
    StopResult,
)
from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceRecord
from tldw_Server_API.app.core.VLLM_Management.ssh_launcher import build_ssh_launcher_command


def _derive_base_url(instance: VLLMInstanceRecord) -> str:
    explicit = (
        instance.routing_policy.get("base_url")
        or instance.launch_spec.get("base_url")
        or instance.transport_config.get("base_url")
    )
    if explicit:
        return str(explicit).rstrip("/")
    host = instance.transport_config.get("base_url") or instance.transport_config.get("host")
    port = instance.launch_spec.get("port") or instance.transport_config.get("service_port") or 8000
    scheme = instance.transport_config.get("scheme") or instance.launch_spec.get("scheme") or "http"
    return f"{scheme}://{host}:{port}/v1"


def _resolve_transport_user(transport: dict[str, Any]) -> str | None:
    user = transport.get("user")
    if user is None:
        user = transport.get("username")
    if user is None:
        return None
    return str(user)


class SSHVLLMExecutor:
    def __init__(
        self,
        *,
        ssh_runner: Any,
        launcher_command_builder: Callable[..., list[str]] = build_ssh_launcher_command,
        probe_func: Callable[[VLLMInstanceRecord], ProbeResult] | None = None,
    ) -> None:
        self._ssh_runner = ssh_runner
        self._launcher_command_builder = launcher_command_builder
        self._probe_func = probe_func

    def start(self, instance: VLLMInstanceRecord) -> LifecycleResult:
        transport = instance.transport_config or {}
        command = self._launcher_command_builder(
            "start",
            launcher_path=str(transport["launcher_path"]),
            launch_spec=instance.launch_spec,
            instance_id=instance.instance_id,
        )
        result = self._ssh_runner.run(
            command,
            host=transport["host"],
            port=int(transport.get("port") or 22),
            user=_resolve_transport_user(transport),
            auth=transport.get("auth"),
        )
        handle = dict(result) if isinstance(result, dict) else {"result": result}
        handle["launcher_command"] = command
        return LifecycleResult(status="started", base_url=_derive_base_url(instance), handle=handle)

    def stop(self, instance: VLLMInstanceRecord, handle: dict[str, Any]) -> StopResult:
        transport = instance.transport_config or {}
        command = self._launcher_command_builder(
            "stop",
            launcher_path=str(transport["launcher_path"]),
            instance_id=instance.instance_id,
            handle=handle,
        )
        self._ssh_runner.run(
            command,
            host=transport["host"],
            port=int(transport.get("port") or 22),
            user=_resolve_transport_user(transport),
            auth=transport.get("auth"),
        )
        return StopResult(status="stopped", forced=False)

    def probe(self, instance: VLLMInstanceRecord) -> ProbeResult:
        if self._probe_func is not None:
            return self._probe_func(instance)
        return ProbeResult(status="unknown", reachable=False, base_url=_derive_base_url(instance))
