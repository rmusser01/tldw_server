"""Local process executor for managed vLLM instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from tldw_Server_API.app.core.VLLM_Management.command_builder import build_vllm_serve_argv
from tldw_Server_API.app.core.VLLM_Management.executors.base import (
    LifecycleResult,
    ProbeResult,
    StopResult,
)
from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceRecord
from tldw_Server_API.app.core.Workflows.subprocess_utils import (
    SubprocessTask,
    start_process,
    terminate_process,
)


def _derive_base_url(instance: VLLMInstanceRecord) -> str:
    explicit = (
        instance.routing_policy.get("base_url")
        or instance.launch_spec.get("base_url")
        or instance.transport_config.get("base_url")
    )
    if explicit:
        return str(explicit).rstrip("/")
    host = instance.launch_spec.get("host") or instance.transport_config.get("host") or "127.0.0.1"
    port = instance.launch_spec.get("port") or instance.transport_config.get("port") or 8000
    scheme = instance.launch_spec.get("scheme") or instance.transport_config.get("scheme") or "http"
    return f"{scheme}://{host}:{port}/v1"


class LocalVLLMExecutor:
    def __init__(
        self,
        *,
        command_builder: Callable[[dict[str, Any]], list[str]] = build_vllm_serve_argv,
        process_starter: Callable[[list[str], str | Path, str | Path], SubprocessTask] = start_process,
        process_terminator: Callable[[SubprocessTask, int], tuple[bool, bool]] = terminate_process,
        probe_func: Callable[[VLLMInstanceRecord], ProbeResult] | None = None,
        log_root: str | Path = "Databases/vllm_logs",
    ) -> None:
        self._command_builder = command_builder
        self._process_starter = process_starter
        self._process_terminator = process_terminator
        self._probe_func = probe_func
        self._log_root = Path(log_root)

    def start(self, instance: VLLMInstanceRecord) -> LifecycleResult:
        argv = self._command_builder(instance.launch_spec)
        workdir = instance.transport_config.get("workdir") or "."
        log_dir = instance.transport_config.get("log_dir") or (self._log_root / instance.instance_id)
        task = self._process_starter(argv, workdir, log_dir)
        task_cmd = list(getattr(task, "cmd", argv))
        task_workdir = str(getattr(task, "workdir", workdir))
        stdout_path = str(getattr(task, "stdout_path"))
        stderr_path = str(getattr(task, "stderr_path"))
        return LifecycleResult(
            status="started",
            base_url=_derive_base_url(instance),
            handle={
                "pid": task.pid,
                "pgid": task.pgid,
                "command": task_cmd,
                "workdir": task_workdir,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "started_at": float(getattr(task, "started_at", 0.0)),
            },
            log_handle={
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
            },
        )

    def stop(self, instance: VLLMInstanceRecord, handle: dict[str, Any]) -> StopResult:
        task = SubprocessTask(
            cmd=list(handle.get("command") or []),
            workdir=Path(str(handle.get("workdir") or ".")),
            stdout_path=Path(str(handle.get("stdout_path") or "stdout.log")),
            stderr_path=Path(str(handle.get("stderr_path") or "stderr.log")),
            pid=int(handle["pid"]) if handle.get("pid") is not None else None,
            pgid=int(handle["pgid"]) if handle.get("pgid") is not None else None,
            started_at=float(handle.get("started_at") or 0.0),
        )
        terminated, forced = self._process_terminator(task, 5000)
        return StopResult(status="stopped" if terminated else "failed", forced=forced)

    def probe(self, instance: VLLMInstanceRecord) -> ProbeResult:
        if self._probe_func is not None:
            return self._probe_func(instance)
        return ProbeResult(status="unknown", reachable=False, base_url=_derive_base_url(instance))
