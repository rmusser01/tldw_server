"""Jobs-backed lifecycle service for managed vLLM instances."""

from __future__ import annotations

import json
import os
import shlex
import subprocess  # nosec B404 - required for explicit argv-based SSH launcher execution
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from loguru import logger

from tldw_Server_API.app.core.VLLM_Management import (
    derive_effective_capabilities,
    get_default_vllm_instance_repository,
    normalize_capabilities,
)
from tldw_Server_API.app.core.VLLM_Management.executors.agent import AgentVLLMExecutor
from tldw_Server_API.app.core.VLLM_Management.executors.base import ProbeResult, VLLMExecutor
from tldw_Server_API.app.core.VLLM_Management.executors.local import LocalVLLMExecutor
from tldw_Server_API.app.core.VLLM_Management.executors.ssh import SSHVLLMExecutor
from tldw_Server_API.app.core.VLLM_Management.repository import VLLMInstanceRepository

VLLM_MANAGEMENT_DOMAIN = "vllm_management"
VLLM_MANAGEMENT_QUEUE = "default"
VLLM_JOB_TYPE_BY_ACTION = {
    "start": "vllm_instance_start",
    "stop": "vllm_instance_stop",
    "restart": "vllm_instance_restart",
    "probe": "vllm_instance_probe",
}


class ShellSSHRunner:
    """Run remote launcher commands through the local ``ssh`` binary."""

    def __init__(self, *, connect_timeout_seconds: int = 10) -> None:
        self._connect_timeout_seconds = int(connect_timeout_seconds)

    def run(
        self,
        command: list[str],
        *,
        host: str,
        port: int,
        user: str | None,
        auth: dict[str, Any] | None,
    ) -> dict[str, Any]:
        argv = [
            "ssh",
            "-p",
            str(port),
            "-o",
            f"ConnectTimeout={self._connect_timeout_seconds}",
        ]
        auth = dict(auth or {})
        identity_file = auth.get("identity_file") or auth.get("private_key_path")
        if identity_file:
            argv.extend(["-i", str(identity_file)])
        if auth.get("strict_host_key_checking") is False:
            argv.extend(["-o", "StrictHostKeyChecking=no"])
        target = f"{user}@{host}" if user else str(host)
        remote_command = " ".join(shlex.quote(str(part)) for part in command)
        argv.extend([target, remote_command])
        completed = subprocess.run(  # nosec B603 - argv is built from validated structured fields; shell is disabled
            argv,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip() or "ssh command failed"
            raise RuntimeError(detail)
        stdout = (completed.stdout or "").strip()
        if not stdout:
            return {}
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            return {"stdout": stdout}
        return parsed if isinstance(parsed, dict) else {"result": parsed}


def vllm_management_queue() -> str:
    return str(os.getenv("VLLM_MANAGEMENT_JOBS_QUEUE") or VLLM_MANAGEMENT_QUEUE).strip() or VLLM_MANAGEMENT_QUEUE


def _probe_http_endpoint(base_url: str) -> ProbeResult:
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        with urlopen(models_url, timeout=3) as response:  # nosec B310 - trusted operator-configured target
            _ = response.read()
        return ProbeResult(status="healthy", reachable=True, base_url=base_url)
    except HTTPError as exc:
        return ProbeResult(status="unhealthy", reachable=False, base_url=base_url, detail=str(exc))
    except URLError as exc:
        return ProbeResult(status="unhealthy", reachable=False, base_url=base_url, detail=str(exc))
    except OSError as exc:
        return ProbeResult(status="unhealthy", reachable=False, base_url=base_url, detail=str(exc))


def build_default_executor_map() -> dict[str, VLLMExecutor]:
    local_executor = LocalVLLMExecutor(
        probe_func=lambda instance: _probe_http_endpoint(
            str(instance.last_known_base_url or instance.launch_spec.get("base_url") or f"http://127.0.0.1:{instance.launch_spec.get('port') or 8000}/v1")
        )
    )
    ssh_executor = SSHVLLMExecutor(
        ssh_runner=ShellSSHRunner(),
        probe_func=lambda instance: _probe_http_endpoint(
            str(
                instance.last_known_base_url
                or instance.transport_config.get("base_url")
                or instance.launch_spec.get("base_url")
                or f"http://{instance.transport_config.get('host')}:{instance.launch_spec.get('port') or instance.transport_config.get('service_port') or 8000}/v1"
            )
        ),
    )
    return {
        "local": local_executor,
        "ssh": ssh_executor,
        "agent": AgentVLLMExecutor(),
    }


class VLLMManagementService:
    def __init__(
        self,
        *,
        repository: VLLMInstanceRepository | None = None,
        job_manager: Any | None = None,
        executors: dict[str, VLLMExecutor] | None = None,
        queue: str | None = None,
    ) -> None:
        self.repository = repository or get_default_vllm_instance_repository()
        self.job_manager = job_manager
        self.executors = executors or build_default_executor_map()
        self.queue = queue or vllm_management_queue()

    def enqueue_start(self, instance_id: str, owner_user_id: str | None = None) -> dict[str, Any]:
        return self._enqueue_action("start", instance_id, owner_user_id=owner_user_id)

    def enqueue_stop(self, instance_id: str, owner_user_id: str | None = None) -> dict[str, Any]:
        return self._enqueue_action("stop", instance_id, owner_user_id=owner_user_id)

    def enqueue_restart(self, instance_id: str, owner_user_id: str | None = None) -> dict[str, Any]:
        return self._enqueue_action("restart", instance_id, owner_user_id=owner_user_id)

    def enqueue_probe(self, instance_id: str, owner_user_id: str | None = None) -> dict[str, Any]:
        return self._enqueue_action("probe", instance_id, owner_user_id=owner_user_id)

    def _enqueue_action(self, action: str, instance_id: str, *, owner_user_id: str | None) -> dict[str, Any]:
        self._require_job_manager()
        self._get_instance(instance_id)
        return self.job_manager.create_job(
            domain=VLLM_MANAGEMENT_DOMAIN,
            queue=self.queue,
            job_type=VLLM_JOB_TYPE_BY_ACTION[action],
            payload={"instance_id": instance_id, "action": action},
            owner_user_id=str(owner_user_id) if owner_user_id is not None else None,
            priority=5,
            max_retries=1,
        )

    def execute_action(self, action: str, instance_id: str) -> dict[str, Any]:
        if action == "start":
            return self.start_instance(instance_id)
        if action == "stop":
            return self.stop_instance(instance_id)
        if action == "restart":
            return self.restart_instance(instance_id)
        if action == "probe":
            return self.probe_instance(instance_id)
        raise ValueError(f"Unsupported managed vLLM action: {action}")

    def start_instance(self, instance_id: str) -> dict[str, Any]:
        instance = self._get_instance(instance_id)
        executor = self._get_executor(instance.execution_mode)
        self.repository.update_instance_runtime(
            instance_id,
            {
                "desired_state": "running",
                "observed_state": "starting",
                "last_error": None,
            },
        )
        try:
            lifecycle = executor.start(instance)
            handle = dict(lifecycle.handle or {})
            if lifecycle.log_handle:
                handle["log_handle"] = dict(lifecycle.log_handle)
            probe = executor.probe(instance)
            updated = self._apply_probe_result(
                instance_id=instance_id,
                instance=instance,
                desired_state="running",
                probe=probe,
                handle=handle,
                fallback_base_url=lifecycle.base_url,
            )
            return {
                "instance_id": instance_id,
                "action": "start",
                "status": updated.observed_state,
            }
        except Exception as exc:
            self.repository.update_instance_runtime(
                instance_id,
                {
                    "desired_state": "running",
                    "observed_state": "failed",
                    "last_error": str(exc),
                },
            )
            raise

    def stop_instance(self, instance_id: str) -> dict[str, Any]:
        instance = self._get_instance(instance_id)
        if instance.desired_state == "stopped" and instance.observed_state == "stopped":
            return {"instance_id": instance_id, "action": "stop", "status": "stopped"}
        executor = self._get_executor(instance.execution_mode)
        self.repository.update_instance_runtime(
            instance_id,
            {
                "desired_state": "stopped",
                "observed_state": "stopping",
                "last_error": None,
            },
        )
        result = executor.stop(instance, dict(instance.executor_handle or {}))
        observed_state = "stopped" if result.status == "stopped" else "failed"
        self.repository.update_instance_runtime(
            instance_id,
            {
                "desired_state": "stopped",
                "observed_state": observed_state,
                "last_known_base_url": None if observed_state == "stopped" else instance.last_known_base_url,
                "last_error": None if observed_state == "stopped" else result.detail,
                "executor_handle": {} if observed_state == "stopped" else instance.executor_handle,
            },
        )
        return {"instance_id": instance_id, "action": "stop", "status": observed_state}

    def restart_instance(self, instance_id: str) -> dict[str, Any]:
        self.stop_instance(instance_id)
        return self.start_instance(instance_id)

    def probe_instance(self, instance_id: str) -> dict[str, Any]:
        instance = self._get_instance(instance_id)
        executor = self._get_executor(instance.execution_mode)
        probe = executor.probe(instance)
        updated = self._apply_probe_result(
            instance_id=instance_id,
            instance=instance,
            desired_state=instance.desired_state,
            probe=probe,
            handle=dict(instance.executor_handle or {}),
            fallback_base_url=probe.base_url or instance.last_known_base_url,
        )
        return {"instance_id": instance_id, "action": "probe", "status": updated.observed_state}

    def _apply_probe_result(
        self,
        *,
        instance_id: str,
        instance: Any,
        desired_state: str,
        probe: ProbeResult,
        handle: dict[str, Any],
        fallback_base_url: str | None,
    ):
        if probe.capabilities:
            effective_capabilities = derive_effective_capabilities(
                declared_capabilities=instance.declared_capabilities,
                probed_capabilities=probe.capabilities,
            )
        elif probe.reachable:
            effective_capabilities = normalize_capabilities(instance.declared_capabilities)
        else:
            effective_capabilities = {}
        observed_state = "healthy" if probe.reachable else ("stopped" if desired_state == "stopped" else "unhealthy")
        return self.repository.update_instance_runtime(
            instance_id,
            {
                "desired_state": desired_state,
                "observed_state": observed_state,
                "last_known_base_url": probe.base_url or fallback_base_url,
                "last_error": None if probe.reachable else probe.detail,
                "probed_capabilities": dict(probe.capabilities or {}),
                "effective_capabilities": dict(effective_capabilities or {}),
                "executor_handle": handle,
            },
        )

    def _get_instance(self, instance_id: str):
        instance = self.repository.get_instance(instance_id)
        if instance is None:
            raise ValueError(f"Managed vLLM instance '{instance_id}' was not found")
        return instance

    def _get_executor(self, execution_mode: str) -> VLLMExecutor:
        executor = self.executors.get(str(execution_mode))
        if executor is None:
            raise ValueError(f"Unsupported managed vLLM execution mode: {execution_mode}")
        return executor

    def _require_job_manager(self) -> None:
        if self.job_manager is None:
            raise RuntimeError("Job manager is required for managed vLLM enqueue operations")


__all__ = [
    "VLLM_MANAGEMENT_DOMAIN",
    "VLLM_JOB_TYPE_BY_ACTION",
    "VLLMManagementService",
    "build_default_executor_map",
    "vllm_management_queue",
]
