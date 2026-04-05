"""Jobs-backed lifecycle service for managed vLLM instances."""

from __future__ import annotations

import json
import os
import shlex
import subprocess  # nosec B404 - required for explicit argv-based SSH launcher execution
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
DEFAULT_STARTUP_TIMEOUT_SECONDS = 300


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
        if not identity_file:
            secret_ref = str(auth.get("secret_ref") or "").strip()
            if secret_ref:
                resolved_identity = str(os.getenv(secret_ref) or "").strip()
                if resolved_identity:
                    identity_file = resolved_identity
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


def build_probe_headers(instance: Any) -> dict[str, str]:
    """Build health-check headers for a managed instance."""

    launch_spec = dict(getattr(instance, "launch_spec", {}) or {})
    transport_config = dict(getattr(instance, "transport_config", {}) or {})
    headers: dict[str, str] = {}

    probe_headers = transport_config.get("probe_headers") or launch_spec.get("probe_headers") or {}
    if isinstance(probe_headers, dict):
        for key, value in probe_headers.items():
            if value is None:
                continue
            headers[str(key)] = str(value)

    api_key = launch_spec.get("api_key")
    if api_key is not None:
        header_name = str(launch_spec.get("api_key_header_name") or "Authorization").strip() or "Authorization"
        header_prefix = launch_spec.get("api_key_header_prefix")
        if header_prefix is None:
            header_prefix = "Bearer" if header_name.lower() == "authorization" else ""
        header_prefix = str(header_prefix).strip()
        api_key_value = str(api_key)
        headers[header_name] = f"{header_prefix} {api_key_value}".strip() if header_prefix else api_key_value

    return headers


def _probe_http_endpoint(base_url: str, *, headers: dict[str, str] | None = None) -> ProbeResult:
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        request = Request(models_url, headers=dict(headers or {}), method="GET")
        with urlopen(request, timeout=3) as response:  # nosec B310 - trusted operator-configured target
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
            str(instance.last_known_base_url or instance.launch_spec.get("base_url") or f"http://127.0.0.1:{instance.launch_spec.get('port') or 8000}/v1"),
            headers=build_probe_headers(instance),
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
            ),
            headers=build_probe_headers(instance),
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
        instance = self.repository.update_instance_runtime(
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
            handle = self._ensure_startup_started_at(instance=instance, handle=handle)
            probe = executor.probe(instance)
            updated = self._apply_probe_result(
                instance_id=instance_id,
                instance=instance,
                desired_state="running",
                probe=probe,
                handle=handle,
                fallback_base_url=lifecycle.base_url,
                allow_starting=not probe.reachable,
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
        allow_starting: bool = False,
    ):
        if probe.capabilities:
            effective_capabilities = derive_effective_capabilities(
                declared_capabilities=instance.declared_capabilities,
                probed_capabilities=probe.capabilities,
            )
        elif probe.reachable:
            effective_capabilities = derive_effective_capabilities(
                declared_capabilities=instance.declared_capabilities,
                probed_capabilities={},
            )
        else:
            effective_capabilities = {}
        handle = self._ensure_startup_started_at(instance=instance, handle=handle)
        observed_state = self._determine_observed_state(
            instance=instance,
            desired_state=desired_state,
            probe=probe,
            handle=handle,
            allow_starting=allow_starting,
        )
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

    def _determine_observed_state(
        self,
        *,
        instance: Any,
        desired_state: str,
        probe: ProbeResult,
        handle: dict[str, Any],
        allow_starting: bool,
    ) -> str:
        if probe.reachable:
            return "healthy"
        if desired_state == "stopped":
            return "stopped"
        if (allow_starting or str(instance.observed_state) == "starting") and not self._startup_timed_out(
            instance=instance,
            handle=handle,
        ):
            return "starting"
        return "unhealthy"

    def _startup_timed_out(self, *, instance: Any, handle: dict[str, Any]) -> bool:
        started_at = self._resolve_start_reference(instance=instance, handle=handle)
        if started_at is None:
            return False
        return (datetime.now(timezone.utc) - started_at).total_seconds() >= self._startup_timeout_seconds()

    def _ensure_startup_started_at(self, *, instance: Any, handle: dict[str, Any]) -> dict[str, Any]:
        normalized_handle = dict(handle or {})
        if normalized_handle.get("started_at"):
            return normalized_handle
        for candidate in (getattr(instance, "updated_at", None), getattr(instance, "created_at", None)):
            if self._parse_timestamp(candidate) is not None:
                normalized_handle["started_at"] = str(candidate)
                return normalized_handle
        normalized_handle["started_at"] = datetime.now(timezone.utc).isoformat()
        return normalized_handle

    def _resolve_start_reference(self, *, instance: Any, handle: dict[str, Any]) -> datetime | None:
        handle_started_at = handle.get("started_at")
        if handle_started_at:
            parsed = self._parse_timestamp(handle_started_at)
            if parsed is not None:
                return parsed
        created_at = getattr(instance, "created_at", None)
        parsed = self._parse_timestamp(created_at)
        if parsed is not None:
            return parsed
        return None

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _startup_timeout_seconds() -> int:
        raw_value = str(os.getenv("VLLM_MANAGEMENT_STARTUP_TIMEOUT_SECONDS") or "").strip()
        if not raw_value:
            return DEFAULT_STARTUP_TIMEOUT_SECONDS
        try:
            timeout_seconds = int(raw_value)
        except ValueError:
            logger.warning(
                "Invalid VLLM_MANAGEMENT_STARTUP_TIMEOUT_SECONDS=%r; using default %s",
                raw_value,
                DEFAULT_STARTUP_TIMEOUT_SECONDS,
            )
            return DEFAULT_STARTUP_TIMEOUT_SECONDS
        return max(timeout_seconds, 0)

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
    "build_probe_headers",
    "build_default_executor_map",
    "vllm_management_queue",
]
