from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ..image_store import ImageStoreValidationError, SandboxImageStore
from ..limits import collect_limited_artifacts
from ..macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
)
from ..models import RunPhase, RunSpec, RunStatus, RuntimeType
from ..policy import SandboxPolicyConfig
from ..runtime_capabilities import RuntimePreflightResult
from ..streams import get_hub
from ..utils import coerce_optional_nonempty_string
from .resource_limits import log_limit_counters
from .vz_common import _VZ_NONCRITICAL_EXCEPTIONS, VZBaseRunner, vz_host_facts

_VZ_LINUX_RUNNER_NONCRITICAL_EXCEPTIONS = (
    OSError,
    PermissionError,
    RuntimeError,
    TypeError,
    ValueError,
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
)
_OUTPUT_COUNTER_KEYS = frozenset(
    {
        "output_limit_bytes",
        "stdout_bytes_original",
        "stderr_bytes_original",
        "stdout_bytes_returned",
        "stderr_bytes_returned",
        "stdout_truncated",
        "stderr_truncated",
        "guest_output_limit_bytes",
        "guest_output_limit_exceeded",
        "guest_stdout_bytes_observed",
        "guest_stderr_bytes_observed",
        "guest_stdout_bytes_returned",
        "guest_stderr_bytes_returned",
    }
)


class VZLinuxRunner(VZBaseRunner):
    runtime_type = RuntimeType.vz_linux
    fake_exec_env_key = "TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC"
    max_helper_output_bytes = 256 * 1024 * 1024
    available_env_key = "TLDW_SANDBOX_VZ_LINUX_AVAILABLE"
    version_env_key = "TLDW_SANDBOX_VZ_LINUX_VERSION"
    template_ready_env_key = "TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY"
    template_missing_reason = "vz_linux_template_missing"
    helper_client_cls = MacOSVirtualizationHelperClient
    _active_lock = threading.RLock()
    _active_vm: dict[str, str] = {}
    _active_run_dir: dict[str, str] = {}

    def __init__(self, session_control_store: Any | None = None) -> None:
        self._session_control_store = session_control_store

    def preflight(self, network_policy: str | None = None) -> RuntimePreflightResult:
        host = vz_host_facts()
        reasons: list[str] = []
        execution_mode = "none"

        if host.get("os") != "darwin":
            reasons.append("macos_required")
        if not bool(host.get("apple_silicon")):
            reasons.append("apple_silicon_required")

        helper_ready = False
        if not reasons:
            try:
                helper_result = self.helper_client_cls().validate_vz_linux_host(
                    {
                        "runtime": self.runtime_type.value,
                        "network_policy": str(network_policy or "deny_all").strip().lower() or "deny_all",
                    }
                )
            except (MacOSVirtualizationHelperUnavailable, MacOSVirtualizationHelperProtocolError) as exc:
                default_reason = (
                    "macos_virtualization_helper_protocol_mismatch"
                    if isinstance(exc, MacOSVirtualizationHelperProtocolError)
                    else "macos_virtualization_helper_unavailable"
                )
                helper_result = {
                    "available": False,
                    "reasons": [str(exc) or default_reason],
                    "execution_mode": "none",
                }
            helper_reasons = [str(reason) for reason in helper_result.get("reasons", []) if str(reason).strip()]
            reasons.extend(helper_reasons)
            helper_ready = bool(helper_result.get("available"))
            execution_mode = str(helper_result.get("execution_mode") or "none").strip().lower() or "none"

        requested_policy = str(network_policy or "deny_all").strip().lower() or "deny_all"
        if requested_policy == "allowlist":
            reasons.append("strict_allowlist_not_supported")

        available = not reasons
        return RuntimePreflightResult(
            runtime=self.runtime_type,
            available=available,
            reasons=reasons,
            execution_mode=execution_mode,
            host={str(k): v for k, v in host.items()},
            enforcement_ready={"deny_all": helper_ready and execution_mode == "real", "allowlist": False},
        )

    @staticmethod
    def _path_within_root(root: Path, candidate: Path) -> bool:
        try:
            resolved = candidate.resolve(strict=False)
            root_resolved = root.resolve(strict=False)
        except _VZ_NONCRITICAL_EXCEPTIONS:
            return False
        return resolved == root_resolved or root_resolved in resolved.parents

    @classmethod
    def _register_active_run(cls, run_id: str, vm_id: str, run_dir: str | None = None) -> None:
        with cls._active_lock:
            cls._active_vm[run_id] = vm_id
            if run_dir:
                cls._active_run_dir[run_id] = run_dir

    @classmethod
    def _clear_active_run(cls, run_id: str) -> tuple[str | None, str | None]:
        with cls._active_lock:
            vm_id = cls._active_vm.pop(run_id, None)
            run_dir = cls._active_run_dir.pop(run_id, None)
        return vm_id, run_dir

    @staticmethod
    def _write_inline_files(workspace: str, files_inline: list[tuple[str, bytes]] | None) -> None:
        workspace_root = Path(workspace)
        if workspace_root.is_symlink():
            raise ValueError("workspace root must not be a symlink")
        for relative_path, data in files_inline or []:
            normalized = str(relative_path or "").replace("\\", "/").lstrip("/")
            parts = [part for part in normalized.split("/") if part]
            if not parts or any(part in {".", ".."} for part in parts):
                raise ValueError(f"invalid inline file path: {relative_path}")
            target = workspace_root.joinpath(*parts)
            if not VZLinuxRunner._path_within_root(workspace_root, target.parent):
                raise ValueError(f"inline file path escapes workspace: {relative_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.is_symlink():
                raise ValueError(f"inline file target must not be a symlink: {relative_path}")
            if not VZLinuxRunner._path_within_root(workspace_root, target):
                raise ValueError(f"inline file path escapes workspace: {relative_path}")
            target.write_bytes(data)

    @staticmethod
    def _collect_artifacts(workspace: str, capture_patterns: list[str] | None) -> dict[str, bytes]:
        return VZLinuxRunner._collect_limited_artifacts(
            workspace,
            capture_patterns,
            max_file_bytes=VZLinuxRunner._max_artifact_file_bytes(),
            max_total_bytes=VZLinuxRunner._max_artifact_total_bytes(),
        )[0]

    @staticmethod
    def _policy_cfg() -> SandboxPolicyConfig:
        try:
            return SandboxPolicyConfig.from_settings()
        except _VZ_LINUX_RUNNER_NONCRITICAL_EXCEPTIONS:
            return SandboxPolicyConfig()

    @staticmethod
    def _positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except _VZ_LINUX_RUNNER_NONCRITICAL_EXCEPTIONS:
            return default
        return parsed if parsed > 0 else default

    @classmethod
    def _max_log_bytes(cls) -> int:
        cfg = cls._policy_cfg()
        requested = cls._positive_int(getattr(cfg, "max_log_bytes", None), 10 * 1024 * 1024)
        return min(requested, cls.max_helper_output_bytes)

    @classmethod
    def _max_artifact_file_bytes(cls) -> int:
        cfg = cls._policy_cfg()
        return cls._positive_int(getattr(cfg, "max_artifact_file_bytes", None), 64 * 1024 * 1024)

    @classmethod
    def _max_artifact_total_bytes(cls) -> int:
        cfg = cls._policy_cfg()
        return cls._positive_int(getattr(cfg, "max_artifact_total_bytes", None), 256 * 1024 * 1024)

    @classmethod
    def _collect_limited_artifacts(
        cls,
        workspace: str,
        capture_patterns: list[str] | None,
        *,
        max_file_bytes: int,
        max_total_bytes: int,
    ) -> tuple[dict[str, bytes], dict[str, int]]:
        try:
            result = collect_limited_artifacts(
                workspace,
                capture_patterns,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
            )
        except _VZ_LINUX_RUNNER_NONCRITICAL_EXCEPTIONS:
            return {}, {}
        return result.artifacts, result.counters

    @staticmethod
    def _parse_helper_output_counter(value: Any) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "on"}:
                return 1
            if normalized in {"false", "no", "off", ""}:
                return 0
            try:
                parsed = int(normalized)
            except ValueError:
                return None
            return parsed if parsed >= 0 else None
        return None

    @classmethod
    def _output_counters_from_details(cls, details: dict[str, Any] | None) -> dict[str, int]:
        if not isinstance(details, dict):
            return {}
        counters: dict[str, int] = {}
        for key in _OUTPUT_COUNTER_KEYS:
            if key not in details:
                continue
            value = cls._parse_helper_output_counter(details.get(key))
            if value is not None:
                counters[key] = value
        return counters

    @classmethod
    def _helper_generation_from_details(cls, details: dict[str, Any] | None) -> tuple[str | None, str | None]:
        """Extract helper-generation identifiers from helper response details."""
        if not isinstance(details, dict):
            return None, None
        return (
            coerce_optional_nonempty_string(details.get("helper_instance_id")),
            coerce_optional_nonempty_string(details.get("helper_started_at")),
        )

    @classmethod
    def _session_status_reusable(
        cls,
        *,
        status: Any,
        session_control: dict[str, Any],
        session_id: str | None,
    ) -> bool:
        """Return true only when live VM metadata proves safe same-session reuse."""
        if not bool(getattr(status, "healthy", False)):
            return False

        metadata = getattr(status, "metadata", None)
        if metadata is None:
            return False
        owner = str(getattr(metadata, "owner", "") or "").strip()
        runtime = str(getattr(metadata, "runtime", "") or "").strip()
        if owner != "tldw" or runtime != RuntimeType.vz_linux.value:
            return False

        requested_session_id = str(session_id or "").strip()
        live_session_id = str(getattr(metadata, "session_id", "") or "").strip()
        if live_session_id and live_session_id != requested_session_id:
            return False

        stored_instance = coerce_optional_nonempty_string(session_control.get("helper_instance_id"))
        stored_started_at = coerce_optional_nonempty_string(session_control.get("helper_started_at"))
        live_instance, live_started_at = cls._helper_generation_from_details(getattr(status, "details", None))
        if stored_instance and stored_started_at and live_instance and live_started_at:
            return stored_instance == live_instance and stored_started_at == live_started_at

        return bool(
            requested_session_id
            and live_session_id == requested_session_id
            and bool(getattr(metadata, "session_mode", False))
        )

    def _load_session_control(self, session_id: str | None) -> dict[str, Any] | None:
        sid = str(session_id or "").strip()
        if not sid or self._session_control_store is None:
            return None
        getter = getattr(self._session_control_store, "get_vz_session_control", None)
        if not callable(getter):
            return None
        row = getter(sid)
        return dict(row) if isinstance(row, dict) else None

    def _store_session_control(
        self,
        *,
        session_id: str | None,
        vm_id: str,
        template_id: str | None,
        workspace_mount: str | None,
        helper_instance_id: str | None = None,
        helper_started_at: str | None = None,
    ) -> None:
        sid = str(session_id or "").strip()
        if not sid or self._session_control_store is None:
            return
        putter = getattr(self._session_control_store, "put_vz_session_control", None)
        if not callable(putter):
            return
        putter(
            session_id=sid,
            runtime=self.runtime_type.value,
            vm_id=str(vm_id),
            template_id=(str(template_id) if template_id is not None else None),
            workspace_mount=(str(workspace_mount) if workspace_mount is not None else None),
            agent_ready=True,
            helper_instance_id=coerce_optional_nonempty_string(helper_instance_id),
            helper_started_at=coerce_optional_nonempty_string(helper_started_at),
        )

    def _delete_session_control(self, session_id: str | None) -> None:
        sid = str(session_id or "").strip()
        if not sid or self._session_control_store is None:
            return
        deleter = getattr(self._session_control_store, "delete_vz_session_control", None)
        if callable(deleter):
            deleter(sid)

    def _image_store(self) -> SandboxImageStore | None:
        root_text = str(os.getenv("TLDW_SANDBOX_IMAGE_STORE_ROOT") or "").strip()
        if not root_text:
            return None
        try:
            return SandboxImageStore(root_path=root_text)
        except (ImageStoreValidationError, OSError, ValueError) as exc:
            logger.warning("vz_linux_image_store_unavailable root={} error={}", root_text, exc)
            return None

    def _looks_like_image_store_template_id(self, value: str) -> bool:
        return ":" in value and "/" not in value and "\\" not in value

    def _resolve_template_request(
        self,
        *,
        base_image: str | None,
    ) -> tuple[str, dict[str, str], SandboxImageStore | None]:
        template_text = str(base_image or "").strip()
        if not template_text or not self._looks_like_image_store_template_id(template_text):
            return template_text, {}, None

        store = self._image_store()
        if store is None:
            return template_text, {}, None
        record = store.get_template(template_text)
        if record is None:
            return template_text, {}, None
        if not record.source_path:
            raise RuntimeError("image_store_template_source_missing")
        return record.source_path, {
            "planning_source": "image_store",
            "template_id": record.template_id,
        }, store

    @classmethod
    def cancel_run(cls, run_id: str) -> bool:
        vm_id, run_dir = cls._clear_active_run(run_id)
        if not vm_id:
            return False
        with contextlib.suppress(_VZ_LINUX_RUNNER_NONCRITICAL_EXCEPTIONS):
            cls.helper_client_cls().terminate_vm(vm_id)
        if run_dir:
            with contextlib.suppress(_VZ_NONCRITICAL_EXCEPTIONS):
                shutil.rmtree(run_dir, ignore_errors=True)
        return True

    def start_run(
        self,
        run_id: str,
        spec: RunSpec,
        session_workspace: str | None = None,
    ) -> RunStatus:
        if os.getenv(self.fake_exec_env_key) and super()._execution_ready():
            return self._run_fake(run_id, message=f"{self.runtime_type.value} fake execution")
        return self._run_real(run_id, spec, session_workspace)

    def _run_real(
        self,
        run_id: str,
        spec: RunSpec,
        session_workspace: str | None = None,
    ) -> RunStatus:
        started = datetime.utcnow()
        finished = started
        hub = get_hub()
        artifacts_map: dict[str, bytes] = {}
        message = "vz_linux execution failed"
        exit_code: int | None = None
        phase = RunPhase.failed
        workspace = session_workspace
        created_workspace = False
        vm_id: str | None = None
        template_ref: str | None = None
        session_mode = bool(str(spec.session_id or "").strip())
        should_terminate_vm = True
        max_log_bytes = self._max_log_bytes()
        max_artifact_file_bytes = self._max_artifact_file_bytes()
        max_artifact_total_bytes = self._max_artifact_total_bytes()
        output_counters: dict[str, int] = {}
        artifact_counters: dict[str, int] = {}

        with contextlib.suppress(_VZ_NONCRITICAL_EXCEPTIONS):
            hub.publish_event(
                run_id,
                "start",
                {
                    "ts": started.isoformat(),
                    "runtime": self.runtime_type.value,
                    "transport": "vsock",
                    "workspace_mount": "virtiofs",
                },
            )

        try:
            if not workspace:
                workspace = tempfile.mkdtemp(prefix="tldw_vz_linux_")
                created_workspace = True
            os.makedirs(workspace, exist_ok=True)
            self._write_inline_files(workspace, spec.files_inline)

            helper = self.helper_client_cls()
            template_request, template_request_metadata, image_store = self._resolve_template_request(
                base_image=spec.base_image,
            )
            session_control = self._load_session_control(spec.session_id)
            if (
                session_mode
                and isinstance(session_control, dict)
                and str(session_control.get("runtime") or "").strip().lower() == self.runtime_type.value
                and bool(session_control.get("agent_ready"))
                and str(session_control.get("vm_id") or "").strip()
            ):
                candidate_vm_id = str(session_control.get("vm_id") or "").strip()
                try:
                    status = helper.get_vm_status(candidate_vm_id)
                except (MacOSVirtualizationHelperUnavailable, MacOSVirtualizationHelperProtocolError):
                    raise
                if status is not None and self._session_status_reusable(
                    status=status,
                    session_control=session_control,
                    session_id=spec.session_id,
                ):
                    vm_id = candidate_vm_id
                    template_ref = str(session_control.get("template_id") or "").strip() or spec.base_image
                    should_terminate_vm = False
                else:
                    self._delete_session_control(spec.session_id)
            if not vm_id:
                template_validation = helper.validate_template(
                    {
                        "runtime": self.runtime_type.value,
                        "template": template_request,
                    }
                )
                if not bool(template_validation.get("ready")):
                    template_reasons = [
                        str(reason) for reason in template_validation.get("reasons", []) if str(reason).strip()
                    ]
                    reason_text = ", ".join(template_reasons) if template_reasons else "template_invalid"
                    raise RuntimeError(reason_text)
                metadata_template_id = str(template_request_metadata.get("template_id") or "").strip()
                template_ref = (
                    metadata_template_id
                    or str(template_validation.get("template_id") or "").strip()
                    or spec.base_image
                )
                template_source = (
                    str(template_validation.get("source") or "").strip()
                    or template_request
                    or spec.base_image
                )
                create_vm_metadata = {
                    key: value
                    for key, value in template_request_metadata.items()
                    if key not in {"template", "template_id"}
                }
                should_persist_run_manifest = (
                    image_store is not None
                    and str(template_request_metadata.get("planning_source") or "").strip() == "image_store"
                    and bool(metadata_template_id)
                )
                if should_persist_run_manifest and image_store is not None:
                    create_vm_metadata["run_manifest_path"] = str(
                        image_store.root_path / "runs" / run_id / "manifest.json"
                    )
                vm = helper.create_vm(
                    {
                        "owner": "tldw",
                        "runtime": self.runtime_type.value,
                        "vm_name": run_id,
                        "run_id": run_id,
                        "session_id": str(spec.session_id or "").strip(),
                        "session_mode": session_mode,
                        "workspace_path": workspace,
                        "workspace_mount": "virtiofs",
                        "template_id": template_ref,
                        "template": template_source,
                        "network_policy": str(spec.network_policy or "deny_all").strip().lower() or "deny_all",
                        "timeout_sec": int(spec.startup_timeout_sec or spec.timeout_sec or 300),
                        **create_vm_metadata,
                    }
                )
                vm_id = vm.vm_id
                if should_persist_run_manifest and image_store is not None and metadata_template_id:
                    image_store.prepare_run_clone(template_id=metadata_template_id, run_id=run_id)
                should_terminate_vm = not session_mode
                if session_mode:
                    self._store_session_control(
                        session_id=spec.session_id,
                        vm_id=vm.vm_id,
                        template_id=template_ref,
                        workspace_mount=workspace,
                        helper_instance_id=vm.details.get("helper_instance_id") if isinstance(vm.details, dict) else None,
                        helper_started_at=vm.details.get("helper_started_at") if isinstance(vm.details, dict) else None,
                    )
            self._register_active_run(run_id, vm_id, workspace if created_workspace else None)

            reply = helper.exec_guest(
                vm_id=vm_id,
                request={
                    "argv": list(spec.command or []),
                    "cwd": "/workspace",
                    "env": dict(spec.env or {}),
                    "timeout_sec": int(spec.timeout_sec or 300),
                    "max_output_bytes": max_log_bytes,
                },
            )
            output_counters = self._output_counters_from_details(reply.details)

            stdout_data = bytes(reply.stdout or b"")
            stderr_data = bytes(reply.stderr or b"")
            if stdout_data:
                hub.publish_stdout(run_id, stdout_data, max_log_bytes=max_log_bytes)
            if stderr_data:
                hub.publish_stderr(run_id, stderr_data, max_log_bytes=max_log_bytes)

            exit_code = int(reply.exit_code)
            phase = RunPhase.completed if exit_code == 0 else RunPhase.failed
            message = (
                "vz_linux execution finished"
                if exit_code == 0
                else f"vz_linux execution failed (exit={exit_code})"
            )
            artifacts_map, artifact_counters = self._collect_limited_artifacts(
                workspace,
                spec.capture_patterns,
                max_file_bytes=max_artifact_file_bytes,
                max_total_bytes=max_artifact_total_bytes,
            )
        except _VZ_LINUX_RUNNER_NONCRITICAL_EXCEPTIONS as exc:
            logger.error("vz_linux execution error for run {}: {}", run_id, exc)
            message = f"vz_linux execution error: {exc}"
        finally:
            if vm_id and should_terminate_vm:
                with contextlib.suppress(_VZ_LINUX_RUNNER_NONCRITICAL_EXCEPTIONS):
                    self.helper_client_cls().terminate_vm(vm_id)
            _active_vm_id, run_dir = self._clear_active_run(run_id)
            del _active_vm_id
            cleanup_dir = run_dir if run_dir else (workspace if created_workspace else None)
            if cleanup_dir:
                with contextlib.suppress(_VZ_NONCRITICAL_EXCEPTIONS):
                    shutil.rmtree(cleanup_dir, ignore_errors=True)
            finished = datetime.utcnow()

        with contextlib.suppress(_VZ_NONCRITICAL_EXCEPTIONS):
            hub.publish_event(run_id, "end", {"exit_code": exit_code})

        try:
            total_log_bytes = int(hub.get_log_bytes(run_id))
        except _VZ_NONCRITICAL_EXCEPTIONS:
            total_log_bytes = 0
        artifact_bytes = sum(len(value) for value in artifacts_map.values()) if artifacts_map else 0
        usage = {
            "cpu_time_sec": 0,
            "wall_time_sec": int(max(0.0, (finished - started).total_seconds())),
            "peak_rss_mb": 0,
            "log_bytes": int(total_log_bytes),
            "artifact_bytes": int(artifact_bytes),
        }
        usage.update(log_limit_counters(hub, run_id, max_log_bytes))
        usage.update(output_counters)
        usage.update(artifact_counters)

        return RunStatus(
            id="",
            phase=phase,
            runtime=self.runtime_type,
            runtime_version=self._version(),
            base_image=spec.base_image,
            started_at=started,
            finished_at=finished,
            exit_code=exit_code,
            message=message,
            resource_usage=usage,
            artifacts=artifacts_map or None,
        )
