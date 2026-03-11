from __future__ import annotations

import contextlib
import fnmatch
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ..macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperUnavailable,
)
from ..models import RunPhase, RunSpec, RunStatus, RuntimeType
from ..runtime_capabilities import RuntimePreflightResult
from ..streams import get_hub
from .vz_common import VZBaseRunner, _VZ_NONCRITICAL_EXCEPTIONS
from .vz_common import vz_host_facts

_VZ_LINUX_RUNNER_NONCRITICAL_EXCEPTIONS = (
    OSError,
    PermissionError,
    RuntimeError,
    TypeError,
    ValueError,
    MacOSVirtualizationHelperUnavailable,
)


class VZLinuxRunner(VZBaseRunner):
    runtime_type = RuntimeType.vz_linux
    fake_exec_env_key = "TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC"
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
                    {"network_policy": str(network_policy or "deny_all").strip().lower() or "deny_all"}
                )
            except MacOSVirtualizationHelperUnavailable as exc:
                helper_result = {
                    "available": False,
                    "reasons": [str(exc) or "macos_virtualization_helper_unavailable"],
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
        if not capture_patterns:
            return {}

        artifacts_map: dict[str, bytes] = {}
        workspace_root = Path(workspace)
        if workspace_root.is_symlink():
            return {}
        try:
            for root, _dirs, files in os.walk(workspace):
                root_path = Path(root)
                if not VZLinuxRunner._path_within_root(workspace_root, root_path):
                    continue
                for file_name in files:
                    full_path = root_path / file_name
                    if full_path.is_symlink():
                        continue
                    if not VZLinuxRunner._path_within_root(workspace_root, full_path):
                        continue
                    full = os.path.join(root, file_name)
                    rel = os.path.relpath(full, workspace)
                    rel_posix = rel.replace(os.sep, "/")
                    if any(fnmatch.fnmatchcase(rel_posix, pattern) for pattern in capture_patterns):
                        artifacts_map[rel_posix] = full_path.read_bytes()
        except _VZ_NONCRITICAL_EXCEPTIONS:
            return {}
        return artifacts_map

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
        )

    def _delete_session_control(self, session_id: str | None) -> None:
        sid = str(session_id or "").strip()
        if not sid or self._session_control_store is None:
            return
        deleter = getattr(self._session_control_store, "delete_vz_session_control", None)
        if callable(deleter):
            deleter(sid)

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
            session_control = self._load_session_control(spec.session_id)
            if (
                session_mode
                and isinstance(session_control, dict)
                and str(session_control.get("runtime") or "").strip().lower() == self.runtime_type.value
                and bool(session_control.get("agent_ready"))
                and str(session_control.get("vm_id") or "").strip()
            ):
                candidate_vm_id = str(session_control.get("vm_id") or "").strip()
                status = helper.get_vm_status(candidate_vm_id)
                if bool(status.healthy):
                    vm_id = candidate_vm_id
                    template_ref = str(session_control.get("template_id") or "").strip() or spec.base_image
                    should_terminate_vm = False
                else:
                    self._delete_session_control(spec.session_id)
            if not vm_id:
                template_validation = helper.validate_template(
                    {
                        "runtime": self.runtime_type.value,
                        "template": spec.base_image,
                    }
                )
                if not bool(template_validation.get("ready")):
                    template_reasons = [
                        str(reason) for reason in template_validation.get("reasons", []) if str(reason).strip()
                    ]
                    reason_text = ", ".join(template_reasons) if template_reasons else "template_invalid"
                    raise RuntimeError(reason_text)
                template_ref = str(template_validation.get("template_id") or "").strip() or spec.base_image
                vm = helper.create_vm(
                    {
                        "runtime": self.runtime_type.value,
                        "vm_name": run_id,
                        "run_id": run_id,
                        "session_mode": session_mode,
                        "workspace_path": workspace,
                        "workspace_mount": "virtiofs",
                        "template": template_ref,
                        "network_policy": str(spec.network_policy or "deny_all").strip().lower() or "deny_all",
                    }
                )
                vm_id = vm.vm_id
                should_terminate_vm = not session_mode
                if session_mode:
                    self._store_session_control(
                        session_id=spec.session_id,
                        vm_id=vm.vm_id,
                        template_id=template_ref,
                        workspace_mount=workspace,
                    )
            self._register_active_run(run_id, vm_id, workspace if created_workspace else None)

            reply = helper.exec_guest(
                vm_id=vm_id,
                request={
                    "argv": list(spec.command or []),
                    "cwd": "/workspace",
                    "env": dict(spec.env or {}),
                    "timeout_sec": int(spec.timeout_sec or 300),
                },
            )

            stdout_data = bytes(reply.stdout or b"")
            stderr_data = bytes(reply.stderr or b"")
            if stdout_data:
                hub.publish_stdout(run_id, stdout_data)
            if stderr_data:
                hub.publish_stderr(run_id, stderr_data)

            exit_code = int(reply.exit_code)
            phase = RunPhase.completed if exit_code == 0 else RunPhase.failed
            message = (
                "vz_linux execution finished"
                if exit_code == 0
                else f"vz_linux execution failed (exit={exit_code})"
            )
            artifacts_map = self._collect_artifacts(workspace, spec.capture_patterns)
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
