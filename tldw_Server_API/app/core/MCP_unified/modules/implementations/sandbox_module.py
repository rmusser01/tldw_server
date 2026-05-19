"""
Sandbox Module for Unified MCP

Provides a management tool `sandbox.run` that triggers code execution via the
server's sandbox service. This is a stub-level integration intended to expose
the tool schema and a basic, policy-aware execution path per the PRD.

Notes:
- Write-capable tool; includes a custom validator.
- Execution uses the internal SandboxService scaffold; honors admin policy.
- For streaming logs or artifact downloads, clients should use the REST/WS endpoints.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from loguru import logger

from ....Sandbox.models import RunSpec
from ....Sandbox.models import RuntimeType as SbxRuntimeType
from ....Sandbox.models import TrustLevel
from ....Sandbox.service import SandboxService
from ..base import BaseModule


def _runtime_values() -> list[str]:
    return [runtime.value for runtime in SbxRuntimeType]


class SandboxModule(BaseModule):
    async def on_initialize(self) -> None:
        logger.info(f"Initializing Sandbox module: {self.name}")
        self._svc = SandboxService()

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down Sandbox module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        checks = {"initialized": hasattr(self, "_svc") and self._svc is not None}
        # Minimal health: service scaffold exists
        return checks

    async def get_tools(self) -> list[dict[str, Any]]:
        # Return explicit schema with oneOf to reflect PRD (session vs one-shot)
        return [
            {
                "name": "sandbox.run",
                "description": "Execute a run in the code sandbox.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "runtime": {
                            "type": "string",
                            "enum": _runtime_values(),
                        },
                        "session_id": {"type": "string"},
                        "base_image": {"type": "string"},
                        "command": {"type": "array", "items": {"type": "string"}},
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "content_b64": {"type": "string"}
                                },
                                "required": ["path", "content_b64"],
                                "additionalProperties": False
                            }
                        },
                        "timeout_sec": {"type": "integer", "minimum": 1},
                        "cpu": {"type": "number", "minimum": 0},
                        "memory_mb": {"type": "integer", "minimum": 64},
                        "network_policy": {"type": "string", "enum": ["deny_all", "allowlist"]},
                        "env": {"type": "object"},
                        "trust_level": {"type": "string", "enum": ["trusted", "standard", "untrusted"]},
                        "persona_id": {"type": "string"},
                        "workspace_id": {"type": "string"},
                        "workspace_group_id": {"type": "string"},
                        "scope_snapshot_id": {"type": "string"},
                        "spec_version": {"type": "string"},
                        "idempotency_key": {"type": "string"},
                    },
                    "oneOf": [
                        {"required": ["session_id", "command"]},
                        {"required": ["base_image", "command"]}
                    ],
                    "additionalProperties": False
                },
                # Categorize as management and mark filesystem/process usage explicitly.
                # Path scoping must fail closed here because the command can touch
                # arbitrary workspace paths beyond any inline file list.
                "metadata": {
                    "category": "management",
                    "notes": "Session-based vs one-shot per PRD",
                    "uses_filesystem": True,
                    "uses_processes": True,
                    "path_boundable": False,
                    "path_argument_hints": ["cwd", "files[].path"],
                },
            }
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        args = self.sanitize_input(arguments)
        if tool_name != "sandbox.run":
            raise ValueError(f"Unknown tool: {tool_name}")
        # Validate strictly (write-capable tool)
        self.validate_tool_arguments(tool_name, args)

        # Require authenticated principal binding; no synthetic fallback identity.
        user_id = getattr(context, "user_id", None) if context is not None else None
        if user_id is None or not str(user_id).strip():
            raise PermissionError("sandbox.run requires an authenticated user context")
        user_id = str(user_id)

        runtime = self._coerce_runtime(args.get("runtime"))
        base_image = (str(args.get("base_image")).strip() if args.get("base_image") is not None else None) or None
        env = args.get("env") or {}
        if not isinstance(env, dict):
            env = {}
        timeout = int(args.get("timeout_sec") or 300)
        cpu = float(args.get("cpu")) if args.get("cpu") is not None else None
        memory_mb = int(args.get("memory_mb")) if args.get("memory_mb") is not None else None
        network_policy = (str(args.get("network_policy")).strip() if args.get("network_policy") is not None else None) or None
        trust_level = self._coerce_trust_level(args.get("trust_level"))
        persona_id = (str(args.get("persona_id")) if args.get("persona_id") is not None else None)
        workspace_id = (str(args.get("workspace_id")) if args.get("workspace_id") is not None else None)
        workspace_group_id = (str(args.get("workspace_group_id")) if args.get("workspace_group_id") is not None else None)
        scope_snapshot_id = (str(args.get("scope_snapshot_id")) if args.get("scope_snapshot_id") is not None else None)

        session_id = args.get("session_id")
        if session_id is not None:
            session_id = str(session_id).strip() or None
        if session_id:
            owner = self._svc.get_session_owner(session_id)
            if owner is None:
                raise ValueError("session_not_found")
            if not self._is_admin(context) and str(owner) != user_id:
                raise PermissionError("sandbox.run session not found")

        files_inline = self._decode_inline_files(args.get("files") or [])
        command = [str(x) for x in (args.get("command") or [])]
        spec = RunSpec(
            session_id=session_id,
            runtime=runtime,
            base_image=base_image,
            command=command,
            env={str(k): str(v) for k, v in env.items()},
            timeout_sec=timeout,
            cpu=cpu,
            memory_mb=memory_mb,
            network_policy=network_policy,
            files_inline=files_inline,
            capture_patterns=[],
            trust_level=trust_level,
            persona_id=persona_id,
            workspace_id=workspace_id,
            workspace_group_id=workspace_group_id,
            scope_snapshot_id=scope_snapshot_id,
        )

        spec_version = str(args.get("spec_version") or "1.0")
        idem_key = None
        try:
            idem_key = str(args.get("idempotency_key") or args.get("idempotencyKey") or "") or None
        except Exception:
            idem_key = None

        try:
            status = self._svc.start_run_scaffold(
                user_id=user_id,
                spec=spec,
                spec_version=spec_version,
                idem_key=idem_key,
                raw_body=args,
                explicit_fields={str(key) for key in args.keys()},
            )
        except Exception as e:
            raise RuntimeError(f"Sandbox execution failed: {e}") from e

        # Return a compact result for MCP
        def _iso(dt):
            try:
                return dt.isoformat() if dt else None
            except Exception:
                return None

        return {
            "id": status.id,
            "phase": status.phase.value,
            "exit_code": status.exit_code,
            "runtime": status.runtime.value if status.runtime else None,
            "base_image": status.base_image,
            "policy_hash": status.policy_hash,
            "image_digest": status.image_digest,
            "started_at": _iso(status.started_at),
            "finished_at": _iso(status.finished_at),
            "message": status.message,
            "session_id": status.session_id,
            "persona_id": status.persona_id,
            "workspace_id": status.workspace_id,
            "workspace_group_id": status.workspace_group_id,
            "scope_snapshot_id": status.scope_snapshot_id,
        }

    def _is_admin(self, context: Any | None) -> bool:
        try:
            if bool(getattr(context, "is_admin", False)):
                return True
            metadata = getattr(context, "metadata", {}) or {}
            roles = metadata.get("roles")
            if isinstance(roles, str):
                roles = [roles]
            if isinstance(roles, list) and any(str(r).strip().lower() == "admin" for r in roles):
                return True
            permissions = metadata.get("permissions")
            if isinstance(permissions, str):
                permissions = [permissions]
            if isinstance(permissions, list):
                normalized = {str(permission).strip().lower() for permission in permissions if str(permission).strip()}
                if "*" in normalized or "system.configure" in normalized:
                    return True
            return False
        except Exception:
            return False

    def _coerce_runtime(self, value: Any) -> SbxRuntimeType | None:
        runtime_raw = (str(value).strip().lower() if value is not None else "")
        if not runtime_raw:
            return None
        try:
            return SbxRuntimeType(runtime_raw)
        except ValueError:
            pass
        return None

    def _coerce_trust_level(self, value: Any) -> TrustLevel | None:
        trust_raw = (str(value).strip().lower() if value is not None else "")
        if trust_raw in ("trusted", "standard", "untrusted"):
            return TrustLevel(trust_raw)
        return None

    def _decode_inline_files(self, files: list[dict[str, Any]]) -> list[tuple[str, bytes]]:
        decoded: list[tuple[str, bytes]] = []
        for index, file_entry in enumerate(files):
            try:
                path = str(file_entry.get("path", ""))
                content_b64 = str(file_entry.get("content_b64", ""))
                data = base64.b64decode(content_b64, validate=True)
            except (TypeError, ValueError, binascii.Error, AttributeError) as exc:
                raise ValueError(f"invalid inline file at index {index}") from exc
            decoded.append((path, data))
        return decoded

    def sanitize_input(self, input_data: Any, _depth: int = 0) -> Any:
        """
        Relaxed sanitizer for sandbox payloads.

        Allows CLI-style args and comment tokens while still stripping control chars
        and guarding against overly deep payloads.
        """
        if _depth > 20:
            raise ValueError("Input too deeply nested")

        def _clean_str(s: str) -> str:
            # Strip NULs and control chars only.
            return "".join(ch for ch in s if ch >= " " or ch == "\n")

        if isinstance(input_data, str):
            return _clean_str(input_data)
        if isinstance(input_data, dict):
            return {k: self.sanitize_input(v, _depth + 1) for k, v in input_data.items()}
        if isinstance(input_data, list):
            return [self.sanitize_input(v, _depth + 1) for v in input_data]
        return input_data

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):
        # Enforce PRD oneOf and types
        cmd = arguments.get("command")
        sess = arguments.get("session_id")
        img = arguments.get("base_image")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and x for x in cmd):
            raise ValueError("command must be a non-empty array of strings")
        if not ((isinstance(sess, str) and sess) or (isinstance(img, str) and img)):
            raise ValueError("Either session_id or base_image is required")
        if (isinstance(sess, str) and sess) and (isinstance(img, str) and img):
            raise ValueError("Provide only one of session_id or base_image, not both")
        rt = arguments.get("runtime")
        runtime_values = tuple(_runtime_values())
        if rt is not None and str(rt).strip().lower() not in runtime_values:
            raise ValueError(
                f"runtime must be {'|'.join(runtime_values)} when provided"
            )
        if arguments.get("timeout_sec") is not None:
            try:
                ts = int(arguments.get("timeout_sec"))
                if ts <= 0:
                    raise ValueError
            except Exception:
                raise ValueError("timeout_sec must be a positive integer") from None
        files = arguments.get("files")
        if files is not None:
            if not isinstance(files, list):
                raise ValueError("files must be an array when provided")
            for i, f in enumerate(files):
                if not isinstance(f, dict) or not f.get("path") or not f.get("content_b64"):
                    raise ValueError(f"files[{i}] must include path and content_b64")
                try:
                    base64.b64decode(str(f.get("content_b64", "")), validate=True)
                except (TypeError, ValueError, binascii.Error):
                    raise ValueError(f"files[{i}].content_b64 must be valid base64") from None
        if arguments.get("cpu") is not None:
            try:
                cpu = float(arguments.get("cpu"))
                if cpu < 0:
                    raise ValueError
            except Exception:
                raise ValueError("cpu must be a non-negative number") from None
        if arguments.get("memory_mb") is not None:
            try:
                memory_mb = int(arguments.get("memory_mb"))
                if memory_mb < 64:
                    raise ValueError
            except Exception:
                raise ValueError("memory_mb must be an integer >= 64") from None
        if arguments.get("network_policy") is not None and str(arguments.get("network_policy")) not in {"deny_all", "allowlist"}:
            raise ValueError("network_policy must be deny_all|allowlist when provided")
        if arguments.get("trust_level") is not None and str(arguments.get("trust_level")).lower() not in {"trusted", "standard", "untrusted"}:
            raise ValueError("trust_level must be trusted|standard|untrusted when provided")
