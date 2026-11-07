from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.managed_secret_refs_repo import (
    ManagedSecretRefsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import (
    parse_managed_secret_ref_id,
)
from tldw_Server_API.app.core.AuthNZ.secret_backends.registry import (
    get_secret_backend,
)
from tldw_Server_API.app.services.mcp_hub_external_auth_service import (
    ManagedExternalAuthBridge,
)


@dataclass
class McpHubExternalAccessResolver:
    """Resolve effective external server access for an assignment."""

    repo: McpHubRepo
    auth_bridge: ManagedExternalAuthBridge | None = None

    def __post_init__(self) -> None:
        if self.auth_bridge is None:
            self.auth_bridge = ManagedExternalAuthBridge()

    async def resolve(
        self,
        *,
        assignment_id: int,
        effective_policy: dict[str, Any],
    ) -> dict[str, Any]:
        assignment = await self.repo.get_policy_assignment(int(assignment_id))
        if not assignment:
            return {"servers": []}
        return await self.resolve_for_sources(
            sources=[
                {
                    "assignment_id": int(assignment.get("id")),
                    "profile_id": assignment.get("profile_id"),
                }
            ],
            effective_policy=effective_policy,
        )

    async def resolve_for_sources(
        self,
        *,
        sources: list[dict[str, Any]],
        effective_policy: dict[str, Any],
    ) -> dict[str, Any]:
        states: dict[str, dict[str, Any]] = {}

        for source in sources:
            profile_id = source.get("profile_id")
            if profile_id is not None:
                for binding in await self.repo.list_credential_bindings(
                    binding_target_type="profile",
                    binding_target_id=str(profile_id),
                ):
                    server_id = str(binding.get("external_server_id") or "")
                    if not server_id or str(binding.get("binding_mode") or "grant") != "grant":
                        continue
                    await self._apply_binding(
                        states=states,
                        server_id=server_id,
                        slot_name=binding.get("slot_name"),
                        credential_ref=binding.get("credential_ref"),
                        binding_mode="grant",
                        granted_by="profile",
                    )

            assignment_id = source.get("assignment_id")
            if assignment_id is None:
                continue
            for binding in await self.repo.list_credential_bindings(
                binding_target_type="assignment",
                binding_target_id=str(assignment_id),
            ):
                server_id = str(binding.get("external_server_id") or "")
                if not server_id:
                    continue
                await self._apply_binding(
                    states=states,
                    server_id=server_id,
                    slot_name=binding.get("slot_name"),
                    credential_ref=binding.get("credential_ref"),
                    binding_mode=str(binding.get("binding_mode") or "grant"),
                    granted_by="assignment",
                )

        return {"servers": await self._build_server_rows(states=states, effective_policy=effective_policy)}

    async def _apply_binding(
        self,
        *,
        states: dict[str, dict[str, Any]],
        server_id: str,
        slot_name: Any,
        credential_ref: Any,
        binding_mode: str,
        granted_by: str,
    ) -> None:
        managed_secret_ref_id = parse_managed_secret_ref_id(credential_ref)
        server_state = states.setdefault(
            server_id,
            {
                "granted_by": None,
                "disabled_by_assignment": False,
                "managed_ref_id": None,
                "slots": {},
            },
        )
        slots = await self.repo.list_external_server_credential_slots(server_id=server_id)
        normalized_slot_name = str(slot_name or "").strip().lower()
        target_slot_names: list[str] = []
        if slots:
            if normalized_slot_name:
                target_slot_names = [normalized_slot_name]
            else:
                default_slot = await self.repo.get_external_server_default_slot(server_id=server_id)
                if default_slot is not None:
                    target_slot_names = [str(default_slot.get("slot_name") or "").strip().lower()]
        if target_slot_names:
            slot_states = server_state.setdefault("slots", {})
            for target_slot_name in target_slot_names:
                slot_state = slot_states.setdefault(
                target_slot_name,
                    {
                        "granted_by": None,
                        "disabled_by_assignment": False,
                        "managed_ref_id": None,
                    },
                )
                if binding_mode == "disable":
                    slot_state["disabled_by_assignment"] = True
                    if slot_state.get("granted_by") is None:
                        slot_state["granted_by"] = granted_by
                else:
                    slot_state["granted_by"] = granted_by
                    slot_state["disabled_by_assignment"] = False
                    slot_state["managed_ref_id"] = managed_secret_ref_id
            return

        if binding_mode == "disable":
            server_state["disabled_by_assignment"] = True
            if server_state.get("granted_by") is None:
                server_state["granted_by"] = granted_by
        else:
            server_state["granted_by"] = granted_by
            server_state["disabled_by_assignment"] = False
            server_state["managed_ref_id"] = managed_secret_ref_id

    async def _build_server_rows(
        self,
        *,
        states: dict[str, dict[str, Any]],
        effective_policy: dict[str, Any],
    ) -> list[dict[str, Any]]:
        policy_document = dict(effective_policy.get("policy_document") or {})
        capabilities = set(
            effective_policy.get("capabilities")
            or policy_document.get("capabilities")
            or []
        )
        allows_external = "network.external" in capabilities
        managed_secret_ref_ready_cache = await self._load_managed_secret_ref_ready_cache(states)
        rows: list[dict[str, Any]] = []
        for server_id in sorted(states.keys()):
            server = await self.repo.get_external_server(server_id)
            if not server:
                continue
            state = states.get(server_id) or {}
            slots = await self.repo.list_external_server_credential_slots(server_id=server_id)
            server_source = str(server.get("server_source") or "managed")
            superseded_by = server.get("superseded_by_server_id")
            enabled = bool(server.get("enabled"))
            if slots:
                slot_rows: list[dict[str, Any]] = []
                active_slot_states = dict(state.get("slots") or {})
                requested_slots = self.auth_bridge.get_required_slot_names(server_config=server)
                for slot in slots:
                    slot_name = str(slot.get("slot_name") or "").strip().lower()
                    slot_state = dict(active_slot_states.get(slot_name) or {})
                    granted_by = slot_state.get("granted_by")
                    disabled = bool(slot_state.get("disabled_by_assignment"))
                    secret_available = bool(slot.get("secret_configured")) or bool(
                        managed_secret_ref_ready_cache.get(int(slot_state.get("managed_ref_id") or 0), False)
                    )
                    runtime_usable = bool(
                        granted_by
                        and allows_external
                        and not disabled
                        and secret_available
                        and server_source == "managed"
                        and not superseded_by
                        and enabled
                    )
                    blocked_reason = None
                    if disabled:
                        blocked_reason = "disabled_by_assignment"
                    elif server_source != "managed":
                        blocked_reason = "legacy_server_not_bindable"
                    elif superseded_by:
                        blocked_reason = "server_superseded"
                    elif not enabled:
                        blocked_reason = "server_disabled"
                    elif not allows_external:
                        blocked_reason = "external_capability_not_granted"
                    elif granted_by and not secret_available:
                        blocked_reason = "missing_secret"
                    elif granted_by is None:
                        blocked_reason = "not_granted"
                    slot_rows.append(
                        {
                            "slot_name": slot.get("slot_name"),
                            "display_name": slot.get("display_name"),
                            "granted_by": granted_by,
                            "disabled_by_assignment": disabled,
                            "secret_available": secret_available,
                            "runtime_usable": runtime_usable,
                            "blocked_reason": blocked_reason,
                        }
                    )

                slot_by_name = {
                    str(slot.get("slot_name") or "").strip().lower(): slot
                    for slot in slot_rows
                    if str(slot.get("slot_name") or "").strip()
                }
                bound_slots = [
                    slot_name
                    for slot_name in requested_slots
                    if isinstance(slot_by_name.get(slot_name), dict)
                    and bool(slot_by_name[slot_name].get("granted_by"))
                    and not bool(slot_by_name[slot_name].get("disabled_by_assignment"))
                ]
                missing_bound_slots = [
                    slot_name for slot_name in requested_slots if slot_name not in bound_slots
                ]
                missing_secret_slots = [
                    slot_name
                    for slot_name in requested_slots
                    if slot_name in bound_slots
                    and isinstance(slot_by_name.get(slot_name), dict)
                    and not bool(slot_by_name[slot_name].get("secret_available"))
                ]
                runtime_executable = (
                    bool(requested_slots)
                    and not missing_bound_slots
                    and not missing_secret_slots
                    and allows_external
                    and server_source == "managed"
                    and not superseded_by
                    and enabled
                ) if requested_slots else any(bool(slot.get("runtime_usable")) for slot in slot_rows)
                disabled = any(bool(slot.get("disabled_by_assignment")) for slot in slot_rows)
                secret_available = all(
                    bool(slot.get("secret_available"))
                    for slot in slot_rows
                    if str(slot.get("slot_name") or "").strip().lower() in requested_slots
                ) if requested_slots else any(bool(slot.get("secret_available")) for slot in slot_rows)
                blocked_reason = None
                if requested_slots and missing_bound_slots:
                    blocked_reason = "required_slot_not_granted"
                elif requested_slots and missing_secret_slots:
                    blocked_reason = "required_slot_secret_missing"
                elif disabled and not runtime_executable:
                    blocked_reason = "disabled_by_assignment"
                elif server_source != "managed":
                    blocked_reason = "legacy_server_not_bindable"
                elif superseded_by:
                    blocked_reason = "server_superseded"
                elif not enabled:
                    blocked_reason = "server_disabled"
                elif not allows_external:
                    blocked_reason = "external_capability_not_granted"
                elif not secret_available:
                    blocked_reason = "missing_secret"
                elif not runtime_executable:
                    blocked_reason = "no_usable_slots"

                rows.append(
                    {
                        "server_id": server_id,
                        "server_name": server.get("name"),
                        "granted_by": state.get("granted_by"),
                        "disabled_by_assignment": disabled,
                        "server_source": server_source,
                        "superseded_by_server_id": superseded_by,
                        "secret_available": secret_available,
                        "runtime_executable": runtime_executable,
                        "blocked_reason": blocked_reason,
                        "requested_slots": requested_slots,
                        "bound_slots": bound_slots,
                        "missing_bound_slots": missing_bound_slots,
                        "missing_secret_slots": missing_secret_slots,
                        "slots": slot_rows,
                    }
                )
                continue

            disabled = bool(state.get("disabled_by_assignment"))
            secret_available = bool(server.get("secret_configured")) or bool(
                managed_secret_ref_ready_cache.get(int(state.get("managed_ref_id") or 0), False)
            )
            auth = dict(dict(server.get("config") or {}).get("auth") or {})
            requires_secret = str(auth.get("mode") or "none").strip().lower() not in {"", "none"}
            runtime_executable = bool(
                allows_external
                and not disabled
                and (secret_available or not requires_secret)
                and server_source == "managed"
                and not superseded_by
                and enabled
            )
            blocked_reason = None
            if disabled:
                blocked_reason = "disabled_by_assignment"
            elif server_source != "managed":
                blocked_reason = "legacy_server_not_bindable"
            elif superseded_by:
                blocked_reason = "server_superseded"
            elif not enabled:
                blocked_reason = "server_disabled"
            elif not allows_external:
                blocked_reason = "external_capability_not_granted"
            elif requires_secret and not secret_available:
                blocked_reason = "missing_secret"

            rows.append(
                {
                    "server_id": server_id,
                    "server_name": server.get("name"),
                    "granted_by": state.get("granted_by"),
                    "disabled_by_assignment": disabled,
                    "server_source": server_source,
                    "superseded_by_server_id": superseded_by,
                    "secret_available": secret_available,
                    "runtime_executable": runtime_executable,
                    "blocked_reason": blocked_reason,
                    "slots": [],
                }
            )
        return rows

    @staticmethod
    def _collect_managed_secret_ref_ids(states: dict[str, dict[str, Any]]) -> list[int]:
        ref_ids: set[int] = set()
        for state in states.values():
            managed_ref_id = state.get("managed_ref_id")
            if managed_ref_id is not None:
                ref_ids.add(int(managed_ref_id))
            for slot_state in dict(state.get("slots") or {}).values():
                if not isinstance(slot_state, dict):
                    continue
                managed_ref_id = slot_state.get("managed_ref_id")
                if managed_ref_id is not None:
                    ref_ids.add(int(managed_ref_id))
        return sorted(ref_ids)

    async def _load_managed_secret_ref_ready_cache(
        self,
        states: dict[str, dict[str, Any]],
    ) -> dict[int, bool]:
        managed_repo = ManagedSecretRefsRepo(self.repo.db_pool)
        await managed_repo.ensure_tables()
        ref_ids = self._collect_managed_secret_ref_ids(states)
        refs_by_id = await managed_repo.list_refs_by_ids(ref_ids, include_revoked=True)
        now = datetime.now(timezone.utc)
        backend_cache: dict[str, Any] = {}
        readiness: dict[int, bool] = {}

        for ref_id in ref_ids:
            ref = refs_by_id.get(int(ref_id))
            if not ref or ref.get("revoked_at"):
                readiness[int(ref_id)] = False
                continue

            raw_status = str(ref.get("status") or "").strip().lower()
            expires_at = ref.get("expires_at")
            expires_at_dt: datetime | None = None
            if isinstance(expires_at, str) and expires_at.strip():
                try:
                    expires_at_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                except ValueError:
                    expires_at_dt = None
            elif isinstance(expires_at, datetime):
                expires_at_dt = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=timezone.utc)

            if raw_status not in {"active", "enabled", "ready"}:
                readiness[int(ref_id)] = False
                continue
            if expires_at_dt is not None and expires_at_dt <= now:
                readiness[int(ref_id)] = False
                continue

            backend_name = str(ref.get("backend_name") or "").strip()
            if not backend_name:
                readiness[int(ref_id)] = False
                continue

            backend = backend_cache.get(backend_name)
            if backend is None:
                try:
                    backend = get_secret_backend(backend_name, db_pool=self.repo.db_pool)
                except ValueError:
                    readiness[int(ref_id)] = False
                    continue
                backend_cache[backend_name] = backend

            status = await backend.describe_status(int(ref_id))
            state = str(status.get("state") or "").strip().lower()
            readiness[int(ref_id)] = state in {"active", "enabled", "ready"}

        return readiness


async def get_mcp_hub_external_access_resolver() -> McpHubExternalAccessResolver:
    """Resolve the external-access resolver backed by the current AuthNZ database."""
    pool = await get_db_pool()
    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    return McpHubExternalAccessResolver(repo=repo)
