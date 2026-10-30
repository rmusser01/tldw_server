from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.core.Provider_Credentials.byok_helpers import is_byok_enabled
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Provider_Credentials.repos.byok_validation_runs_repo import (
    AuthnzByokValidationRunsRepo,
)
from tldw_Server_API.app.core.Provider_Credentials.user_provider_secrets import normalize_provider_name
from tldw_Server_API.app.core.exceptions import (
    ByokValidationActiveRunError,
    ByokValidationDisabledError,
    ByokValidationRunNotFoundError,
)
from tldw_Server_API.app.services import admin_scope_service


def _requested_by_label(principal: AuthPrincipal) -> str | None:
    """Return a stable operator label for persisted validation runs."""
    return principal.email or principal.username or principal.subject


@dataclass
class AdminByokValidationService:
    """Service layer for authoritative admin BYOK validation runs."""

    repo: AuthnzByokValidationRunsRepo

    async def create_run(
        self,
        principal: AuthPrincipal,
        *,
        org_id: int | None,
        provider: str | None,
    ) -> dict[str, object]:
        """Validate create inputs, enforce scope, and persist a queued validation run."""
        if not is_byok_enabled():
            raise ByokValidationDisabledError("BYOK is disabled in this deployment")

        if org_id is not None:
            await admin_scope_service.enforce_admin_org_access(principal, org_id, require_admin=True)

        provider_norm = normalize_provider_name(provider) if provider else None

        if await self.repo.has_active_run():
            raise ByokValidationActiveRunError("active_validation_run_exists")

        try:
            return await self.repo.create_run(
                org_id=org_id,
                provider=provider_norm,
                requested_by_user_id=principal.user_id,
                requested_by_label=_requested_by_label(principal),
                scope_summary=self._scope_summary(org_id=org_id, provider=provider_norm),
            )
        except Exception as exc:
            if "idx_byok_validation_runs_active" in str(exc):
                raise ByokValidationActiveRunError("active_validation_run_exists") from exc
            raise

    async def list_runs(self, *, limit: int, offset: int) -> tuple[list[dict[str, object]], int]:
        """Return queued and historical BYOK validation runs."""
        return await self.repo.list_runs(limit=limit, offset=offset)

    async def get_run(self, run_id: str) -> dict[str, object]:
        """Return a BYOK validation run by id or raise 404 when absent."""
        item = await self.repo.get_run(run_id)
        if item is None:
            raise ByokValidationRunNotFoundError("byok_validation_run_not_found")
        return item

    @staticmethod
    def _scope_summary(*, org_id: int | None, provider: str | None) -> str:
        """Return the persisted human-readable scope summary."""
        parts: list[str] = []
        if org_id is not None:
            parts.append(f"org={org_id}")
        else:
            parts.append("global")
        if provider is not None:
            parts.append(f"provider={provider}")
        return ", ".join(parts)
