"""Runtime resolver for context integrity enforcement."""

from __future__ import annotations

from tldw_Server_API.app.core.Context_Integrity.models import (
    ContextAssetState,
    ContextIntegrityBootState,
    ContextIntegrityFinding,
)

_ENFORCING_MODES = frozenset(("enforce", "hardened"))


class ContextIntegrityBlocked(RuntimeError):
    """Raised when a prompt-bearing asset is quarantined or unavailable."""

    def __init__(self, asset_id: str, state: ContextAssetState) -> None:
        self.asset_id = asset_id
        self.state = state
        super().__init__(f"Context integrity quarantined asset {asset_id!r}; state={state}.")


class ContextIntegrityResolver:
    """Current-process resolver backed by verified boot state."""

    def __init__(self, boot_state: ContextIntegrityBootState) -> None:
        self.boot_state = boot_state
        self._findings_by_asset_id: dict[str, ContextIntegrityFinding] = {
            finding.asset_id: finding for finding in boot_state.findings
        }

    def finding_for(self, asset_id: str) -> ContextIntegrityFinding | None:
        """Return the current boot finding for an asset, if any."""
        return self._findings_by_asset_id.get(asset_id)

    def require_allowed(self, asset_id: str, *, purpose: str) -> None:
        """Raise if the asset is not allowed for the requested runtime purpose."""
        if self.boot_state.mode == "audit_only":
            return

        if self.boot_state.degraded and not purpose.startswith("admin_review"):
            raise ContextIntegrityBlocked(
                asset_id=asset_id,
                state="degraded_integrity",
            )

        finding = self.finding_for(asset_id)
        if finding is not None and finding.state != "trusted":
            raise ContextIntegrityBlocked(asset_id=asset_id, state=finding.state)

        if self.boot_state.mode in _ENFORCING_MODES and asset_id not in self.boot_state.approved_digests_by_asset_id:
            raise ContextIntegrityBlocked(asset_id=asset_id, state="new_unapproved")

    def require_digest_allowed(
        self,
        asset_id: str,
        *,
        current_digest: str,
        purpose: str,
        changed_state: ContextAssetState = "changed_approved_executable",
    ) -> None:
        """Raise if the asset is disallowed or its live digest is no longer approved."""
        self.require_allowed(asset_id, purpose=purpose)
        if self.boot_state.mode == "audit_only":
            return

        approved_digest = self.boot_state.approved_digests_by_asset_id.get(asset_id)
        if approved_digest is None:
            raise ContextIntegrityBlocked(asset_id=asset_id, state="new_unapproved")
        if approved_digest != current_digest:
            raise ContextIntegrityBlocked(asset_id=asset_id, state=changed_state)


_global_resolver: ContextIntegrityResolver | None = None


def set_global_context_integrity_resolver(resolver: ContextIntegrityResolver | None) -> None:
    """Set the process-wide Context Integrity resolver compatibility bridge."""
    global _global_resolver
    _global_resolver = resolver


def get_global_context_integrity_resolver() -> ContextIntegrityResolver | None:
    """Return the process-wide Context Integrity resolver, if configured."""
    return _global_resolver


def clear_global_context_integrity_resolver() -> None:
    """Clear the process-wide Context Integrity resolver."""
    set_global_context_integrity_resolver(None)
