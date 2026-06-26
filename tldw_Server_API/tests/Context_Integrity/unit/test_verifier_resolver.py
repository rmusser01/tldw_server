from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _asset(asset_id: str, digest: str, *, executable: bool = False):
    from tldw_Server_API.app.core.Context_Integrity.models import ContextAssetDescriptor

    return ContextAssetDescriptor(
        asset_id=asset_id,
        source_type="skill_file",
        digest=digest,
        display_name=asset_id,
        executable=executable,
        owner_scope="user:1",
    )


def test_verifier_detects_changed_executable_asset() -> None:
    from tldw_Server_API.app.core.Context_Integrity.verifier import verify_inventory

    findings = verify_inventory(
        current_assets=[_asset("skill:user:1/demo", "sha256:current", executable=True)],
        approved_entries=[
            {
                "asset_id": "skill:user:1/demo",
                "source_type": "skill_file",
                "digest": "sha256:approved",
                "display_name": "demo",
                "executable": True,
                "required": False,
                "owner_scope": "user:1",
            }
        ],
    )

    assert len(findings) == 1
    assert findings[0].state == "changed_approved_executable"
    assert findings[0].severity == "error"


def test_verifier_uses_approved_executable_flag_for_changed_asset() -> None:
    from tldw_Server_API.app.core.Context_Integrity.verifier import verify_inventory

    findings = verify_inventory(
        current_assets=[_asset("skill:user:1/demo", "sha256:current", executable=False)],
        approved_entries=[
            {
                "asset_id": "skill:user:1/demo",
                "source_type": "skill_file",
                "digest": "sha256:approved",
                "display_name": "demo",
                "executable": True,
                "required": False,
                "owner_scope": "user:1",
            }
        ],
    )

    assert len(findings) == 1
    assert findings[0].state == "changed_approved_executable"
    assert findings[0].severity == "error"


def test_verifier_detects_changed_non_executable_asset() -> None:
    from tldw_Server_API.app.core.Context_Integrity.verifier import verify_inventory

    findings = verify_inventory(
        current_assets=[_asset("skill:user:1/readme", "sha256:current")],
        approved_entries=[
            {
                "asset_id": "skill:user:1/readme",
                "source_type": "skill_file",
                "digest": "sha256:approved",
                "display_name": "readme",
                "executable": False,
                "required": False,
                "owner_scope": "user:1",
            }
        ],
    )

    assert len(findings) == 1
    assert findings[0].state == "changed_approved_non_executable"
    assert findings[0].severity == "warning"


def test_verifier_detects_new_unapproved_asset() -> None:
    from tldw_Server_API.app.core.Context_Integrity.verifier import verify_inventory

    findings = verify_inventory(
        current_assets=[_asset("skill:user:1/new", "sha256:new")],
        approved_entries=[],
    )

    assert findings[0].state == "new_unapproved"


def test_verifier_detects_duplicate_live_asset_ids() -> None:
    from tldw_Server_API.app.core.Context_Integrity.verifier import verify_inventory

    findings = verify_inventory(
        current_assets=[
            _asset("skill:user:1/duplicate", "sha256:first"),
            _asset("skill:user:1/duplicate", "sha256:second"),
        ],
        approved_entries=[],
    )

    duplicate = [finding for finding in findings if finding.state == "verification_error"]
    assert len(duplicate) == 1
    assert duplicate[0].asset_id == "skill:user:1/duplicate"
    assert duplicate[0].severity == "error"


def test_verifier_detects_missing_assets_by_required_flag() -> None:
    from tldw_Server_API.app.core.Context_Integrity.verifier import verify_inventory

    findings = verify_inventory(
        current_assets=[],
        approved_entries=[
            {
                "asset_id": "skill:user:1/required",
                "source_type": "skill_file",
                "digest": "sha256:required",
                "display_name": "required",
                "executable": False,
                "required": True,
                "owner_scope": "user:1",
            },
            {
                "asset_id": "skill:user:1/optional",
                "source_type": "skill_file",
                "digest": "sha256:optional",
                "display_name": "optional",
                "executable": False,
                "required": False,
                "owner_scope": "user:1",
            },
        ],
    )

    findings_by_asset_id = {finding.asset_id: finding for finding in findings}
    assert findings_by_asset_id["skill:user:1/required"].state == "missing_required"
    assert findings_by_asset_id["skill:user:1/required"].severity == "error"
    assert findings_by_asset_id["skill:user:1/optional"].state == "missing_optional"
    assert findings_by_asset_id["skill:user:1/optional"].severity == "warning"


def test_resolver_blocks_quarantined_asset() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import (
        ContextIntegrityBootState,
        ContextIntegrityFinding,
    )
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityBlocked,
        ContextIntegrityResolver,
    )

    finding = ContextIntegrityFinding(
        asset_id="skill:user:1/demo",
        state="changed_approved_executable",
        severity="error",
        summary="changed",
        remediation="review",
        source_type="skill_file",
    )
    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            findings=(finding,),
        )
    )

    with pytest.raises(ContextIntegrityBlocked, match="quarantined"):
        resolver.require_allowed("skill:user:1/demo", purpose="skill_execution")


def test_resolver_detects_live_digest_mismatch_at_use() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityBlocked,
        ContextIntegrityResolver,
    )

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            approved_digests_by_asset_id={"skill:user:1/demo": "sha256:approved"},
        )
    )

    with pytest.raises(ContextIntegrityBlocked, match="quarantined"):
        resolver.require_digest_allowed(
            "skill:user:1/demo",
            current_digest="sha256:changed",
            purpose="skill_execution",
        )


def test_resolver_blocks_unknown_asset_in_enforce_mode() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityBlocked,
        ContextIntegrityResolver,
    )

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            approved_digests_by_asset_id={"skill:user:1/known": "sha256:approved"},
        )
    )

    with pytest.raises(ContextIntegrityBlocked) as exc_info:
        resolver.require_allowed("skill:user:1/live-added", purpose="skill_discovery")

    assert exc_info.value.state == "new_unapproved"


def test_resolver_blocks_unknown_asset_in_hardened_mode() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityBlocked,
        ContextIntegrityResolver,
    )

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="hardened",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            approved_digests_by_asset_id={"skill:user:1/known": "sha256:approved"},
        )
    )

    with pytest.raises(ContextIntegrityBlocked) as exc_info:
        resolver.require_allowed("skill:user:1/live-added", purpose="skill_discovery")

    assert exc_info.value.state == "new_unapproved"


def test_resolver_blocks_degraded_injection_use() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityBlocked,
        ContextIntegrityResolver,
    )

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=True,
            manifest_sequence=None,
            manifest_digest=None,
            approved_digests_by_asset_id={"prompt_file:demo.prompts.md": "sha256:approved"},
        )
    )

    with pytest.raises(ContextIntegrityBlocked) as exc_info:
        resolver.require_allowed("prompt_file:demo.prompts.md", purpose="prompt_load")

    assert exc_info.value.state == "degraded_integrity"


def test_resolver_allows_degraded_admin_review_purpose() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityResolver

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="hardened",
            degraded=True,
            manifest_sequence=None,
            manifest_digest=None,
            approved_digests_by_asset_id={"prompt_file:demo.prompts.md": "sha256:approved"},
        )
    )

    resolver.require_allowed(
        "prompt_file:demo.prompts.md",
        purpose="admin_review_status",
    )


def test_resolver_audit_only_allows_quarantined_and_unknown_assets() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import (
        ContextIntegrityBootState,
        ContextIntegrityFinding,
    )
    from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityResolver

    finding = ContextIntegrityFinding(
        asset_id="skill:user:1/demo",
        state="changed_approved_executable",
        severity="error",
        summary="changed",
        remediation="review",
        source_type="skill_file",
    )
    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="audit_only",
            degraded=True,
            manifest_sequence=None,
            manifest_digest=None,
            findings=(finding,),
        )
    )

    resolver.require_allowed("skill:user:1/demo", purpose="skill_execution")
    resolver.require_allowed("skill:user:1/unknown", purpose="skill_discovery")


def test_resolver_allows_matching_digest_at_use() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityResolver

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            approved_digests_by_asset_id={"prompt_file:demo.prompts.md": "sha256:approved"},
        )
    )

    resolver.require_digest_allowed(
        "prompt_file:demo.prompts.md",
        current_digest="sha256:approved",
        purpose="prompt_load",
    )


def test_global_resolver_can_be_set_and_cleared() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityResolver,
        clear_global_context_integrity_resolver,
        get_global_context_integrity_resolver,
        set_global_context_integrity_resolver,
    )

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
        )
    )
    try:
        set_global_context_integrity_resolver(resolver)
        assert get_global_context_integrity_resolver() is resolver
    finally:
        clear_global_context_integrity_resolver()
    assert get_global_context_integrity_resolver() is None
