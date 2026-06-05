from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest


async def _make_repo(tmp_path: Path, monkeypatch: Any):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    return repo


async def _update_trust_policy(service: Any, payload: dict[str, Any], *, actor_id: int) -> dict[str, Any]:
    current_policy = await service.get_policy()
    request_payload = dict(payload)
    request_payload["policy_fingerprint"] = current_policy["policy_fingerprint"]
    return await service.update_policy(request_payload, actor_id=actor_id)


def _fixture_pack_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "fixtures"
        / "governance_packs"
        / "minimal_researcher_pack"
    )


def _init_git_pack_repo(tmp_path: Path, *, subpath: str = "packs/researcher") -> tuple[str, str]:
    repo_root = tmp_path / "git-pack"
    repo_root.mkdir()
    pack_root = repo_root / subpath
    shutil.copytree(_fixture_pack_path(), pack_root)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "codex@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Codex"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "Add governance pack fixture"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "tag", "v1.0.0"], cwd=repo_root, check=True, capture_output=True, text=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo_root.as_uri(), commit


class _FakeGitTrustService:
    def __init__(
        self,
        *,
        canonical_repository: str = "github.com/example/researcher-pack",
        verification_required: bool = False,
        trusted_git_key_fingerprints: list[str] | None = None,
    ) -> None:
        self.canonical_repository = canonical_repository
        self.verification_required = verification_required
        self.trusted_git_key_fingerprints = list(trusted_git_key_fingerprints or [])
        self.calls: list[dict[str, str]] = []

    async def evaluate_git_source(self, repo_url: str, *, ref_kind: str) -> dict[str, Any]:
        self.calls.append({"repo_url": repo_url, "ref_kind": ref_kind})
        return {
            "allowed": True,
            "reason": None,
            "canonical_repository": self.canonical_repository,
            "verification_required": self.verification_required,
            "trusted_git_key_fingerprints": list(self.trusted_git_key_fingerprints),
        }

    async def evaluate_signer_for_repository(self, signer_fingerprint: str, repo_url: str) -> dict[str, Any]:
        cleaned = str(signer_fingerprint or "").strip().upper()
        if cleaned in {value.upper() for value in self.trusted_git_key_fingerprints}:
            return {
                "allowed": True,
                "reason": None,
                "result_code": "signer_trusted_for_repo",
                "canonical_repository": self.canonical_repository,
                "signer_fingerprint": cleaned,
            }
        return {
            "allowed": False,
            "reason": "signer_not_allowed_for_repo",
            "result_code": "signer_not_allowed_for_repo",
            "canonical_repository": self.canonical_repository,
            "signer_fingerprint": cleaned,
        }


@pytest.mark.asyncio
async def test_governance_pack_trust_service_enforces_local_root_allowlist(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    inside = allowed_root / "packs" / "researcher"
    inside.mkdir(parents=True)
    outside = tmp_path / "outside" / "researcher"
    outside.mkdir(parents=True)

    await _update_trust_policy(
        service,
        {
            "allow_local_path_sources": True,
            "allowed_local_roots": [str(allowed_root)],
        },
        actor_id=7,
    )

    allowed = await service.evaluate_local_path(str(inside))
    assert allowed["allowed"] is True
    assert allowed["resolved_path"] == str(inside.resolve())

    denied = await service.evaluate_local_path(str(outside))
    assert denied["allowed"] is False
    assert denied["reason"] == "path_not_allowed"


@pytest.mark.asyncio
async def test_governance_pack_trust_service_enforces_git_repo_and_ref_policy(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    await _update_trust_policy(
        service,
        {
            "allow_git_sources": True,
            "allowed_git_hosts": ["github.com"],
            "allowed_git_repositories": ["github.com/example/researcher-pack"],
            "allowed_git_ref_kinds": ["tag"],
            "require_git_signature_verification": True,
        },
        actor_id=7,
    )

    allowed = await service.evaluate_git_source(
        "https://github.com/example/researcher-pack.git",
        ref_kind="tag",
    )
    assert allowed["allowed"] is True
    assert allowed["verification_required"] is True
    assert allowed["canonical_repository"] == "github.com/example/researcher-pack"

    denied_repo = await service.evaluate_git_source(
        "https://github.com/example/other-pack.git",
        ref_kind="tag",
    )
    assert denied_repo["allowed"] is False
    assert denied_repo["reason"] == "repository_not_allowed"

    denied_ref = await service.evaluate_git_source(
        "https://github.com/example/researcher-pack.git",
        ref_kind="branch",
    )
    assert denied_ref["allowed"] is False
    assert denied_ref["reason"] == "ref_kind_not_allowed"


@pytest.mark.asyncio
async def test_governance_pack_trust_policy_persists_as_deployment_wide_config(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    updated = await _update_trust_policy(
        service,
        {
            "allow_local_path_sources": True,
            "allowed_local_roots": ["/srv/packs"],
            "allow_git_sources": True,
            "allowed_git_hosts": ["github.com"],
            "allowed_git_repositories": ["github.com/example/researcher-pack"],
            "allowed_git_ref_kinds": ["commit", "tag"],
            "require_git_signature_verification": True,
        },
        actor_id=11,
    )

    expected_local_roots = [str(Path("/srv/packs").resolve())]
    assert updated["allowed_local_roots"] == expected_local_roots
    assert updated["allowed_git_ref_kinds"] == ["commit", "tag"]
    assert updated["require_git_signature_verification"] is True

    stored = await repo.get_governance_pack_trust_policy()
    assert stored["policy_document"]["allowed_local_roots"] == expected_local_roots
    assert stored["updated_by"] == 11


@pytest.mark.asyncio
async def test_governance_pack_trust_service_normalizes_structured_signers_and_legacy_fingerprints(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    policy = await _update_trust_policy(
        service,
        {
            "allow_git_sources": True,
            "allowed_git_hosts": ["github.com"],
            "allowed_git_repositories": ["github.com/example/packs"],
            "allowed_git_ref_kinds": ["tag"],
            "require_git_signature_verification": True,
            "trusted_signers": [
                {
                    "fingerprint": "abc123",
                    "display_name": "Release Bot",
                    "repo_bindings": [
                        "github.com/example/packs",
                        "github.com/example/",
                    ],
                    "status": "active",
                },
                {
                    "fingerprint": "def456",
                    "repo_bindings": ["github.com/example"],
                    "status": "inactive",
                },
            ],
            "trusted_git_key_fingerprints": ["legacy789"],
        },
        actor_id=1,
    )

    assert policy["trusted_signers"][0]["fingerprint"] == "ABC123"
    assert policy["trusted_signers"][0]["display_name"] == "Release Bot"
    assert policy["trusted_signers"][0]["repo_bindings"] == [
        "github.com/example/packs",
        "github.com/example/",
    ]
    assert policy["trusted_signers"][0]["status"] == "active"
    assert policy["trusted_signers"][1]["fingerprint"] == "DEF456"
    assert policy["trusted_signers"][1]["repo_bindings"] == ["github.com/example"]
    assert policy["trusted_signers"][1]["status"] == "inactive"
    assert policy["trusted_signers"][2]["fingerprint"] == "LEGACY789"
    assert policy["trusted_signers"][2]["repo_bindings"] == ["github.com/example/packs"]
    assert policy["trusted_signers"][2]["status"] == "active"
    assert "trusted_git_key_fingerprints" not in policy

    stored = await repo.get_governance_pack_trust_policy()
    assert "trusted_git_key_fingerprints" not in stored["policy_document"]


@pytest.mark.asyncio
async def test_governance_pack_trust_service_updates_legacy_policy_rows_without_false_stale_conflicts(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    legacy_policy = {
        "allow_git_sources": True,
        "allowed_git_hosts": ["github.com"],
        "allowed_git_repositories": ["github.com/example/packs"],
        "allowed_git_ref_kinds": ["tag"],
        "require_git_signature_verification": True,
        "trusted_git_key_fingerprints": ["legacy789"],
    }
    await repo.db_pool.execute(
        """
        INSERT INTO mcp_governance_pack_trust_policy (id, policy_document_json, updated_by, updated_at)
        VALUES (1, ?, ?, CURRENT_TIMESTAMP)
        """,
        (json.dumps(legacy_policy), 99),
    )

    current = await service.get_policy()
    updated = await service.update_policy(
        {
            "policy_fingerprint": current["policy_fingerprint"],
            "allow_git_sources": True,
            "allowed_git_hosts": ["github.com"],
            "allowed_git_repositories": ["github.com/example/packs"],
            "allowed_git_ref_kinds": ["tag"],
            "require_git_signature_verification": True,
            "trusted_signers": [
                {
                    "fingerprint": "ABC123",
                    "repo_bindings": ["github.com/example/packs"],
                    "status": "active",
                }
            ],
        },
        actor_id=7,
    )

    assert updated["trusted_signers"] == [
        {
            "fingerprint": "ABC123",
            "display_name": None,
            "repo_bindings": ["github.com/example/packs"],
            "status": "active",
        }
    ]
    stored = await repo.get_governance_pack_trust_policy()
    assert "trusted_git_key_fingerprints" not in stored["policy_document"]
    assert stored["updated_by"] == 7


@pytest.mark.asyncio
async def test_governance_pack_trust_service_keeps_structured_signer_status_authoritative_over_legacy_fingerprints(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    policy = await _update_trust_policy(
        service,
        {
            "allow_git_sources": True,
            "allowed_git_hosts": ["github.com"],
            "allowed_git_repositories": ["github.com/example/packs"],
            "allowed_git_ref_kinds": ["tag"],
            "trusted_signers": [
                {
                    "fingerprint": "abc123",
                    "display_name": "Release Bot",
                    "repo_bindings": ["github.com/example/packs"],
                    "status": "revoked",
                }
            ],
            "trusted_git_key_fingerprints": ["abc123"],
        },
        actor_id=1,
    )

    assert policy["trusted_signers"] == [
        {
            "fingerprint": "ABC123",
            "display_name": "Release Bot",
            "repo_bindings": ["github.com/example/packs"],
            "status": "revoked",
        }
    ]
    assert "trusted_git_key_fingerprints" not in policy


@pytest.mark.asyncio
async def test_governance_pack_trust_service_rejects_duplicate_structured_signer_fingerprints(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    with pytest.raises(ValueError, match="duplicate trusted signer fingerprint"):
        await _update_trust_policy(
            service,
            {
                "allow_git_sources": True,
                "allowed_git_hosts": ["github.com"],
                "allowed_git_repositories": ["github.com/example/packs"],
                "allowed_git_ref_kinds": ["tag"],
                "trusted_signers": [
                    {
                        "fingerprint": "abc123",
                        "repo_bindings": ["github.com/example/packs"],
                        "status": "active",
                    },
                    {
                        "fingerprint": "ABC123",
                        "repo_bindings": ["github.com/example/"],
                        "status": "revoked",
                    },
                ],
            },
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_governance_pack_trust_service_rejects_invalid_structured_signer_status(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    with pytest.raises(ValueError, match="invalid signer status"):
        await _update_trust_policy(
            service,
            {
                "allow_git_sources": True,
                "allowed_git_hosts": ["github.com"],
                "allowed_git_repositories": ["github.com/example/packs"],
                "allowed_git_ref_kinds": ["tag"],
                "trusted_signers": [
                    {
                        "fingerprint": "abc123",
                        "repo_bindings": ["github.com/example/packs"],
                        "status": "pending",
                    }
                ],
            },
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_governance_pack_trust_service_rejects_blank_repo_binding(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    with pytest.raises(ValueError, match="repo binding entries cannot be blank"):
        await _update_trust_policy(
            service,
            {
                "allow_git_sources": True,
                "allowed_git_hosts": ["github.com"],
                "allowed_git_repositories": ["github.com/example/packs"],
                "allowed_git_ref_kinds": ["tag"],
                "trusted_signers": [
                    {
                        "fingerprint": "abc123",
                        "repo_bindings": ["   "],
                        "status": "active",
                    }
                ],
            },
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_governance_pack_trust_service_rejects_empty_structured_repo_bindings(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    with pytest.raises(ValueError, match="trusted signer repo_bindings must not be empty"):
        await _update_trust_policy(
            service,
            {
                "allow_git_sources": True,
                "allowed_git_hosts": ["github.com"],
                "allowed_git_repositories": ["github.com/example/packs"],
                "allowed_git_ref_kinds": ["tag"],
                "trusted_signers": [
                    {
                        "fingerprint": "abc123",
                        "repo_bindings": [],
                        "status": "active",
                    }
                ],
            },
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_governance_pack_trust_service_rejects_blank_legacy_fingerprint(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    with pytest.raises(ValueError, match="fingerprint entries cannot be blank"):
        await _update_trust_policy(
            service,
            {
                "allow_git_sources": True,
                "allowed_git_hosts": ["github.com"],
                "allowed_git_repositories": ["github.com/example/packs"],
                "allowed_git_ref_kinds": ["tag"],
                "trusted_git_key_fingerprints": ["   "],
            },
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_governance_pack_trust_service_rejects_invalid_legacy_fingerprint_type(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    with pytest.raises(ValueError, match="fingerprint must be a string or list/tuple/set of strings"):
        await _update_trust_policy(
            service,
            {
                "allow_git_sources": True,
                "allowed_git_hosts": ["github.com"],
                "allowed_git_repositories": ["github.com/example/packs"],
                "allowed_git_ref_kinds": ["tag"],
                "trusted_git_key_fingerprints": 123,
            },
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_governance_pack_trust_service_denies_git_evaluation_when_persisted_policy_is_invalid(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    await repo.upsert_governance_pack_trust_policy(
        policy_document={
            "allow_git_sources": True,
            "allowed_git_hosts": ["github.com"],
            "allowed_git_repositories": ["github.com/example/packs"],
            "allowed_git_ref_kinds": ["tag"],
            "trusted_git_key_fingerprints": ["   "],
        },
        actor_id=7,
    )

    evaluation = await service.evaluate_git_source(
        "https://github.com/example/packs.git",
        ref_kind="tag",
    )

    assert evaluation["allowed"] is False
    assert evaluation["reason"] == "invalid_trust_policy"
    assert evaluation["canonical_repository"] == "github.com/example/packs"
    assert evaluation["trusted_signers"] == []
    assert evaluation["trusted_git_key_fingerprints"] == []


@pytest.mark.asyncio
async def test_governance_pack_trust_service_rejects_stale_write_after_precheck_race(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        GovernancePackTrustPolicyStaleError,
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    initial = await _update_trust_policy(
        service,
        {
            "allow_git_sources": True,
            "allowed_git_hosts": ["github.com"],
            "allowed_git_repositories": ["github.com/example/packs"],
            "allowed_git_ref_kinds": ["tag"],
            "trusted_signers": [
                {
                    "fingerprint": "abc123",
                    "repo_bindings": ["github.com/example/packs"],
                    "status": "active",
                }
            ],
        },
        actor_id=1,
    )

    original_upsert = repo.upsert_governance_pack_trust_policy
    raced = False

    async def _racing_upsert(**kwargs):
        nonlocal raced
        if not raced:
            raced = True
            await original_upsert(
                policy_document={
                    "allow_git_sources": True,
                    "allowed_git_hosts": ["github.com"],
                    "allowed_git_repositories": ["github.com/example/packs"],
                    "allowed_git_ref_kinds": ["tag"],
                    "require_git_signature_verification": False,
                    "trusted_signers": [
                        {
                            "fingerprint": "race999",
                            "repo_bindings": ["github.com/example/packs"],
                            "status": "active",
                        }
                    ],
                },
                actor_id=99,
            )
        return await original_upsert(**kwargs)

    repo.upsert_governance_pack_trust_policy = _racing_upsert  # type: ignore[method-assign]

    with pytest.raises(GovernancePackTrustPolicyStaleError, match="stale governance pack trust policy write"):
        await service.update_policy(
            {
                "policy_fingerprint": initial["policy_fingerprint"],
                "allow_git_sources": True,
                "allowed_git_hosts": ["github.com"],
                "allowed_git_repositories": ["github.com/example/packs"],
                "allowed_git_ref_kinds": ["tag"],
                "trusted_signers": [
                    {
                        "fingerprint": "def456",
                        "repo_bindings": ["github.com/example/packs"],
                        "status": "active",
                    }
                ],
            },
            actor_id=2,
        )


@pytest.mark.asyncio
async def test_governance_pack_trust_service_filters_signers_by_repo_bindings_when_evaluating_git_sources(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    await _update_trust_policy(
        service,
        {
            "allow_git_sources": True,
            "allowed_git_hosts": ["github.com"],
            "allowed_git_repositories": [
                "github.com/example/researcher-pack",
                "github.com/example/other-pack",
            ],
            "allowed_git_ref_kinds": ["tag"],
            "require_git_signature_verification": True,
            "trusted_signers": [
                {
                    "fingerprint": "EXACT123",
                    "repo_bindings": ["github.com/example/researcher-pack"],
                    "status": "active",
                },
                {
                    "fingerprint": "PREFIX123",
                    "repo_bindings": ["github.com/example/"],
                    "status": "active",
                },
                {
                    "fingerprint": "OTHER123",
                    "repo_bindings": ["github.com/example/other-pack"],
                    "status": "active",
                },
            ],
            "trusted_git_key_fingerprints": ["legacy999"],
        },
        actor_id=3,
    )

    researcher_pack = await service.evaluate_git_source(
        "https://github.com/example/researcher-pack.git",
        ref_kind="tag",
    )
    other_pack = await service.evaluate_git_source(
        "https://github.com/example/other-pack.git",
        ref_kind="tag",
    )

    assert [signer["fingerprint"] for signer in researcher_pack["trusted_signers"]] == [
        "EXACT123",
        "PREFIX123",
        "LEGACY999",
    ]
    assert researcher_pack["trusted_git_key_fingerprints"] == [
        "EXACT123",
        "PREFIX123",
        "LEGACY999",
    ]
    assert [signer["fingerprint"] for signer in other_pack["trusted_signers"]] == [
        "PREFIX123",
        "OTHER123",
        "LEGACY999",
    ]
    assert other_pack["trusted_git_key_fingerprints"] == [
        "PREFIX123",
        "OTHER123",
        "LEGACY999",
    ]


@pytest.mark.asyncio
async def test_governance_pack_trust_service_excludes_inactive_and_revoked_signers_from_git_evaluation(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackTrustService(repo=repo)

    await _update_trust_policy(
        service,
        {
            "allow_git_sources": True,
            "allowed_git_hosts": ["github.com"],
            "allowed_git_repositories": ["github.com/example/researcher-pack"],
            "allowed_git_ref_kinds": ["tag"],
            "require_git_signature_verification": True,
            "trusted_signers": [
                {
                    "fingerprint": "ACTIVE123",
                    "repo_bindings": ["github.com/example/researcher-pack"],
                    "status": "active",
                },
                {
                    "fingerprint": "INACTIVE123",
                    "repo_bindings": ["github.com/example/researcher-pack"],
                    "status": "inactive",
                },
                {
                    "fingerprint": "REVOKED123",
                    "repo_bindings": ["github.com/example/"],
                    "status": "revoked",
                },
            ],
            "trusted_git_key_fingerprints": ["legacy999"],
        },
        actor_id=4,
    )

    evaluation = await service.evaluate_git_source(
        "https://github.com/example/researcher-pack.git",
        ref_kind="tag",
    )

    assert [signer["fingerprint"] for signer in evaluation["trusted_signers"]] == [
        "ACTIVE123",
        "LEGACY999",
    ]
    assert evaluation["trusted_git_key_fingerprints"] == [
        "ACTIVE123",
        "LEGACY999",
    ]


@pytest.mark.asyncio
async def test_distribution_service_resolves_allowed_local_pack_and_digest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    trust_service = McpHubGovernancePackTrustService(repo=repo)
    allowed_root = tmp_path / "allowed"
    pack_path = allowed_root / "researcher-pack"
    allowed_root.mkdir()
    shutil.copytree(_fixture_pack_path(), pack_path)
    await _update_trust_policy(
        trust_service,
        {
            "allow_local_path_sources": True,
            "allowed_local_roots": [str(allowed_root)],
        },
        actor_id=5,
    )

    service = McpHubGovernancePackDistributionService(trust_service=trust_service)
    resolved = await service.resolve_local_path(str(pack_path))

    assert resolved.manifest.pack_id == "researcher-pack"
    assert resolved.source_type == "local_path"
    assert resolved.source_location == str(pack_path.resolve())
    assert resolved.pack_content_digest


@pytest.mark.asyncio
async def test_distribution_service_resolve_local_path_uses_thread_offload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services import mcp_hub_governance_pack_distribution_service as distribution_module
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    trust_service = McpHubGovernancePackTrustService(repo=repo)
    allowed_root = tmp_path / "allowed"
    pack_path = allowed_root / "researcher-pack"
    allowed_root.mkdir()
    shutil.copytree(_fixture_pack_path(), pack_path)
    await _update_trust_policy(
        trust_service,
        {
            "allow_local_path_sources": True,
            "allowed_local_roots": [str(allowed_root)],
        },
        actor_id=5,
    )

    service = McpHubGovernancePackDistributionService(trust_service=trust_service)
    calls: list[str] = []

    async def _fake_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        calls.append(getattr(func, "__name__", "<anonymous>"))
        return func(*args, **kwargs)

    monkeypatch.setattr(distribution_module.asyncio, "to_thread", _fake_to_thread)

    resolved = await service.resolve_local_path(str(pack_path))

    assert resolved.manifest.pack_id == "researcher-pack"
    assert calls == ["_load_local_pack_sync"]


@pytest.mark.asyncio
async def test_distribution_service_rejects_local_pack_outside_allowlist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    trust_service = McpHubGovernancePackTrustService(repo=repo)
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    pack_path = outside_root / "researcher-pack"
    shutil.copytree(_fixture_pack_path(), pack_path)
    await _update_trust_policy(
        trust_service,
        {
            "allow_local_path_sources": True,
            "allowed_local_roots": [str(allowed_root)],
        },
        actor_id=5,
    )

    service = McpHubGovernancePackDistributionService(trust_service=trust_service)
    with pytest.raises(ValueError, match="path_not_allowed"):
        await service.resolve_local_path(str(pack_path))


@pytest.mark.asyncio
async def test_distribution_service_resolves_git_source_to_exact_commit_and_digest(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    repo_url, commit = _init_git_pack_repo(tmp_path)
    trust_service = _FakeGitTrustService()
    service = McpHubGovernancePackDistributionService(trust_service=trust_service)

    resolved = await service.resolve_git_source(
        repo_url,
        ref="v1.0.0",
        ref_kind="tag",
        subpath="packs/researcher",
    )

    assert resolved.manifest.pack_id == "researcher-pack"
    assert resolved.source_type == "git"
    assert resolved.source_location == repo_url
    assert resolved.source_ref_requested == "v1.0.0"
    assert resolved.source_ref_kind == "tag"
    assert resolved.source_subpath == "packs/researcher"
    assert resolved.source_commit_resolved == commit
    assert resolved.source_path is None
    assert resolved.pack_content_digest


@pytest.mark.asyncio
async def test_distribution_service_checkout_git_source_uses_thread_offload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services import mcp_hub_governance_pack_distribution_service as distribution_module
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    service = McpHubGovernancePackDistributionService(trust_service=_FakeGitTrustService())
    calls: list[str] = []

    async def _fake_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        calls.append(getattr(func, "__name__", "<anonymous>"))
        return func(*args, **kwargs)

    def _fake_checkout_sync(
        *,
        repo_url: str,
        ref: str | None,
        ref_kind: str,
        checkout_root: Path,
    ) -> str:
        assert repo_url == "https://example.com/researcher-pack.git"
        assert ref == "main"
        assert ref_kind == "branch"
        assert checkout_root == tmp_path / "checkout"
        return "abc123"

    monkeypatch.setattr(distribution_module.asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(service, "_checkout_git_source_sync", _fake_checkout_sync)

    commit = await service._checkout_git_source(
        repo_url="https://example.com/researcher-pack.git",
        ref="main",
        ref_kind="branch",
        checkout_root=tmp_path / "checkout",
    )

    assert commit == "abc123"
    assert calls == ["_fake_checkout_sync"]


@pytest.mark.asyncio
async def test_distribution_service_verify_git_revision_uses_thread_offload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services import mcp_hub_governance_pack_distribution_service as distribution_module
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    service = McpHubGovernancePackDistributionService(
        trust_service=_FakeGitTrustService(trusted_git_key_fingerprints=["ABCD1234"])
    )
    calls: list[str] = []

    async def _fake_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        calls.append(getattr(func, "__name__", "<anonymous>"))
        return func(*args, **kwargs)

    def _fake_verify_sync(
        *,
        checkout_root: Path,
        repo_url: str,
        ref: str | None,
        ref_kind: str,
        commit: str,
        trusted_git_key_fingerprints: list[str] | None = None,
    ) -> dict[str, Any]:
        assert checkout_root == tmp_path / "checkout"
        assert repo_url == "https://example.com/researcher-pack.git"
        assert ref == "v1.0.0"
        assert ref_kind == "tag"
        assert commit == "abc123"
        assert trusted_git_key_fingerprints == ["ABCD1234"]
        return {
            "verified": True,
            "verification_mode": "git_signature",
            "verified_object_type": "tag",
            "signer_fingerprint": "ABCD1234",
            "signer_identity": "Release Bot <bot@example.com>",
            "result_code": "verified_and_trusted",
            "warning_code": None,
        }

    monkeypatch.setattr(distribution_module.asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(service, "_verify_git_revision_sync", _fake_verify_sync)

    verified = await service._verify_git_revision(
        checkout_root=tmp_path / "checkout",
        repo_url="https://example.com/researcher-pack.git",
        ref="v1.0.0",
        ref_kind="tag",
        commit="abc123",
        trusted_git_key_fingerprints=["ABCD1234"],
    )

    assert verified["verified"] is True
    assert verified["result_code"] == "verified_and_trusted"
    assert calls == ["_fake_verify_sync"]


@pytest.mark.asyncio
async def test_distribution_service_verify_git_revision_awaits_async_signer_trust(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    class _Trust:
        async def evaluate_signer_for_repository(self, signer_fingerprint: str, repo_url: str) -> dict[str, Any]:
            assert signer_fingerprint == "ABCD1234"
            assert repo_url == "https://example.com/researcher-pack.git"
            return {
                "allowed": True,
                "reason": None,
                "result_code": "signer_trusted_for_repo",
                "canonical_repository": "github.com/example/researcher-pack",
                "signer_fingerprint": signer_fingerprint,
            }

    service = McpHubGovernancePackDistributionService(trust_service=_Trust())

    def _fake_verify_sync(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["trusted_git_key_fingerprints"] == ["ABCD1234"]
        return {
            "verified": True,
            "verification_mode": "git_signature",
            "verified_object_type": "tag",
            "signer_fingerprint": "ABCD1234",
            "signer_identity": "Release Bot <bot@example.com>",
            "result_code": "verified_locally",
            "warning_code": None,
        }

    monkeypatch.setattr(service, "_verify_git_revision_sync", _fake_verify_sync)

    result = await service._verify_git_revision(
        checkout_root=tmp_path / "checkout",
        repo_url="https://example.com/researcher-pack.git",
        ref="v1.0.0",
        ref_kind="tag",
        commit="abc123",
        trusted_git_key_fingerprints=["ABCD1234"],
    )

    assert result["verified"] is True
    assert result["result_code"] == "verified_and_trusted"
    assert result["signer_fingerprint"] == "ABCD1234"


def test_distribution_service_verify_git_revision_sync_returns_structured_result_for_signed_tag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services import mcp_hub_governance_pack_distribution_service as distribution_module
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    service = McpHubGovernancePackDistributionService(
        trust_service=_FakeGitTrustService(trusted_git_key_fingerprints=["ABCD1234"])
    )

    def _fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        assert command[-4:] == ["verify-tag", "--raw", "--", "v1.0.0"]
        return SimpleNamespace(
            returncode=0,
            stdout="",
            stderr='[GNUPG:] VALIDSIG ABCD1234\n[GNUPG:] GOODSIG ABCD1234 Release Bot\ngpg: Good signature from "Release Bot <bot@example.com>"',
        )

    monkeypatch.setattr(distribution_module, "_git_executable", lambda: "git")
    monkeypatch.setattr(distribution_module.subprocess, "run", _fake_run)

    result = service._verify_git_revision_sync(
        checkout_root=tmp_path / "checkout",
        repo_url="https://github.com/example/researcher-pack.git",
        ref="v1.0.0",
        ref_kind="tag",
        commit="abc123",
        trusted_git_key_fingerprints=["ABCD1234"],
    )

    assert result["verified"] is True
    assert result["verification_mode"] == "git_signature"
    assert result["verified_object_type"] == "tag"
    assert result["signer_fingerprint"] == "ABCD1234"
    assert result["signer_identity"] == "Release Bot <bot@example.com>"
    assert result["result_code"] == "verified_and_trusted"
    assert result["warning_code"] is None


def test_distribution_service_verify_git_revision_sync_returns_signature_invalid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services import mcp_hub_governance_pack_distribution_service as distribution_module
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    service = McpHubGovernancePackDistributionService(trust_service=_FakeGitTrustService())

    def _fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        assert command[-4:] == ["verify-commit", "--raw", "--", "abc123"]
        return SimpleNamespace(returncode=1, stdout="", stderr="gpg: BAD signature")

    monkeypatch.setattr(distribution_module, "_git_executable", lambda: "git")
    monkeypatch.setattr(distribution_module.subprocess, "run", _fake_run)

    result = service._verify_git_revision_sync(
        checkout_root=tmp_path / "checkout",
        repo_url="https://github.com/example/researcher-pack.git",
        ref=None,
        ref_kind="commit",
        commit="abc123",
        trusted_git_key_fingerprints=["ABCD1234"],
    )

    assert result["verified"] is False
    assert result["verified_object_type"] == "commit"
    assert result["result_code"] == "signature_invalid"
    assert result["warning_code"] is None


def test_distribution_service_verify_git_revision_sync_returns_signer_unknown_without_validsig(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services import mcp_hub_governance_pack_distribution_service as distribution_module
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    service = McpHubGovernancePackDistributionService(
        trust_service=_FakeGitTrustService(trusted_git_key_fingerprints=["ABCD1234"])
    )

    def _fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout="",
            stderr='gpg: Good signature from "Release Bot <bot@example.com>"',
        )

    monkeypatch.setattr(distribution_module, "_git_executable", lambda: "git")
    monkeypatch.setattr(distribution_module.subprocess, "run", _fake_run)

    result = service._verify_git_revision_sync(
        checkout_root=tmp_path / "checkout",
        repo_url="https://github.com/example/researcher-pack.git",
        ref="v1.0.0",
        ref_kind="tag",
        commit="abc123",
        trusted_git_key_fingerprints=["ABCD1234"],
    )

    assert result["verified"] is False
    assert result["result_code"] == "signer_unknown"
    assert result["warning_code"] is None


def test_distribution_service_verify_git_revision_sync_reports_unsupported_signature_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services import mcp_hub_governance_pack_distribution_service as distribution_module
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    service = McpHubGovernancePackDistributionService(
        trust_service=_FakeGitTrustService(trusted_git_key_fingerprints=["ABCD1234"])
    )

    def _fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout='Good "git" signature for abc123 with ED25519 key SHA256:deadbeef',
            stderr="",
        )

    monkeypatch.setattr(distribution_module, "_git_executable", lambda: "git")
    monkeypatch.setattr(distribution_module.subprocess, "run", _fake_run)

    result = service._verify_git_revision_sync(
        checkout_root=tmp_path / "checkout",
        repo_url="https://github.com/example/researcher-pack.git",
        ref=None,
        ref_kind="commit",
        commit="abc123",
        trusted_git_key_fingerprints=["ABCD1234"],
    )

    assert result["verified"] is False
    assert result["result_code"] == "unsupported_signature_backend"
    assert result["warning_code"] is None


@pytest.mark.asyncio
async def test_distribution_service_uses_canonicalized_repo_for_trust_matching(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    trust_service = McpHubGovernancePackTrustService(repo=repo)
    await _update_trust_policy(
        trust_service,
        {
            "allow_git_sources": True,
            "allowed_git_hosts": ["github.com"],
            "allowed_git_repositories": ["github.com/example/researcher-pack"],
            "allowed_git_ref_kinds": ["tag"],
        },
        actor_id=7,
    )

    checkout_root = tmp_path / "checkout"
    shutil.copytree(_fixture_pack_path(), checkout_root)
    service = McpHubGovernancePackDistributionService(trust_service=trust_service)

    async def _fake_checkout(*args: Any, **kwargs: Any) -> str:
        requested_checkout_root = Path(kwargs["checkout_root"])
        shutil.copytree(checkout_root, requested_checkout_root)
        return "abc123"

    monkeypatch.setattr(service, "_checkout_git_source", _fake_checkout)

    resolved = await service.resolve_git_source(
        "git@github.com:example/researcher-pack.git",
        ref="v1.0.0",
        ref_kind="tag",
        subpath=None,
    )

    assert resolved.source_location == "git@github.com:example/researcher-pack.git"
    assert resolved.source_ref_kind == "tag"
    assert resolved.source_commit_resolved == "abc123"


@pytest.mark.asyncio
async def test_distribution_service_rejects_git_subpath_escape(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    repo_url, _commit = _init_git_pack_repo(tmp_path)
    service = McpHubGovernancePackDistributionService(trust_service=_FakeGitTrustService())

    with pytest.raises(ValueError, match="subpath"):
        await service.resolve_git_source(
            repo_url,
            ref="v1.0.0",
            ref_kind="tag",
            subpath="../escape",
        )


@pytest.mark.asyncio
async def test_distribution_service_rejects_git_symlink_subpath_escape(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    repo_url, _commit = _init_git_pack_repo(tmp_path)
    repo_root = Path(repo_url.removeprefix("file://"))
    link_parent = repo_root / "packs"
    link_parent.mkdir(exist_ok=True)
    (link_parent / "escape").symlink_to("../../outside-pack")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "Add symlink escape"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    service = McpHubGovernancePackDistributionService(trust_service=_FakeGitTrustService())
    with pytest.raises(ValueError, match="repository root"):
        await service.resolve_git_source(
            repo_url,
            ref=None,
            ref_kind="branch",
            subpath="packs/escape",
        )


@pytest.mark.asyncio
async def test_distribution_service_rejects_git_urls_with_embedded_credentials() -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    service = McpHubGovernancePackDistributionService(trust_service=_FakeGitTrustService())

    with pytest.raises(ValueError, match="credentials"):
        await service.resolve_git_source(
            "https://token@example@example.com/researcher-pack.git",
            ref="main",
            ref_kind="branch",
            subpath=None,
        )


@pytest.mark.asyncio
async def test_distribution_service_requires_git_signature_verification_when_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    repo_url, _commit = _init_git_pack_repo(tmp_path)
    trust_service = _FakeGitTrustService(verification_required=True)
    service = McpHubGovernancePackDistributionService(trust_service=trust_service)

    async def _fake_verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "verified": False,
            "verification_mode": "git_signature",
            "verified_object_type": "tag",
            "signer_fingerprint": None,
            "signer_identity": None,
            "result_code": "signature_invalid",
            "warning_code": None,
        }

    monkeypatch.setattr(service, "_verify_git_revision", _fake_verify)

    with pytest.raises(ValueError, match="verification"):
        await service.resolve_git_source(
            repo_url,
            ref="v1.0.0",
            ref_kind="tag",
            subpath="packs/researcher",
        )


@pytest.mark.asyncio
async def test_distribution_service_passes_trusted_key_fingerprints_to_verifier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    repo_url, commit = _init_git_pack_repo(tmp_path)
    trust_service = _FakeGitTrustService(
        verification_required=True,
        trusted_git_key_fingerprints=["ABCD1234"],
    )
    service = McpHubGovernancePackDistributionService(trust_service=trust_service)

    async def _fake_verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["repo_url"] == repo_url
        assert kwargs["trusted_git_key_fingerprints"] == ["ABCD1234"]
        return {
            "verified": True,
            "verification_mode": "git_signature",
            "verified_object_type": "tag",
            "signer_fingerprint": "ABCD1234",
            "signer_identity": "Release Bot <bot@example.com>",
            "result_code": "verified_and_trusted",
            "warning_code": None,
        }

    monkeypatch.setattr(service, "_verify_git_revision", _fake_verify)

    resolved = await service.resolve_git_source(
        repo_url,
        ref="v1.0.0",
        ref_kind="tag",
        subpath="packs/researcher",
    )

    assert resolved.source_commit_resolved == commit


@pytest.mark.asyncio
async def test_distribution_service_derives_summary_fields_from_structured_verification_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    repo_url, commit = _init_git_pack_repo(tmp_path)
    trust_service = _FakeGitTrustService(
        verification_required=True,
        trusted_git_key_fingerprints=["ABCD1234"],
    )
    service = McpHubGovernancePackDistributionService(trust_service=trust_service)

    async def _fake_verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "verified": True,
            "verification_mode": "git_signature",
            "verified_object_type": "tag",
            "signer_fingerprint": "ABCD1234",
            "signer_identity": "Release Bot <bot@example.com>",
            "result_code": "verified_and_trusted",
            "warning_code": None,
        }

    monkeypatch.setattr(service, "_verify_git_revision", _fake_verify)

    resolved = await service.resolve_git_source(
        repo_url,
        ref="v1.0.0",
        ref_kind="tag",
        subpath="packs/researcher",
    )

    assert resolved.source_verified is True
    assert resolved.source_verification_mode == "git_signature"
    assert resolved.source_commit_resolved == commit


@pytest.mark.asyncio
async def test_distribution_service_revalidates_prepared_candidate_with_revoked_signer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    repo_url, _commit = _init_git_pack_repo(tmp_path)
    trust_service = _FakeGitTrustService(
        verification_required=True,
        trusted_git_key_fingerprints=["ABCD1234"],
    )
    service = McpHubGovernancePackDistributionService(
        trust_service=trust_service,
        repo=repo,
    )

    async def _verify_trusted(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "verified": True,
            "verification_mode": "git_signature",
            "verified_object_type": "commit",
            "signer_fingerprint": "ABCD1234",
            "signer_identity": "Release Bot <bot@example.com>",
            "result_code": "verified_and_trusted",
            "warning_code": None,
        }

    async def _verify_revoked(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "verified": False,
            "verification_mode": "git_signature",
            "verified_object_type": "commit",
            "signer_fingerprint": "ABCD1234",
            "signer_identity": "Release Bot <bot@example.com>",
            "result_code": "signer_revoked",
            "warning_code": None,
        }

    monkeypatch.setattr(service, "_verify_git_revision", _verify_trusted)
    prepared = await service.prepare_source_candidate(
        source={
            "source_type": "git",
            "repo_url": repo_url,
            "ref": "HEAD",
            "ref_kind": "commit",
            "subpath": "packs/researcher",
        },
        actor_id=7,
    )

    monkeypatch.setattr(service, "_verify_git_revision", _verify_revoked)
    with pytest.raises(ValueError, match="verification requirements"):
        await service.load_prepared_candidate(
            int(prepared["candidate"]["id"]),
            actor_id=7,
            revalidate_trust=True,
        )


@pytest.mark.asyncio
async def test_distribution_service_rejects_option_like_git_refs(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )

    repo_url, _commit = _init_git_pack_repo(tmp_path)
    service = McpHubGovernancePackDistributionService(trust_service=_FakeGitTrustService())

    with pytest.raises(ValueError, match="must not start"):
        await service.resolve_git_source(
            repo_url,
            ref="--help",
            ref_kind="tag",
            subpath="packs/researcher",
        )
