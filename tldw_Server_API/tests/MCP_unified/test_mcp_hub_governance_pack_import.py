from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import pytest

if TYPE_CHECKING:
    from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo


async def _make_repo(tmp_path, monkeypatch) -> McpHubRepo:
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
    pack_root.parent.mkdir(parents=True, exist_ok=True)
    import shutil

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
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo_root.as_uri(), commit


def _repo_path_from_file_uri(repo_url: str) -> Path:
    parsed = urlparse(repo_url)
    if parsed.scheme != "file":
        return Path(repo_url)

    path = unquote(parsed.path)
    if parsed.netloc:
        path = f"//{parsed.netloc}{path}"
    elif os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


def _git_head_branch(repo_url: str) -> str:
    repo_root = _repo_path_from_file_uri(repo_url)
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _update_git_pack_manifest(
    repo_url: str,
    *,
    pack_version: str | None = None,
    pack_id: str | None = None,
    description: str | None = None,
    subpath: str = "packs/researcher",
) -> str:
    repo_root = _repo_path_from_file_uri(repo_url)
    manifest_path = repo_root / subpath / "manifest.yaml"
    updated_lines: list[str] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if pack_version is not None and line.startswith("pack_version:"):
            updated_lines.append(f"pack_version: {pack_version}")
            continue
        if pack_id is not None and line.startswith("pack_id:"):
            updated_lines.append(f"pack_id: {pack_id}")
            continue
        if description is not None and line.startswith("description:"):
            updated_lines.append(f"description: {description}")
            continue
        updated_lines.append(line)
    manifest_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "Update governance pack fixture"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class _AllowGitTrustService:
    async def evaluate_git_source(self, repo_url: str, *, ref_kind: str) -> dict[str, object]:
        return {
            "allowed": True,
            "reason": None,
            "canonical_repository": repo_url,
            "verification_required": False,
            "trusted_git_key_fingerprints": [],
            "ref_kind": ref_kind,
        }


class _CanonicalizingGitTrustService:
    async def evaluate_git_source(self, repo_url: str, *, ref_kind: str) -> dict[str, object]:
        return {
            "allowed": True,
            "reason": None,
            "canonical_repository": "github.com/example/researcher-pack",
            "verification_required": False,
            "trusted_git_key_fingerprints": [],
            "ref_kind": ref_kind,
        }


def _pack_source_metadata(pack) -> dict[str, object]:
    return {
        "source_type": pack.source_type,
        "source_location": pack.source_location,
        "source_ref_requested": pack.source_ref_requested,
        "source_ref_kind": getattr(pack, "source_ref_kind", None),
        "source_subpath": pack.source_subpath,
        "source_commit_resolved": pack.source_commit_resolved,
        "pack_content_digest": pack.pack_content_digest,
        "source_verified": pack.source_verified,
        "source_verification_mode": pack.source_verification_mode,
        "signer_fingerprint": getattr(pack, "signer_fingerprint", None),
        "signer_identity": getattr(pack, "signer_identity", None),
        "verified_object_type": getattr(pack, "verified_object_type", None),
        "verification_result_code": getattr(pack, "verification_result_code", None),
        "verification_warning_code": getattr(pack, "verification_warning_code", None),
    }


async def _update_trust_policy(service: Any, payload: dict[str, Any], *, actor_id: int) -> dict[str, Any]:
    current_policy = await service.get_policy()
    request_payload = dict(payload)
    request_payload["policy_fingerprint"] = current_policy["policy_fingerprint"]
    return await service.update_policy(request_payload, actor_id=actor_id)


async def _seed_research_capability_mappings(
    repo: McpHubRepo,
    *,
    actor_id: int = 7,
) -> None:
    await repo.create_capability_adapter_mapping(
        mapping_id="filesystem.read.global",
        owner_scope_type="global",
        owner_scope_id=None,
        capability_name="filesystem.read",
        adapter_contract_version=1,
        resolved_policy_document={"allowed_tools": ["files.read"]},
        supported_environment_requirements=["workspace_bounded_read"],
        is_active=True,
        actor_id=actor_id,
    )
    await repo.create_capability_adapter_mapping(
        mapping_id="tool.invoke.research.global",
        owner_scope_type="global",
        owner_scope_id=None,
        capability_name="tool.invoke.research",
        adapter_contract_version=1,
        resolved_policy_document={"allowed_tools": ["web.search"]},
        supported_environment_requirements=[],
        is_active=True,
        actor_id=actor_id,
    )


@pytest.mark.asyncio
async def test_dry_run_governance_pack_uses_live_capability_mappings(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackService(repo=repo)
    pack = load_governance_pack_fixture("minimal_researcher_pack")

    report = await service.dry_run_pack(
        pack=pack,
        owner_scope_type="team",
        owner_scope_id=21,
    )

    assert report.verdict == "blocked"
    assert report.resolved_capabilities == []
    assert sorted(report.unresolved_capabilities) == [
        "filesystem.read",
        "tool.invoke.research",
    ]
    assert report.capability_mapping_summary == []

    await repo.create_capability_adapter_mapping(
        mapping_id="filesystem.read.global",
        owner_scope_type="global",
        owner_scope_id=None,
        capability_name="filesystem.read",
        adapter_contract_version=1,
        resolved_policy_document={"allowed_tools": ["files.read"]},
        supported_environment_requirements=["workspace_bounded_read"],
        is_active=True,
        actor_id=7,
    )
    await repo.create_capability_adapter_mapping(
        mapping_id="tool.invoke.research.team-21",
        owner_scope_type="team",
        owner_scope_id=21,
        capability_name="tool.invoke.research",
        adapter_contract_version=1,
        resolved_policy_document={"allowed_tools": ["web.search"]},
        supported_environment_requirements=[],
        is_active=True,
        actor_id=7,
    )

    report = await service.dry_run_pack(
        pack=pack,
        owner_scope_type="team",
        owner_scope_id=21,
    )

    assert report.verdict == "importable"
    assert sorted(report.resolved_capabilities) == [
        "filesystem.read",
        "tool.invoke.research",
    ]
    assert report.unresolved_capabilities == []
    assert sorted(summary["mapping_id"] for summary in report.capability_mapping_summary) == [
        "filesystem.read.global",
        "tool.invoke.research.team-21",
    ]


@pytest.mark.asyncio
async def test_dry_run_governance_pack_warns_when_profile_requirement_not_guaranteed(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackService(repo=repo)
    pack = load_governance_pack_fixture("minimal_researcher_pack")
    pack.profiles[0].environment_requirements = ["workspace_bounded_write"]

    await repo.create_capability_adapter_mapping(
        mapping_id="filesystem.read.global",
        owner_scope_type="global",
        owner_scope_id=None,
        capability_name="filesystem.read",
        adapter_contract_version=1,
        resolved_policy_document={"allowed_tools": ["files.read"]},
        supported_environment_requirements=[],
        is_active=True,
        actor_id=7,
    )
    await repo.create_capability_adapter_mapping(
        mapping_id="tool.invoke.research.global",
        owner_scope_type="global",
        owner_scope_id=None,
        capability_name="tool.invoke.research",
        adapter_contract_version=1,
        resolved_policy_document={"allowed_tools": ["web.search"]},
        supported_environment_requirements=[],
        is_active=True,
        actor_id=7,
    )

    report = await service.dry_run_pack(
        pack=pack,
        owner_scope_type="global",
        owner_scope_id=None,
    )

    assert report.verdict == "importable"
    assert report.unresolved_capabilities == []
    assert (
        "profile:researcher.profile requires environment requirement 'workspace_bounded_write' "
        "but current capability mappings do not guarantee it"
    ) in report.warnings


@pytest.mark.asyncio
async def test_import_governance_pack_materializes_immutable_base_objects_with_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)

    pack = load_governance_pack_fixture("minimal_researcher_pack")
    service = McpHubGovernancePackService(repo=repo)

    result = await service.import_pack(
        pack=pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
    )

    assert result.blocked_objects == []
    assert result.imported_object_counts["approval_policies"] == 1
    assert result.imported_object_counts["permission_profiles"] == 1
    assert result.imported_object_counts["policy_assignments"] == 1

    governance_pack = await repo.get_governance_pack(result.governance_pack_id)
    assert governance_pack is not None
    assert governance_pack["pack_id"] == "researcher-pack"
    assert governance_pack["pack_version"] == "1.0.0"
    assert governance_pack["is_active_install"] is True
    assert len(str(governance_pack["bundle_digest"])) == 64

    approval_policy = await repo.get_approval_policy(result.imported_object_ids["approval_policies"][0])
    assert approval_policy is not None
    assert approval_policy["is_immutable"] is True
    assert approval_policy["mode"] == "ask_every_time"

    permission_profile = await repo.get_permission_profile(
        result.imported_object_ids["permission_profiles"][0]
    )
    assert permission_profile is not None
    assert permission_profile["is_immutable"] is True
    assert permission_profile["mode"] == "preset"
    assert permission_profile["policy_document"]["capabilities"] == [
        "filesystem.read",
        "tool.invoke.research",
    ]
    assert permission_profile["policy_document"]["environment_requirements"] == [
        "workspace_bounded_read",
    ]

    profile_link = await repo.get_governance_pack_object(
        object_type="permission_profile",
        object_id=permission_profile["id"],
    )
    assert profile_link is not None
    assert profile_link["source_object_id"] == "researcher.profile"

    assignment = await repo.get_policy_assignment(result.imported_object_ids["policy_assignments"][0])
    assert assignment is not None
    assert assignment["is_immutable"] is True
    assert assignment["target_type"] == "default"
    assert int(assignment["profile_id"]) == int(permission_profile["id"])
    assert int(assignment["approval_policy_id"]) == int(approval_policy["id"])

    override = await repo.upsert_policy_override(
        int(assignment["id"]),
        override_policy_document={"allowed_tools": ["Read"]},
        broadens_access=False,
        grant_authority_snapshot={"source": "local-overlay"},
        actor_id=8,
        is_active=True,
    )
    assert override is not None
    assert override["override_policy_document"]["allowed_tools"] == ["Read"]


@pytest.mark.asyncio
async def test_governance_pack_repo_tracks_active_install_state_and_upgrade_lineage(
    tmp_path,
    monkeypatch,
) -> None:
    repo = await _make_repo(tmp_path, monkeypatch)

    created = await repo.create_governance_pack(
        pack_id="researcher-pack",
        pack_version="1.0.0",
        pack_schema_version=1,
        capability_taxonomy_version=1,
        adapter_contract_version=1,
        title="Researcher Pack",
        description="Initial install",
        owner_scope_type="user",
        owner_scope_id=7,
        bundle_digest="a" * 64,
        manifest={"pack_id": "researcher-pack", "pack_version": "1.0.0"},
        normalized_ir={"profiles": []},
        actor_id=7,
    )

    assert created["is_active_install"] is True
    assert created["superseded_by_governance_pack_id"] is None

    upgraded = await repo.create_governance_pack(
        pack_id="researcher-pack",
        pack_version="1.1.0",
        pack_schema_version=1,
        capability_taxonomy_version=1,
        adapter_contract_version=1,
        title="Researcher Pack",
        description="Upgrade target",
        owner_scope_type="user",
        owner_scope_id=7,
        bundle_digest="b" * 64,
        manifest={"pack_id": "researcher-pack", "pack_version": "1.1.0"},
        normalized_ir={"profiles": []},
        actor_id=7,
        is_active_install=False,
    )

    lineage = await repo.create_governance_pack_upgrade(
        pack_id="researcher-pack",
        owner_scope_type="user",
        owner_scope_id=7,
        from_governance_pack_id=int(created["id"]),
        to_governance_pack_id=int(upgraded["id"]),
        from_pack_version="1.0.0",
        to_pack_version="1.1.0",
        status="planned",
        planned_by=7,
        plan_summary={"changed_objects": 2},
    )

    assert lineage["from_pack_version"] == "1.0.0"
    assert lineage["to_pack_version"] == "1.1.0"
    assert lineage["status"] == "planned"

    history = await repo.list_governance_pack_upgrades(
        pack_id="researcher-pack",
        owner_scope_type="user",
        owner_scope_id=7,
    )

    assert [item["to_pack_version"] for item in history] == ["1.1.0"]


@pytest.mark.asyncio
async def test_governance_pack_repo_tracks_source_provenance_and_candidate_storage(
    tmp_path,
    monkeypatch,
) -> None:
    repo = await _make_repo(tmp_path, monkeypatch)

    created = await repo.create_governance_pack(
        pack_id="researcher-pack",
        pack_version="1.0.0",
        pack_schema_version=1,
        capability_taxonomy_version=1,
        adapter_contract_version=1,
        title="Researcher Pack",
        description="Initial install",
        owner_scope_type="user",
        owner_scope_id=7,
        bundle_digest="a" * 64,
        manifest={"pack_id": "researcher-pack", "pack_version": "1.0.0"},
        normalized_ir={"profiles": []},
        actor_id=7,
        source_type="git",
        source_location="https://github.com/example/researcher-pack.git",
        source_ref_requested="main",
        source_subpath="packs/researcher",
        source_commit_resolved="abc123",
        pack_content_digest="c" * 64,
        source_verified=True,
        source_verification_mode="git-commit",
        signer_fingerprint="ABCD1234",
        signer_identity="Release Bot <bot@example.com>",
        verified_object_type="commit",
        verification_result_code="verified_and_trusted",
        verification_warning_code=None,
        source_fetched_at=datetime.now(timezone.utc),
        fetched_by=7,
    )

    assert created["source_type"] == "git"
    assert created["source_location"] == "https://github.com/example/researcher-pack.git"
    assert created["source_ref_requested"] == "main"
    assert created["source_subpath"] == "packs/researcher"
    assert created["source_commit_resolved"] == "abc123"
    assert created["pack_content_digest"] == "c" * 64
    assert created["source_verified"] is True
    assert created["source_verification_mode"] == "git-commit"
    assert created["signer_fingerprint"] == "ABCD1234"
    assert created["signer_identity"] == "Release Bot <bot@example.com>"
    assert created["verified_object_type"] == "commit"
    assert created["verification_result_code"] == "verified_and_trusted"
    assert created["verification_warning_code"] is None
    assert created["fetched_by"] == 7

    listed = await repo.list_governance_packs(owner_scope_type="user", owner_scope_id=7)
    assert listed[0]["source_type"] == "git"
    assert listed[0]["source_location"] == "https://github.com/example/researcher-pack.git"
    assert listed[0]["pack_content_digest"] == "c" * 64

    candidate = await repo.create_governance_pack_source_candidate(
        source_type="git",
        source_location="https://github.com/example/researcher-pack.git",
        source_ref_requested="main",
        source_subpath="packs/researcher",
        source_commit_resolved="abc123",
        pack_content_digest="c" * 64,
        source_verified=True,
        source_verification_mode="git-commit",
        signer_fingerprint="ABCD1234",
        signer_identity="Release Bot <bot@example.com>",
        verified_object_type="commit",
        verification_result_code="verified_and_trusted",
        verification_warning_code=None,
        fetched_by=7,
    )
    assert candidate["source_type"] == "git"
    assert candidate["pack_content_digest"] == "c" * 64
    assert candidate["signer_fingerprint"] == "ABCD1234"
    assert candidate["verification_result_code"] == "verified_and_trusted"

    candidate_by_id = await repo.get_governance_pack_source_candidate(int(candidate["id"]))
    assert candidate_by_id is not None
    assert candidate_by_id["source_commit_resolved"] == "abc123"
    assert candidate_by_id["source_subpath"] == "packs/researcher"
    assert candidate_by_id["signer_identity"] == "Release Bot <bot@example.com>"

    candidate_list = await repo.list_governance_pack_source_candidates()
    assert candidate_list[0]["source_location"] == "https://github.com/example/researcher-pack.git"
    assert candidate_list[0]["verified_object_type"] == "commit"

    superseding = await repo.create_governance_pack(
        pack_id="researcher-pack",
        pack_version="1.1.0",
        pack_schema_version=1,
        capability_taxonomy_version=1,
        adapter_contract_version=1,
        title="Researcher Pack",
        description="Upgrade target",
        owner_scope_type="user",
        owner_scope_id=7,
        bundle_digest="b" * 64,
        manifest={"pack_id": "researcher-pack", "pack_version": "1.1.0"},
        normalized_ir={"profiles": []},
        actor_id=7,
        source_type="git",
        source_location="https://github.com/example/researcher-pack.git",
        source_ref_requested="main",
        source_subpath="packs/researcher",
        source_commit_resolved="def456",
        pack_content_digest="d" * 64,
        source_verified=False,
        source_verification_mode="git-commit",
        source_fetched_at=datetime.now(timezone.utc),
        fetched_by=7,
        is_active_install=False,
    )

    updated = await repo.update_governance_pack_install_state(
        int(created["id"]),
        is_active_install=False,
        superseded_by_governance_pack_id=int(superseding["id"]),
        actor_id=7,
    )
    assert updated is not None
    assert updated["is_active_install"] is False
    assert updated["superseded_by_governance_pack_id"] == int(superseding["id"])
    assert updated["source_type"] == "git"
    assert updated["source_location"] == "https://github.com/example/researcher-pack.git"
    assert updated["source_commit_resolved"] == "abc123"
    assert updated["pack_content_digest"] == "c" * 64


@pytest.mark.asyncio
async def test_governance_pack_source_candidate_dry_run_and_import_persists_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    import shutil

    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    pack_root = tmp_path / "allowed" / "researcher-pack"
    pack_root.parent.mkdir(parents=True)
    shutil.copytree(_fixture_pack_path(), pack_root)

    trust_service = McpHubGovernancePackTrustService(repo=repo)
    await _update_trust_policy(
        trust_service,
        {
            "allow_local_path_sources": True,
            "allowed_local_roots": [str(pack_root.parent)],
        },
        actor_id=7,
    )

    distribution_service = McpHubGovernancePackDistributionService(
        trust_service=trust_service,
        repo=repo,
    )
    governance_service = McpHubGovernancePackService(repo=repo)
    await _seed_research_capability_mappings(repo)

    prepared = await distribution_service.prepare_source_candidate(
        source={
            "source_type": "local_path",
            "local_path": str(pack_root),
        },
        actor_id=7,
    )
    stored_candidate = await repo.get_governance_pack_source_candidate(int(prepared["candidate"]["id"]))
    loaded = await distribution_service.load_prepared_candidate(int(prepared["candidate"]["id"]))

    report = await governance_service.dry_run_pack_document(
        document=loaded["pack_document"],
        owner_scope_type="user",
        owner_scope_id=7,
    )
    imported = await governance_service.import_pack_document(
        document=loaded["pack_document"],
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
        source_metadata=loaded["candidate"],
    )

    installed = await repo.get_governance_pack(int(imported["governance_pack_id"]))
    assert report.verdict == "importable"
    assert stored_candidate is not None
    assert stored_candidate["pack_document"]["manifest"]["pack_id"] == "researcher-pack"
    assert installed is not None
    assert installed["source_type"] == "local_path"
    assert installed["source_location"] == str(pack_root.resolve())
    assert installed["pack_content_digest"] == prepared["candidate"]["pack_content_digest"]


@pytest.mark.asyncio
async def test_git_source_candidate_and_import_persist_signer_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    class _VerifiedGitTrustService:
        async def evaluate_git_source(self, repo_url: str, *, ref_kind: str) -> dict[str, object]:
            return {
                "allowed": True,
                "reason": None,
                "canonical_repository": repo_url,
                "verification_required": True,
                "trusted_git_key_fingerprints": ["ABCD1234"],
                "ref_kind": ref_kind,
            }

        async def evaluate_signer_for_repository(self, signer_fingerprint: str, repo_url: str) -> dict[str, object]:
            return {
                "allowed": True,
                "reason": None,
                "result_code": "signer_trusted_for_repo",
                "canonical_repository": repo_url,
                "signer_fingerprint": signer_fingerprint,
            }

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    governance_service = McpHubGovernancePackService(repo=repo)
    distribution_service = McpHubGovernancePackDistributionService(
        trust_service=_VerifiedGitTrustService(),
        repo=repo,
    )

    repo_url, commit = _init_git_pack_repo(tmp_path)
    branch_name = _git_head_branch(repo_url)

    async def _fake_verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["repo_url"] == repo_url
        return {
            "verified": True,
            "verification_mode": "git_signature",
            "verified_object_type": "commit",
            "signer_fingerprint": "ABCD1234",
            "signer_identity": "Release Bot <bot@example.com>",
            "result_code": "verified_and_trusted",
            "warning_code": None,
        }

    monkeypatch.setattr(distribution_service, "_verify_git_revision", _fake_verify)

    prepared = await distribution_service.prepare_source_candidate(
        source={
            "source_type": "git",
            "repo_url": repo_url,
            "ref": branch_name,
            "ref_kind": "branch",
            "subpath": "packs/researcher",
        },
        actor_id=7,
    )
    stored_candidate = await repo.get_governance_pack_source_candidate(int(prepared["candidate"]["id"]))
    loaded = await distribution_service.load_prepared_candidate(int(prepared["candidate"]["id"]))

    imported = await governance_service.import_pack_document(
        document=loaded["pack_document"],
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
        source_metadata=loaded["candidate"],
    )

    installed = await repo.get_governance_pack(int(imported["governance_pack_id"]))
    assert stored_candidate is not None
    assert stored_candidate["source_commit_resolved"] == commit
    assert stored_candidate["source_verified"] is True
    assert stored_candidate["source_verification_mode"] == "git_signature"
    assert stored_candidate["signer_fingerprint"] == "ABCD1234"
    assert stored_candidate["signer_identity"] == "Release Bot <bot@example.com>"
    assert stored_candidate["verified_object_type"] == "commit"
    assert stored_candidate["verification_result_code"] == "verified_and_trusted"
    assert stored_candidate["verification_warning_code"] is None

    assert installed is not None
    assert installed["source_type"] == "git"
    assert installed["source_location"] == repo_url
    assert installed["source_commit_resolved"] == commit
    assert installed["source_verified"] is True
    assert installed["source_verification_mode"] == "git_signature"
    assert installed["signer_fingerprint"] == "ABCD1234"
    assert installed["signer_identity"] == "Release Bot <bot@example.com>"
    assert installed["verified_object_type"] == "commit"
    assert installed["verification_result_code"] == "verified_and_trusted"
    assert installed["verification_warning_code"] is None


@pytest.mark.asyncio
async def test_prepared_source_candidate_revalidates_actor_and_trust_policy(
    tmp_path,
    monkeypatch,
) -> None:
    import shutil

    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
        McpHubGovernancePackTrustService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    pack_root = tmp_path / "allowed" / "researcher-pack"
    pack_root.parent.mkdir(parents=True)
    shutil.copytree(_fixture_pack_path(), pack_root)

    trust_service = McpHubGovernancePackTrustService(repo=repo)
    await _update_trust_policy(
        trust_service,
        {
            "allow_local_path_sources": True,
            "allowed_local_roots": [str(pack_root.parent)],
        },
        actor_id=7,
    )
    distribution_service = McpHubGovernancePackDistributionService(
        trust_service=trust_service,
        repo=repo,
    )
    prepared = await distribution_service.prepare_source_candidate(
        source={
            "source_type": "local_path",
            "local_path": str(pack_root),
        },
        actor_id=7,
    )

    with pytest.raises(ValueError, match="different actor"):
        await distribution_service.load_prepared_candidate(
            int(prepared["candidate"]["id"]),
            actor_id=8,
            revalidate_trust=True,
        )

    await _update_trust_policy(
        trust_service,
        {
            "allow_local_path_sources": False,
            "allowed_local_roots": [],
        },
        actor_id=7,
    )

    with pytest.raises(ValueError, match="trust policy"):
        await distribution_service.load_prepared_candidate(
            int(prepared["candidate"]["id"]),
            actor_id=7,
            revalidate_trust=True,
        )


@pytest.mark.asyncio
async def test_git_governance_pack_update_check_detects_newer_version_and_prepares_upgrade_candidate(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    governance_service = McpHubGovernancePackService(repo=repo)
    distribution_service = McpHubGovernancePackDistributionService(
        trust_service=_CanonicalizingGitTrustService(),
        repo=repo,
    )

    repo_url, _initial_commit = _init_git_pack_repo(tmp_path)
    branch_name = _git_head_branch(repo_url)
    initial_pack = await distribution_service.resolve_git_source(
        repo_url,
        ref=branch_name,
        ref_kind="branch",
        subpath="packs/researcher",
    )
    imported = await governance_service.import_pack(
        pack=initial_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
        source_metadata=_pack_source_metadata(initial_pack),
    )
    installed = await repo.get_governance_pack(imported.governance_pack_id)
    assert installed is not None
    assert installed["source_location"] == repo_url

    _update_git_pack_manifest(repo_url, pack_version="1.1.0")

    check = await distribution_service.check_for_updates(imported.governance_pack_id)
    assert check["status"] == "newer_version_available"
    assert check["candidate_manifest"]["pack_version"] == "1.1.0"
    assert installed["source_ref_kind"] == "branch"

    prepared = await distribution_service.prepare_upgrade_candidate(
        governance_pack_id=imported.governance_pack_id,
        actor_id=7,
    )
    loaded = await distribution_service.load_prepared_candidate(int(prepared["candidate"]["id"]))
    plan = await governance_service.dry_run_upgrade_document(
        source_governance_pack_id=imported.governance_pack_id,
        document=loaded["pack_document"],
        owner_scope_type="user",
        owner_scope_id=7,
    )

    assert prepared["status"] == "newer_version_available"
    assert plan.upgradeable is True


@pytest.mark.asyncio
async def test_git_governance_pack_update_check_preserves_tag_ref_kind(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    governance_service = McpHubGovernancePackService(repo=repo)
    distribution_service = McpHubGovernancePackDistributionService(
        trust_service=_AllowGitTrustService(),
        repo=repo,
    )

    repo_url, _initial_commit = _init_git_pack_repo(tmp_path)
    repo_root = _repo_path_from_file_uri(repo_url)
    subprocess.run(["git", "tag", "release"], cwd=repo_root, check=True, capture_output=True, text=True)

    initial_pack = await distribution_service.resolve_git_source(
        repo_url,
        ref="release",
        ref_kind="tag",
        subpath="packs/researcher",
    )
    imported = await governance_service.import_pack(
        pack=initial_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
        source_metadata=_pack_source_metadata(initial_pack),
    )

    subprocess.run(["git", "branch", "release"], cwd=repo_root, check=True, capture_output=True, text=True)
    updated_commit = _update_git_pack_manifest(repo_url, pack_version="1.1.0")
    subprocess.run(["git", "tag", "-f", "release", updated_commit], cwd=repo_root, check=True, capture_output=True, text=True)

    check = await distribution_service.check_for_updates(imported.governance_pack_id)

    assert check["status"] == "newer_version_available"
    assert check["candidate_manifest"]["pack_version"] == "1.1.0"


@pytest.mark.asyncio
async def test_git_governance_pack_update_check_reports_same_version_source_drift(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    governance_service = McpHubGovernancePackService(repo=repo)
    distribution_service = McpHubGovernancePackDistributionService(
        trust_service=_AllowGitTrustService(),
        repo=repo,
    )

    repo_url, _initial_commit = _init_git_pack_repo(tmp_path)
    branch_name = _git_head_branch(repo_url)
    initial_pack = await distribution_service.resolve_git_source(
        repo_url,
        ref=branch_name,
        ref_kind="branch",
        subpath="packs/researcher",
    )
    imported = await governance_service.import_pack(
        pack=initial_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
        source_metadata=_pack_source_metadata(initial_pack),
    )

    _update_git_pack_manifest(repo_url, description="Same version drifted pack content")

    check = await distribution_service.check_for_updates(imported.governance_pack_id)
    assert check["status"] == "source_drift_same_version"
    assert check["candidate_manifest"]["pack_version"] == "1.0.0"


@pytest.mark.asyncio
async def test_git_governance_pack_update_checks_reject_mismatched_pack_id_and_stale_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    governance_service = McpHubGovernancePackService(repo=repo)
    distribution_service = McpHubGovernancePackDistributionService(
        trust_service=_AllowGitTrustService(),
        repo=repo,
    )

    repo_url, _initial_commit = _init_git_pack_repo(tmp_path)
    branch_name = _git_head_branch(repo_url)
    initial_pack = await distribution_service.resolve_git_source(
        repo_url,
        ref=branch_name,
        ref_kind="branch",
        subpath="packs/researcher",
    )
    imported = await governance_service.import_pack(
        pack=initial_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
        source_metadata=_pack_source_metadata(initial_pack),
    )

    _update_git_pack_manifest(repo_url, pack_version="1.1.0")
    prepared = await distribution_service.prepare_upgrade_candidate(
        governance_pack_id=imported.governance_pack_id,
        actor_id=7,
    )

    _update_git_pack_manifest(repo_url, pack_version="1.2.0")

    with pytest.raises(ValueError, match="stale"):
        await distribution_service.validate_prepared_upgrade_candidate(
            governance_pack_id=imported.governance_pack_id,
            candidate_id=int(prepared["candidate"]["id"]),
        )

    _update_git_pack_manifest(repo_url, pack_id="other-pack")

    with pytest.raises(ValueError, match="pack_id"):
        await distribution_service.check_for_updates(imported.governance_pack_id)


@pytest.mark.asyncio
async def test_git_governance_pack_update_check_warns_on_trusted_signer_rotation(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    class _VerifiedRotationTrustService:
        async def evaluate_git_source(self, repo_url: str, *, ref_kind: str) -> dict[str, object]:
            return {
                "allowed": True,
                "reason": None,
                "canonical_repository": repo_url,
                "verification_required": True,
                "trusted_git_key_fingerprints": ["AAAA1111", "BBBB2222"],
                "ref_kind": ref_kind,
            }

        async def evaluate_signer_for_repository(self, signer_fingerprint: str, repo_url: str) -> dict[str, object]:
            return {
                "allowed": True,
                "reason": None,
                "result_code": "signer_trusted_for_repo",
                "canonical_repository": repo_url,
                "signer_fingerprint": signer_fingerprint,
            }

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    governance_service = McpHubGovernancePackService(repo=repo)
    distribution_service = McpHubGovernancePackDistributionService(
        trust_service=_VerifiedRotationTrustService(),
        repo=repo,
    )

    repo_url, _initial_commit = _init_git_pack_repo(tmp_path)
    branch_name = _git_head_branch(repo_url)
    verify_calls = 0

    async def _fake_verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 1:
            fingerprint = "AAAA1111"
            identity = "Primary Bot <primary@example.com>"
        else:
            fingerprint = "BBBB2222"
            identity = "Backup Bot <backup@example.com>"
        return {
            "verified": True,
            "verification_mode": "git_signature",
            "verified_object_type": "commit",
            "signer_fingerprint": fingerprint,
            "signer_identity": identity,
            "result_code": "verified_and_trusted",
            "warning_code": None,
        }

    monkeypatch.setattr(distribution_service, "_verify_git_revision", _fake_verify)

    initial_pack = await distribution_service.resolve_git_source(
        repo_url,
        ref=branch_name,
        ref_kind="branch",
        subpath="packs/researcher",
    )
    imported = await governance_service.import_pack(
        pack=initial_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
        source_metadata=_pack_source_metadata(initial_pack),
    )

    _update_git_pack_manifest(repo_url, pack_version="1.1.0")

    check = await distribution_service.check_for_updates(imported.governance_pack_id)
    prepared = await distribution_service.prepare_upgrade_candidate(
        governance_pack_id=imported.governance_pack_id,
        actor_id=7,
    )

    assert check["verification_warning_code"] == "signer_rotated_trusted"
    assert check["signer_fingerprint"] == "BBBB2222"
    assert prepared["candidate"]["verification_warning_code"] == "signer_rotated_trusted"
    assert prepared["candidate"]["signer_fingerprint"] == "BBBB2222"


@pytest.mark.asyncio
async def test_git_governance_pack_update_check_warns_when_previous_signer_unknown(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    class _VerifiedTrustService:
        async def evaluate_git_source(self, repo_url: str, *, ref_kind: str) -> dict[str, object]:
            return {
                "allowed": True,
                "reason": None,
                "canonical_repository": repo_url,
                "verification_required": True,
                "trusted_git_key_fingerprints": ["ABCD1234"],
                "ref_kind": ref_kind,
            }

        async def evaluate_signer_for_repository(self, signer_fingerprint: str, repo_url: str) -> dict[str, object]:
            return {
                "allowed": True,
                "reason": None,
                "result_code": "signer_trusted_for_repo",
                "canonical_repository": repo_url,
                "signer_fingerprint": signer_fingerprint,
            }

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    governance_service = McpHubGovernancePackService(repo=repo)
    distribution_service = McpHubGovernancePackDistributionService(
        trust_service=_VerifiedTrustService(),
        repo=repo,
    )

    repo_url, _initial_commit = _init_git_pack_repo(tmp_path)
    branch_name = _git_head_branch(repo_url)

    async def _fake_verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "verified": True,
            "verification_mode": "git_signature",
            "verified_object_type": "commit",
            "signer_fingerprint": "ABCD1234",
            "signer_identity": "Release Bot <bot@example.com>",
            "result_code": "verified_and_trusted",
            "warning_code": None,
        }

    monkeypatch.setattr(distribution_service, "_verify_git_revision", _fake_verify)

    initial_pack = await distribution_service.resolve_git_source(
        repo_url,
        ref=branch_name,
        ref_kind="branch",
        subpath="packs/researcher",
    )
    source_metadata = _pack_source_metadata(initial_pack)
    source_metadata["signer_fingerprint"] = None
    source_metadata["signer_identity"] = None
    source_metadata["verified_object_type"] = None
    source_metadata["verification_result_code"] = None
    source_metadata["verification_warning_code"] = None
    imported = await governance_service.import_pack(
        pack=initial_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
        source_metadata=source_metadata,
    )

    _update_git_pack_manifest(repo_url, pack_version="1.1.0")

    check = await distribution_service.check_for_updates(imported.governance_pack_id)
    assert check["verification_warning_code"] == "unknown_previous_signer"
    assert check["signer_fingerprint"] == "ABCD1234"


@pytest.mark.asyncio
async def test_git_governance_pack_update_checks_block_revoked_signer_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
        McpHubGovernancePackDistributionService,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    class _VerifiedTrustService:
        async def evaluate_git_source(self, repo_url: str, *, ref_kind: str) -> dict[str, object]:
            return {
                "allowed": True,
                "reason": None,
                "canonical_repository": repo_url,
                "verification_required": True,
                "trusted_git_key_fingerprints": ["ABCD1234"],
                "ref_kind": ref_kind,
            }

        async def evaluate_signer_for_repository(self, signer_fingerprint: str, repo_url: str) -> dict[str, object]:
            return {
                "allowed": True,
                "reason": None,
                "result_code": "signer_trusted_for_repo",
                "canonical_repository": repo_url,
                "signer_fingerprint": signer_fingerprint,
            }

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    governance_service = McpHubGovernancePackService(repo=repo)
    distribution_service = McpHubGovernancePackDistributionService(
        trust_service=_VerifiedTrustService(),
        repo=repo,
    )

    repo_url, _initial_commit = _init_git_pack_repo(tmp_path)
    branch_name = _git_head_branch(repo_url)
    verify_calls = 0

    async def _fake_verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 1:
            return {
                "verified": True,
                "verification_mode": "git_signature",
                "verified_object_type": "commit",
                "signer_fingerprint": "ABCD1234",
                "signer_identity": "Release Bot <bot@example.com>",
                "result_code": "verified_and_trusted",
                "warning_code": None,
            }
        return {
            "verified": False,
            "verification_mode": "git_signature",
            "verified_object_type": "commit",
            "signer_fingerprint": "ABCD1234",
            "signer_identity": "Release Bot <bot@example.com>",
            "result_code": "signer_revoked",
            "warning_code": None,
        }

    monkeypatch.setattr(distribution_service, "_verify_git_revision", _fake_verify)

    initial_pack = await distribution_service.resolve_git_source(
        repo_url,
        ref=branch_name,
        ref_kind="branch",
        subpath="packs/researcher",
    )
    imported = await governance_service.import_pack(
        pack=initial_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
        source_metadata=_pack_source_metadata(initial_pack),
    )

    _update_git_pack_manifest(repo_url, pack_version="1.1.0")

    with pytest.raises(ValueError, match="signer_revoked"):
        await distribution_service.check_for_updates(imported.governance_pack_id)

    with pytest.raises(ValueError, match="signer_revoked"):
        await distribution_service.prepare_upgrade_candidate(
            governance_pack_id=imported.governance_pack_id,
            actor_id=7,
        )


@pytest.mark.asyncio
async def test_import_governance_pack_rejects_duplicate_scope_identity(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        GovernancePackAlreadyExistsError,
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)

    pack = load_governance_pack_fixture("minimal_researcher_pack")
    service = McpHubGovernancePackService(repo=repo)

    await service.import_pack(
        pack=pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
    )

    with pytest.raises(GovernancePackAlreadyExistsError):
        await service.import_pack(
            pack=pack,
            owner_scope_type="user",
            owner_scope_id=7,
            actor_id=7,
        )


@pytest.mark.asyncio
async def test_imported_governance_pack_denied_capabilities_narrow_runtime_policy(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    repo = await _make_repo(tmp_path, monkeypatch)
    service = McpHubGovernancePackService(repo=repo)
    pack = load_governance_pack_fixture("minimal_researcher_pack")
    pack.profiles[0].capabilities.deny = ["tool.invoke.docs"]

    await repo.create_capability_adapter_mapping(
        mapping_id="filesystem.read.global",
        owner_scope_type="global",
        owner_scope_id=None,
        capability_name="filesystem.read",
        adapter_contract_version=1,
        resolved_policy_document={"allowed_tools": ["files.read"]},
        supported_environment_requirements=["workspace_bounded_read"],
        is_active=True,
        actor_id=7,
    )
    await repo.create_capability_adapter_mapping(
        mapping_id="tool.invoke.research.global",
        owner_scope_type="global",
        owner_scope_id=None,
        capability_name="tool.invoke.research",
        adapter_contract_version=1,
        resolved_policy_document={"allowed_tools": ["web.search"]},
        supported_environment_requirements=[],
        is_active=True,
        actor_id=7,
    )
    await repo.create_capability_adapter_mapping(
        mapping_id="tool.invoke.docs.global",
        owner_scope_type="global",
        owner_scope_id=None,
        capability_name="tool.invoke.docs",
        adapter_contract_version=1,
        resolved_policy_document={"allowed_tools": ["docs.search"]},
        supported_environment_requirements=[],
        is_active=True,
        actor_id=7,
    )

    await service.import_pack(
        pack=pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
    )

    resolver = McpHubPolicyResolver(repo=repo)
    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={"mcp_policy_context_enabled": True},
    )

    assert sorted(policy["allowed_tools"]) == ["files.read", "web.search"]
    assert policy["denied_tools"] == ["docs.search"]
    assert any(
        summary["capability_name"] == "tool.invoke.docs" and summary["resolution_intent"] == "deny"
        for summary in policy["capability_mapping_summary"]
    )

    inventory = await repo.list_governance_packs(owner_scope_type="user", owner_scope_id=7)
    assert len(inventory) == 1


@pytest.mark.asyncio
async def test_import_governance_pack_rolls_back_partial_objects_on_failure(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)

    original_create_policy_assignment = repo.create_policy_assignment

    async def _boom_create_policy_assignment(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("assignment insert failed")

    repo.create_policy_assignment = _boom_create_policy_assignment  # type: ignore[method-assign]

    pack = load_governance_pack_fixture("minimal_researcher_pack")
    service = McpHubGovernancePackService(repo=repo)

    with pytest.raises(RuntimeError, match="assignment insert failed"):
        await service.import_pack(
            pack=pack,
            owner_scope_type="user",
            owner_scope_id=7,
            actor_id=7,
        )

    repo.create_policy_assignment = original_create_policy_assignment  # type: ignore[method-assign]

    assert await repo.list_governance_packs(owner_scope_type="user", owner_scope_id=7) == []
    assert await repo.list_permission_profiles(owner_scope_type="user", owner_scope_id=7) == []
    assert await repo.list_approval_policies(owner_scope_type="user", owner_scope_id=7) == []
    assert await repo.list_policy_assignments(owner_scope_type="user", owner_scope_id=7) == []


@pytest.mark.asyncio
async def test_dry_run_upgrade_accepts_newer_same_pack_and_reports_fingerprints(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    service = McpHubGovernancePackService(repo=repo)
    source_pack = load_governance_pack_fixture("minimal_researcher_pack")
    imported = await service.import_pack(
        pack=source_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
    )

    target_pack = deepcopy(source_pack)
    target_pack.manifest.pack_version = "1.0.1"
    target_pack.manifest.description = "Minor pack refresh"

    plan = await service.dry_run_upgrade_pack(
        source_governance_pack_id=imported.governance_pack_id,
        pack=target_pack,
        owner_scope_type="user",
        owner_scope_id=7,
    )

    assert plan.upgradeable is True
    assert plan.source_manifest["pack_version"] == "1.0.0"
    assert plan.target_manifest["pack_version"] == "1.0.1"
    assert plan.structural_conflicts == []
    assert plan.behavioral_conflicts == []
    assert len(plan.planner_inputs_fingerprint) == 64
    assert len(plan.adapter_state_fingerprint) == 64


@pytest.mark.asyncio
async def test_dry_run_upgrade_rejects_equal_version_and_cross_scope_target(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    service = McpHubGovernancePackService(repo=repo)
    source_pack = load_governance_pack_fixture("minimal_researcher_pack")
    imported = await service.import_pack(
        pack=source_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
    )

    same_version_plan = await service.dry_run_upgrade_pack(
        source_governance_pack_id=imported.governance_pack_id,
        pack=deepcopy(source_pack),
        owner_scope_type="user",
        owner_scope_id=7,
    )
    assert same_version_plan.upgradeable is False
    assert any("newer than the installed version" in item for item in same_version_plan.structural_conflicts)

    moved_scope_pack = deepcopy(source_pack)
    moved_scope_pack.manifest.pack_version = "1.0.1"
    moved_scope_plan = await service.dry_run_upgrade_pack(
        source_governance_pack_id=imported.governance_pack_id,
        pack=moved_scope_pack,
        owner_scope_type="team",
        owner_scope_id=21,
    )
    assert moved_scope_plan.upgradeable is False
    assert any("same owner scope" in item for item in moved_scope_plan.structural_conflicts)


@pytest.mark.asyncio
async def test_dry_run_upgrade_blocks_removed_profile_with_local_assignment_dependency(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    service = McpHubGovernancePackService(repo=repo)
    source_pack = load_governance_pack_fixture("minimal_researcher_pack")
    imported = await service.import_pack(
        pack=source_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
    )

    await repo.create_policy_assignment(
        target_type="persona",
        target_id="local.persona",
        owner_scope_type="user",
        owner_scope_id=7,
        profile_id=imported.imported_object_ids["permission_profiles"][0],
        inline_policy_document={"source": "local-overlay"},
        approval_policy_id=imported.imported_object_ids["approval_policies"][0],
        actor_id=8,
        is_active=True,
        is_immutable=False,
    )

    target_pack = deepcopy(source_pack)
    target_pack.manifest.pack_version = "1.1.0"
    target_pack.profiles[0].profile_id = "researcher.profile.v2"
    target_pack.personas[0].capability_profile_id = "researcher.profile.v2"
    target_pack.assignments[0].capability_profile_id = "researcher.profile.v2"

    plan = await service.dry_run_upgrade_pack(
        source_governance_pack_id=imported.governance_pack_id,
        pack=target_pack,
        owner_scope_type="user",
        owner_scope_id=7,
    )

    assert plan.upgradeable is False
    assert any(
        "permission_profile:researcher.profile" in item and "policy assignment" in item
        for item in plan.structural_conflicts
    )
    assert any(
        item["object_type"] == "permission_profile"
        and item["source_object_id"] == "researcher.profile"
        and item["impact"] == "structural_conflict"
        for item in plan.dependency_impact
    )


@pytest.mark.asyncio
async def test_dry_run_upgrade_blocks_semantic_profile_change_with_local_assignment_dependency(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    service = McpHubGovernancePackService(repo=repo)
    source_pack = load_governance_pack_fixture("minimal_researcher_pack")
    imported = await service.import_pack(
        pack=source_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
    )

    await repo.create_policy_assignment(
        target_type="persona",
        target_id="local.persona",
        owner_scope_type="user",
        owner_scope_id=7,
        profile_id=imported.imported_object_ids["permission_profiles"][0],
        inline_policy_document={"source": "local-overlay"},
        approval_policy_id=imported.imported_object_ids["approval_policies"][0],
        actor_id=8,
        is_active=True,
        is_immutable=False,
    )

    target_pack = deepcopy(source_pack)
    target_pack.manifest.pack_version = "1.1.0"
    target_pack.profiles[0].environment_requirements = []

    plan = await service.dry_run_upgrade_pack(
        source_governance_pack_id=imported.governance_pack_id,
        pack=target_pack,
        owner_scope_type="user",
        owner_scope_id=7,
    )

    assert plan.upgradeable is False
    assert any(
        "permission_profile:researcher.profile" in item and "materially changes" in item
        for item in plan.behavioral_conflicts
    )
    assert any(
        item["object_type"] == "permission_profile"
        and item["source_object_id"] == "researcher.profile"
        and item["impact"] == "behavioral_conflict"
        for item in plan.dependency_impact
    )


@pytest.mark.asyncio
async def test_dry_run_upgrade_reports_workspace_rebind_for_modified_assignment(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    service = McpHubGovernancePackService(repo=repo)
    source_pack = load_governance_pack_fixture("minimal_researcher_pack")
    imported = await service.import_pack(
        pack=source_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
    )

    await repo.add_policy_assignment_workspace(
        int(imported.imported_object_ids["policy_assignments"][0]),
        workspace_id="workspace-alpha",
        actor_id=8,
    )

    target_pack = deepcopy(source_pack)
    target_pack.manifest.pack_version = "1.0.1"
    target_pack.assignments[0].approval_template_id = None

    plan = await service.dry_run_upgrade_pack(
        source_governance_pack_id=imported.governance_pack_id,
        pack=target_pack,
        owner_scope_type="user",
        owner_scope_id=7,
    )

    assert plan.upgradeable is True
    assert plan.structural_conflicts == []
    assert plan.behavioral_conflicts == []
    assert any(
        item["object_type"] == "policy_assignment"
        and item["source_object_id"] == "researcher.default"
        and item["impact"] == "rebind_required"
        and item["dependent_type"] == "policy_assignment_workspace"
        and item["reference_field"] == "assignment_id"
        and item["target_id"] == "workspace-alpha"
        for item in plan.dependency_impact
    )


@pytest.mark.asyncio
async def test_execute_upgrade_rebinds_dependents_and_marks_pack_lineage(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    service = McpHubGovernancePackService(repo=repo)
    source_pack = load_governance_pack_fixture("minimal_researcher_pack")
    imported = await service.import_pack(
        pack=source_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
    )

    local_assignment = await repo.create_policy_assignment(
        target_type="persona",
        target_id="local.persona",
        owner_scope_type="user",
        owner_scope_id=7,
        profile_id=imported.imported_object_ids["permission_profiles"][0],
        inline_policy_document={"source": "local-overlay"},
        approval_policy_id=imported.imported_object_ids["approval_policies"][0],
        actor_id=8,
        is_active=True,
        is_immutable=False,
    )
    await repo.upsert_policy_override(
        int(imported.imported_object_ids["policy_assignments"][0]),
        override_policy_document={"allowed_tools": ["Read"]},
        broadens_access=False,
        grant_authority_snapshot={"source": "local-overlay"},
        actor_id=8,
        is_active=True,
    )
    await repo.add_policy_assignment_workspace(
        int(imported.imported_object_ids["policy_assignments"][0]),
        workspace_id="workspace-alpha",
        actor_id=8,
    )

    target_pack = deepcopy(source_pack)
    target_pack.manifest.pack_version = "1.0.1"
    target_pack.manifest.description = "Upgrade target"
    plan = await service.dry_run_upgrade_pack(
        source_governance_pack_id=imported.governance_pack_id,
        pack=target_pack,
        owner_scope_type="user",
        owner_scope_id=7,
    )

    result = await service.execute_upgrade_pack(
        source_governance_pack_id=imported.governance_pack_id,
        pack=target_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
        planner_inputs_fingerprint=plan.planner_inputs_fingerprint,
        adapter_state_fingerprint=plan.adapter_state_fingerprint,
    )

    assert result.from_pack_version == "1.0.0"
    assert result.to_pack_version == "1.0.1"

    old_pack = await repo.get_governance_pack(imported.governance_pack_id)
    new_pack = await repo.get_governance_pack(result.target_governance_pack_id)
    assert old_pack is not None and new_pack is not None
    assert old_pack["is_active_install"] is False
    assert old_pack["superseded_by_governance_pack_id"] == result.target_governance_pack_id
    assert new_pack["is_active_install"] is True
    assert new_pack["installed_from_upgrade_id"] == result.upgrade_id

    new_objects = await repo.list_governance_pack_objects(result.target_governance_pack_id)
    new_profile_id = int(
        next(
            item["object_id"]
            for item in new_objects
            if item["object_type"] == "permission_profile"
            and item["source_object_id"] == "researcher.profile"
        )
    )
    new_approval_id = int(
        next(
            item["object_id"]
            for item in new_objects
            if item["object_type"] == "approval_policy"
            and item["source_object_id"] == "researcher.ask"
        )
    )
    new_assignment_id = int(
        next(
            item["object_id"]
            for item in new_objects
            if item["object_type"] == "policy_assignment"
            and item["source_object_id"] == "researcher.default"
        )
    )

    rebound_assignment = await repo.get_policy_assignment(int(local_assignment["id"]))
    assert rebound_assignment is not None
    assert int(rebound_assignment["profile_id"]) == new_profile_id
    assert int(rebound_assignment["approval_policy_id"]) == new_approval_id

    assert (
        await repo.get_policy_override_by_assignment(
            int(imported.imported_object_ids["policy_assignments"][0])
        )
        is None
    )
    rebound_override = await repo.get_policy_override_by_assignment(new_assignment_id)
    assert rebound_override is not None
    assert rebound_override["override_policy_document"]["allowed_tools"] == ["Read"]
    assert await repo.list_policy_assignment_workspaces(
        int(imported.imported_object_ids["policy_assignments"][0])
    ) == []
    assert [item["workspace_id"] for item in await repo.list_policy_assignment_workspaces(new_assignment_id)] == [
        "workspace-alpha"
    ]

    history = await repo.list_governance_pack_upgrades(
        pack_id="researcher-pack",
        owner_scope_type="user",
        owner_scope_id=7,
    )
    assert history[-1]["status"] == "executed"


@pytest.mark.asyncio
async def test_execute_upgrade_rejects_stale_plan_fingerprints(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    service = McpHubGovernancePackService(repo=repo)
    source_pack = load_governance_pack_fixture("minimal_researcher_pack")
    imported = await service.import_pack(
        pack=source_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
    )

    target_pack = deepcopy(source_pack)
    target_pack.manifest.pack_version = "1.0.1"
    plan = await service.dry_run_upgrade_pack(
        source_governance_pack_id=imported.governance_pack_id,
        pack=target_pack,
        owner_scope_type="user",
        owner_scope_id=7,
    )

    await repo.create_policy_assignment(
        target_type="persona",
        target_id="late.local.persona",
        owner_scope_type="user",
        owner_scope_id=7,
        profile_id=imported.imported_object_ids["permission_profiles"][0],
        inline_policy_document={"source": "late-local-overlay"},
        approval_policy_id=None,
        actor_id=8,
        is_active=True,
        is_immutable=False,
    )

    with pytest.raises(ValueError, match="stale"):
        await service.execute_upgrade_pack(
            source_governance_pack_id=imported.governance_pack_id,
            pack=target_pack,
            owner_scope_type="user",
            owner_scope_id=7,
            actor_id=7,
            planner_inputs_fingerprint=plan.planner_inputs_fingerprint,
            adapter_state_fingerprint=plan.adapter_state_fingerprint,
        )

    source_row = await repo.get_governance_pack(imported.governance_pack_id)
    assert source_row is not None
    assert source_row["is_active_install"] is True


@pytest.mark.asyncio
async def test_execute_upgrade_rolls_back_when_staging_insert_fails(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.governance_packs import (
        load_governance_pack_fixture,
    )
    from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
        McpHubGovernancePackService,
    )

    repo = await _make_repo(tmp_path, monkeypatch)
    await _seed_research_capability_mappings(repo)
    service = McpHubGovernancePackService(repo=repo)
    source_pack = load_governance_pack_fixture("minimal_researcher_pack")
    imported = await service.import_pack(
        pack=source_pack,
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
    )

    target_pack = deepcopy(source_pack)
    target_pack.manifest.pack_version = "1.0.1"
    plan = await service.dry_run_upgrade_pack(
        source_governance_pack_id=imported.governance_pack_id,
        pack=target_pack,
        owner_scope_type="user",
        owner_scope_id=7,
    )

    original_create_policy_assignment = repo.create_policy_assignment

    async def _boom_create_policy_assignment(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("upgrade assignment insert failed")

    repo.create_policy_assignment = _boom_create_policy_assignment  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="upgrade assignment insert failed"):
            await service.execute_upgrade_pack(
                source_governance_pack_id=imported.governance_pack_id,
                pack=target_pack,
                owner_scope_type="user",
                owner_scope_id=7,
                actor_id=7,
                planner_inputs_fingerprint=plan.planner_inputs_fingerprint,
                adapter_state_fingerprint=plan.adapter_state_fingerprint,
            )
    finally:
        repo.create_policy_assignment = original_create_policy_assignment  # type: ignore[method-assign]

    inventory = await repo.list_governance_packs(owner_scope_type="user", owner_scope_id=7)
    assert [(item["pack_version"], item["is_active_install"]) for item in inventory] == [("1.0.0", True)]
