"""Scheduled Agent execution evidence harness tests."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "Helper_Scripts"
    / "Testing-related"
    / "scheduled_agent_execution_certification.py"
)
NOW = datetime(2026, 8, 26, 20, 0, tzinfo=timezone.utc)
REQUIREMENT_IDS = (
    "isolation_attestation",
    "hostile_boundary",
    "scheduled_transcript_non_disclosure",
    "adapter_dispatch_recovery",
    "monotonic_execution_evidence",
    "brokered_credentials_and_mediation",
    "operational_fail_closed",
)
MANIFEST_KEYS = {
    "schema_version",
    "evidence_id",
    "deployment_class_id",
    "source_commit",
    "created_at",
    "valid_until",
    "outcome",
    "reason_codes",
    "requirements",
    "commands",
}
REQUIREMENT_KEYS = {
    "requirement_id",
    "state",
    "verification",
    "subject_id",
    "observed_at",
    "valid_until",
    "reason_codes",
    "evidence_sha256",
    "safety_boundary_breached",
}
COMMAND_KEYS = {
    "id",
    "description",
    "invocation_template",
    "parameter_names",
    "safe_to_run_by_default",
    "required_environment_names",
}


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "scheduled_agent_execution_certification",
        SCRIPT_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _inputs(module: ModuleType, *, runtime: str = "docker"):
    return module.CertificationInputs(
        host_os=platform.system().lower(),
        host_arch=platform.machine().lower(),
        runtime=runtime,
        auth_mode="single_user",
        adapter_id="acp",
        adapter_version="1",
        source_commit="a" * 40,
        server_build_sha="b" * 40,
        image_digest="unverified",
        mount_policy_hash="unverified",
        egress_policy_hash="unverified",
        credential_policy_hash="unverified",
        tenant_boundary_policy_hash="unverified",
        mediation_policy_hash="unverified",
        isolation_profile_version="phase4d0f-baseline",
    )


def _sentinels(module: ModuleType):
    return module.CharacterizationSentinels(
        prompt="PROMPT-SENTINEL-9ca92c",
        credential="CREDENTIAL-SENTINEL-8e448f",
        host_path="/private/sensitive/path/SENTINEL-51db",
        hostname="internal-host-SENTINEL-71c3.example",
        tool_argument="--secret-argument=SENTINEL-3df2",
        environment_value="ENVIRONMENT-SENTINEL-88a1",
    )


def _base_cli_args(runtime: str = "docker") -> list[str]:
    return [
        "--host-os",
        platform.system().lower(),
        "--host-arch",
        platform.machine().lower(),
        "--runtime",
        runtime,
        "--auth-mode",
        "single_user",
        "--adapter-id",
        "acp",
        "--adapter-version",
        "1",
        "--source-commit",
        "a" * 40,
        "--server-build-sha",
        "b" * 40,
        "--image-digest",
        "unverified",
        "--mount-policy-hash",
        "unverified",
        "--egress-policy-hash",
        "unverified",
        "--credential-policy-hash",
        "unverified",
        "--tenant-boundary-policy-hash",
        "unverified",
        "--mediation-policy-hash",
        "unverified",
        "--isolation-profile-version",
        "phase4d0f-baseline",
    ]


def test_manifest_has_exact_sanitized_schema_and_all_seven_domains(
    tmp_path: Path,
) -> None:
    """A missing or extra public field must not silently change the evidence API."""

    module = _load_module()

    manifest = module.build_evidence_manifest(
        _inputs(module),
        now=NOW,
        temporary_directory=tmp_path,
        sentinels=_sentinels(module),
    )

    assert set(manifest) == MANIFEST_KEYS
    assert manifest["schema_version"] == (
        "scheduled-agent-execution-certification.v1"
    )
    assert manifest["outcome"] == "draft_only"
    assert [item["requirement_id"] for item in manifest["requirements"]] == list(
        REQUIREMENT_IDS
    )
    assert all(set(item) == REQUIREMENT_KEYS for item in manifest["requirements"])
    assert [command["id"] for command in manifest["commands"]] == list(
        REQUIREMENT_IDS
    )
    assert all(set(command) == COMMAND_KEYS for command in manifest["commands"])


def test_repository_characterization_never_emits_certified(
    tmp_path: Path,
) -> None:
    """Static and temporary local characterization must not become authority."""

    module = _load_module()

    manifest = module.build_evidence_manifest(
        _inputs(module),
        now=NOW,
        temporary_directory=tmp_path,
    )

    assert manifest["outcome"] != "certified"
    assert all(
        item["verification"] == "repository_characterization"
        for item in manifest["requirements"]
    )
    assert all(item["state"] == "missing" for item in manifest["requirements"])


def test_ineligible_worktree_runtime_is_unsupported(tmp_path: Path) -> None:
    """Host-local repository worktrees must not be reported as draft-ready."""

    module = _load_module()

    manifest = module.build_evidence_manifest(
        _inputs(module, runtime="worktree"),
        now=NOW,
        temporary_directory=tmp_path,
    )

    assert manifest["outcome"] == "unsupported"
    assert "runtime_not_untrusted_eligible" in manifest["reason_codes"]
    assert "runtime_strict_deny_all_unavailable" in manifest["reason_codes"]


def test_prompt_and_other_sensitive_values_never_serialize(tmp_path: Path) -> None:
    """Characterization values must be reduced to bounded hashes and reason codes."""

    module = _load_module()
    sentinels = _sentinels(module)
    manifest = module.build_evidence_manifest(
        _inputs(module),
        now=NOW,
        temporary_directory=tmp_path,
        sentinels=sentinels,
    )

    rendered = module.render_manifest_json(manifest) + module.render_manifest_markdown(
        manifest
    )

    for sentinel in vars(sentinels).values():
        assert sentinel not in rendered
    assert str(tmp_path) not in rendered


def test_command_manifest_contains_names_but_no_values_or_urls(
    tmp_path: Path,
) -> None:
    """A runnable manifest must not publish local arguments or environment values."""

    module = _load_module()
    manifest = module.build_evidence_manifest(
        _inputs(module),
        now=NOW,
        temporary_directory=tmp_path,
    )
    serialized = json.dumps(manifest["commands"], sort_keys=True)

    assert "Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py" in serialized
    assert "--runtime" in serialized
    assert "http://" not in serialized
    assert "https://" not in serialized
    assert str(tmp_path) not in serialized
    assert "single_user" not in serialized
    assert "docker" not in serialized
    assert "b" * 40 not in serialized


def test_evidence_id_covers_canonical_sanitized_content(tmp_path: Path) -> None:
    """Changing an artifact after generation must invalidate its evidence identity."""

    module = _load_module()
    manifest = module.build_evidence_manifest(
        _inputs(module),
        now=NOW,
        temporary_directory=tmp_path,
    )

    assert manifest["evidence_id"] == module.compute_manifest_evidence_id(manifest)

    tampered = json.loads(json.dumps(manifest))
    tampered["outcome"] = "certified"
    with pytest.raises(ValueError, match="evidence_id"):
        module.validate_manifest(tampered)


def test_transcript_characterization_uses_real_store_and_fork_without_leaking(
    tmp_path: Path,
) -> None:
    """The ordinary ACP leakage finding must come from persisted behavior."""

    module = _load_module()
    sentinel = "PROMPT-STORE-SENTINEL-4dc6"

    result = module.characterize_ordinary_acp_transcript(
        database_path=tmp_path / "acp.db",
        prompt_sentinel=sentinel,
    )

    assert result == {
        "ordinary_prompt_retrievable": True,
        "ordinary_fork_copies_prompt": True,
        "prompt_sha256": module.sha256_text(sentinel),
        "reason_code": "scheduled_transcript_mode_unimplemented",
    }
    assert sentinel not in json.dumps(result, sort_keys=True)


def test_characterization_records_current_partial_primitives(tmp_path: Path) -> None:
    """Generic primitives must remain gaps when scheduled bindings are absent."""

    module = _load_module()

    facts = module.characterize_current_primitives(
        database_path=tmp_path / "acp.db",
        prompt_sentinel="PROMPT-PARTIAL-SENTINEL-1d2f",
    )

    assert facts["adapter_dispatch_recovery"] == {
        "sandbox_idempotency_available": True,
        "acp_dispatch_token_parameter": False,
        "acp_dispatch_token_persisted": False,
    }
    assert facts["monotonic_execution_evidence"]["ordered_attempt_journal"] is False
    assert facts["brokered_credentials_and_mediation"] == {
        "managed_credential_broker_available": True,
        "acp_session_env_channel_present": True,
        "scheduled_grant_action_binding": False,
    }


def test_write_and_validate_artifact_pair(tmp_path: Path) -> None:
    """JSON and Markdown outputs must describe the same immutable result."""

    module = _load_module()
    manifest = module.build_evidence_manifest(
        _inputs(module),
        now=NOW,
        temporary_directory=tmp_path,
    )
    json_path = tmp_path / "baseline.json"
    markdown_path = tmp_path / "baseline.md"

    module.write_artifacts(
        manifest,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    module.validate_artifact_pair(json_path, markdown_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == manifest
    assert manifest["evidence_id"] in markdown_path.read_text(encoding="utf-8")


def test_artifact_pair_rejects_markdown_from_another_manifest(
    tmp_path: Path,
) -> None:
    """A stale Markdown summary must not validate beside newer JSON evidence."""

    module = _load_module()
    docker = module.build_evidence_manifest(
        _inputs(module),
        now=NOW,
        temporary_directory=tmp_path,
    )
    worktree = module.build_evidence_manifest(
        _inputs(module, runtime="worktree"),
        now=NOW,
        temporary_directory=tmp_path,
    )
    json_path = tmp_path / "baseline.json"
    markdown_path = tmp_path / "baseline.md"
    json_path.write_text(module.render_manifest_json(docker), encoding="utf-8")
    markdown_path.write_text(
        module.render_manifest_markdown(worktree),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Markdown"):
        module.validate_artifact_pair(json_path, markdown_path)


@pytest.mark.parametrize(
    "missing_field",
    [
        "evidence_dir",
        "server_url",
        "api_key_environment_name",
        "attestation_reference",
    ],
)
def test_hostile_probe_refuses_before_launch_when_prerequisite_is_missing(
    tmp_path: Path,
    missing_field: str,
) -> None:
    """Hostile execution must fail closed before any adapter or network call."""

    module = _load_module()
    values = {
        "evidence_dir": tmp_path,
        "server_url": "http://127.0.0.1:8000",
        "api_key_environment_name": "TLDW_TEST_API_KEY",
        "attestation_reference": "sha256:" + "c" * 64,
    }
    values[missing_field] = None
    request = module.HostileProbeRequest(
        deployment_class_id="sha256:" + "d" * 64,
        **values,
    )

    result = module.evaluate_hostile_probe_admission(
        request,
        environment={"TLDW_TEST_API_KEY": "secret-value"},
    )

    assert result.allowed is False
    assert result.reason_code.startswith("hostile_probe_blocked_")


def test_hostile_probe_refuses_nonlocal_server_url(tmp_path: Path) -> None:
    """A hostile vector must never target a remote or user-controlled host."""

    module = _load_module()
    request = module.HostileProbeRequest(
        deployment_class_id="sha256:" + "d" * 64,
        evidence_dir=tmp_path,
        server_url="https://example.com",
        api_key_environment_name="TLDW_TEST_API_KEY",
        attestation_reference="sha256:" + "c" * 64,
    )

    result = module.evaluate_hostile_probe_admission(
        request,
        environment={"TLDW_TEST_API_KEY": "secret-value"},
    )

    assert result.allowed is False
    assert result.reason_code == "hostile_probe_blocked_nonlocal_server"


def test_hostile_probe_remains_blocked_without_server_verifier(tmp_path: Path) -> None:
    """Supplying every CLI value must not substitute for attestation verification."""

    module = _load_module()
    request = module.HostileProbeRequest(
        deployment_class_id="sha256:" + "d" * 64,
        evidence_dir=tmp_path,
        server_url="http://127.0.0.1:8000",
        api_key_environment_name="TLDW_TEST_API_KEY",
        attestation_reference="sha256:" + "c" * 64,
    )

    result = module.evaluate_hostile_probe_admission(
        request,
        environment={"TLDW_TEST_API_KEY": "secret-value"},
    )

    assert result.allowed is False
    assert result.reason_code == (
        "hostile_probe_blocked_server_attestation_verifier_unimplemented"
    )


def test_cli_emits_draft_json_for_eligible_runtime(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default CLI path must produce machine-readable draft evidence."""

    module = _load_module()

    result = module.main([*_base_cli_args(), "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert result == 0
    assert payload["outcome"] == "draft_only"
    assert captured.err == ""


def test_cli_emits_unsupported_markdown_for_worktree(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The human-readable CLI path must preserve the unsupported outcome."""

    module = _load_module()

    result = module.main([*_base_cli_args("worktree"), "--format", "markdown"])
    captured = capsys.readouterr()

    assert result == 0
    assert "Outcome | `unsupported`" in captured.out
    assert captured.err == ""


def test_cli_rejects_claimed_host_without_characterization_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An operator typo must not mint evidence for a different host subject."""

    module = _load_module()
    args = _base_cli_args()
    host_index = args.index("--host-os") + 1
    args[host_index] = "windows" if platform.system().lower() != "windows" else "linux"

    result = module.main([*args, "--format", "json"])
    captured = capsys.readouterr()

    assert result == 2
    assert captured.out == ""
    assert "does not match the observed host" in captured.err


def test_cli_hostile_refusal_does_not_write_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A blocked hostile probe must not leave an artifact that resembles a pass."""

    module = _load_module()
    monkeypatch.setenv("TLDW_TEST_API_KEY", "secret-value")
    json_path = tmp_path / "must-not-exist.json"
    markdown_path = tmp_path / "must-not-exist.md"
    args = [
        *_base_cli_args(),
        "--run-hostile",
        "--evidence-dir",
        str(tmp_path),
        "--server-url",
        "http://127.0.0.1:8000",
        "--api-key-env-name",
        "TLDW_TEST_API_KEY",
        "--attestation-reference",
        "sha256:" + "c" * 64,
        "--output-json",
        str(json_path),
        "--output-markdown",
        str(markdown_path),
    ]

    result = module.main(args)
    captured = capsys.readouterr()

    assert result == 2
    assert "server_attestation_verifier_unimplemented" in captured.err
    assert not json_path.exists()
    assert not markdown_path.exists()


def test_cli_validates_existing_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Operators need a deterministic success code for unchanged evidence pairs."""

    module = _load_module()
    manifest = module.build_evidence_manifest(
        _inputs(module),
        now=NOW,
        temporary_directory=tmp_path,
    )
    json_path = tmp_path / "baseline.json"
    markdown_path = tmp_path / "baseline.md"
    module.write_artifacts(
        manifest,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    result = module.main(
        ["--validate-artifacts", str(json_path), str(markdown_path)]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.out == "Artifacts valid.\n"
    assert captured.err == ""
