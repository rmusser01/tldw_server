#!/usr/bin/env python3
"""Generate sanitized Scheduled Agent execution feasibility evidence.

This helper characterizes current repository behavior. It cannot construct the
authoritative receipt required for certification and cannot launch hostile
probes until a later server-side attestation verifier exists.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import re
import secrets
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tldw_Server_API.app.core.Sandbox.runtime_capabilities import (  # noqa: E402
    runtime_isolation_metadata,
    runtime_network_policy_metadata,
)
from tldw_Server_API.app.core.Scheduled_Tasks.execution_certification import (  # noqa: E402
    CERTIFICATION_REASON_CODES,
    REQUIRED_EVIDENCE_DOMAINS,
    DeploymentClass,
    IsolationProfile,
    RequirementEvidence,
    RuntimeEligibility,
    evaluate_execution_certification,
)

SCHEMA_VERSION = "scheduled-agent-execution-certification.v1"
VALIDITY_WINDOW = timedelta(hours=24)
STATIC_RUNTIME_VALUES = (
    "docker",
    "firecracker",
    "lima",
    "vz_linux",
    "vz_macos",
    "seatbelt",
    "worktree",
)
HELPER_PATH = (
    "Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py"
)
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_MANIFEST_KEYS = {
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
_REQUIREMENT_KEYS = {
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
_COMMAND_KEYS = {
    "id",
    "description",
    "invocation_template",
    "parameter_names",
    "safe_to_run_by_default",
    "required_environment_names",
}
_COMMON_PARAMETER_NAMES = [
    "host-os",
    "host-arch",
    "runtime",
    "auth-mode",
    "adapter-id",
    "adapter-version",
    "source-commit",
    "server-build-sha",
    "image-digest",
    "mount-policy-hash",
    "egress-policy-hash",
    "credential-policy-hash",
    "tenant-boundary-policy-hash",
    "mediation-policy-hash",
    "isolation-profile-version",
]
_INVOCATION_TEMPLATE = " ".join(
    ["python", HELPER_PATH, *[f"--{name}" for name in _COMMON_PARAMETER_NAMES]]
)
_COMMAND_DESCRIPTIONS = {
    "isolation_attestation": (
        "Characterize static runtime isolation metadata; no attestation is issued."
    ),
    "hostile_boundary": (
        "Enumerate the hostile boundary vector; launch remains attestation-gated."
    ),
    "scheduled_transcript_non_disclosure": (
        "Characterize ordinary ACP prompt persistence and fork behavior."
    ),
    "adapter_dispatch_recovery": (
        "Characterize generic idempotency and missing ACP dispatch bindings."
    ),
    "monotonic_execution_evidence": (
        "Characterize cancellation primitives and the missing ordered journal."
    ),
    "brokered_credentials_and_mediation": (
        "Characterize managed credentials, session environment, and grant gaps."
    ),
    "operational_fail_closed": (
        "Evaluate the current fail-closed outcome for the exact deployment class."
    ),
}


def _aware_utc(value: datetime) -> datetime:
    """Normalize one aware evidence timestamp to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("evidence timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _canonical_json(value: object) -> str:
    """Serialize sanitized evidence into canonical JSON."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sha256_text(value: str) -> str:
    """Return a bounded one-way identity for sensitive characterization text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_slug(name: str, value: str) -> str:
    """Validate one bounded non-secret identifier."""

    normalized = str(value or "").strip()
    if not _SLUG_PATTERN.fullmatch(normalized):
        raise ValueError(f"{name} must be a bounded identifier")
    return normalized


def _require_commit(name: str, value: str) -> str:
    """Validate and normalize one source/build commit digest."""

    normalized = str(value or "").strip()
    if not _COMMIT_PATTERN.fullmatch(normalized):
        raise ValueError(f"{name} must be a 40- or 64-character commit digest")
    return normalized.lower()


def _require_hash_or_unverified(name: str, value: str) -> str:
    """Validate an opaque SHA-256 identity or explicit unverified marker."""

    normalized = str(value or "").strip().lower()
    if normalized != "unverified" and not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{name} must be unverified or a sha256 identity")
    return normalized


@dataclass(frozen=True)
class CertificationInputs:
    """Explicit private inputs used only to derive opaque evidence subjects."""

    host_os: str
    host_arch: str
    runtime: str
    auth_mode: str
    adapter_id: str
    adapter_version: str
    source_commit: str
    server_build_sha: str
    image_digest: str
    mount_policy_hash: str
    egress_policy_hash: str
    credential_policy_hash: str
    tenant_boundary_policy_hash: str
    mediation_policy_hash: str
    isolation_profile_version: str

    def __post_init__(self) -> None:
        """Normalize and validate all explicit private input identities."""

        normalized = {
            "host_os": _require_slug("host_os", self.host_os).lower(),
            "host_arch": _require_slug("host_arch", self.host_arch).lower(),
            "runtime": _require_slug("runtime", self.runtime).lower(),
            "auth_mode": _require_slug("auth_mode", self.auth_mode).lower(),
            "adapter_id": _require_slug("adapter_id", self.adapter_id).lower(),
            "adapter_version": _require_slug(
                "adapter_version", self.adapter_version
            ),
            "source_commit": _require_commit("source_commit", self.source_commit),
            "server_build_sha": _require_commit(
                "server_build_sha", self.server_build_sha
            ),
            "image_digest": _require_hash_or_unverified(
                "image_digest", self.image_digest
            ),
            "mount_policy_hash": _require_hash_or_unverified(
                "mount_policy_hash", self.mount_policy_hash
            ),
            "egress_policy_hash": _require_hash_or_unverified(
                "egress_policy_hash", self.egress_policy_hash
            ),
            "credential_policy_hash": _require_hash_or_unverified(
                "credential_policy_hash", self.credential_policy_hash
            ),
            "tenant_boundary_policy_hash": _require_hash_or_unverified(
                "tenant_boundary_policy_hash", self.tenant_boundary_policy_hash
            ),
            "mediation_policy_hash": _require_hash_or_unverified(
                "mediation_policy_hash", self.mediation_policy_hash
            ),
            "isolation_profile_version": _require_slug(
                "isolation_profile_version", self.isolation_profile_version
            ),
        }
        for field_name, value in normalized.items():
            object.__setattr__(self, field_name, value)

    def deployment_class(self) -> DeploymentClass:
        """Build the exact private subject whose digest is published."""

        profile = IsolationProfile(
            runtime_image_digest=self.image_digest,
            mount_policy_hash=self.mount_policy_hash,
            egress_policy_hash=self.egress_policy_hash,
            credential_policy_hash=self.credential_policy_hash,
            tenant_boundary_policy_hash=self.tenant_boundary_policy_hash,
            mediation_policy_hash=self.mediation_policy_hash,
            isolation_profile_version=self.isolation_profile_version,
        )
        return DeploymentClass(
            host_os_family=self.host_os,
            host_architecture=self.host_arch,
            auth_mode=self.auth_mode,
            sandbox_runtime=self.runtime,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            server_build_sha=self.server_build_sha,
            isolation_profile=profile,
        )


@dataclass(frozen=True)
class CharacterizationSentinels:
    """Values exercised by probes but replaced with hashes before serialization."""

    prompt: str
    credential: str
    host_path: str
    hostname: str
    tool_argument: str
    environment_value: str


def _random_sentinels() -> CharacterizationSentinels:
    """Generate per-run sensitive values that must never be serialized."""

    return CharacterizationSentinels(
        prompt="prompt-" + secrets.token_hex(16),
        credential="credential-" + secrets.token_hex(16),
        host_path="/private/characterization/" + secrets.token_hex(16),
        hostname="host-" + secrets.token_hex(16) + ".invalid",
        tool_argument="--characterization=" + secrets.token_hex(16),
        environment_value="environment-" + secrets.token_hex(16),
    )


def characterize_ordinary_acp_transcript(
    *,
    database_path: Path,
    prompt_sentinel: str,
) -> dict[str, object]:
    """Use ordinary ACP storage to prove current prompt and fork disclosure."""

    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB

    database = ACPSessionsDB(str(database_path))
    run_identity = secrets.token_hex(8)
    source_session_id = f"phase4d0f-source-{run_identity}"
    fork_session_id = f"phase4d0f-fork-{run_identity}"
    try:
        database.register_session(
            session_id=source_session_id,
            user_id=1,
            agent_type="custom",
        )
        database.record_prompt(
            source_session_id,
            [{"role": "user", "content": prompt_sentinel}],
            {"content": "characterization response"},
        )
        source_messages = database.get_messages(source_session_id)
        database.fork_session(
            source_session_id,
            fork_session_id,
            message_index=0,
            user_id=1,
        )
        fork_messages = database.get_messages(fork_session_id)
        source_retrievable = any(
            prompt_sentinel in str(message.get("content") or "")
            for message in source_messages
        )
        fork_copies_prompt = any(
            prompt_sentinel in str(message.get("content") or "")
            for message in fork_messages
        )
    finally:
        database.close()
    return {
        "ordinary_prompt_retrievable": source_retrievable,
        "ordinary_fork_copies_prompt": fork_copies_prompt,
        "prompt_sha256": sha256_text(prompt_sentinel),
        "reason_code": "scheduled_transcript_mode_unimplemented",
    }


def characterize_current_primitives(
    *,
    database_path: Path,
    prompt_sentinel: str,
) -> dict[str, dict[str, object]]:
    """Characterize existing typed APIs without treating partials as proof."""

    from tldw_Server_API.app.core.Agent_Client_Protocol.sandbox_runner_client import (
        ACPSandboxRunnerManager,
    )
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
    from tldw_Server_API.app.core.Sandbox.service import SandboxService
    from tldw_Server_API.app.services.mcp_credential_broker_service import (
        McpCredentialBrokerService,
    )

    acp_create_parameters = inspect.signature(
        ACPSandboxRunnerManager.create_session
    ).parameters
    sandbox_create_parameters = inspect.signature(SandboxService.create_session).parameters
    persisted_session_parameters = inspect.signature(
        ACPSessionsDB.register_session
    ).parameters
    transcript = characterize_ordinary_acp_transcript(
        database_path=database_path,
        prompt_sentinel=prompt_sentinel,
    )
    return {
        "isolation_attestation": {
            "server_attestation_verifier": False,
        },
        "hostile_boundary": {
            "hostile_probe_launch_allowed": False,
        },
        "scheduled_transcript_non_disclosure": transcript,
        "adapter_dispatch_recovery": {
            "sandbox_idempotency_available": "idem_key" in sandbox_create_parameters,
            "acp_dispatch_token_parameter": "dispatch_token" in acp_create_parameters,
            "acp_dispatch_token_persisted": (
                "dispatch_token" in persisted_session_parameters
            ),
        },
        "monotonic_execution_evidence": {
            "cancellation_primitive_available": hasattr(
                ACPSandboxRunnerManager,
                "cancel",
            ),
            "terminal_state_primitive_available": hasattr(
                ACPSessionsDB,
                "set_session_status",
            ),
            "ordered_attempt_journal": hasattr(
                ACPSessionsDB,
                "append_execution_event",
            ),
        },
        "brokered_credentials_and_mediation": {
            "managed_credential_broker_available": hasattr(
                McpCredentialBrokerService,
                "get_slot_status",
            ),
            "acp_session_env_channel_present": "session_env" in acp_create_parameters,
            "scheduled_grant_action_binding": False,
        },
        "operational_fail_closed": {
            "unified_certification_gate_available": True,
            "authoritative_evidence_ingestion_available": False,
        },
    }


def _runtime_eligibility(runtime: str) -> RuntimeEligibility:
    """Map typed Sandbox runtime metadata to certification eligibility."""

    try:
        isolation = runtime_isolation_metadata(runtime)
        deny_all = runtime_network_policy_metadata(runtime).deny_all
    except ValueError:
        return RuntimeEligibility(untrusted_eligible=False, strict_deny_all=False)
    return RuntimeEligibility(
        untrusted_eligible=isolation.untrusted_eligible,
        strict_deny_all=(
            deny_all.strict_enforcement
            and deny_all.support_state in {"supported", "host_gated"}
        ),
    )


def _requirement_record(
    *,
    requirement_id: str,
    subject_id: str,
    observed_at: datetime,
    valid_until: datetime,
    facts: Mapping[str, object],
    sentinel_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Build one sanitized missing-evidence requirement record."""

    evidence_sha256 = sha256_text(
        _canonical_json(
            {
                "facts": dict(facts),
                "requirement_id": requirement_id,
                "sentinel_hashes": dict(sentinel_hashes),
                "subject_id": subject_id,
            }
        )
    )
    return {
        "requirement_id": requirement_id,
        "state": "missing",
        "verification": "repository_characterization",
        "subject_id": subject_id,
        "observed_at": observed_at.isoformat(),
        "valid_until": valid_until.isoformat(),
        "reason_codes": [f"{requirement_id}_missing"],
        "evidence_sha256": evidence_sha256,
        "safety_boundary_breached": False,
    }


def _command_manifest() -> list[dict[str, object]]:
    """Return the closed deterministic command metadata manifest."""

    commands: list[dict[str, object]] = []
    for requirement_id in REQUIRED_EVIDENCE_DOMAINS:
        parameter_names = list(_COMMON_PARAMETER_NAMES)
        safe_to_run = requirement_id != "hostile_boundary"
        if requirement_id == "hostile_boundary":
            parameter_names.extend(
                [
                    "run-hostile",
                    "evidence-dir",
                    "server-url",
                    "api-key-env-name",
                    "attestation-reference",
                ]
            )
        commands.append(
            {
                "id": requirement_id,
                "description": _COMMAND_DESCRIPTIONS[requirement_id],
                "invocation_template": (
                    _INVOCATION_TEMPLATE
                    + (" --run-hostile" if not safe_to_run else "")
                ),
                "parameter_names": parameter_names,
                "safe_to_run_by_default": safe_to_run,
                "required_environment_names": (
                    ["configured_credential_environment"] if not safe_to_run else []
                ),
            }
        )
    return commands


def compute_manifest_evidence_id(manifest: Mapping[str, object]) -> str:
    """Hash canonical sanitized content while excluding its identity field."""

    payload = dict(manifest)
    payload.pop("evidence_id", None)
    return sha256_text(_canonical_json(payload))


def validate_manifest(manifest: Mapping[str, object]) -> None:
    """Validate the closed characterization schema and integrity digest."""

    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError("manifest fields do not match the v1 schema")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported evidence schema_version")
    if manifest.get("evidence_id") != compute_manifest_evidence_id(manifest):
        raise ValueError("manifest evidence_id does not match sanitized content")
    if manifest.get("outcome") not in {"draft_only", "unsupported"}:
        raise ValueError("repository characterization cannot be certified")
    if not _COMMIT_PATTERN.fullmatch(str(manifest.get("source_commit") or "")):
        raise ValueError("manifest source_commit is invalid")
    if not _SHA256_PATTERN.fullmatch(
        str(manifest.get("deployment_class_id") or "")
    ):
        raise ValueError("manifest deployment_class_id is invalid")

    reasons = manifest.get("reason_codes")
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or reason not in CERTIFICATION_REASON_CODES
        for reason in reasons
    ):
        raise ValueError("manifest reason_codes are invalid")

    requirements = manifest.get("requirements")
    if not isinstance(requirements, list):
        raise ValueError("manifest requirements must be a list")
    if [item.get("requirement_id") for item in requirements] != list(
        REQUIRED_EVIDENCE_DOMAINS
    ):
        raise ValueError("manifest requirements are incomplete or out of order")
    for item in requirements:
        if not isinstance(item, dict) or set(item) != _REQUIREMENT_KEYS:
            raise ValueError("manifest requirement fields are invalid")
        if item.get("verification") != "repository_characterization":
            raise ValueError("manifest requirement is not repository characterization")
        if item.get("state") != "missing":
            raise ValueError("current characterization requirement must remain missing")
        if item.get("subject_id") != manifest.get("deployment_class_id"):
            raise ValueError("manifest requirement subject mismatch")
        if not _SHA256_PATTERN.fullmatch(str(item.get("evidence_sha256") or "")):
            raise ValueError("manifest requirement evidence digest is invalid")

    commands = manifest.get("commands")
    if not isinstance(commands, list):
        raise ValueError("manifest commands must be a list")
    if [item.get("id") for item in commands] != list(REQUIRED_EVIDENCE_DOMAINS):
        raise ValueError("manifest commands are incomplete or out of order")
    if any(not isinstance(item, dict) or set(item) != _COMMAND_KEYS for item in commands):
        raise ValueError("manifest command fields are invalid")
    if commands != _command_manifest():
        raise ValueError("manifest command metadata does not match the v1 schema")


def build_evidence_manifest(
    inputs: CertificationInputs,
    *,
    now: datetime,
    temporary_directory: Path,
    sentinels: CharacterizationSentinels | None = None,
) -> dict[str, object]:
    """Build one exact, sanitized repository-characterization manifest."""

    observed_at = _aware_utc(now)
    valid_until = observed_at + VALIDITY_WINDOW
    subject = inputs.deployment_class()
    active_sentinels = sentinels or _random_sentinels()
    temporary_directory.mkdir(parents=True, exist_ok=True)
    facts = characterize_current_primitives(
        database_path=temporary_directory / "scheduled-agent-characterization.db",
        prompt_sentinel=active_sentinels.prompt,
    )
    sentinel_hashes = {
        name: sha256_text(value)
        for name, value in vars(active_sentinels).items()
    }
    requirements = [
        _requirement_record(
            requirement_id=requirement_id,
            subject_id=subject.deployment_class_id,
            observed_at=observed_at,
            valid_until=valid_until,
            facts=facts[requirement_id],
            sentinel_hashes=sentinel_hashes,
        )
        for requirement_id in REQUIRED_EVIDENCE_DOMAINS
    ]
    domain_evidence = tuple(
        RequirementEvidence(
            requirement_id=str(item["requirement_id"]),
            state="missing",
            verification="repository_characterization",
            subject_id=str(item["subject_id"]),
            observed_at=observed_at,
            valid_until=valid_until,
            evidence_sha256=str(item["evidence_sha256"]),
        )
        for item in requirements
    )
    certification = evaluate_execution_certification(
        subject,
        domain_evidence,
        None,
        runtime_eligibility=_runtime_eligibility(inputs.runtime),
        now=observed_at,
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": "",
        "deployment_class_id": subject.deployment_class_id,
        "source_commit": inputs.source_commit,
        "created_at": observed_at.isoformat(),
        "valid_until": valid_until.isoformat(),
        "outcome": certification.outcome,
        "reason_codes": list(certification.reason_codes),
        "requirements": requirements,
        "commands": _command_manifest(),
    }
    manifest["evidence_id"] = compute_manifest_evidence_id(manifest)
    validate_manifest(manifest)
    return manifest


def render_manifest_json(manifest: Mapping[str, object]) -> str:
    """Render stable, human-diffable JSON."""

    validate_manifest(manifest)
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _static_runtime_eligibility_rows() -> list[tuple[str, str, str]]:
    """Derive the appendix rows from typed runtime metadata."""

    rows: list[tuple[str, str, str]] = []
    for runtime in STATIC_RUNTIME_VALUES:
        eligibility = _runtime_eligibility(runtime)
        if eligibility.untrusted_eligible and eligibility.strict_deny_all:
            rows.append(
                (
                    runtime,
                    "draft_only",
                    "required_server_verified_evidence_incomplete",
                )
            )
            continue
        reasons: list[str] = []
        if not eligibility.untrusted_eligible:
            reasons.append("runtime_not_untrusted_eligible")
        if not eligibility.strict_deny_all:
            reasons.append("runtime_strict_deny_all_unavailable")
        rows.append((runtime, "unsupported", ", ".join(reasons)))
    return rows


def render_manifest_markdown(manifest: Mapping[str, object]) -> str:
    """Render a bounded Markdown summary of the same manifest."""

    validate_manifest(manifest)
    reasons = manifest["reason_codes"]
    requirements = manifest["requirements"]
    if not isinstance(reasons, list) or not isinstance(requirements, list):
        raise ValueError("validated manifest collections are invalid")
    lines = [
        "# Scheduled Agent Execution Feasibility Evidence",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Evidence ID | `{manifest['evidence_id']}` |",
        f"| Deployment class | `{manifest['deployment_class_id']}` |",
        f"| Source commit | `{manifest['source_commit']}` |",
        f"| Created | `{manifest['created_at']}` |",
        f"| Valid until | `{manifest['valid_until']}` |",
        f"| Outcome | `{manifest['outcome']}` |",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- `{reason}`" for reason in reasons)
    lines.extend(
        [
            "",
            "## Requirements",
            "",
            "| Requirement | State | Verification | Evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        "| `{requirement_id}` | `{state}` | `{verification}` | "
        "`{evidence_sha256}` |".format(**item)
        for item in requirements
    )
    lines.extend(
        [
            "",
            "## Repository-Static Runtime Eligibility",
            "",
            "This appendix is derived from typed runtime metadata. It is not "
            "deployment certification evidence.",
            "",
            "| Runtime | Default outcome | Primary reason |",
            "| --- | --- | --- |",
        ]
    )
    lines.extend(
        f"| `{runtime}` | `{outcome}` | `{reason}` |"
        for runtime, outcome, reason in _static_runtime_eligibility_rows()
    )
    lines.extend(
        [
            "",
            "Repository characterization is not deployment certification. Host-gated "
            "raw evidence is retained outside the repository.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    """Write one validated artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_artifacts(
    manifest: Mapping[str, object],
    *,
    json_path: Path | None,
    markdown_path: Path | None,
) -> None:
    """Validate then atomically replace each requested sanitized artifact."""

    validate_manifest(manifest)
    json_content = render_manifest_json(manifest)
    markdown_content = render_manifest_markdown(manifest)
    if json_path is not None:
        _atomic_write(json_path, json_content)
    if markdown_path is not None:
        _atomic_write(markdown_path, markdown_content)


def validate_artifact_pair(json_path: Path, markdown_path: Path) -> None:
    """Verify integrity and identity parity across JSON and Markdown artifacts."""

    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("JSON evidence must contain one manifest object")
    validate_manifest(manifest)
    markdown = markdown_path.read_text(encoding="utf-8")
    if markdown != render_manifest_markdown(manifest):
        raise ValueError("Markdown artifact does not match JSON evidence")


@dataclass(frozen=True)
class HostileProbeRequest:
    """Prerequisites for a future attested hostile boundary run."""

    deployment_class_id: str
    evidence_dir: Path | None
    server_url: str | None
    api_key_environment_name: str | None
    attestation_reference: str | None


@dataclass(frozen=True)
class HostileProbeAdmission:
    """Bounded fail-closed admission result without launch details."""

    allowed: bool
    reason_code: str


def _local_server_url(value: str) -> bool:
    """Return whether a URL targets an explicit loopback HTTP server."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        and parsed.username is None
        and parsed.password is None
    )


def evaluate_hostile_probe_admission(
    request: HostileProbeRequest,
    *,
    environment: Mapping[str, str],
) -> HostileProbeAdmission:
    """Refuse hostile launch until every prerequisite is server-verified."""

    if request.evidence_dir is None:
        return HostileProbeAdmission(False, "hostile_probe_blocked_evidence_dir_missing")
    if not request.evidence_dir.is_dir() or request.evidence_dir.is_symlink():
        return HostileProbeAdmission(
            False,
            "hostile_probe_blocked_evidence_dir_unavailable",
        )
    if not request.server_url:
        return HostileProbeAdmission(False, "hostile_probe_blocked_server_url_missing")
    if not _local_server_url(request.server_url):
        return HostileProbeAdmission(False, "hostile_probe_blocked_nonlocal_server")
    environment_name = str(request.api_key_environment_name or "")
    if not environment_name:
        return HostileProbeAdmission(
            False,
            "hostile_probe_blocked_api_key_environment_missing",
        )
    if not _ENVIRONMENT_NAME_PATTERN.fullmatch(environment_name):
        return HostileProbeAdmission(
            False,
            "hostile_probe_blocked_api_key_environment_invalid",
        )
    if not str(environment.get(environment_name) or ""):
        return HostileProbeAdmission(
            False,
            "hostile_probe_blocked_api_key_unavailable",
        )
    if not request.attestation_reference:
        return HostileProbeAdmission(
            False,
            "hostile_probe_blocked_attestation_reference_missing",
        )
    if not _SHA256_PATTERN.fullmatch(request.attestation_reference):
        return HostileProbeAdmission(
            False,
            "hostile_probe_blocked_attestation_reference_invalid",
        )
    if not _SHA256_PATTERN.fullmatch(request.deployment_class_id):
        return HostileProbeAdmission(
            False,
            "hostile_probe_blocked_deployment_class_invalid",
        )
    return HostileProbeAdmission(
        False,
        "hostile_probe_blocked_server_attestation_verifier_unimplemented",
    )


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser for generation and validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    for option in _COMMON_PARAMETER_NAMES:
        parser.add_argument(f"--{option}")
    parser.add_argument("--format", choices=("json", "markdown"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--repository-characterization-only", action="store_true")
    parser.add_argument("--run-hostile", action="store_true")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--server-url")
    parser.add_argument("--api-key-env-name")
    parser.add_argument("--attestation-reference")
    parser.add_argument(
        "--validate-artifacts",
        nargs=2,
        type=Path,
        metavar=("JSON", "MARKDOWN"),
    )
    return parser


def _inputs_from_args(args: argparse.Namespace) -> CertificationInputs:
    """Build validated certification inputs from parsed CLI arguments."""

    missing = [
        name
        for name in _COMMON_PARAMETER_NAMES
        if not getattr(args, name.replace("-", "_"), None)
    ]
    if missing:
        raise ValueError("missing required parameters: " + ", ".join(missing))
    return CertificationInputs(
        host_os=args.host_os,
        host_arch=args.host_arch,
        runtime=args.runtime,
        auth_mode=args.auth_mode,
        adapter_id=args.adapter_id,
        adapter_version=args.adapter_version,
        source_commit=args.source_commit,
        server_build_sha=args.server_build_sha,
        image_digest=args.image_digest,
        mount_policy_hash=args.mount_policy_hash,
        egress_policy_hash=args.egress_policy_hash,
        credential_policy_hash=args.credential_policy_hash,
        tenant_boundary_policy_hash=args.tenant_boundary_policy_hash,
        mediation_policy_hash=args.mediation_policy_hash,
        isolation_profile_version=args.isolation_profile_version,
    )


def _validate_observed_host(inputs: CertificationInputs) -> None:
    """Reject host identity claims that differ from local observation."""

    observed_os = platform.system().lower()
    observed_arch = platform.machine().lower()
    if inputs.host_os != observed_os or inputs.host_arch != observed_arch:
        raise ValueError(
            "claimed host does not match the observed host; use "
            "--repository-characterization-only for a non-host claim"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run characterization, validation, or fail-closed hostile admission."""

    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.validate_artifacts:
            validate_artifact_pair(*args.validate_artifacts)
            print("Artifacts valid.")
            return 0

        inputs = _inputs_from_args(args)
        if not args.repository_characterization_only:
            _validate_observed_host(inputs)
        with tempfile.TemporaryDirectory(prefix="scheduled-agent-certification-") as temp:
            manifest = build_evidence_manifest(
                inputs,
                now=datetime.now(timezone.utc),
                temporary_directory=Path(temp),
            )

        if args.run_hostile:
            admission = evaluate_hostile_probe_admission(
                HostileProbeRequest(
                    deployment_class_id=str(manifest["deployment_class_id"]),
                    evidence_dir=args.evidence_dir,
                    server_url=args.server_url,
                    api_key_environment_name=args.api_key_env_name,
                    attestation_reference=args.attestation_reference,
                ),
                environment=os.environ,
            )
            if not admission.allowed:
                print(admission.reason_code, file=sys.stderr)
                return 2

        write_artifacts(
            manifest,
            json_path=args.output_json,
            markdown_path=args.output_markdown,
        )
        output_format = args.format
        if output_format is None and not args.output_json and not args.output_markdown:
            output_format = "json"
        if output_format == "json":
            print(render_manifest_json(manifest), end="")
        elif output_format == "markdown":
            print(render_manifest_markdown(manifest), end="")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Scheduled Agent characterization failed: {}", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
