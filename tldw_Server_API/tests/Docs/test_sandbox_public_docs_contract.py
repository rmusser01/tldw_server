"""Contracts for public sandbox runtime support documentation."""

from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.schemas.sandbox_schemas import SandboxRuntimeInfo


REPO_ROOT = Path(__file__).resolve().parents[3]

PUBLIC_API_DOCS = (
    REPO_ROOT / "Docs/API-related/Sandbox_API.md",
    REPO_ROOT / "Docs/Published/API-related/Sandbox_API.md",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized_text(path: Path) -> str:
    return " ".join(_text(path).split())


def _require(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


@pytest.mark.parametrize("doc_path", PUBLIC_API_DOCS)
def test_public_sandbox_api_docs_reference_runtime_contract_docs(doc_path: Path) -> None:
    """Public API guides should point callers at the current runtime contracts."""
    text = _text(doc_path)

    for contract_path in (
        "Docs/Sandbox/sandbox-runtime-capability-inventory.md",
        "Docs/Sandbox/sandbox-security-policy-matrix.md",
    ):
        _require(
            contract_path in text,
            f"{doc_path} should reference {contract_path}",
        )


@pytest.mark.parametrize("doc_path", PUBLIC_API_DOCS)
def test_public_sandbox_api_docs_do_not_overclaim_runtime_support(doc_path: Path) -> None:
    """Public API guides should name current runtimes without overstating guarantees."""
    text = _text(doc_path)

    for runtime in (
        "docker",
        "firecracker",
        "lima",
        "vz_linux",
        "vz_macos",
        "seatbelt",
        "worktree",
    ):
        _require(runtime in text, f"{doc_path} should mention runtime {runtime}")

    for field_name in (
        "boundary_class",
        "vm_grade_isolation",
        "untrusted_eligible",
        "network_policy_contract",
        "normalized_reason_details",
        "session_contract",
    ):
        _require(
            field_name in text,
            f"{doc_path} should document runtime discovery field {field_name}",
        )

    for host_local_runtime in ("seatbelt", "worktree"):
        _require(
            f"`{host_local_runtime}` is host-local" in text,
            f"{doc_path} should describe {host_local_runtime} as host-local",
        )
        _require(
            f"`{host_local_runtime}` is not `untrusted`-eligible" in text,
            f"{doc_path} should not classify {host_local_runtime} as untrusted-eligible",
        )

    _require(
        "`vz_macos` real execution is not implemented" in text,
        f"{doc_path} should state vz_macos real execution is not implemented",
    )


def test_code_interpreter_prd_reconciles_current_runtime_status() -> None:
    """Historical product PRD should defer current runtime status to contract docs."""
    text = _normalized_text(
        REPO_ROOT / "Docs/Product/Sandbox/Code_Interpreter_Sandbox_PRD.md"
    )

    for snippet in (
        "Current Runtime Reconciliation",
        "Docs/Sandbox/sandbox-runtime-capability-inventory.md",
        "Docs/Sandbox/sandbox-security-policy-matrix.md",
        "`seatbelt` and `worktree` are host-local",
        "`seatbelt` and `worktree` are not `untrusted`-eligible",
        "`vz_macos` real execution is not implemented",
    ):
        _require(snippet in text, f"Sandbox PRD should include: {snippet}")


def test_sandbox_runtime_isolation_schema_fields_are_required() -> None:
    schema = SandboxRuntimeInfo.model_json_schema()
    required = set(schema.get("required", []))

    for field_name in (
        "boundary_class",
        "vm_grade_isolation",
        "untrusted_eligible",
    ):
        _require(
            field_name in required,
            f"SandboxRuntimeInfo should require {field_name}",
        )
        field_schema = schema["properties"][field_name]
        serialized = str(field_schema)
        _require(
            "'type': 'null'" not in serialized,
            f"SandboxRuntimeInfo field {field_name} should not be nullable",
        )


def test_sandbox_runtime_network_policy_schema_field_is_required() -> None:
    schema = SandboxRuntimeInfo.model_json_schema()
    required = set(schema.get("required", []))

    _require(
        "network_policy_contract" in required,
        "SandboxRuntimeInfo should require network_policy_contract",
    )
    field_schema = schema["properties"]["network_policy_contract"]
    serialized = str(field_schema)
    _require(
        "'type': 'null'" not in serialized,
        "SandboxRuntimeInfo field network_policy_contract should not be nullable",
    )


def test_sandbox_runtime_session_contract_schema_field_is_required() -> None:
    schema = SandboxRuntimeInfo.model_json_schema()
    required = set(schema.get("required", []))

    _require(
        "session_contract" in required,
        "SandboxRuntimeInfo should require session_contract",
    )
    field_schema = schema["properties"]["session_contract"]
    serialized = str(field_schema)
    _require(
        "'type': 'null'" not in serialized,
        "SandboxRuntimeInfo field session_contract should not be nullable",
    )


def test_sandbox_runtime_schema_exposes_reason_details() -> None:
    schema = SandboxRuntimeInfo.model_json_schema()

    _require(
        "normalized_reason_details" in schema["properties"],
        "SandboxRuntimeInfo should expose normalized_reason_details",
    )
    serialized = str(schema["properties"]["normalized_reason_details"])
    _require(
        "'type': 'null'" not in serialized,
        "SandboxRuntimeInfo field normalized_reason_details should not be nullable",
    )
