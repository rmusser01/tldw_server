"""Guard required prototype workspace operational documentation."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_doc(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").lower()


def test_operator_runbook_documents_support_fields() -> None:
    """Risk Gate 7 operator docs must expose support and diagnosis fields."""
    runbook = _read_doc("Docs/Operations/Prototype_Workspaces_Runbook.md")

    required_terms = [
        "runtime_status",
        "preview_status",
        "preview_health",
        "canonical_preview_status",
        "publish_validation_status",
        "promotion request",
        "job_id",
        "job_type",
        "idempotency_key",
        "category",
        "frontend_state",
        "retryable",
        "signing secret",
        "quota",
        "audit",
    ]

    for term in required_terms:
        assert term in runbook


def test_user_guide_documents_required_failure_examples() -> None:
    """Risk Gate 7 user docs must cover owner and collaborator edge cases."""
    user_guide = _read_doc("Docs/User_Guides/Prototype_Workspaces.md")

    required_examples = [
        "password-protected",
        "single-use",
        "resume cookie",
        "revoked link",
        "archived workspace",
        "exhausted link",
        "promotion conflict",
        "validation failure",
    ]

    for example in required_examples:
        assert example in user_guide


def test_api_and_contract_docs_link_gate_7_artifacts() -> None:
    """Existing prototype API docs should point support readers to Gate 7 docs."""
    api_doc = _read_doc("Docs/API-related/Prototype_Workspaces_API.md")
    contract_doc = _read_doc("Docs/API-related/Prototype_Workspaces_Contract_Matrix.md")

    assert "prototype_workspaces_runbook.md" in api_doc
    assert "prototype_workspaces.md" in api_doc
    assert "operational support fields" in contract_doc
