import pytest

from tldw_Server_API.tests.lint._moved_endpoint_code import REPO_ROOT, moved_files
from Helper_Scripts.ci.path_classifier import classify_paths


def test_ui_only_change_disables_backend_and_coverage() -> None:
    flags = classify_paths(
        [
            "apps/tldw-frontend/src/app/page.tsx",
            "apps/packages/ui/src/components/Option/Playground/Foo.tsx",
        ]
    )
    assert flags["backend_changed"] is False
    assert flags["coverage_required"] is False
    assert flags["frontend_changed"] is True
    assert flags["tldw_frontend_changed"] is True
    assert flags["admin_ui_changed"] is False
    assert flags["e2e_changed"] is True
    assert flags["family_guardrails_changed"] is False


def test_family_guardrails_change_enables_targeted_family_gate() -> None:
    flags = classify_paths(
        [
            "apps/tldw-frontend/e2e/workflows/family-guardrails-wizard.spec.ts",
            "apps/packages/ui/src/components/Option/Settings/FamilyGuardrailsWizard.tsx",
        ]
    )
    assert flags["frontend_changed"] is True
    assert flags["tldw_frontend_changed"] is True
    assert flags["e2e_changed"] is True
    assert flags["family_guardrails_changed"] is True


def test_admin_ui_change_enables_frontend_without_e2e() -> None:
    flags = classify_paths(
        [
            "admin-ui/app/monitoring/page.tsx",
            "admin-ui/lib/api-client.ts",
        ]
    )
    assert flags["frontend_changed"] is True
    assert flags["admin_ui_changed"] is True
    assert flags["tldw_frontend_changed"] is False
    assert flags["e2e_changed"] is False


@pytest.mark.parametrize(
    ("path", "backend_changed", "tldw_frontend_changed", "admin_ui_changed"),
    (
        ("apps/packages/api-types/paths.d.ts", True, True, False),
        ("apps/packages/api-types/package.json", True, True, False),
        ("apps/packages/api-types/scripts/generate-api-types.mjs", True, True, False),
        ("Helper_Scripts/export_openapi_schema.py", True, False, False),
        (
            "Helper_Scripts/ci/vitest_base_ratchet.py",
            True,
            True,
            True,
        ),
        (
            "Helper_Scripts/ci/vitest_deterministic.config.ts",
            False,
            True,
            False,
        ),
        (
            "Helper_Scripts/ci/vitest_execution_order_reporter.mjs",
            False,
            True,
            True,
        ),
        (
            "Helper_Scripts/ci/emit_ci_gate_flags.py",
            True,
            False,
            False,
        ),
        (
            "Helper_Scripts/ci/path_classifier.py",
            True,
            False,
            False,
        ),
        (
            "admin-ui/scripts/ci/vitest-safety-reporter.mjs",
            False,
            False,
            True,
        ),
    ),
)
@pytest.mark.unit
def test_shared_contract_and_runner_paths_select_every_consuming_gate(
    path: str,
    backend_changed: bool,
    tldw_frontend_changed: bool,
    admin_ui_changed: bool,
) -> None:
    flags = classify_paths([path])
    assert flags["backend_changed"] is backend_changed
    assert flags["frontend_changed"] is (
        tldw_frontend_changed or admin_ui_changed
    )
    assert flags["tldw_frontend_changed"] is tldw_frontend_changed
    assert flags["admin_ui_changed"] is admin_ui_changed
    assert flags["e2e_changed"] is tldw_frontend_changed


def test_api_schema_change_enables_e2e() -> None:
    flags = classify_paths(
        [
            "tldw_Server_API/app/api/v1/endpoints/chat.py",
            "tldw_Server_API/app/api/v1/schemas/chat.py",
        ]
    )
    assert flags["backend_changed"] is True
    assert flags["coverage_required"] is True
    assert flags["e2e_changed"] is True


def test_moved_endpoint_code_still_enables_e2e() -> None:
    for path in (
        "tldw_Server_API/app/api/v1/utils/chatops/ingress.py",
        "tldw_Server_API/app/core/Sandbox/default_service.py",
    ):
        assert classify_paths([path])["e2e_changed"] is True


def test_every_moved_endpoint_file_still_enables_e2e() -> None:
    """A file moved out of endpoints/ must keep the required E2E trigger (TASK-13950.20)."""
    paths = {p.relative_to(REPO_ROOT).as_posix() for p in moved_files()}
    paths.add("tldw_Server_API/app/core/Prompt_Management/prompt_studio/ws_connections.py")
    missing = sorted(p for p in paths if not classify_paths([p])["e2e_changed"])
    assert missing == []


def test_backlog_py_change_runs_backend_gate_without_coverage() -> None:
    """The task editor's tests and the backlog task-format ratchet run in backend-required."""
    flags = classify_paths(["tools/backlog-py/src/backlog_py/markdown/task_parser.py"])
    assert flags["backend_changed"] is True
    assert flags["coverage_required"] is False


def test_merge_queue_implementation_only_change_runs_backend_safety_suites() -> None:
    """Queue code alone must run the backend gate that owns its safety regressions."""
    flags = classify_paths(["Helper_Scripts/ci/merge_queue.py"])
    assert flags["backend_changed"] is True  # nosec B101 - pytest regression assertion.
    assert flags["security_relevant_changed"] is True  # nosec B101 - pytest regression assertion.
    assert flags["coverage_required"] is False  # nosec B101 - pytest regression assertion.


def test_workflow_only_change_keeps_backend_gate_but_skips_coverage() -> None:
    flags = classify_paths(
        [
            ".github/workflows/e2e-required.yml",
        ]
    )
    assert flags["backend_changed"] is True
    assert flags["coverage_required"] is False


def test_emitter_writes_github_output(tmp_path, monkeypatch) -> None:
    out = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))

    from Helper_Scripts.ci.emit_ci_gate_flags import emit

    emit(["apps/tldw-frontend/src/app/page.tsx"])
    text = out.read_text(encoding="utf-8")
    assert "frontend_changed=true" in text
    assert "backend_changed=false" in text


@pytest.mark.parametrize(
    "path",
    (
        "tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py",
        "tldw_Server_API/app/core/DB_Management/chacha/message_store.py",
        "tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py",
        "tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py",
        "tldw_Server_API/app/core/DB_Management/scope_context.py",
        "tldw_Server_API/app/core/DB_Management/backends/factory.py",
        "tldw_Server_API/app/core/DB_Management/backends/base.py",
        "tldw_Server_API/app/core/WebClipper/service.py",
        "tldw_Server_API/app/core/Character_Chat/chacha_db_provider.py",
        "tldw_Server_API/tests/ChaChaNotesDB/conftest.py",
        "tldw_Server_API/tests/_plugins/postgres.py",
        "tldw_Server_API/tests/DB_Management/test_core_chat_owner_isolation.py",
        ".github/workflows/backend-required.yml",
        "Helper_Scripts/ci/path_classifier.py",
        "pyproject.toml",
        "uv.lock",
    ),
)
def test_chacha_owner_scope_paths_enable_the_owner_isolation_job(path: str) -> None:
    assert classify_paths([path])["chacha_owner_scope_changed"] is True


def test_unrelated_backend_change_leaves_the_owner_isolation_job_off() -> None:
    flags = classify_paths(["tldw_Server_API/app/core/RAG/pipeline.py"])
    assert flags["backend_changed"] is True
    assert flags["chacha_owner_scope_changed"] is False


@pytest.mark.parametrize(
    "path",
    (
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py",
        "tldw_Server_API/app/core/RAG/rag_schemas_unified.py",
        "tldw_Server_API/app/api/v1/endpoints/rag_unified.py",
        "tldw_Server_API/app/api/v1/schemas/rag_schemas_simple.py",
        "tldw_Server_API/tests/RAG_NEW/unit/test_hyde.py",
        "tldw_Server_API/tests/RAG/test_unified_schema_and_pipeline.py",
        "tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py",
        "pyproject.toml",
        ".github/workflows/backend-required.yml",
    ),
)
def test_rag_pipeline_paths_enable_the_rag_job(path: str) -> None:
    assert classify_paths([path])["rag_pipeline_changed"] is True


def test_unrelated_backend_change_leaves_the_rag_job_off() -> None:
    flags = classify_paths(["tldw_Server_API/app/core/Chat/chat_service.py"])
    assert flags["backend_changed"] is True
    assert flags["rag_pipeline_changed"] is False
