from __future__ import annotations

from Helper_Scripts.ci.check_frontend_license_gate import blocked_changes, evaluate


def test_owner_changes_are_allowed() -> None:
    result = evaluate(
        author="rmusser01",
        owner="rmusser01",
        paths=["apps/extension/entrypoints/popup.tsx", "tldw_Server_API/app/main.py"],
    )

    assert result == []


def test_external_protected_and_governance_changes_are_blocked() -> None:
    paths = [
        "apps/packages/ui/src/index.ts",
        "admin-ui/package.json",
        "LICENSES/README.md",
        "THIRD_PARTY_NOTICES.txt",
    ]

    assert blocked_changes(paths) == paths


def test_external_api_declaration_changes_are_conservatively_blocked() -> None:
    paths = [
        "tldw_Server_API/app/main.py",
        "tldw_Server_API/app/api/v1/endpoints/chat.py",
        "tldw_Server_API/app/api/v1/schemas/chat.py",
    ]

    assert blocked_changes(paths) == paths


def test_external_unrelated_backend_and_docs_changes_remain_allowed() -> None:
    result = evaluate(
        author="contributor",
        owner="rmusser01",
        paths=[
            "tldw_Server_API/app/core/RAG/service.py",
            "Docs/Development/RAG.md",
        ],
    )

    assert result == []
