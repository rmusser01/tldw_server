from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Notes_Graph.suggestion_api import NotesGraphSuggestionsAPI
from tldw_Server_API.app.core.Notes_Graph.suggestion_provider import (
    ResolvedSuggestionProvider,
    resolve_generation_capability,
)
from tldw_Server_API.app.services import notes_graph_suggestions_worker

pytestmark = pytest.mark.unit


def test_worker_uses_the_shared_provider_capability_resolver() -> None:
    assert (
        notes_graph_suggestions_worker.resolve_generation_capability
        is resolve_generation_capability
    )


def test_api_preflight_and_admission_default_to_the_worker_shared_resolver() -> None:
    api = NotesGraphSuggestionsAPI(
        store=object(),
        owner_user_id="owner-1",
        dataset_id="dataset-1",
        admission_service=object(),
        cancellation_coordinator=object(),
        decision_service=object(),
        worker_ready=lambda: True,
        feature_ready=lambda: True,
        cursor_codec=object(),
    )

    assert api.capability_resolver is resolve_generation_capability
    assert (
        api.capability_resolver
        is notes_graph_suggestions_worker.resolve_generation_capability
    )


def test_shared_resolver_returns_one_typed_revision_bound_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Adapter:
        @staticmethod
        def capabilities() -> dict[str, bool]:
            return {"response_format": True}

    class Registry:
        @staticmethod
        def resolve_provider_name(provider: str) -> str:
            return provider.strip().lower()

        @staticmethod
        def get_adapter(provider: str):
            return Adapter() if provider == "openai" else None

    import tldw_Server_API.app.core.Notes_Graph.suggestion_provider as provider_module

    monkeypatch.setattr(provider_module.adapter_registry, "get_registry", lambda: Registry())
    monkeypatch.setattr(
        provider_module,
        "loaded_config_data",
        {"openai_api": {"api_key": "secret", "api_url": "https://example.test/v1"}},
    )
    monkeypatch.setattr(
        provider_module,
        "build_suggestion_capabilities",
        lambda contract: type(
            "Capabilities",
            (),
            {
                "provider": contract.adapter,
                "model": contract.model,
                "revision": "sha256:shared-revision",
                "generation_available": True,
            },
        )(),
    )

    resolved = resolve_generation_capability(provider="OpenAI", model="model-a")

    assert isinstance(resolved, ResolvedSuggestionProvider)
    assert resolved.capabilities.revision == "sha256:shared-revision"
    assert resolved.provider.adapter == "openai"
    assert resolved.provider.model == "model-a"
