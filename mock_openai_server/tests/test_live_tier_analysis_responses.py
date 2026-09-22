"""Regression coverage for deterministic live-tier media analysis responses."""

import json
from pathlib import Path

import pytest

from mock_openai_server.mock_openai.config import MockConfig


@pytest.mark.unit
def test_live_tier_analysis_prompts_select_distinct_static_responses() -> None:
    """Live-tier marker prompts must select distinct committed response fixtures."""
    repo_root = Path(__file__).resolve().parents[2]
    config_root = (
        repo_root / "apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs"
    )

    for config_name, model in (
        ("local-success.json", "local-uat-chat"),
        ("hosted-success.json", "gpt-4.1-mini"),
    ):
        config_path = config_root / config_name
        config = MockConfig.from_file(config_path)
        response_config = config.responses["chat_completions"]

        first_file = response_config.find_matching_response(
            {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return exactly LIVE_TIER_ANALYSIS_ONE and nothing else.",
                    },
                    {"role": "user", "content": "media body"},
                ],
            }
        )
        second_file = response_config.find_matching_response(
            {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return exactly LIVE_TIER_ANALYSIS_TWO and nothing else.",
                    },
                    {"role": "user", "content": "media body"},
                ],
            }
        )

        assert first_file == "chat/analysis-one.json"
        assert second_file == "chat/analysis-two.json"
        assert first_file != second_file

        response_root = (config_path.parent / str(config.response_base_dir)).resolve()
        first = json.loads((response_root / first_file).read_text(encoding="utf8"))
        second = json.loads((response_root / second_file).read_text(encoding="utf8"))
        assert first["choices"][0]["message"]["content"] == "LIVE_TIER_ANALYSIS_ONE"
        assert second["choices"][0]["message"]["content"] == "LIVE_TIER_ANALYSIS_TWO"


@pytest.mark.unit
@pytest.mark.parametrize("system,content,expected", [
    ("Generate 5 flashcards.", "Generate flashcards from: DNA stands for deoxyribonucleic acid.", "ci-biology-cards"),
    ("Use the source", "Rowan Observatory: Mira Vale, Cedar Ridge, Friday 18:00. Who directs it?", "ci-rowan-facts"),
    ("Use the source", "What is the ticket price at Rowan Observatory?", "ci-rowan-price"),
    ("Use the source", "Who directs Rowan Observatory?", "default"),
    ("Generate 5 flashcards.", "An unrelated biology source.", "default"),
])
def test_ci_journey_provider_uses_only_declared_source_fixtures(system: str, content: str, expected: str) -> None:
    """The CI provider selects bounded fixtures and does not invent answers for missing context."""
    root = Path(__file__).resolve().parents[2] / "apps/tldw-frontend/e2e/onboarding-uat/mock-openai"
    config = MockConfig.from_file(root / "configs/ci-journeys.json")
    selected = config.responses["chat_completions"].find_matching_response({
        "model": "gpt-4.1-mini", "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": content},
        ],
    })
    assert selected == f"chat/{expected}.json"
    response = json.loads((root / "responses" / selected).read_text(encoding="utf-8"))
    assert response["choices"][0]["finish_reason"] == "stop"
    if expected == "ci-biology-cards":
        cards = json.loads(response["choices"][0]["message"]["content"])
        assert len(cards) == 5
        assert all(card["front"] and card["back"] for card in cards)


@pytest.mark.unit
def test_ci_provider_accepts_workflow_credentials_and_requires_authenticated_readiness() -> None:
    """The actual provider accepts the backend credential and the readiness probe uses it."""
    import yaml
    from fastapi.testclient import TestClient

    from mock_openai_server.mock_openai.server import app, get_config_instance

    root = Path(__file__).resolve().parents[2]
    critical = yaml.safe_load((root / ".github/workflows/frontend-e2e-tiers.yml").read_text())["jobs"]["critical"]
    config = MockConfig.from_file(root / "apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/ci-journeys.json")
    app.dependency_overrides[get_config_instance] = lambda: config
    try:
        with TestClient(app) as client:
            assert client.get("/v1/models").status_code == 401
            response = client.get("/v1/models", headers={"Authorization": f"Bearer {critical['env']['OPENAI_API_KEY']}"})
            assert response.status_code == 200
            assert critical["env"]["TLDW_UAT390_MODEL"] in [model["id"] for model in response.json()["data"]]
    finally:
        app.dependency_overrides.pop(get_config_instance, None)
    readiness = next(step["run"] for step in critical["steps"] if step.get("name") == "Start deterministic downstream provider")
    assert '-H "Authorization: Bearer $OPENAI_API_KEY"' in readiness


@pytest.mark.unit
def test_ci_provider_exposes_explicit_configuration_failure_without_breaking_success() -> None:
    """CI must exercise backend error recovery without fabricating application responses."""
    import yaml
    from fastapi.testclient import TestClient

    from mock_openai_server.mock_openai.server import app, get_config_instance

    root = Path(__file__).resolve().parents[2]
    critical = yaml.safe_load((root / ".github/workflows/frontend-e2e-tiers.yml").read_text())["jobs"]["critical"]
    config = MockConfig.from_file(root / "apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/ci-journeys.json")
    app.dependency_overrides[get_config_instance] = lambda: config
    try:
        with TestClient(app) as client:
            headers = {"Authorization": f"Bearer {critical['env']['OPENAI_API_KEY']}"}
            model = critical["env"]["TLDW_UAT390_MODEL"]
            payload = {"model": model, "messages": [{"role": "user", "content": "Trigger the real provider configuration failure."}]}
            response = client.post("/v1/chat/completions", headers=headers, json=payload)
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "model_not_found"
            payload["messages"][0]["content"] = "Reply with one short sentence for Phase 7."
            assert client.post("/v1/chat/completions", headers=headers, json=payload).status_code == 200
            assert critical["env"]["TLDW_E2E_CHARACTER_PROVIDER_FAILURE_MODEL"] == f"tldw:openai:{model}"
    finally:
        app.dependency_overrides.pop(get_config_instance, None)
