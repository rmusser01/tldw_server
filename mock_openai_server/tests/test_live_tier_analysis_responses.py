"""Regression coverage for deterministic live-tier media analysis responses."""

import json
from pathlib import Path

from mock_openai_server.mock_openai.config import MockConfig


def test_live_tier_analysis_prompts_select_distinct_static_responses() -> None:
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
