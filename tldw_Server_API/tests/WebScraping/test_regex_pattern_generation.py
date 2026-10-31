import dataclasses
import json

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import schema as schema_strategy


def test_generate_regex_pattern_from_llm(monkeypatch):
    html = "<html><body>Order #12345 confirmed.</body></html>"
    payload = json.dumps({"pattern": r"Order\s+#(\d+)", "flags": "i", "group": 1})

    def _fake_call(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": payload,
                    }
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
            "model": "gpt-test",
        }

    dependencies = dataclasses.replace(
        build_default_dependencies(),
        perform_chat_api_call=_fake_call,
    )
    monkeypatch.setattr(schema_strategy, "build_default_dependencies", lambda: dependencies)

    result = ael.generate_regex_pattern_from_llm(
        html,
        "https://example.com",
        label="order_id",
        query="Find order IDs",
        llm_settings={"provider": "openai"},
    )

    assert result["success"] is True
    assert result["pattern"] == r"Order\s+#(\d+)"
    assert result.get("sample_match") == "12345"
