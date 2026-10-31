import dataclasses

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import schema as schema_strategy


def test_generate_schema_rules_from_llm_parses_schema(monkeypatch):
    html = """
    <html>
      <body>
        <article>
          <h1>Title</h1>
          <div class="content">Body</div>
        </article>
      </body>
    </html>
    """

    def _fake_call(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "```json\n"
                            '{"schema": {"name": "Article", "baseSelector": "//article",'
                            ' "fields": ['
                            '{"name": "title", "selector": "//article/h1", "type": "text"},'
                            '{"name": "content", "selector": "//article/div[@class=\'content\']", "type": "text"}'
                            "]}}\n"
                            "```"
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17},
            "model": "gpt-test",
        }

    dependencies = dataclasses.replace(
        build_default_dependencies(),
        perform_chat_api_call=_fake_call,
    )
    monkeypatch.setattr(schema_strategy, "build_default_dependencies", lambda: dependencies)

    result = ael.generate_schema_rules_from_llm(
        html,
        "https://example.com",
        llm_settings={"provider": "openai"},
        query="Extract article title and body",
    )

    assert result["success"] is True
    schema = result.get("schema_rules") or {}
    assert schema.get("baseSelector") == "//article"
    assert schema.get("fields", [])[0].get("name") == "title"
    assert result.get("schema_validation", {}).get("errors") == []


def test_generate_schema_rules_uses_unknown_for_missing_response_model(monkeypatch):
    def _fake_call(**_kwargs):
        return {
            "choices": [{"message": {"content": '{"schema": {"fields": []}}'}}],
            "usage": {},
        }

    dependencies = dataclasses.replace(
        build_default_dependencies(),
        perform_chat_api_call=_fake_call,
    )
    monkeypatch.setattr(schema_strategy, "build_default_dependencies", lambda: dependencies)

    result = ael.generate_schema_rules_from_llm(
        "<html><body>Body</body></html>",
        "https://example.com",
        llm_settings={"provider": "openai"},
    )

    assert result["success"] is True
    assert result["llm_model"] == "unknown"
