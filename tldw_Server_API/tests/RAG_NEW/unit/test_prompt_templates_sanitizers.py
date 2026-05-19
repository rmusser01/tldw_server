"""Sanitizer coverage for RAG prompt template logs."""

import pytest

from tldw_Server_API.app.core.RAG.rag_service import prompt_templates
from tldw_Server_API.app.core.RAG.rag_service.prompt_templates import (
    PromptTemplate,
    TemplateType,
)


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message, *args, **kwargs):
        _ = (args, kwargs)
        self.warnings.append(str(message))


def test_missing_variable_warning_omits_raw_key_error(monkeypatch):
    logger_stub = _LoggerStub()
    template_text = "Context: {super_secret_token_abc123}"
    template = PromptTemplate(
        name="rag_template",
        template=template_text,
        type=TemplateType.FULL,
    )

    monkeypatch.setattr(prompt_templates, "logger", logger_stub)

    result = template.format()

    assert result == template_text
    assert logger_stub.warnings == ["Missing variable in template 'rag_template'"]
    joined = "\n".join(logger_stub.warnings)
    assert "super_secret_token_abc123" not in joined
