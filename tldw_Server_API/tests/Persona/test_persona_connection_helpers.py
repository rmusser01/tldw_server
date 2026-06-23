import pytest

from tldw_Server_API.app.core.Persona import connections as persona_connections


pytestmark = pytest.mark.unit


def test_render_template_value_logs_warning_for_invalid_format(monkeypatch: pytest.MonkeyPatch):
    warnings: list[str] = []

    monkeypatch.setattr(
        persona_connections.logger,
        "warning",
        lambda message, *args: warnings.append(str(message).format(*args)),
    )

    rendered = persona_connections.render_template_value(
        "Bearer sk-live-literal {secret",
        {"secret": "token"},
    )

    assert rendered == "Bearer sk-live-literal {secret"
    assert warnings
    assert "failed to render persona connection template" in warnings[0].lower()
    assert "sk-live-literal" not in warnings[0]
