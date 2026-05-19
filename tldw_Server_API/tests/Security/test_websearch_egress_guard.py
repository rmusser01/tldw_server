import types
import pytest


def test_brave_search_blocked_by_egress(monkeypatch):


    """Unit-test that WebSearch honors centralized egress policy (denied)."""
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    # Stub evaluate_url_policy to deny
    from tldw_Server_API.app.core.Security import egress as eg
    pol = types.SimpleNamespace(allowed=False, reason="deny_test")
    monkeypatch.setattr(eg, 'evaluate_url_policy', lambda url: pol)

    # Patch requests usage via session to avoid real network
    with pytest.raises(ValueError, match="Blocked by outbound policy: deny_test"):
        ws.search_web_brave("query", "US", "en", "en", 5, brave_api_key="k", result_filter="webpages", search_type="ai")
