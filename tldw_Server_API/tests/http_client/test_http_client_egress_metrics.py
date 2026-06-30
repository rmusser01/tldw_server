import pytest


pytestmark = pytest.mark.unit


def _has_httpx():
    try:
        import httpx  # noqa: F401
        return True
    except Exception:
        return False


requires_httpx = pytest.mark.skipif(not _has_httpx(), reason="httpx not installed")


@requires_httpx
def test_egress_denial_increments_metric(monkeypatch):
    from tldw_Server_API.app.core.http_client import fetch_json
    from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
    from tldw_Server_API.app.core.exceptions import EgressPolicyError

    # Ensure private IP blocking is enabled for this egress denial test
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")

    reg = get_metrics_registry()
    before_total = reg.get_metric_stats("http_client_egress_denials_total").get("sum", 0) or 0

    with pytest.raises(EgressPolicyError) as ei:
        # 127.0.0.1 should be denied by default egress policy
        fetch_json(method="GET", url="http://127.0.0.1/")

    # Error should be clear
    msg = str(ei.value).lower()
    assert any(kw in msg for kw in ("egress", "not allowed", "private", "allowlist"))

    after_total = reg.get_metric_stats("http_client_egress_denials_total").get("sum", 0) or 0
    assert after_total >= before_total + 1
