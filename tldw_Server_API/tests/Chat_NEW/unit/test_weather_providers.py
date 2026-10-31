import httpx
import pytest

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.Integrations import weather_providers


@pytest.fixture
def reject_direct_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep RED tests local if production still tries its legacy client."""

    def fail_direct_client(*_args: object, **_kwargs: object) -> None:
        """Fail if weather code bypasses the central HTTP client."""
        pytest.fail("weather requests must use the central HTTP client")

    monkeypatch.setattr(
        weather_providers,
        "http_client_factory",
        fail_direct_client,
        raising=False,
    )


@pytest.mark.unit
def test_get_weather_client_falls_back_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the no-key provider when OpenWeather credentials are absent."""
    monkeypatch.setenv("WEATHER_PROVIDER", "openweather")
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    client = weather_providers.get_weather_client()
    assert isinstance(client, weather_providers.NoKeyWeatherClient)


@pytest.mark.unit
def test_get_weather_client_openweather_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build the configured OpenWeather client from environment settings."""
    monkeypatch.setenv("WEATHER_PROVIDER", "openweather")
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-key")
    monkeypatch.setenv("WEATHER_UNITS", "imperial")
    monkeypatch.setenv("WEATHER_LANG", "es")
    monkeypatch.setenv("WEATHER_TIMEOUT_MS", "2500")
    client = weather_providers.get_weather_client()
    assert isinstance(client, weather_providers.OpenWeatherClient)
    assert client.units == "imperial"
    assert client.lang == "es"
    assert abs(client.timeout_seconds - 2.5) < 0.001


@pytest.mark.unit
def test_openweather_http_error_response(monkeypatch, reject_direct_httpx):
    client = weather_providers.OpenWeatherClient(api_key="k")

    class FakeResponse:
        status_code = 503

        @staticmethod
        def json():
            return {}

    monkeypatch.setattr(
        weather_providers,
        "fetch",
        lambda **_kwargs: FakeResponse(),
        raising=False,
    )
    result = client.get_current(location="Boston")
    assert not result.ok
    assert result.metadata.get("error") == "http_error"
    assert result.metadata.get("status_code") == 503


@pytest.mark.unit
def test_openweather_exception_path(monkeypatch, reject_direct_httpx):
    client = weather_providers.OpenWeatherClient(api_key="k")

    def fake_fetch(**_kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(weather_providers, "fetch", fake_fetch, raising=False)
    result = client.get_current(location="Boston")
    assert not result.ok
    assert result.metadata.get("error") == "exception"
    assert result.metadata.get("provider") == "openweather"


@pytest.mark.unit
def test_openweather_exception_metadata_does_not_expose_raw_details(
    monkeypatch,
    reject_direct_httpx,
):
    client = weather_providers.OpenWeatherClient(api_key="secret-key")

    def fake_fetch(**_kwargs):
        raise httpx.RequestError("failed with appid=secret-key")

    monkeypatch.setattr(weather_providers, "fetch", fake_fetch, raising=False)
    result = client.get_current(location="Boston")
    combined = f"{result.summary} {result.metadata}"
    assert not result.ok
    assert result.metadata.get("error") == "exception"
    assert result.metadata.get("exception_type") == "RequestError"
    assert "details" not in result.metadata
    assert "secret-key" not in combined


@pytest.mark.unit
def test_openweather_central_policy_denial_returns_sanitized_error(
    monkeypatch,
    reject_direct_httpx,
):
    client = weather_providers.OpenWeatherClient(api_key="secret-key")
    monkeypatch.setenv("WORKFLOWS_EGRESS_PROFILE", "strict")
    monkeypatch.delenv("WORKFLOWS_EGRESS_ALLOWLIST", raising=False)
    monkeypatch.delenv("EGRESS_ALLOWLIST", raising=False)
    monkeypatch.setattr(
        http_client,
        "_get_httpx_client",
        lambda **_kwargs: pytest.fail("policy denial must precede network I/O"),
    )
    result = client.get_current(location="Boston")
    combined = f"{result.summary} {result.metadata}"
    assert not result.ok
    assert result.metadata.get("error") == "exception"
    assert result.metadata.get("exception_type") == "EgressPolicyError"
    assert "details" not in result.metadata
    assert "secret-key" not in combined


@pytest.mark.unit
def test_openweather_strict_policy_allows_documented_host(
    monkeypatch,
    reject_direct_httpx,
):
    client = weather_providers.OpenWeatherClient(api_key="secret-key")
    monkeypatch.setenv("WORKFLOWS_EGRESS_PROFILE", "strict")
    monkeypatch.setenv("EGRESS_ALLOWLIST", "api.openweathermap.org")
    monkeypatch.delenv("WORKFLOWS_EGRESS_ALLOWLIST", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.openweathermap.org"
        return httpx.Response(
            200,
            request=request,
            json={
                "name": "Boston",
                "sys": {"country": "US"},
                "main": {"temp": 12.6},
                "weather": [{"description": "clear sky"}],
            },
        )

    transport_client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(http_client, "_get_httpx_client", lambda **_kwargs: transport_client)
    try:
        result = client.get_current(location="Boston")
    finally:
        transport_client.close()
    assert result.ok


@pytest.mark.unit
def test_openweather_does_not_follow_provider_redirects(
    monkeypatch,
    reject_direct_httpx,
):
    client = weather_providers.OpenWeatherClient(api_key="secret-key")
    monkeypatch.setenv("WORKFLOWS_EGRESS_PROFILE", "permissive")
    monkeypatch.delenv("WORKFLOWS_EGRESS_ALLOWLIST", raising=False)
    monkeypatch.delenv("EGRESS_ALLOWLIST", raising=False)
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if request.url.host == "api.openweathermap.org":
            return httpx.Response(
                302,
                request=request,
                headers={"Location": "https://example.com/capture"},
            )
        return httpx.Response(200, request=request, json={})

    transport_client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(http_client, "_get_httpx_client", lambda **_kwargs: transport_client)
    try:
        result = client.get_current(location="Boston")
    finally:
        transport_client.close()
    assert not result.ok
    assert len(requested_urls) == 1
    assert requested_urls[0].startswith("https://api.openweathermap.org/")


@pytest.mark.unit
def test_openweather_uses_central_http_fetch_for_requests(
    monkeypatch,
    reject_direct_httpx,
):
    client = weather_providers.OpenWeatherClient(api_key="k", units="metric")
    calls = []

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "name": "Boston",
                "sys": {"country": "US"},
                "main": {"temp": 12.6},
                "weather": [{"description": "clear sky"}],
            }

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr(weather_providers, "fetch", fake_fetch, raising=False)
    result = client.get_current(location="Boston")
    assert result.ok
    assert len(calls) == 1
    call = calls[0]
    assert call["method"] == "GET"
    assert call["url"] == weather_providers.OpenWeatherClient._BASE_URL
    assert call["params"] == {
        "appid": "k",
        "units": "metric",
        "lang": "en",
        "q": "Boston",
    }
    assert call["timeout"] == client.timeout_seconds
    assert call["retry"].attempts == 1
    assert call["allow_redirects"] is False
    assert call["sensitive_observability"] is True


@pytest.mark.unit
def test_openweather_rejects_oversized_location_before_network(
    monkeypatch,
    reject_direct_httpx,
):
    client = weather_providers.OpenWeatherClient(api_key="k")

    def failing_fetch(**_kwargs):
        raise AssertionError("network should not be used for invalid input")

    monkeypatch.setattr(weather_providers, "fetch", failing_fetch, raising=False)
    result = client.get_current(location="x" * 300)
    assert not result.ok
    assert result.metadata.get("error") == "invalid_location"


@pytest.mark.unit
def test_openweather_rejects_out_of_range_coordinates_before_network(
    monkeypatch,
    reject_direct_httpx,
):
    client = weather_providers.OpenWeatherClient(api_key="k")

    def failing_fetch(**_kwargs):
        raise AssertionError("network should not be used for invalid input")

    monkeypatch.setattr(weather_providers, "fetch", failing_fetch, raising=False)
    result = client.get_current(lat=95.0, lon=0.0)
    assert not result.ok
    assert result.metadata.get("error") == "invalid_coordinates"


@pytest.mark.unit
def test_openweather_success_response(monkeypatch, reject_direct_httpx):
    client = weather_providers.OpenWeatherClient(api_key="k", units="metric")

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "name": "Boston",
                "sys": {"country": "US"},
                "main": {"temp": 12.6},
                "weather": [{"description": "clear sky"}],
            }

    monkeypatch.setattr(
        weather_providers,
        "fetch",
        lambda **_kwargs: FakeResponse(),
        raising=False,
    )
    result = client.get_current(location="Boston")
    assert result.ok
    assert "Weather for Boston, US" in result.summary
    assert result.metadata.get("provider") == "openweather"
