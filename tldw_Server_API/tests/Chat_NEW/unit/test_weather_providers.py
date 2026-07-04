import httpx
import pytest

from tldw_Server_API.app.core.exceptions import EgressPolicyError
from tldw_Server_API.app.core.Integrations import weather_providers


@pytest.mark.unit
def test_get_weather_client_falls_back_without_api_key(monkeypatch):
    monkeypatch.setenv("WEATHER_PROVIDER", "openweather")
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)

    client = weather_providers.get_weather_client()
    assert isinstance(client, weather_providers.NoKeyWeatherClient)


@pytest.mark.unit
def test_get_weather_client_openweather_when_configured(monkeypatch):
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
def test_openweather_http_error_response(monkeypatch):
    client = weather_providers.OpenWeatherClient(api_key="k")

    class FakeResponse:
        status_code = 503

        @staticmethod
        def json():
            return {}

    def fake_fetch(**kwargs):
        return FakeResponse()

    monkeypatch.setattr(weather_providers, "fetch", fake_fetch)

    result = client.get_current(location="Boston")
    assert not result.ok
    assert result.metadata.get("error") == "http_error"
    assert result.metadata.get("status_code") == 503


@pytest.mark.unit
def test_openweather_exception_path(monkeypatch):
    client = weather_providers.OpenWeatherClient(api_key="k")

    def fake_fetch(**kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(weather_providers, "fetch", fake_fetch)

    result = client.get_current(location="Boston")
    assert not result.ok
    assert result.metadata.get("error") == "exception"
    assert result.metadata.get("provider") == "openweather"


@pytest.mark.unit
def test_openweather_exception_metadata_does_not_expose_raw_details(monkeypatch):
    client = weather_providers.OpenWeatherClient(api_key="secret-key")

    def fake_fetch(**kwargs):
        raise httpx.RequestError("failed with appid=secret-key")

    monkeypatch.setattr(weather_providers, "fetch", fake_fetch)

    result = client.get_current(location="Boston")
    combined = f"{result.summary} {result.metadata}"
    assert not result.ok
    assert result.metadata.get("error") == "exception"
    assert result.metadata.get("exception_type") == "RequestError"
    assert "details" not in result.metadata
    assert "secret-key" not in combined


@pytest.mark.unit
def test_openweather_central_policy_denial_returns_sanitized_error(monkeypatch):
    client = weather_providers.OpenWeatherClient(api_key="secret-key")

    def fake_fetch(**kwargs):
        raise EgressPolicyError("blocked appid=secret-key")

    monkeypatch.setattr(weather_providers, "fetch", fake_fetch)

    result = client.get_current(location="Boston")
    combined = f"{result.summary} {result.metadata}"
    assert not result.ok
    assert result.metadata.get("error") == "exception"
    assert result.metadata.get("exception_type") == "EgressPolicyError"
    assert "details" not in result.metadata
    assert "secret-key" not in combined


@pytest.mark.unit
def test_openweather_uses_central_http_fetch_for_requests(monkeypatch):
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


@pytest.mark.unit
def test_openweather_rejects_oversized_location_before_network(monkeypatch):
    client = weather_providers.OpenWeatherClient(api_key="k")

    def failing_fetch(**kwargs):
        raise AssertionError("network should not be used for invalid input")

    monkeypatch.setattr(weather_providers, "fetch", failing_fetch)

    result = client.get_current(location="x" * 300)
    assert not result.ok
    assert result.metadata.get("error") == "invalid_location"


@pytest.mark.unit
def test_openweather_rejects_out_of_range_coordinates_before_network(monkeypatch):
    client = weather_providers.OpenWeatherClient(api_key="k")

    def failing_fetch(**kwargs):
        raise AssertionError("network should not be used for invalid input")

    monkeypatch.setattr(weather_providers, "fetch", failing_fetch)

    result = client.get_current(lat=95.0, lon=0.0)
    assert not result.ok
    assert result.metadata.get("error") == "invalid_coordinates"


@pytest.mark.unit
def test_openweather_success_response(monkeypatch):
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

    def fake_fetch(**kwargs):
        return FakeResponse()

    monkeypatch.setattr(weather_providers, "fetch", fake_fetch)

    result = client.get_current(location="Boston")
    assert result.ok
    assert "Weather for Boston, US" in result.summary
    assert result.metadata.get("provider") == "openweather"
