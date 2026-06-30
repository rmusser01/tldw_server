# Integrations

Integrations currently contains the weather provider used by chat slash commands. The broader Slack, Discord, and Telegram connection control plane exists in API and service modules outside this package; this core package only owns weather provider selection, OpenWeather request handling, and a no-key fallback client.

## Start Here

- `weather_providers.py` defines the weather result model, provider protocol, no-key fallback, OpenWeather client, and environment-based client factory.
- Related chat surface: `tldw_Server_API/app/core/Chat/command_router.py`.
- Related control-plane API surface: `tldw_Server_API/app/api/v1/endpoints/integrations_control_plane.py`.
- Related control-plane service: `tldw_Server_API/app/services/integrations_control_plane_service.py`.
- Related tests: `tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py`, `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py`, and `tldw_Server_API/tests/Integrations/`.

## Responsibilities

- Select a weather client from `WEATHER_PROVIDER`.
- Read OpenWeather settings from `OPENWEATHER_API_KEY`, `WEATHER_UNITS`, `WEATHER_LANG`, and `WEATHER_TIMEOUT_MS`.
- Return a safe unavailable response when no weather API key is configured.
- Call the OpenWeather current-weather API and normalize the response into `WeatherResult`.
- Provide an injectable HTTP client factory for tests.

## Module Map

- `weather_providers.py`: weather provider protocol, no-key fallback, OpenWeather implementation, settings, and factory.
- `__init__.py`: package marker currently describing this package as a small integration stub.

## How It Connects

- `Chat/command_router.py` uses `get_weather_client()` to implement the `/weather` slash command.
- The command router also handles command allowlists, output limits, optional RBAC, rate limiting, and metrics outside this package.
- `integrations_control_plane.py` exposes `/integrations` state for Slack, Discord, and Telegram, but that control plane is backed by `IntegrationsControlPlaneService` and AuthNZ repositories rather than files in this core package.

## Architecture Notes

### Core Flow

- The chat command router recognizes `/weather`, applies command-level authorization and limits, then asks `get_weather_client()` for a provider client.
- `weather_providers.py` returns `NoKeyWeatherClient` when OpenWeather configuration is absent, or an `OpenWeatherClient` that normalizes the current-weather response into `WeatherResult`.
- Weather output is formatted by the chat command path; this package only owns provider selection and normalized weather data.

### State And Data

- There is no database state in this core package.
- Provider behavior is configured through `WEATHER_PROVIDER`, `OPENWEATHER_API_KEY`, `WEATHER_UNITS`, `WEATHER_LANG`, and `WEATHER_TIMEOUT_MS`.
- Slack, Discord, and Telegram connection state belongs to the integrations control-plane service and AuthNZ repositories outside this package.

### Security And Operations

- The no-key client must not make outbound network requests.
- OpenWeather failures should return normalized unavailable/error metadata without leaking API keys.
- The injectable HTTP client factory is the test boundary for timeout, status-code, and response-shape behavior.

### Extension Checklist

- New weather provider: implement the `WeatherClient` protocol, extend `get_weather_client()`, and add provider tests.
- New slash command behavior: update `Chat/command_router.py` and command router tests.
- New connection control-plane feature: update `integrations_control_plane.py`, `IntegrationsControlPlaneService`, and `tests/Integrations/` rather than expanding this weather-only package.

## Extension Points

- Add a weather provider by implementing the `WeatherClient` protocol and extending `get_weather_client()`.
- Add provider-specific settings in `weather_providers.py` and cover them in chat weather tests.
- Extend chat command behavior in `Chat/command_router.py`, not in this package.
- Extend Slack, Discord, or Telegram connection state in `app/services/integrations_control_plane_service.py` and the `/integrations` endpoint.

## Testing

- Weather provider behavior is covered in `tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py`.
- Slash command behavior is covered in `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py` and related chat command injection tests.
- The separate integrations control plane is covered in `tldw_Server_API/tests/Integrations/`.

## Gotchas

- This package does not own the full integrations control plane despite the package name.
- Without an OpenWeather API key, the weather command intentionally returns an unavailable result instead of making a network request.
