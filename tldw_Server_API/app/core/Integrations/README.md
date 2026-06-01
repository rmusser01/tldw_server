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
