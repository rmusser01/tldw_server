"""
weather_providers.py

Weather provider abstraction for slash commands and template integrations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from typing import Any

import httpx

from tldw_Server_API.app.core.exceptions import (
    EgressPolicyError,
    NetworkError,
    RetryExhaustedError,
)
from tldw_Server_API.app.core.http_client import RetryPolicy, fetch

_WEATHER_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    EgressPolicyError,
    KeyError,
    LookupError,
    NetworkError,
    OSError,
    RetryExhaustedError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    httpx.HTTPError,
)

_MAX_LOCATION_CHARS = 120
_MIN_LATITUDE = -90.0
_MAX_LATITUDE = 90.0
_MIN_LONGITUDE = -180.0
_MAX_LONGITUDE = 180.0


@dataclass
class WeatherResult:
    ok: bool
    summary: str
    metadata: dict[str, Any]


class WeatherClient:
    def get_current(
        self,
        location: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ) -> WeatherResult:
        raise NotImplementedError


class NoKeyWeatherClient(WeatherClient):
    def get_current(
        self,
        location: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ) -> WeatherResult:
        loc = location or (f"{lat},{lon}" if lat is not None and lon is not None else "your area")
        return WeatherResult(
            ok=False,
            summary=f"Weather information is unavailable for {loc}.",
            metadata={"provider": "noop", "location": loc},
        )


class OpenWeatherClient(WeatherClient):
    _BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = 1.5,
        units: str = "metric",
        lang: str = "en",
    ):
        self.api_key = api_key
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.units = units if units in {"metric", "imperial"} else "metric"
        self.lang = (lang or "en").strip() or "en"

    def _temp_unit(self) -> str:
        return "F" if self.units == "imperial" else "C"

    def _build_params(
        self,
        *,
        location: str | None,
        lat: float | None,
        lon: float | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "appid": self.api_key,
            "units": self.units,
            "lang": self.lang,
        }
        if lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        elif location:
            params["q"] = location
        return params

    def _invalid_location_result(self) -> WeatherResult:
        """Build the stable response for invalid location input."""
        return WeatherResult(
            ok=False,
            summary="Weather information is unavailable for the requested location.",
            metadata={
                "provider": "openweather",
                "error": "invalid_location",
                "max_location_chars": _MAX_LOCATION_CHARS,
            },
        )

    def _invalid_coordinates_result(self) -> WeatherResult:
        """Build the stable response for invalid coordinate input."""
        return WeatherResult(
            ok=False,
            summary="Weather information is unavailable for the requested coordinates.",
            metadata={"provider": "openweather", "error": "invalid_coordinates"},
        )

    def _validate_inputs(
        self,
        *,
        location: str | None,
        lat: float | None,
        lon: float | None,
    ) -> tuple[str | None, float | None, float | None, WeatherResult | None]:
        """Normalize and validate location and coordinate inputs before HTTP use."""
        normalized_location: str | None = None
        if location is not None:
            if not isinstance(location, str):
                return None, None, None, self._invalid_location_result()
            normalized_location = location.strip() or None
            if normalized_location and len(normalized_location) > _MAX_LOCATION_CHARS:
                return None, None, None, self._invalid_location_result()

        normalized_lat: float | None = None
        normalized_lon: float | None = None
        if lat is not None or lon is not None:
            if lat is None or lon is None:
                if normalized_location:
                    return normalized_location, None, None, None
                return None, None, None, self._invalid_coordinates_result()
            try:
                normalized_lat = float(lat)
                normalized_lon = float(lon)
            except (TypeError, ValueError):
                return None, None, None, self._invalid_coordinates_result()
            if (
                not isfinite(normalized_lat)
                or not isfinite(normalized_lon)
                or not (_MIN_LATITUDE <= normalized_lat <= _MAX_LATITUDE)
                or not (_MIN_LONGITUDE <= normalized_lon <= _MAX_LONGITUDE)
            ):
                return None, None, None, self._invalid_coordinates_result()

        return normalized_location, normalized_lat, normalized_lon, None

    def _parse_summary(self, data: dict[str, Any], location_hint: str | None) -> tuple[str, dict[str, Any]]:
        weather_desc = ""
        weather = data.get("weather")
        if isinstance(weather, list) and weather:
            first = weather[0]
            if isinstance(first, dict):
                weather_desc = str(first.get("description", "") or "")

        main = data.get("main") if isinstance(data.get("main"), dict) else {}
        temp = main.get("temp")
        try:
            temp_f = float(temp)
            temp_str = f"{round(temp_f)}°{self._temp_unit()}"
        except (TypeError, ValueError):
            temp_str = "unknown"

        name = str(data.get("name") or "").strip()
        country = ""
        sys_val = data.get("sys")
        if isinstance(sys_val, dict):
            country = str(sys_val.get("country") or "").strip()

        loc = ", ".join([x for x in [name, country] if x]) if (name or country) else (location_hint or "your area")
        cond = weather_desc or "conditions unavailable"
        summary = f"Weather for {loc}: {temp_str}, {cond}."
        metadata = {
            "provider": "openweather",
            "location": loc,
            "temperature": temp,
            "units": self.units,
            "description": weather_desc,
        }
        return summary, metadata

    def get_current(
        self,
        location: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ) -> WeatherResult:
        location, lat, lon, validation_error = self._validate_inputs(location=location, lat=lat, lon=lon)
        if validation_error is not None:
            return validation_error

        params = self._build_params(location=location, lat=lat, lon=lon)
        if "q" not in params and ("lat" not in params or "lon" not in params):
            return WeatherResult(
                ok=False,
                summary="Weather information is unavailable for your area.",
                metadata={"provider": "openweather", "error": "missing_location"},
            )

        try:
            response = fetch(
                method="GET",
                url=self._BASE_URL,
                params=params,
                retry=RetryPolicy(attempts=1),
                timeout=self.timeout_seconds,
                allow_redirects=False,
                sensitive_observability=True,
            )
            if response.status_code >= 400:
                return WeatherResult(
                    ok=False,
                    summary=f"Weather information is unavailable for {location or 'your area'}.",
                    metadata={
                        "provider": "openweather",
                        "error": "http_error",
                        "status_code": response.status_code,
                    },
                )
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Unexpected weather provider payload")
            summary, metadata = self._parse_summary(data, location_hint=location)
            return WeatherResult(ok=True, summary=summary, metadata=metadata)
        except _WEATHER_NONCRITICAL_EXCEPTIONS as exc:
            return WeatherResult(
                ok=False,
                summary=f"Weather information is unavailable for {location or 'your area'}.",
                metadata={
                    "provider": "openweather",
                    "error": "exception",
                    "exception_type": type(exc).__name__,
                },
            )


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(str(raw))
    except (TypeError, ValueError):
        return default


def _str_env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip()


def get_weather_client() -> WeatherClient:
    provider = _str_env("WEATHER_PROVIDER", "openweather").lower()
    api_key = _str_env("OPENWEATHER_API_KEY", "")
    units = _str_env("WEATHER_UNITS", "metric").lower() or "metric"
    lang = _str_env("WEATHER_LANG", "en")
    timeout_ms = _float_env("WEATHER_TIMEOUT_MS", 1500.0)

    if provider in {"", "noop", "none", "disabled"}:
        return NoKeyWeatherClient()
    if provider == "openweather" and api_key:
        return OpenWeatherClient(
            api_key=api_key,
            timeout_seconds=max(0.1, timeout_ms / 1000.0),
            units=units,
            lang=lang,
        )
    return NoKeyWeatherClient()
