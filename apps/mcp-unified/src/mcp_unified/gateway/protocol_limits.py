"""Resource limits for the strict MCP protocol layer."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _bounded_int(value: object, name: str, lower: int, upper: int) -> int:
    """Return an integer within inclusive bounds, rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < lower or value > upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _bounded_finite_number(
    value: object,
    name: str,
    lower_exclusive: float,
    upper: float,
) -> float:
    """Return a finite number within the configured interval."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= lower_exclusive or numeric > upper:
        raise ValueError(
            f"{name} must be finite and greater than {lower_exclusive} "
            f"and at most {upper}"
        )
    return numeric


@dataclass(frozen=True, slots=True)
class GatewayLimits:
    """Validated resource limits for one strict stdio server."""

    max_input_line_bytes: int = 1_048_576
    max_output_line_bytes: int = 1_048_576
    max_result_bytes: int = 786_432
    max_json_depth: int = 64
    max_in_flight: int = 16
    default_catalog_page_size: int = 50
    max_catalog_page_size: int = 100
    max_catalog_items: int = 10_000
    max_batch_items: int = 100
    max_requests_per_minute: int = 600
    request_burst: int = 32
    max_schema_bytes: int = 262_144
    max_schema_depth: int = 32
    max_schema_subschemas: int = 1_024
    max_schema_refs: int = 256
    max_schema_pattern_chars: int = 4_096
    max_schema_validation_processes: int = 4
    schema_validation_timeout_seconds: float = 1.0
    graceful_shutdown_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        """Reject invalid limits before any protocol work begins."""

        _bounded_int(self.max_input_line_bytes, "max_input_line_bytes", 1, 16_777_216)
        _bounded_int(self.max_output_line_bytes, "max_output_line_bytes", 1, 16_777_216)
        _bounded_int(self.max_result_bytes, "max_result_bytes", 1, 16_777_216)
        _bounded_int(self.max_json_depth, "max_json_depth", 1, 256)
        _bounded_int(self.max_in_flight, "max_in_flight", 1, 1_024)
        _bounded_int(
            self.default_catalog_page_size,
            "default_catalog_page_size",
            1,
            1_000,
        )
        _bounded_int(self.max_catalog_page_size, "max_catalog_page_size", 1, 1_000)
        _bounded_int(self.max_catalog_items, "max_catalog_items", 1, 100_000)
        _bounded_int(self.max_batch_items, "max_batch_items", 1, 1_000)
        _bounded_int(
            self.max_requests_per_minute,
            "max_requests_per_minute",
            1,
            60_000,
        )
        _bounded_int(self.request_burst, "request_burst", 1, 10_000)
        _bounded_int(self.max_schema_bytes, "max_schema_bytes", 1, 4_194_304)
        _bounded_int(self.max_schema_depth, "max_schema_depth", 1, 128)
        _bounded_int(
            self.max_schema_subschemas,
            "max_schema_subschemas",
            1,
            10_000,
        )
        _bounded_int(self.max_schema_refs, "max_schema_refs", 1, 4_096)
        _bounded_int(
            self.max_schema_pattern_chars,
            "max_schema_pattern_chars",
            1,
            65_536,
        )
        _bounded_int(
            self.max_schema_validation_processes,
            "max_schema_validation_processes",
            1,
            32,
        )
        _bounded_finite_number(
            self.schema_validation_timeout_seconds,
            "schema_validation_timeout_seconds",
            0.0,
            10.0,
        )
        _bounded_finite_number(
            self.graceful_shutdown_timeout_seconds,
            "graceful_shutdown_timeout_seconds",
            0.0,
            60.0,
        )

        if self.max_result_bytes > self.max_output_line_bytes:
            raise ValueError("max_result_bytes must not exceed max_output_line_bytes")
        if self.default_catalog_page_size > self.max_catalog_page_size:
            raise ValueError(
                "default_catalog_page_size must not exceed max_catalog_page_size"
            )
        if self.max_catalog_items < self.max_catalog_page_size:
            raise ValueError(
                "max_catalog_items must not be less than max_catalog_page_size"
            )
        if self.request_burst > self.max_requests_per_minute:
            raise ValueError(
                "request_burst must not exceed max_requests_per_minute"
            )


__all__ = ["GatewayLimits"]
