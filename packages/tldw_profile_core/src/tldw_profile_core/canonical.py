import hashlib
import hmac
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from math import isfinite
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import BeforeValidator
from rfc8785 import dumps

I_JSON_MAX_INTEGER = 2**53 - 1
CANONICAL_DATETIME_FORMAT = "utc-milliseconds-v1"
PORTABLE_DATETIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$"
)


def normalize_datetime(value: datetime) -> str:
    """Return the portable V1 UTC millisecond representation of an aware datetime."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    if value.utcoffset().total_seconds() % 60:
        raise ValueError("timestamp offset must use whole minutes")
    if value.microsecond % 1_000:
        raise ValueError("timestamp precision must not exceed milliseconds")
    try:
        value = value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise ValueError("timestamp must normalize within years 0001-9999") from error
    milliseconds = value.microsecond // 1_000
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}T"
        f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}."
        f"{milliseconds:03d}Z"
    )


def parse_portable_datetime(value: Any) -> datetime:
    """Parse a V1 wire timestamp or validate an already-constructed datetime."""

    if isinstance(value, datetime):
        normalize_datetime(value)
        return value.astimezone(UTC)
    if not isinstance(value, str) or not PORTABLE_DATETIME_PATTERN.fullmatch(value):
        raise ValueError("timestamp must use the portable V1 RFC 3339 syntax")
    try:
        parsed = datetime.fromisoformat(
            value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else "")
        )
    except ValueError as error:
        raise ValueError("timestamp must be a valid RFC 3339 value") from error
    normalize_datetime(parsed)
    return parsed.astimezone(UTC)


PortableDateTime = Annotated[
    datetime,
    BeforeValidator(parse_portable_datetime),
    PlainSerializer(normalize_datetime, return_type=str, when_used="json"),
]


def parse_json_integer(value: Any) -> int:
    """Apply JSON Schema integer semantics without Pydantic coercion."""

    if type(value) is int:
        return value
    if type(value) is float and isfinite(value) and value.is_integer():
        return int(value)
    raise ValueError("value must be a finite JSON integer")


def parse_json_number(value: Any) -> float:
    """Apply finite JSON number semantics without Pydantic coercion."""

    if type(value) not in (int, float):
        raise ValueError("value must be a finite JSON number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("value must be a finite JSON number") from error
    if not isfinite(number):
        raise ValueError("value must be a finite JSON number")
    return number


def parse_version_one(value: Any) -> int:
    """Accept JSON numeric one while rejecting bool and string coercion."""

    if parse_json_integer(value) != 1:
        raise ValueError("version must be numeric one")
    return 1


JsonInteger = Annotated[int, BeforeValidator(parse_json_integer)]
JsonNumber = Annotated[float, BeforeValidator(parse_json_number)]
Confidence = Annotated[
    float,
    Field(ge=0, le=1),
    BeforeValidator(parse_json_number),
]
VersionOne = Annotated[Literal[1], BeforeValidator(parse_version_one)]


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return normalize_datetime(value)
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data with RFC 8785 JCS."""

    return dumps(_json_value(value))


def canonical_bytes(value: BaseModel) -> bytes:
    payload = value.model_dump(mode="python", exclude_none=False, by_alias=True)
    return canonical_json_bytes(payload)


def integrity_tag(value: BaseModel, key: bytes) -> str:
    if len(key) != 32:
        raise ValueError("integrity key must be exactly 32 bytes")
    return f"hmac-sha256-v1:{hmac.new(key, canonical_bytes(value), hashlib.sha256).hexdigest()}"
