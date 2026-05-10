from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OracleFixture:
    name: str
    agent_critical: bool
    command: list[str] | None = None
    resource: str | None = None
    tool: str | None = None
    classification: str = "golden-required"


@dataclass(frozen=True)
class OracleManifest:
    upstream_version: str
    source_kind: str
    source_ref: str
    package_metadata_sha256: str
    generated_at: str
    fixtures: tuple[OracleFixture, ...]


def load_oracle_manifest(path: Path) -> OracleManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Oracle manifest must contain a mapping: {path}")

    return OracleManifest(
        upstream_version=str(raw["upstream_version"]),
        source_kind=str(raw["source_kind"]),
        source_ref=str(raw["source_ref"]),
        package_metadata_sha256=str(raw["package_metadata_sha256"]),
        generated_at=str(raw["generated_at"]),
        fixtures=_parse_fixtures(raw["fixtures"]),
    )


def _parse_fixtures(raw_fixtures: Any) -> tuple[OracleFixture, ...]:
    if not isinstance(raw_fixtures, list):
        raise ValueError("Oracle manifest fixtures must be a list")

    return tuple(_parse_fixture(item) for item in raw_fixtures)


def _parse_fixture(raw_fixture: Any) -> OracleFixture:
    if not isinstance(raw_fixture, dict):
        raise ValueError("Oracle manifest fixture entries must be mappings")

    return OracleFixture(
        name=str(raw_fixture["name"]),
        agent_critical=bool(raw_fixture["agent_critical"]),
        command=_optional_string_list(raw_fixture.get("command")),
        resource=_optional_string(raw_fixture.get("resource")),
        tool=_optional_string(raw_fixture.get("tool")),
        classification=str(raw_fixture.get("classification", "golden-required")),
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Oracle manifest command must be a list")
    return [str(item) for item in value]
