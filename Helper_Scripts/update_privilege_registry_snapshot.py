"""Regenerate the privilege route registry snapshot used by CI.

Usage:
    python Helper_Scripts/update_privilege_registry_snapshot.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _apply_test_env_defaults() -> None:
    """Align environment with pytest defaults so snapshots match CI expectations.

    Sets environment flags to mirror test configuration and route scoping:
    - MINIMAL_TEST_APP: enable minimal app configuration for tests
    - TEST_MODE: activate test mode behaviors
    - SINGLE_USER_API_KEY/AUTH_MODE/DATABASE_URL: mirror pytest's deterministic
      AuthNZ defaults before importing the app
    - OTEL_SDK_DISABLED: disable OpenTelemetry instrumentation
    - AUTH_MODE/SINGLE_USER_*: mirror the deterministic single-user auth
      defaults from tests/conftest.py
    - ROUTES_DISABLE: ensure "research" is disabled and remove "notes" if
      present (parsed from comma/space-delimited values).  Must match
      ``tests/conftest.py`` which only disables "research", NOT "evaluations".
    - ROUTES_ENABLE: ensure "workflows" and "scheduler" are enabled (parsed
      from comma/space-delimited values)
    """
    os.environ["MINIMAL_TEST_APP"] = "1"
    os.environ["TEST_MODE"] = "1"
    os.environ["OTEL_SDK_DISABLED"] = "true"
    os.environ.setdefault("SINGLE_USER_TEST_API_KEY", "test-api-key-12345")
    os.environ["SINGLE_USER_API_KEY"] = os.environ["SINGLE_USER_TEST_API_KEY"]
    os.environ["AUTH_MODE"] = "single_user"
    os.environ.setdefault("DATABASE_URL", "sqlite:///./Databases/users.db")
    os.environ.pop("PROFILE", None)
    # Route inclusion in app startup now keys off explicit pytest runtime, not
    # only TEST_MODE. Mirror pytest's runtime signal so this helper produces
    # the same snapshot shape as tests.
    os.environ.setdefault("PYTEST_CURRENT_TEST", "snapshot_regen::helper (call)")
    # The privilege snapshot test imports the app during collection, before
    # PYTEST_CURRENT_TEST is set, so audio routers are included. Keep the
    # helper aligned with that live app shape.
    os.environ.setdefault("MINIMAL_TEST_INCLUDE_AUDIO", "1")
    logger.debug("Set test environment flags: MINIMAL_TEST_APP, TEST_MODE, OTEL_SDK_DISABLED")
    existing_disable = os.getenv("ROUTES_DISABLE", "")
    disable_parts = [p for p in existing_disable.replace(" ", ",").split(",") if p]
    disable_lower = {p.lower() for p in disable_parts}
    # Only disable "research" to match tests/conftest.py behaviour.
    if "research" not in disable_lower:
        disable_parts.append("research")
        disable_lower.add("research")
    disable_parts = [p for p in disable_parts if p.lower() != "notes"]
    os.environ["ROUTES_DISABLE"] = ",".join(dict.fromkeys(disable_parts))
    logger.debug("Set ROUTES_DISABLE={}", os.environ["ROUTES_DISABLE"])
    existing_enable = os.getenv("ROUTES_ENABLE", "")
    parts = [p for p in existing_enable.replace(" ", ",").split(",") if p]
    lower_parts = {p.lower() for p in parts}
    for key in ("workflows", "scheduler"):
        if key not in lower_parts:
            parts.append(key)
            lower_parts.add(key)
    os.environ["ROUTES_ENABLE"] = ",".join(dict.fromkeys(parts))
    logger.debug("Set ROUTES_ENABLE={}", os.environ["ROUTES_ENABLE"])


def main() -> None:
    _apply_test_env_defaults()
    from tldw_Server_API.app.main import app as fastapi_app
    from tldw_Server_API.app.core.AuthNZ.privilege_catalog import load_catalog
    from tldw_Server_API.app.core.PrivilegeMaps.introspection import (
        collect_privilege_route_registry,
        serialize_route_registry,
    )

    catalog = load_catalog()
    registry = collect_privilege_route_registry(fastapi_app, catalog, strict=False)
    serialized = serialize_route_registry(registry)

    snapshot_path = PROJECT_ROOT / "tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(serialized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Updated snapshot written to {}", snapshot_path)


if __name__ == "__main__":
    main()
