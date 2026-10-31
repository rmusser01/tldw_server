"""
Guardrail audit for AuthNZ lockout and virtual-key counter tables.

This test ensures that low-level tables used for guardrails
(`failed_attempts`, `account_lockouts`, and `vk_*_counters`) are only
referenced from the expected infrastructure modules (rate_limiter,
quotas, migrations, initialize) within the AuthNZ core package.
"""

from pathlib import Path

import pytest


@pytest.mark.unit
def test_guardrail_tables_used_only_in_core_authnz_modules():
    project_root = Path(__file__).resolve().parents[3]
    roots = [project_root / "app" / "core" / name for name in ("AuthNZ", "Provider_Credentials")]
    for root in roots:
        assert root.exists(), f"Expected core path not found: {root}"
    files = [path for root in roots for path in root.rglob("*.py")]
    assert project_root / "app" / "core" / "Provider_Credentials" / "byok_runtime.py" in files

    allowed_files = {
        "app/core/AuthNZ/rate_limiter.py",
        "app/core/AuthNZ/lockout_tracker.py",
        "app/core/AuthNZ/repos/rate_limits_repo.py",
        "app/core/AuthNZ/repos/quotas_repo.py",
        "app/core/AuthNZ/migrations.py",
        "app/core/AuthNZ/quotas.py",
        "app/core/AuthNZ/initialize.py",
        "app/core/AuthNZ/pg_migrations_extra.py",
    }

    markers = (
        "failed_attempts",
        "account_lockouts",
        "vk_jwt_counters",
        "vk_api_key_counters",
    )

    offending: list[str] = []

    for path in files:
        rel = path.relative_to(project_root).as_posix()
        if rel in allowed_files:
            continue
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in markers):
            offending.append(rel)

    assert offending == [], f"Guardrail tables referenced in unexpected modules: {offending}"
