# Usage Quotas PR A (Relief) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A stock install, single-user or multi-user, stops returning usage-quota 402/403/413/429 responses. Every quota check is gated on one master switch that is off by default, and the hosted product keeps its billing-plan enforcement behind that switch.

**Architecture:**
- One switch, `config.usage_quotas_enabled()`. It checks env `USAGE_QUOTAS_ENABLED`, then the legacy env `LIMIT_ENFORCEMENT_ENABLED`, then config.txt `[Usage-Quotas] enabled`, and is otherwise off.
- Each quota family is gated at its existing choke point: one guard per shared function, never per caller.
- Billing checks also require a wired billing repository, which only the hosted product has. They run through one async helper, `enforcement.billing_checks_active()`.
- **Gate the check, never the record.** Usage counters keep being written.

**Tech Stack:** Python 3.12, FastAPI, pytest (`asyncio_mode = "auto"`), loguru, PyYAML, the Backlog CLI (`backlog`), `gh`.

**Spec:** `Docs/Design/2026-10-02-usage-quota-posture-design.md` (PR A of four). PRs B, C and D get their own plans, written against the code once A lands.

## Global Constraints

- All PRs target `dev`. Never `main`.
- Never use `git stash` (the stash is shared across worktrees). Never pass `--no-verify`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- PR bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)` and include the owner's waiver line: `> **Waived by the repository owner (@rmusser01) on 2026-09-22**, by explicit instruction in the session that produced this PR.`
- The Backlog CLI is the ledger. Never hand-edit task files. Use `--append-notes`, never `--notes`.
- Loguru only (`from loguru import logger`). No new dependencies.
- New tests carry `pytestmark = pytest.mark.unit`.
- Run Python with the venv interpreter: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest …`. There is no bare `python` on PATH.
- Postgres fixture tests: run with `TLDW_TEST_NO_DOCKER=1` locally. Only failures that also occur on `origin/dev` count as environmental.
- Any edit to a doc mirrored under `Docs/Published` needs `bash Helper_Scripts/refresh_docs_published.sh`, and the Published diff goes in the same commit.
- **Merge coordination.** Another session (`tldw-server-03`) merges into `dev` in alternation with this one. Push and run CI freely, but merge only after messaging that session and getting its go-ahead, and message it again when the merge lands.
- **Owner prerequisite before merging PR A:** the hosted deploy must set `USAGE_QUOTAS_ENABLED=true` (spec §8). Ask the owner to confirm before the merge.
- **The switch's default is off in production but on in the test suite.** The root `tests/conftest.py` sets `LIMIT_ENFORCEMENT_ENABLED=true` (Task 1), so existing quota suites keep testing enforcement. New stock-default tests must clear both `USAGE_QUOTAS_ENABLED` and `LIMIT_ENFORCEMENT_ENABLED`.

## Review Focus

These are inputs the spec implies but no task's main tests exercise. Each one has a pinning test in the task named at the end of its line.

1. **A hosted deploy wires a billing repository but never sets the switch.** It must warn exactly once, both at startup and on the first check, and must not enforce. Task 2, `test_hosted_with_quotas_off_warns_once`.
2. **An existing deploy explicitly set `LIMIT_ENFORCEMENT_ENABLED=false`.** It must stay off, and `USAGE_QUOTAS_ENABLED`, when set, must win over it. Task 1, `test_new_env_beats_legacy_env_and_config`, `test_legacy_env_still_turns_quotas_off`.
3. **config.txt spells the switch `yes` or `on`.** It must read as on. Task 1, `test_config_txt_truthy_spellings`.
4. **The switch flips between a stream's start and its finish.** `finish_stream` and `finish_job` must not raise when no lease was taken. Task 3, `test_finish_without_lease_is_a_noop`.
5. **A storage caller passes `raise_on_exceed=True`** (file artifacts do). It must not raise when quotas are off. Task 4, `test_storage_never_raises_when_quotas_off`.

---

## PR A — Relief: quotas off by default

Branch: `fix/usage-quotas-off-by-default`, created from `design/usage-quota-posture` (which holds the spec and TASK-13432/13433). Rebase it onto `origin/dev` before pushing.

### Task 0: Ledger and branch

**Files:** none (Backlog CLI and git only).

- [ ] **Step 1: File the parent task**

```bash
backlog task create "Usage quotas off by default, set per user/team/org (spec 2)" --priority high --labels quotas,backend \
  -d "Implements Docs/Design/2026-10-02-usage-quota-posture-design.md in four PRs: A relief (switch + gates), B per-user values and group routes, C storage cut-over and migration, D reporting, docs and ADR-058. Plan for A: Docs/superpowers/plans/2026-10-02-usage-quotas-pr-a-relief.md." \
  --ac "PR A merged: every usage-quota check gated on USAGE_QUOTAS_ENABLED (off by default); billing checks need a wired billing repo" \
  --ac "PR B merged: quota_resolver, limits.* write path with null-as-delete, team/org override routes, non-storage sites read the resolver" \
  --ac "PR C merged: storage writers/readers cut over to limits.storage_quota_mb, migration" \
  --ac "PR D merged: reporting endpoints, ADR-058, docs"
```

Record the new ID as `<PARENT>`. Check that no open PR already claims it (`gh pr list --state open` plus each PR's file list). If one does, renumber with `git mv` and fix the `id:` line, following the repo precedent.

- [ ] **Step 2: Create the branch**

```bash
git fetch origin
git checkout -b fix/usage-quotas-off-by-default design/usage-quota-posture
git rebase origin/dev
```

- [ ] **Step 3: Commit the plan file and the ledger entry**

```bash
git add Docs/superpowers/plans/2026-10-02-usage-quotas-pr-a-relief.md backlog/tasks/
git commit -m "docs(plan): usage quotas PR A (relief) plan; file <PARENT>

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 1: The master switch

**Files:**
- Modify: `tldw_Server_API/app/core/config.py`: add `usage_quotas_enabled()` right after `rg_enabled()` (which ends with `return _as_bool(v, default)`, near line 2869).
- Modify: `tldw_Server_API/Config_Files/config.txt`: add a `[Usage-Quotas]` section just before `[ResourceGovernor]` (near line 825).
- Modify: `tldw_Server_API/tests/conftest.py`: add one env default next to the other `os.environ.setdefault` lines (near line 41).
- Create: `tldw_Server_API/tests/Usage/test_usage_quotas_switch.py`.

**Interfaces:**
- Produces: `config.usage_quotas_enabled() -> bool` and the module flag `config._USAGE_QUOTAS_LEGACY_WARNED: bool`. Every later task imports `from tldw_Server_API.app.core.config import usage_quotas_enabled`.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_usage_quotas_switch.py`:

```python
"""The usage-quota master switch (spec 2, Docs/Design/2026-10-02-usage-quota-posture-design.md §1)."""

import configparser

import pytest
from loguru import logger

from tldw_Server_API.app.core import config as cfg

pytestmark = pytest.mark.unit


def _config(enabled: str | None) -> configparser.ConfigParser:
    """A config.txt stand-in, optionally with [Usage-Quotas] enabled set."""
    cp = configparser.ConfigParser()
    if enabled is not None:
        cp.read_dict({"Usage-Quotas": {"enabled": enabled}})
    return cp


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a stock install: no env, no config section, no warning issued yet."""
    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr(cfg, "_USAGE_QUOTAS_LEGACY_WARNED", False)
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: _config(None))


def test_off_by_default() -> None:
    assert cfg.usage_quotas_enabled() is False


@pytest.mark.parametrize("spelling", ["true", "1", "yes", "on"])
def test_config_txt_truthy_spellings(monkeypatch: pytest.MonkeyPatch, spelling: str) -> None:
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: _config(spelling))
    assert cfg.usage_quotas_enabled() is True


def test_new_env_beats_legacy_env_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: _config("true"))
    monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "true")
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")
    assert cfg.usage_quotas_enabled() is False


def test_legacy_env_beats_config_and_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: _config("false"))
    monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "true")
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        assert cfg.usage_quotas_enabled() is True
        assert cfg.usage_quotas_enabled() is True
    finally:
        logger.remove(handler_id)
    assert sum("LIMIT_ENFORCEMENT_ENABLED is deprecated" in m for m in messages) == 1


def test_legacy_env_still_turns_quotas_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: _config("true"))
    monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "false")
    assert cfg.usage_quotas_enabled() is False


def test_unreadable_config_means_off(monkeypatch: pytest.MonkeyPatch) -> None:
    def _broken() -> configparser.ConfigParser:
        raise configparser.Error("bad file")

    monkeypatch.setattr(cfg, "load_comprehensive_config", _broken)
    assert cfg.usage_quotas_enabled() is False
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_usage_quotas_switch.py -q`
Expected: FAIL with `AttributeError: module ... has no attribute '_USAGE_QUOTAS_LEGACY_WARNED'`.

- [ ] **Step 3: Implement the switch**

In `config.py`, directly after `rg_enabled()`:

```python
_USAGE_QUOTAS_LEGACY_WARNED = False


def usage_quotas_enabled() -> bool:
    """
    Master switch for usage quotas (Docs/Design/2026-10-02-usage-quota-posture-design.md §1).

    Resolution order:
      1) Env var USAGE_QUOTAS_ENABLED
      2) Env var LIMIT_ENFORCEMENT_ENABLED, when explicitly set (legacy spelling; warns once)
      3) [Usage-Quotas] enabled in config.txt
      4) False: quotas are off unless an operator turns them on
    """
    global _USAGE_QUOTAS_LEGACY_WARNED
    v = os.getenv("USAGE_QUOTAS_ENABLED")
    if v is None:
        legacy = os.getenv("LIMIT_ENFORCEMENT_ENABLED")
        if legacy is not None:
            if not _USAGE_QUOTAS_LEGACY_WARNED:
                _USAGE_QUOTAS_LEGACY_WARNED = True
                logger.warning("LIMIT_ENFORCEMENT_ENABLED is deprecated; set USAGE_QUOTAS_ENABLED instead")
            v = legacy
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = cp.get("Usage-Quotas", "enabled", fallback="false") if cp else "false"
        except (FileNotFoundError, configparser.Error, KeyError, ValueError) as exc:
            _log_debug(f"usage_quotas_enabled: config read failed, quotas stay off: {exc}")
            v = "false"
    return _as_bool(v, False)
```

In `config.txt`, just before `[ResourceGovernor]`:

```ini
[Usage-Quotas]
# Master switch for usage quotas: audio minutes, storage, chatbooks, media, workflows,
# evaluations, and billing-plan limits (env: USAGE_QUOTAS_ENABLED; legacy env:
# LIMIT_ENFORCEMENT_ENABLED). Off by default: no quota applies until an operator turns
# this on. Request rate limits are separate ([ResourceGovernor]).
enabled = false
```

In `tldw_Server_API/tests/conftest.py`, next to the other `os.environ.setdefault(...)` lines:

```python
# Usage quotas are off by default in production (spec 2). The existing quota suites
# test enforcement, so keep it on for the test session through the legacy spelling,
# which tests can still flip with setenv. Stock-default tests clear both variables.
os.environ.setdefault("LIMIT_ENFORCEMENT_ENABLED", "true")
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_usage_quotas_switch.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/config.py tldw_Server_API/Config_Files/config.txt tldw_Server_API/tests/conftest.py tldw_Server_API/tests/Usage/test_usage_quotas_switch.py
git commit -m "feat(quotas): USAGE_QUOTAS_ENABLED master switch, off by default (spec 2)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: Billing checks need the switch and a wired billing repository

**Files:**
- Modify: `tldw_Server_API/app/core/Billing/subscription_service.py`: add the `has_billing_repo` property on `SubscriptionService`, and `billing_repo_configured()` after `get_subscription_service()`.
- Modify: `tldw_Server_API/app/core/Billing/enforcement.py`: rewrite `enforcement_enabled()` (near line 975) and add `billing_checks_active()` below it.
- Modify: `tldw_Server_API/app/api/v1/API_Deps/billing_deps.py`: import `billing_checks_active`; replace the gate in `get_billing_org_id`, `resolve_org_id_for_principal`, `require_within_limit`, `require_feature`, `add_billing_headers` and `LimitEnforcer.__aenter__`.
- Modify: `tldw_Server_API/app/core/RAG/rag_service/transport.py`: the gate in `enforce_rag_query_limit_for_org_context`.
- Modify: `tldw_Server_API/app/services/startup_auth_runtime.py`: the startup warning.
- Modify: `tldw_Server_API/tests/conftest.py` (fixture `billing_repo_wired`) and `tldw_Server_API/tests/Billing/conftest.py` (autouse).
- Create: `tldw_Server_API/tests/Usage/test_usage_quotas_billing_gate.py`.

**Interfaces:**
- Consumes: `config.usage_quotas_enabled()` (Task 1).
- Produces:
  - `subscription_service.billing_repo_configured() -> Awaitable[bool]`;
  - `enforcement.billing_checks_active() -> Awaitable[bool]`;
  - the module flag `enforcement._PLAN_LIMITS_UNENFORCED_WARNED: bool`;
  - the pytest fixture `billing_repo_wired`.
  - `enforcement.enforcement_enabled()` keeps its name and becomes an alias of the switch.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_usage_quotas_billing_gate.py`:

```python
"""Billing-plan checks run only with quotas on and a billing repository wired (spec 2 §6)."""

from types import SimpleNamespace

import pytest
from fastapi import Response
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps import billing_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Billing import enforcement, subscription_service
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.core.RAG.rag_service import transport

pytestmark = pytest.mark.unit


async def _wired() -> bool:
    return True


async def _not_wired() -> bool:
    return False


async def _must_not_resolve(*_args: object, **_kwargs: object) -> int:
    raise AssertionError("org resolution must not run")


def _principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=7, is_admin=False)


@pytest.fixture(autouse=True)
def _quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas switched on explicitly; each test picks whether a billing repo is wired."""
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")


async def test_oss_skips_org_resolution_even_with_quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subscription_service, "billing_repo_configured", _not_wired)
    monkeypatch.setattr(billing_deps, "_resolve_org_id", _must_not_resolve)
    assert await billing_deps.get_billing_org_id(principal=_principal(), x_tldw_org_id=None, org_id=None) is None
    assert await billing_deps.resolve_org_id_for_principal(_principal()) is None


async def test_orgless_multi_user_account_gets_no_403(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subscription_service, "billing_repo_configured", _not_wired)
    monkeypatch.setattr(billing_deps, "_allow_orgless_billing_access", lambda: False)
    monkeypatch.setattr(billing_deps, "_resolve_org_id", _must_not_resolve)
    check = billing_deps.require_within_limit(LimitCategory.RAG_QUERIES_DAY)
    result = await check(response=Response(), principal=_principal(), x_tldw_org_id=None, org_id=None)
    assert result.unlimited is True
    feature_check = billing_deps.require_feature("advanced_analytics")
    assert await feature_check(principal=_principal(), x_tldw_org_id=None, org_id=None) is True


async def test_hosted_path_still_resolves_the_org(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subscription_service, "billing_repo_configured", _wired)

    async def _org(*_args: object, **_kwargs: object) -> int:
        return 42

    monkeypatch.setattr(billing_deps, "_resolve_org_id", _org)
    assert await billing_deps.get_billing_org_id(principal=_principal(), x_tldw_org_id=None, org_id=None) == 42


async def test_hosted_with_quotas_off_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")
    monkeypatch.setattr(subscription_service, "billing_repo_configured", _wired)
    monkeypatch.setattr(enforcement, "_PLAN_LIMITS_UNENFORCED_WARNED", False)
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        assert await enforcement.billing_checks_active() is False
        assert await enforcement.billing_checks_active() is False
    finally:
        logger.remove(handler_id)
    assert sum("plan limits are NOT enforced" in m for m in messages) == 1


async def test_rag_transport_check_skips_without_billing_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subscription_service, "billing_repo_configured", _not_wired)
    monkeypatch.setattr(transport, "resolve_org_id_for_rag_context", _must_not_resolve)
    await transport.enforce_rag_query_limit_for_org_context(current_user=SimpleNamespace(id=7), units=1)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_usage_quotas_billing_gate.py -q`
Expected: FAIL with `AttributeError: ... has no attribute 'billing_repo_configured'`.

- [ ] **Step 3: Implement the gate**

`subscription_service.py`: add this property to `SubscriptionService` (after `__init__`):

```python
    @property
    def has_billing_repo(self) -> bool:
        """True when a billing repository (the hosted product's plans) is wired in."""
        return self._billing_repo is not None
```

and this function after `get_subscription_service()`:

```python
async def billing_repo_configured() -> bool:
    """True when the subscription service has a billing repository: commercial (hosted) mode."""
    service = await get_subscription_service()
    return service.has_billing_repo
```

`enforcement.py`: replace `enforcement_enabled()` with the following, and add `billing_checks_active()` after it:

```python
def enforcement_enabled() -> bool:
    """The usage-quota master switch; this name is kept for existing imports (config.usage_quotas_enabled)."""
    from tldw_Server_API.app.core.config import usage_quotas_enabled

    return usage_quotas_enabled()


_PLAN_LIMITS_UNENFORCED_WARNED = False


async def billing_checks_active() -> bool:
    """
    True when org billing-plan limits apply: usage quotas are on and a billing repository
    is wired (the hosted product). OSS installs wire none, so billing checks, org
    resolution and usage aggregation never run there (spec 2 §6).
    """
    global _PLAN_LIMITS_UNENFORCED_WARNED
    from tldw_Server_API.app.core.Billing.subscription_service import billing_repo_configured

    repo_wired = await billing_repo_configured()
    if enforcement_enabled():
        return repo_wired
    if repo_wired and not _PLAN_LIMITS_UNENFORCED_WARNED:
        _PLAN_LIMITS_UNENFORCED_WARNED = True
        logger.warning(
            "A billing repository is configured but usage quotas are off: plan limits are NOT enforced. "
            "Set USAGE_QUOTAS_ENABLED=true to enforce them."
        )
    return False
```

If `os` becomes unused in `enforcement.py`, ruff will report F401. It won't: the `BILLING_*` env reads still use it.

`billing_deps.py`: add `billing_checks_active` to the `from tldw_Server_API.app.core.Billing.enforcement import (...)` list, then make these replacements:
- In `get_billing_org_id`, `resolve_org_id_for_principal`, `require_within_limit` (`_check_limit`), `require_feature` (`_check_feature`) and `add_billing_headers`: replace `if not enforcement_enabled():` with `if not await billing_checks_active():`.
- In `LimitEnforcer.__aenter__`: replace `if enforcement_enabled():` with `if await billing_checks_active():`.
- Leave `LimitEnforcer.__aexit__` unchanged. A `LimitEnforcer` exists only for a resolved billing org, which means the checks are active.
- `enforcement_enabled` stays imported for `__aexit__`.

`transport.py`, in `enforce_rag_query_limit_for_org_context`: change the import list to `LimitCategory, billing_checks_active, get_billing_enforcer`, and replace `if not enforcement_enabled():` with `if not await billing_checks_active():`.

`startup_auth_runtime.py`: in `initialize_auth_runtime_services`, add `await _warn_if_plan_limits_unenforced()` right after `await _init_resource_governor(app)`. Add this helper next to `_init_resource_governor`:

```python
async def _warn_if_plan_limits_unenforced() -> None:
    """Log at startup when a billing repository is wired but usage quotas are off (spec 2 §1)."""
    from tldw_Server_API.app.core.Billing.enforcement import billing_checks_active

    await billing_checks_active()
```

`tests/conftest.py`: add this fixture (module level, with the other fixtures):

```python
@pytest.fixture()
def billing_repo_wired(monkeypatch):
    """Simulate the hosted product: a billing repository is wired into SubscriptionService."""
    from tldw_Server_API.app.core.Billing import subscription_service

    async def _wired() -> bool:
        return True

    monkeypatch.setattr(subscription_service, "billing_repo_configured", _wired)
```

`tests/Billing/conftest.py`: add this autouse fixture:

```python
@pytest.fixture(autouse=True)
def _hosted_billing_repo(billing_repo_wired):
    """Billing tests exercise the hosted path, where a billing repository is wired (spec 2 §6)."""
    yield
```

- [ ] **Step 4: Run the new tests and the billing-dependent suites**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Usage/test_usage_quotas_billing_gate.py tldw_Server_API/tests/Billing tldw_Server_API/tests/RAG_NEW/unit/test_rag_transport_helpers.py tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py tldw_Server_API/tests/Audio/test_audio_transcriptions_hotwords.py`

Expected: the 5 new tests pass. Some existing tests that assert a billing block outside `tests/Billing` now fail, because no repo is wired: `test_enforce_rag_query_limit_for_org_context_uses_rag_daily_limit` and `..._raises_when_blocked` in `test_rag_transport_helpers.py`, and possibly billing-exit cases in the chat test.
- Add `@pytest.mark.usefixtures("billing_repo_wired")` to each failing test that exercises a billing block or a billing org; these exercise the hosted path.
- Re-run until all pass.
- If a failure is not about a billing block, stop and investigate. It is not covered by this step.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Billing tldw_Server_API/app/api/v1/API_Deps/billing_deps.py tldw_Server_API/app/core/RAG/rag_service/transport.py tldw_Server_API/app/services/startup_auth_runtime.py tldw_Server_API/tests
git commit -m "feat(quotas): billing checks need the switch and a wired billing repo (spec 2 §6)

OSS installs never wire a billing repository, so plan checks, org
resolution and usage aggregation stop running there, and orgless
multi-user accounts no longer get 403. The hosted path is unchanged
under the switch; a wired repo with the switch off warns once.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 3: Audio, which covers minutes, concurrency, the worker and the upload cap

**Files:**
- Modify: `tldw_Server_API/app/core/Usage/audio_quota.py`: the import, `_UNLIMITED_AUDIO_LIMITS`, `get_limits_for_user`, `can_start_job` and `can_start_stream`.
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py`: `_max_upload_bytes()` and the upload-cap block (near lines 619-645).
- Create: `tldw_Server_API/tests/Usage/test_usage_quotas_audio.py`.

**Interfaces:**
- Consumes: `config.usage_quotas_enabled()`.
- Produces:
  - `audio_quota.get_limits_for_user()` returns `None` for every limit when quotas are off. That covers the HTTP, WebSocket and realtime paths and the audio Jobs worker, which reads `concurrent_jobs` (`None` means 0, which means unlimited).
  - `audio_transcriptions._max_upload_bytes(limits: dict) -> int`.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_usage_quotas_audio.py`:

```python
"""Audio quotas are unlimited when usage quotas are off (spec 2)."""

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files
from tldw_Server_API.app.core.Usage import audio_quota

pytestmark = pytest.mark.unit


@pytest.fixture()
def quotas_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stock install: neither switch spelling is set."""
    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: None)


class _DenyingGovernor:
    """A governor that fails the test if it is consulted."""

    async def reserve(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("the governor must not be consulted when quotas are off")

    async def release(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("no lease was taken, so none may be released")


async def _denying_governor() -> _DenyingGovernor:
    return _DenyingGovernor()


async def test_limits_are_unlimited_when_quotas_off(quotas_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_tier_lookup(_user_id: int) -> str:
        raise AssertionError("the tier lookup must not run when quotas are off")

    monkeypatch.setattr(audio_quota, "get_user_tier", _no_tier_lookup)
    limits = await audio_quota.get_limits_for_user(1)
    assert limits == {"daily_minutes": None, "concurrent_streams": None, "concurrent_jobs": None, "max_file_size_mb": None}


async def test_daily_minutes_allowed_past_the_old_free_tier(quotas_off: None) -> None:
    allowed, remaining = await audio_quota.check_daily_minutes_allow(1, 31.0)
    assert allowed is True
    assert remaining is None


async def test_concurrency_unlimited_when_quotas_off(quotas_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_quota, "_get_audio_rg_governor", _denying_governor)
    assert await audio_quota.can_start_job(1) == (True, "OK")
    assert await audio_quota.can_start_stream(1) == (True, "OK")


async def test_finish_without_lease_is_a_noop(quotas_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_quota, "_get_audio_rg_governor", _denying_governor)
    await audio_quota.can_start_stream(5)
    await audio_quota.finish_stream(5)
    await audio_quota.can_start_job(5)
    await audio_quota.finish_job(5)


async def test_quotas_on_keeps_the_tier_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")

    async def _free(_user_id: int) -> str:
        return "free"

    async def _no_overrides(_user_id: int) -> dict:
        return {}

    monkeypatch.setattr(audio_quota, "get_user_tier", _free)
    monkeypatch.setattr(audio_quota, "_get_user_override_limits", _no_overrides)
    limits = await audio_quota.get_limits_for_user(1)
    assert limits["daily_minutes"] == audio_quota.TIER_LIMITS["free"]["daily_minutes"]


def test_upload_cap_falls_back_to_the_media_processing_cap() -> None:
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_transcriptions

    assert audio_transcriptions._max_upload_bytes({"max_file_size_mb": None}) == Audio_Files.MAX_FILE_SIZE
    assert audio_transcriptions._max_upload_bytes({}) == Audio_Files.MAX_FILE_SIZE
    assert audio_transcriptions._max_upload_bytes({"max_file_size_mb": "junk"}) == Audio_Files.MAX_FILE_SIZE
    assert audio_transcriptions._max_upload_bytes({"max_file_size_mb": 25}) == 25 * 1024 * 1024
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_usage_quotas_audio.py -q`
Expected: FAIL. `get_limits_for_user` returns the free tier, and `_max_upload_bytes` doesn't exist.

- [ ] **Step 3: Implement**

`audio_quota.py`: add a top-level import outside the optional-RG `try` block, right after the existing `from loguru import logger` import group:

```python
from tldw_Server_API.app.core.config import usage_quotas_enabled
```

Add this constant above `get_limits_for_user`, and make the change shown at the top of the function:

```python
# Every audio limit when usage quotas are off: no tier applies (spec 2).
_UNLIMITED_AUDIO_LIMITS: dict[str, float | None] = {
    "daily_minutes": None,
    "concurrent_streams": None,
    "concurrent_jobs": None,
    "max_file_size_mb": None,
}


async def get_limits_for_user(user_id: int) -> dict[str, float | None]:
    if not usage_quotas_enabled():
        return dict(_UNLIMITED_AUDIO_LIMITS)
    tier = await get_user_tier(user_id)
    # (rest of the function unchanged)
```

Make the first statement of both `can_start_job` and `can_start_stream`, before `gov = await _get_audio_rg_governor()`:

```python
    if not usage_quotas_enabled():
        return True, "OK"
```

`audio_transcriptions.py`: add this module-level helper (near the other private helpers above the route functions):

```python
def _max_upload_bytes(limits: dict[str, Any]) -> int:
    """The user's upload cap from their audio limits, else [Media-Processing] max_audio_file_size_mb."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio_files

    cap_mb = limits.get("max_file_size_mb")
    if cap_mb:
        try:
            return int(float(cap_mb) * 1024 * 1024)
        except (TypeError, ValueError):
            logger.warning("Could not parse max_file_size_mb {!r}; using the media-processing cap", cap_mb)
    return int(audio_files.MAX_FILE_SIZE)
```

Replace the upload-cap block (the `limits = await _audio_shim_attr("get_limits_for_user")(...)` try/except through the `max_file_size = 25 * 1024 * 1024` fallback) with:

```python
    try:
        limits = await _audio_shim_attr("get_limits_for_user")(current_user.id)
    except EXPECTED_DB_EXC as e:
        logger.exception(
            'Failed to get limits for user {} during upload; using the media-processing cap: {}; request_id={}',
            current_user.id,
            e,
            rid,
        )
        limits = {}
    max_file_size = _max_upload_bytes(limits)
```

Check that `Any` is imported in `audio_transcriptions.py` (`from typing import Any`); add it if not.

- [ ] **Step 4: Run the new tests and the audio suites**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Usage tldw_Server_API/tests/Audio tldw_Server_API/tests/AudioJobs tldw_Server_API/tests/Resource_Governance/test_rg_cutover_audio_quota.py`
Expected: all pass. The existing audio quota tests run with the session switch on (`LIMIT_ENFORCEMENT_ENABLED=true`), so their tier assertions still hold. A test that expects the 25 MB fallback after a limits *lookup error* needs updating to `Audio_Files.MAX_FILE_SIZE`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Usage/audio_quota.py tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py tldw_Server_API/tests
git commit -m "feat(quotas): audio limits unlimited when usage quotas are off (spec 2)

get_limits_for_user is the one source for daily minutes, the worker's
concurrent_jobs and the upload cap, so one guard covers HTTP, WebSocket,
realtime and the Jobs worker; consume_daily_minutes still records to the
ledger. Concurrency gates return OK before touching the governor. The
upload cap falls back to [Media-Processing] max_audio_file_size_mb
instead of a hard-coded 25 MB.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 4: Media ingest, storage, workflows and chatbooks

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`: `_resolve_media_budget_context` (near line 245) and its import.
- Modify: `tldw_Server_API/app/services/storage_quota_service.py`: `check_quota` and `check_combined_quota`.
- Modify: `tldw_Server_API/app/api/v1/API_Deps/storage_quota_guard.py`: `_is_enabled`.
- Modify: `tldw_Server_API/app/api/v1/endpoints/workflows.py`: `_enforce_workflows_daily_cap`.
- Modify: `tldw_Server_API/app/core/Chatbooks/quota_manager.py`: `QuotaManager.__init__`.
- Create: `tldw_Server_API/tests/Usage/test_usage_quotas_sites.py`.

**Interfaces:**
- Consumes: `config.usage_quotas_enabled()`.
- Produces: no new names. Each choke point admits everything when quotas are off.
- **Known gap until PR B:** today the media daily-bytes ledger row is written only inside the RG cap block, which this task skips when quotas are off. Until then, uploads made with quotas off are not counted toward a day's bytes. PR B moves that write out of the block so it always runs, as spec §4 requires. Every other counter (audio minutes, workflow runs, chatbooks jobs, storage usage) keeps recording in PR A.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_usage_quotas_sites.py`:

```python
"""Media, storage, workflows and chatbooks quotas admit everything when usage quotas are off (spec 2)."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.API_Deps import storage_quota_guard
from tldw_Server_API.app.api.v1.endpoints import workflows as workflows_ep
from tldw_Server_API.app.core.Chatbooks.quota_manager import QuotaManager
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

pytestmark = pytest.mark.unit


@pytest.fixture()
def quotas_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stock install: neither switch spelling is set."""
    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: None)


@pytest.fixture()
def quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator who turned usage quotas on."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")


def _media_request() -> SimpleNamespace:
    """A request whose app carries a governor and a policy with media caps."""
    loader = SimpleNamespace(get_policy=lambda _pid: {"jobs": {"max_concurrent": 2}})
    app = SimpleNamespace(state=SimpleNamespace(rg_governor=object(), rg_policy_loader=loader))
    return SimpleNamespace(app=app, state=SimpleNamespace())


def test_media_budget_context_is_empty_when_quotas_off(quotas_off: None) -> None:
    gov, _policy_id, policy, entity = persistence._resolve_media_budget_context(
        request=_media_request(), current_user=SimpleNamespace(id=1)
    )
    assert gov is None and policy == {} and entity == ""


def test_media_budget_context_unchanged_when_quotas_on(quotas_on: None) -> None:
    gov, _policy_id, policy, entity = persistence._resolve_media_budget_context(
        request=_media_request(), current_user=SimpleNamespace(id=1)
    )
    assert gov is not None and policy["jobs"]["max_concurrent"] == 2 and entity == "user:1"


def _full_storage_service(monkeypatch: pytest.MonkeyPatch) -> StorageQuotaService:
    """A storage service whose user is exactly at the old 5 GB default."""
    service = StorageQuotaService(db_pool=object(), settings=SimpleNamespace())
    service._initialized = True

    async def _info(_user_id: int) -> dict:
        return {"storage_used_mb": 5120.0, "storage_quota_mb": 5120}

    monkeypatch.setattr(service, "_get_user_storage_info", _info)
    return service


async def test_storage_never_raises_when_quotas_off(quotas_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _full_storage_service(monkeypatch)
    has_quota, info = await service.check_quota(1, 1024 * 1024, raise_on_exceed=True)
    assert has_quota is True and info["has_quota"] is True
    has_combined, combined = await service.check_combined_quota(1, 1024 * 1024, raise_on_exceed=True)
    assert has_combined is True and combined["blocking_level"] is None


async def test_storage_still_enforced_when_quotas_on(quotas_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _full_storage_service(monkeypatch)
    has_quota, _info = await service.check_quota(1, 1024 * 1024)
    assert has_quota is False


def test_storage_pool_guard_disabled_when_quotas_off(quotas_off: None) -> None:
    assert storage_quota_guard._is_enabled() is False


async def test_workflows_cap_skipped_when_quotas_off(quotas_off: None) -> None:
    class _ExplodingRequest:
        """Any attribute access means the cap logic ran."""

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"the workflows cap must not inspect the request ({name})")

    await workflows_ep._enforce_workflows_daily_cap(
        request=_ExplodingRequest(), current_user=SimpleNamespace(id=1), db=None
    )


def test_chatbooks_quotas_follow_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CHATBOOKS_DISABLE_QUOTAS", "TEST_MODE", "TESTING", "PYTEST_CURRENT_TEST", "LIMIT_ENFORCEMENT_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")
    assert QuotaManager(1, "free")._quotas_disabled is True
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    assert QuotaManager(1, "free")._quotas_disabled is False
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_usage_quotas_sites.py -q`
Expected: FAIL. The quotas-off tests fail; the quotas-on tests already pass.

- [ ] **Step 3: Implement the five gates**

`persistence.py`: extend the existing import to `from tldw_Server_API.app.core.config import loaded_config_data, settings, usage_quotas_enabled`. In `_resolve_media_budget_context`, change the first guard to:

```python
    # Usage quotas off (spec 2): no media concurrency or daily-bytes budget applies.
    if request is None or not usage_quotas_enabled():
        return None, _MEDIA_INGESTION_POLICY_ID, {}, ""
```

`storage_quota_service.py`: add the import `from tldw_Server_API.app.core.config import usage_quotas_enabled` with the module's other `tldw_Server_API.app.core` imports. In `check_quota`, replace `has_quota = projected_mb <= quota_mb` with:

```python
        # Usage quotas off (spec 2): report usage, never block.
        has_quota = projected_mb <= quota_mb or not usage_quotas_enabled()
```

In `check_combined_quota`, replace `has_quota = has_user_quota and has_team_quota and has_org_quota` with:

```python
        has_quota = (has_user_quota and has_team_quota and has_org_quota) or not usage_quotas_enabled()
```

`storage_quota_guard.py`: add the import and change `_is_enabled`:

```python
def _is_enabled() -> bool:
    """True when usage quotas are on and storage quota enforcement isn't explicitly disabled."""
    if not usage_quotas_enabled():
        return False
    val = os.getenv("STORAGE_QUOTA_ENFORCEMENT", "1").strip().lower()
    return val not in ("0", "false", "no", "off")
```

`workflows.py`: add the import, and make this the first statement of `_enforce_workflows_daily_cap`:

```python
    # Usage quotas off (spec 2). Runs are still recorded by _record_workflow_run_usage.
    if not usage_quotas_enabled():
        return
```

`quota_manager.py`: add the import, and change the `_quotas_disabled` assignment to:

```python
        self._quotas_disabled = (
            not usage_quotas_enabled()
            or _env_flag("CHATBOOKS_DISABLE_QUOTAS")
            or _env_flag("TEST_MODE")
            or _env_flag("TESTING")
            or bool(os.getenv("PYTEST_CURRENT_TEST"))
        )
```

- [ ] **Step 4: Run the new tests and the affected suites**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Usage tldw_Server_API/tests/Storage tldw_Server_API/tests/Chatbooks tldw_Server_API/tests/MediaIngestion_NEW/unit tldw_Server_API/tests/Workflows tldw_Server_API/tests/Resource_Governance/test_workflows_runs_daily_cap.py tldw_Server_API/tests/Resource_Governance/test_e2e_workflows_daily_cap.py tldw_Server_API/tests/Resource_Governance/test_resource_governor_endpoint.py tldw_Server_API/tests/Billing/test_storage_quota_guard.py tldw_Server_API/tests/Admin/test_admin_storage_quotas.py`
Expected: all pass, because the session switch is on.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py tldw_Server_API/app/services/storage_quota_service.py tldw_Server_API/app/api/v1/API_Deps/storage_quota_guard.py tldw_Server_API/app/api/v1/endpoints/workflows.py tldw_Server_API/app/core/Chatbooks/quota_manager.py tldw_Server_API/tests/Usage/test_usage_quotas_sites.py
git commit -m "feat(quotas): media, storage, workflows and chatbooks quotas follow the switch (spec 2)

One guard per choke point: the media budget context, check_quota and
check_combined_quota (every storage site routes through them), the
storage-pool guard, the workflows daily cap, and QuotaManager. Counters
and usage recording are untouched.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 5: Evaluations daily caps leave the stock policy

The `evals.*` RG policies put the per-minute rate (spec 1, which stays) and the daily cap in the same `evaluations` category, and send both in one reserve. So code can't gate one without the other. PR A removes the daily caps from the stock policy. PR B brings per-user caps back through `limits.evaluations_per_day` and `limits.evaluation_tokens_per_day`, read from the evaluations DB. Until PR B, evaluations have no daily cap even with the switch on. That is acceptable, because no hosted billing plan uses them.

**Files:**
- Modify: `tldw_Server_API/Config_Files/resource_governor_policies.yaml`: the `evals.default` and `evals.{free,basic,premium,enterprise}[.batch]` blocks (near lines 129-171).
- Create: `tldw_Server_API/tests/Usage/test_usage_quotas_policy_defaults.py`.

- [ ] **Step 1: Write the failing test**

```python
"""No stock RG policy carries a usage quota for evaluations (spec 2)."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_POLICIES = Path(__file__).resolve().parents[2] / "Config_Files" / "resource_governor_policies.yaml"


def test_stock_evaluation_policies_have_no_daily_caps() -> None:
    policies = yaml.safe_load(_POLICIES.read_text())["policies"]
    offenders = [
        f"{pid}.{category}"
        for pid, policy in policies.items()
        if pid.startswith("evals.")
        for category, spec in policy.items()
        if isinstance(spec, dict) and "daily_cap" in spec
    ]
    assert offenders == []
```

Check the YAML's top-level key holding the policy map. If it is not `policies`, use the key that `policy_loader` reads.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_usage_quotas_policy_defaults.py -q`
Expected: FAIL listing `evals.default.evaluations`, `evals.default.tokens`, `evals.free.evaluations` and the others.

- [ ] **Step 3: Edit the YAML**

- In `evals.default`, delete the `evaluations: { daily_cap: 500 }` line, and change `tokens: { per_min: 1000000, burst: 1.0, daily_cap: 500000 }` to `tokens: { per_min: 1000000, burst: 1.0 }`.
- In each of the eight tier policies, remove the `daily_cap: N` entry from both the `evaluations` and the `tokens` flow mappings, keeping `per_min` and `burst`. For example, `evals.free` becomes:

```yaml
  evals.free:
    evaluations: { per_min: 10, burst: 1.5 }
    tokens: { per_min: 0, burst: 1.0 }
    scopes: [user, api_key]
```

- Update the comment above `evals.default`: "The Evaluations module reserves `evaluations` (+ optional `tokens`) for per-minute rates; per-user daily caps are UserProfiles `limits.*` keys (spec 2)."

- [ ] **Step 4: Run the test, the RG suite and the evaluations suite**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Usage tldw_Server_API/tests/Resource_Governance tldw_Server_API/tests/Evaluations/unit`
Expected: all pass.
- A test that pinned an `evals.*` stock daily cap fails here, for example `test_rg_cutover_evals_authnz_character_web.py`. Rewrite it to build its own policy with a `daily_cap`, so it keeps covering the governor's daily-cap path without depending on the stock file.
- Tests that already inject their own policies are unaffected.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/Config_Files/resource_governor_policies.yaml tldw_Server_API/tests
git commit -m "feat(quotas): drop evaluations daily caps from the stock RG policy (spec 2)

Per-minute evaluation rates stay. Per-user daily caps return in PR B as
limits.evaluations_per_day / limits.evaluation_tokens_per_day.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 6: Docs

**Files:**
- Modify: `Docs/Operations/Env_Vars.md`: a new `## Usage Quotas` section just above `## Resource Governor (Unified Rate Limiting)`.
- Modify: `Docs/Operations/Rate_Limits_Troubleshooting.md`: the billing row (line 14) and one paragraph.
- Run: `Helper_Scripts/refresh_docs_published.sh` (`Env_Vars.md` is mirrored; publish it at the site root, with no `../` links).

- [ ] **Step 1: Write the Env_Vars section**

```markdown
## Usage Quotas

Usage quotas are per-user budgets: audio minutes, storage, chatbook exports and imports, media ingest bytes and concurrency, workflow runs, evaluation caps, and billing-plan limits. They are **off by default**: a stock install, single-user or multi-user, applies none of them. Request rate limits are separate (Resource Governor, below). Design: `Docs/Design/2026-10-02-usage-quota-posture-design.md`.

- `USAGE_QUOTAS_ENABLED`: master switch for every usage quota (`true|1|false|0`). Resolution: this env var > `LIMIT_ENFORCEMENT_ENABLED` (legacy, when set) > `config.txt` `[Usage-Quotas] enabled` > default `false`. With it off, quota checks never block, and usage is still recorded, so turning it on mid-day counts correctly.
- `LIMIT_ENFORCEMENT_ENABLED`: **deprecated** spelling of `USAGE_QUOTAS_ENABLED`. It is honored only when `USAGE_QUOTAS_ENABLED` is unset, and logs a one-time warning. Its old default was `true`. A deploy that relied on that default must now set `USAGE_QUOTAS_ENABLED=true`.
- Billing-plan limits additionally need a billing repository, which only the hosted product wires in. Without one, billing checks never run, even with quotas on, and accounts without an organization are never refused. If a billing repository is wired while quotas are off, the server logs a warning at startup and on the first billing check.
```

- [ ] **Step 2: Update the troubleshooting row and add a paragraph**

Replace the line-14 row's last cell with: "Usage quotas are off by default (`USAGE_QUOTAS_ENABLED`, see `Env_Vars.md`). Billing checks also need a wired billing repository (hosted product only); otherwise raise the org's plan limits."

Add this paragraph under the table:

```markdown
**Usage quotas versus rate limits.** A 402, a 413 `storage_quota_exceeded`, or a "daily quota" 429 comes from a usage quota, not from the Resource Governor's rate limits. Usage quotas are off unless `USAGE_QUOTAS_ENABLED` is on; see the Usage Quotas section of `Env_Vars.md`.
```

- [ ] **Step 3: Refresh the published mirror and run the docs tests**

```bash
bash Helper_Scripts/refresh_docs_published.sh
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Docs
```

Expected: all pass. Stage only the `Docs/Published` files the script changed.

- [ ] **Step 4: Commit**

```bash
git add Docs/Operations/Env_Vars.md Docs/Operations/Rate_Limits_Troubleshooting.md Docs/Published
git commit -m "docs(quotas): USAGE_QUOTAS_ENABLED, off by default; legacy LIMIT_ENFORCEMENT_ENABLED

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 7: Ship PR A

- [ ] **Step 1: Sweep every suite that touches a changed default**

Changing a global default means grepping every test that references the affected symbols and running each hit:

```bash
git grep -l -e enforcement_enabled -e LIMIT_ENFORCEMENT_ENABLED -e get_limits_for_user -e can_start_job -e can_start_stream \
  -e check_quota -e check_combined_quota -e guard_storage_quota -e _enforce_workflows_daily_cap -e QuotaManager \
  -e _resolve_media_budget_context -e billing_deps -e "evals\." -- tldw_Server_API/tests | sort > /tmp/pra_hits.txt
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 $(cat /tmp/pra_hits.txt) tldw_Server_API/tests/Usage tldw_Server_API/tests/Docs
```

Expected: everything passes, or fails identically on `origin/dev`. To compare, run the failing test IDs on a detached `origin/dev` checkout in this worktree. Fix every new failure before shipping.

- [ ] **Step 2: Ruff and Bandit on the touched production files**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m ruff check $(git diff --name-only origin/dev...HEAD -- '*.py')
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -q -r $(git diff --name-only origin/dev...HEAD -- 'tldw_Server_API/app/*.py')
```

Expected: no new findings.

- [ ] **Step 3: Get the owner's confirmation, then coordinate and open the PR**

1. **The owner.** Ask them to confirm that the hosted deploy sets `USAGE_QUOTAS_ENABLED=true`; this is spec §8 and blocks the merge.
2. **The peer session.** Message `tldw-server-03` with the branch, the touched areas (billing deps, audio quota, storage, workflows, chatbooks, RG policy YAML, Env_Vars and its mirror), and a request for a merge slot. Wait for its go-ahead before merging; pushing and CI may start earlier.
3. **Open the PR.**

```bash
git push -u origin fix/usage-quotas-off-by-default
gh pr create --base dev --title "feat(quotas): usage quotas off by default behind USAGE_QUOTAS_ENABLED (spec 2, PR A)" --body-file <body>
```

The body lists:
- the switch and its precedence;
- each gated site;
- the billing-repo requirement;
- the evals daily-cap removal;
- the owner prerequisite;
- the verification counts.

It ends with the waiver line and the Claude Code footer.

- [ ] **Step 4: Run the Qodo loop and merge**

- **Qodo.** Address every Qodo finding: fix it, or decline it with a posted rationale. Add the accepted titles to the scratchpad `qodo_open.py` ACCEPTED map.
- **Merge.** Only on the peer's go-ahead, run the merge queue for this PR alone.
- **After the merge,** message the peer, and append the merge to `<PARENT>`: `backlog task edit <PARENT> --check-ac 1 --append-notes "PR A #<n> merged <sha>: ..."`.
