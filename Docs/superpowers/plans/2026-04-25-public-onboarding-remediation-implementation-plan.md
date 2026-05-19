# Public Onboarding Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Docker single-user + WebUI, Docker multi-user + Postgres, and local single-user setup usable from a fresh clone with matching prepare, start, verify, first-value, and audio guidance.

**Architecture:** `tldw-setup` becomes the shared profile-aware setup and verification layer. The Makefile delegates public setup/start/verify targets to `tldw-setup` and profile-specific Docker Compose files instead of embedding fragile shell setup logic. Documentation presents the three public profiles as peers and uses the same lifecycle headings.

**Tech Stack:** Python 3.10+, Typer, FastAPI, httpx, pytest, Make, Docker Compose, Markdown docs.

---

## Source Spec

- Design spec: `Docs/superpowers/specs/2026-04-25-public-onboarding-remediation-design.md`
- Review findings: `/tmp/tldw_server_public_onboarding_review_wt/Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-findings.md`
- Review synthesis: `/tmp/tldw_server_public_onboarding_review_wt/Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-synthesis.md`

## Stage Summary

## Stage 1: CLI And Command Contract
**Goal:** Add profile-aware env resolution, prepare, and verification behavior to the existing `tldw-setup` CLI, then make public Make targets delegate to it.
**Success Criteria:** Profile commands write the expected env file, generated secrets are masked in default output, and old aliases cannot report success without verification.
**Tests:** Wizard profile tests, wizard verify tests, Makefile command-boundary tests.
**Status:** Complete

## Stage 2: Docker Single-User + WebUI
**Goal:** Provide a single-user Docker compose path that starts API + WebUI without Postgres and with writable app data.
**Success Criteria:** Clean-volume startup reaches `/health`, `/ready`, `/docs`, `/api/v1/config/quickstart`, and WebUI readiness.
**Tests:** Docker compose contract tests, Makefile target tests, runtime validation.
**Status:** Complete

## Stage 3: Docker Multi-User + Postgres
**Goal:** Provide a multi-user Docker compose path with bundled Postgres, required secrets, env-driven first admin, and bearer-token verification.
**Success Criteria:** Clean-volume startup reaches Postgres readiness, admin login, `/api/v1/auth/me`, first ingest/search, and provider-missing chat readiness.
**Tests:** Docker compose contract tests, wizard verify tests, runtime validation.
**Status:** Complete

## Stage 4: Local Single-User
**Goal:** Split local install, setup, start, and verify commands so local onboarding is predictable and not dev-server-only.
**Success Criteria:** `make install-local` and `make quickstart-install` never start a server; `make start-local-single` starts plain `uvicorn`; `make verify-local-single` validates auth and first value.
**Tests:** Makefile command-boundary tests, wizard verify tests, runtime validation from clean `.venv` and env.
**Status:** Complete

## Stage 5: Docs And Runtime Validation
**Goal:** Update public docs, audio auth examples, Windows/WSL guidance, and capture clean-state validation transcripts.
**Success Criteria:** All public docs show the three peer profiles with the same lifecycle and the validation logs prove the documented paths work.
**Tests:** Docs tests, docs scripts, runtime Docker/local validation commands, Bandit on touched code.
**Status:** Complete

## File Map

### CLI Wizard

- Create `tldw_Server_API/cli/wizard/profiles.py`
  - Owns public profile names, repo-root detection, default env paths, secret generation, and profile env defaults.
- Create `tldw_Server_API/cli/wizard/profile_verify.py`
  - Owns HTTP verification helpers for health/docs/quickstart/auth/provider/first-value checks.
- Modify `tldw_Server_API/cli/wizard/cli.py`
  - Adds `--profile`, `--env-file`, `--base-url`, `--webui-url`, `--first-value`, and Docker/no-spawn verification behavior.
  - Removes scaffold wording from public output.
  - Keeps JSON output stable for tests.

### Makefile And Shell Helpers

- Modify `.gitignore`
  - Ignore the lightweight `.setup-venv` used by Docker-first wizard commands.
- Modify `Makefile`
  - Add a lightweight `.setup-venv` bootstrap for the wizard so Docker-first users do not need the full local Python install path.
  - Add public peer targets:
    - `setup-docker-single`
    - `start-docker-single`
    - `verify-docker-single`
    - `setup-docker-multi`
    - `start-docker-multi`
    - `verify-docker-multi`
    - `install-local`
    - `setup-local-single`
    - `start-local-single`
    - `verify-local-single`
  - Rewire compatibility aliases:
    - `quickstart`
    - `quickstart-docker-webui`
    - `quickstart-docker`
    - `quickstart-install`
    - `quickstart-local`
  - Ensure default output does not print full secrets.

### Docker

- Create `Dockerfiles/docker-compose.single-user.yml`
  - API + Redis + named volumes, no Postgres dependency.
- Create `Dockerfiles/docker-compose.multi-user-postgres.yml`
  - API + Postgres + Redis with `postgres:18-bookworm` volume mounted at `/var/lib/postgresql`.
- Modify `Dockerfiles/docker-compose.webui.yml`
  - Keep same-origin proxy defaults and make it work as an overlay on either public Docker profile when selected.
- Modify `Dockerfiles/entrypoints/tldw-app-first-run.sh`
  - Make auth/bootstrap failures easier to diagnose.
  - Keep env-driven admin creation as the public multi-user path.
- Modify `Dockerfiles/README.md`
  - Document the two profile-specific compose files and volume migration note.

### Docs

- Modify `README.md`
- Modify `Docs/Getting_Started/README.md`
- Modify `Docs/Getting_Started/Profile_Docker_Single_User.md`
- Modify `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- Modify `Docs/Getting_Started/Profile_Local_Single_User.md`
- Modify `Docs/Getting_Started/QUICKSTART.md`
- Modify `Docs/Deployment/setup-wizard-guide.md`
- Modify `Docs/Getting_Started/First_Time_Audio_Setup_CPU.md`
- Modify `Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md`
- Modify `Docs/Website/index.html`
- Modify the matching mirrors under `Docs/Published/Getting_Started/`

### Tests

- Create `tldw_Server_API/tests/wizard/test_cli_profiles.py`
- Create `tldw_Server_API/tests/wizard/test_cli_verify_profiles.py`
- Modify `tldw_Server_API/tests/wizard/test_cli_basic.py`
- Modify `tldw_Server_API/tests/wizard/test_cli_verify.py`
- Modify `tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py`
- Create `tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py`
- Create `tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py`
- Modify `tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py`
- Create `tldw_Server_API/tests/Docs/test_public_onboarding_profile_parity.py`

### Runtime Validation Artifacts

- Create `Docs/superpowers/reviews/public-onboarding-remediation/2026-04-25-runtime-validation.md`
  - Records final clean-state commands, outcomes, and known environment details.

## Task 1: Add Profile Contract Helpers

**Files:**
- Create: `tldw_Server_API/cli/wizard/profiles.py`
- Test: `tldw_Server_API/tests/wizard/test_cli_profiles.py`

- [ ] **Step 1: Write failing profile helper tests**

Create `tldw_Server_API/tests/wizard/test_cli_profiles.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.cli.wizard import profiles


def test_normalize_profile_accepts_public_names() -> None:
    assert profiles.normalize_profile("docker-single-webui").name == "docker-single-webui"
    assert profiles.normalize_profile("docker-multi-postgres").auth_mode == "multi_user"
    assert profiles.normalize_profile("local-single").auth_mode == "single_user"


def test_normalize_profile_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported setup profile"):
        profiles.normalize_profile("docker-team")


def test_repo_checkout_env_defaults_to_config_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "tldw_Server_API" / "Config_Files").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='tldw-server'\n", encoding="utf-8")

    env_path = profiles.resolve_env_path(
        profile=profiles.normalize_profile("docker-single-webui"),
        start_dir=root / "Docs",
        explicit_env_file=None,
    )

    assert env_path == root / "tldw_Server_API" / "Config_Files" / ".env"


def test_explicit_env_file_overrides_repo_default(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.env"

    env_path = profiles.resolve_env_path(
        profile=profiles.normalize_profile("local-single"),
        start_dir=tmp_path,
        explicit_env_file=explicit,
    )

    assert env_path == explicit


def test_single_user_defaults_generate_maskable_api_key() -> None:
    defaults = profiles.build_profile_env(
        profile=profiles.normalize_profile("local-single"),
        existing_env={},
    )

    assert defaults["AUTH_MODE"] == "single_user"
    assert defaults["SINGLE_USER_API_KEY"].startswith("tldw_")
    assert "DATABASE_URL" in defaults


def test_multi_user_defaults_include_required_secrets() -> None:
    defaults = profiles.build_profile_env(
        profile=profiles.normalize_profile("docker-multi-postgres"),
        existing_env={},
        admin_username="admin",
        admin_password="CorrectHorseBatteryStaple1!",
        admin_email="admin@example.com",
    )

    for key in (
        "AUTH_MODE",
        "DATABASE_URL",
        "JWT_SECRET_KEY",
        "SESSION_ENCRYPTION_KEY",
        "MCP_JWT_SECRET",
        "MCP_API_KEY_SALT",
        "BYOK_ENCRYPTION_KEY",
        "ADMIN_USERNAME",
        "ADMIN_PASSWORD",
        "ADMIN_EMAIL",
    ):
        assert defaults[key]
    assert defaults["AUTH_MODE"] == "multi_user"
    assert defaults["DATABASE_URL"].startswith("postgresql://")
```

- [ ] **Step 2: Run the profile helper tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/wizard/test_cli_profiles.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'tldw_Server_API.cli.wizard.profiles'
```

- [ ] **Step 3: Implement profile helpers**

Create `tldw_Server_API/cli/wizard/profiles.py`:

```python
from __future__ import annotations

import base64
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tldw_Server_API.cli.wizard.utils import env as env_utils


@dataclass(frozen=True)
class SetupProfile:
    name: str
    auth_mode: str
    docker: bool
    includes_webui: bool
    includes_postgres: bool
    default_base_url: str = "http://127.0.0.1:8000"
    default_webui_url: str | None = None


_PROFILES: dict[str, SetupProfile] = {
    "docker-single-webui": SetupProfile(
        name="docker-single-webui",
        auth_mode="single_user",
        docker=True,
        includes_webui=True,
        includes_postgres=False,
        default_webui_url="http://127.0.0.1:8080",
    ),
    "docker-multi-postgres": SetupProfile(
        name="docker-multi-postgres",
        auth_mode="multi_user",
        docker=True,
        includes_webui=False,
        includes_postgres=True,
    ),
    "local-single": SetupProfile(
        name="local-single",
        auth_mode="single_user",
        docker=False,
        includes_webui=False,
        includes_postgres=False,
    ),
}


def normalize_profile(value: str | None) -> SetupProfile:
    name = (value or "local-single").strip().lower().replace("_", "-")
    aliases = {
        "docker-single": "docker-single-webui",
        "docker-webui": "docker-single-webui",
        "docker-multi": "docker-multi-postgres",
        "local": "local-single",
    }
    name = aliases.get(name, name)
    try:
        return _PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unsupported setup profile '{value}'. Use one of: {choices}") from exc


def resolve_repo_root(start_dir: Path | None = None) -> Path | None:
    current = (start_dir or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "tldw_Server_API").is_dir():
            return candidate
    return None


def resolve_env_path(
    *,
    profile: SetupProfile,
    start_dir: Path | None = None,
    explicit_env_file: Path | None = None,
) -> Path:
    if explicit_env_file is not None:
        return explicit_env_file.expanduser().resolve()
    repo_root = resolve_repo_root(start_dir)
    if repo_root is not None:
        return repo_root / "tldw_Server_API" / "Config_Files" / ".env"
    return (start_dir or Path.cwd()).resolve() / ".env"


def _secret_token() -> str:
    return secrets.token_urlsafe(32)


def _byok_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def _existing_or_generated(existing_env: Mapping[str, str], key: str, generator) -> str:
    value = os.getenv(key) or existing_env.get(key)
    if value and not value.startswith("CHANGE_ME"):
        return value
    return generator()


def build_profile_env(
    *,
    profile: SetupProfile,
    existing_env: Mapping[str, str],
    admin_username: str | None = None,
    admin_password: str | None = None,
    admin_email: str | None = None,
) -> dict[str, str]:
    values: dict[str, str] = {"AUTH_MODE": profile.auth_mode}
    if profile.auth_mode == "single_user":
        values["DATABASE_URL"] = os.getenv("DATABASE_URL") or existing_env.get("DATABASE_URL") or "sqlite:///./Databases/users.db"
        values["SINGLE_USER_API_KEY"] = _existing_or_generated(
            existing_env,
            "SINGLE_USER_API_KEY",
            env_utils.generate_single_user_api_key,
        )
        return values

    values["DATABASE_URL"] = (
        os.getenv("DATABASE_URL")
        or existing_env.get("DATABASE_URL")
        or "postgresql://tldw_user:TestPassword123!@postgres:5432/tldw_users"
    )
    values["JWT_SECRET_KEY"] = _existing_or_generated(existing_env, "JWT_SECRET_KEY", _secret_token)
    values["SESSION_ENCRYPTION_KEY"] = _existing_or_generated(existing_env, "SESSION_ENCRYPTION_KEY", _secret_token)
    values["MCP_JWT_SECRET"] = _existing_or_generated(existing_env, "MCP_JWT_SECRET", _secret_token)
    values["MCP_API_KEY_SALT"] = _existing_or_generated(existing_env, "MCP_API_KEY_SALT", _secret_token)
    values["BYOK_ENCRYPTION_KEY"] = _existing_or_generated(existing_env, "BYOK_ENCRYPTION_KEY", _byok_key)
    if admin_username:
        values["ADMIN_USERNAME"] = admin_username
    if admin_password:
        values["ADMIN_PASSWORD"] = admin_password
    if admin_email:
        values["ADMIN_EMAIL"] = admin_email
    return values
```

- [ ] **Step 4: Run profile helper tests and verify they pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/wizard/test_cli_profiles.py -v
```

Expected:

```text
6 passed
```

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add tldw_Server_API/cli/wizard/profiles.py tldw_Server_API/tests/wizard/test_cli_profiles.py
git commit -m "feat: add onboarding profile helpers"
```

## Task 2: Make `tldw-setup init` Profile-Aware

**Files:**
- Modify: `tldw_Server_API/cli/wizard/cli.py`
- Test: `tldw_Server_API/tests/wizard/test_cli_basic.py`
- Test: `tldw_Server_API/tests/wizard/test_cli_profiles.py`

- [ ] **Step 1: Add failing CLI init tests**

Append to `tldw_Server_API/tests/wizard/test_cli_profiles.py`:

```python
from typer.testing import CliRunner

from tldw_Server_API.cli.wizard.cli import app
from tldw_Server_API.tests.wizard.helpers import assert_action_field, assert_wizard_json


runner = CliRunner()


def test_init_profile_writes_repo_env_path_in_dry_run(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    (repo / "tldw_Server_API" / "Config_Files").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='tldw-server'\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    result = runner.invoke(
        app,
        ["init", "--profile", "docker-single-webui", "--dry-run", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="init", status="ok")
    assert payload["paths"]["env"].endswith("tldw_Server_API/Config_Files/.env")
    actions = payload.get("actions") or []
    set_env = next(action["set_env"] for action in actions if "set_env" in action)
    assert_action_field(actions, "set_env", "AUTH_MODE", "single_user")
    assert str(set_env["SINGLE_USER_API_KEY"]).startswith("*")


def test_init_multi_user_profile_requires_admin_password_or_generates_recovery_note(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    (repo / "tldw_Server_API" / "Config_Files").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='tldw-server'\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    result = runner.invoke(
        app,
        [
            "init",
            "--profile",
            "docker-multi-postgres",
            "--admin-username",
            "admin",
            "--admin-password",
            "CorrectHorseBatteryStaple1!",
            "--admin-email",
            "admin@example.com",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="init", status="ok")
    actions = payload.get("actions") or []
    set_env = next(action["set_env"] for action in actions if "set_env" in action)
    assert_action_field(actions, "set_env", "AUTH_MODE", "multi_user")
    assert str(set_env["SESSION_ENCRYPTION_KEY"]).startswith("*")
    assert_action_field(actions, "set_env", "ADMIN_USERNAME", "admin")
    assert str(set_env["ADMIN_PASSWORD"]).startswith("*")
```

- [ ] **Step 2: Run tests and verify they fail on missing options**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/wizard/test_cli_profiles.py -v
```

Expected:

```text
No such option: --profile
```

- [ ] **Step 3: Update imports and `init` options**

In `tldw_Server_API/cli/wizard/cli.py`, add the import near existing wizard imports:

```python
from . import profiles as profile_utils
```

Update the `init` signature with these options:

```python
    profile: str | None = typer.Option(None, "--profile", help="Public setup profile"),
    env_file: Path | None = typer.Option(None, "--env-file", help="Explicit env file path"),
    admin_username: str | None = typer.Option(None, "--admin-username", help="Initial multi-user admin username"),
    admin_password: str | None = typer.Option(None, "--admin-password", help="Initial multi-user admin password"),
    admin_email: str | None = typer.Option(None, "--admin-email", help="Initial multi-user admin email"),
```

Inside `init`, replace the `base = Path(install_dir).resolve()` and `env_path = base / ".env"` block with:

```python
    base = Path(install_dir).resolve()
    setup_profile = profile_utils.normalize_profile(profile) if profile else None
    env_path = (
        profile_utils.resolve_env_path(
            profile=setup_profile,
            start_dir=base,
            explicit_env_file=env_file,
        )
        if setup_profile
        else base / ".env"
    )
```

Replace the existing `existing_env`, `auth_mode`, and `updates` initialization with:

```python
    existing_env = env_utils.load_env(env_path)
    updates: dict[str, str | None] = {}
    if setup_profile:
        updates.update(
            profile_utils.build_profile_env(
                profile=setup_profile,
                existing_env=existing_env,
                admin_username=admin_username,
                admin_password=admin_password,
                admin_email=admin_email,
            )
        )
        auth_mode = updates["AUTH_MODE"]
    else:
        auth_mode = os.getenv("AUTH_MODE") or existing_env.get("AUTH_MODE") or ("single_user" if default or yes else "")
```

Remove the dry-run note string `"this is a scaffold; future steps will initialize DBs and verify endpoints"` from the result.

- [ ] **Step 4: Ensure dry-run output masks secrets**

Keep the existing call:

```python
env_utils.mask_env_values({k: v for k, v in updates.items() if v is not None})
```

Verify the masked action is always the `set_env` action in profile dry-run output.

- [ ] **Step 5: Run CLI profile tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/wizard/test_cli_profiles.py tldw_Server_API/tests/wizard/test_cli_basic.py -v
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add tldw_Server_API/cli/wizard/cli.py tldw_Server_API/tests/wizard/test_cli_basic.py tldw_Server_API/tests/wizard/test_cli_profiles.py
git commit -m "feat: make setup wizard init profile aware"
```

## Task 3: Add Profile-Aware Verification

**Files:**
- Create: `tldw_Server_API/cli/wizard/profile_verify.py`
- Modify: `tldw_Server_API/cli/wizard/cli.py`
- Test: `tldw_Server_API/tests/wizard/test_cli_verify_profiles.py`
- Modify: `tldw_Server_API/tests/wizard/test_cli_verify.py`

- [ ] **Step 1: Write failing verification tests**

Create `tldw_Server_API/tests/wizard/test_cli_verify_profiles.py`:

```python
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tldw_Server_API.cli.wizard import profile_verify
from tldw_Server_API.cli.wizard import cli as wizard_cli
from tldw_Server_API.tests.wizard.helpers import assert_action_field, assert_wizard_json


runner = CliRunner()


def test_docker_profile_verify_does_not_spawn_ephemeral_server(monkeypatch) -> None:
    def fake_run_checks(*, profile, base_url, webui_url, env_path, first_value, timeout):
        return {
            "status": "ok",
            "actions": [
                {"server": {"mode": "existing", "profile": profile.name}},
                {"endpoints": {"health": {"ok": True}, "ready": {"ok": True}, "docs": {"ok": True}}},
            ],
            "notes": [],
        }

    def start_ephemeral(*_args, **_kwargs):
        raise AssertionError("docker profile verify must not spawn a local server")

    monkeypatch.setattr(profile_verify, "run_profile_checks", fake_run_checks)
    monkeypatch.setattr(wizard_cli, "_start_ephemeral_server", start_ephemeral)

    result = runner.invoke(wizard_cli.app, ["verify", "--profile", "docker-single-webui", "--json"])

    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="verify", status="ok")
    actions = payload.get("actions") or []
    assert_action_field(actions, "server", "profile", "docker-single-webui")


def test_verify_first_value_reports_provider_missing(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AUTH_MODE=single_user\nSINGLE_USER_API_KEY=tldw_test.key\n", encoding="utf-8")

    def fake_run_checks(*, profile, base_url, webui_url, env_path, first_value, timeout):
        return {
            "status": "ok",
            "actions": [
                {"chat": {"status": "provider_missing", "env_examples": ["OPENAI_API_KEY=sk-..."]}},
                {"first_value": {"ingest": "ok", "search": "ok"}},
            ],
            "notes": ["No provider key configured; chat verification skipped."],
        }

    monkeypatch.setattr(profile_verify, "run_profile_checks", fake_run_checks)

    result = runner.invoke(
        wizard_cli.app,
        [
            "verify",
            "--profile",
            "local-single",
            "--env-file",
            str(env_path),
            "--first-value",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="verify", status="ok")
    actions = payload.get("actions") or []
    assert_action_field(actions, "chat", "status", "provider_missing")
    assert_action_field(actions, "first_value", "search", "ok")
```

- [ ] **Step 2: Run verification tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/wizard/test_cli_verify_profiles.py -v
```

Expected:

```text
ImportError: cannot import name 'profile_verify'
```

- [ ] **Step 3: Implement `profile_verify.py`**

Create `tldw_Server_API/cli/wizard/profile_verify.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from . import profiles as profile_utils
from .utils import env as env_utils


def _request(
    method: str,
    base_url: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, str] | None = None,
    files: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(method, url, headers=headers, data=data, files=files, json=json_body)
        payload: Any
        try:
            payload = response.json()
        except ValueError:
            payload = response.text[:300]
        return {"url": url, "status_code": response.status_code, "ok": response.status_code < 400, "body": payload}
    except (httpx.HTTPError, OSError, TimeoutError, ValueError) as exc:
        return {"url": url, "ok": False, "error": str(exc)}


def _headers_for_profile(profile: profile_utils.SetupProfile, env_values: dict[str, str]) -> dict[str, str]:
    if profile.auth_mode == "single_user":
        return {"X-API-KEY": env_values.get("SINGLE_USER_API_KEY", "")}
    return {}


def _login_multi_user(base_url: str, env_values: dict[str, str], timeout: float) -> dict[str, Any]:
    username = env_values.get("ADMIN_USERNAME", "")
    password = env_values.get("ADMIN_PASSWORD", "")
    if not username or not password:
        return {"ok": False, "error": "admin_credentials_missing"}
    result = _request(
        "POST",
        base_url,
        "/api/v1/auth/login",
        data={"username": username, "password": password},
        timeout=timeout,
    )
    token = None
    if result.get("ok") and isinstance(result.get("body"), dict):
        token = result["body"].get("access_token")
    result["token_present"] = bool(token)
    if token:
        result["token"] = token
    return result


def _provider_check(base_url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    result = _request("GET", base_url, "/api/v1/llm/providers", headers=headers, timeout=timeout)
    if not result.get("ok"):
        return {"status": "endpoint_failed", "result": result}
    body = result.get("body") if isinstance(result.get("body"), dict) else {}
    total = int(body.get("total_configured") or 0)
    if total <= 0:
        return {"status": "provider_missing", "env_examples": ["OPENAI_API_KEY=sk-...", "ANTHROPIC_API_KEY=sk-ant-..."]}
    return {"status": "provider_configured", "total_configured": total}


def _first_value_check(base_url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    content = b"# tldw onboarding sample\n\nUnique phrase: tldw-onboarding-sample-search.\n"
    files = {"files": ("tldw-onboarding-sample.md", content, "text/markdown")}
    data = {
        "media_type": "document",
        "title": "tldw onboarding sample",
        "keywords": "onboarding,first-run",
        "perform_analysis": "false",
        "perform_chunking": "false",
    }
    ingest = _request("POST", base_url, "/api/v1/media/add", headers=headers, data=data, files=files, timeout=timeout)
    search = _request(
        "POST",
        base_url,
        "/api/v1/media/search",
        headers=headers,
        json_body={"query": "tldw-onboarding-sample-search", "fields": ["title", "content"]},
        timeout=timeout,
    )
    return {"ingest": "ok" if ingest.get("ok") else "failed", "search": "ok" if search.get("ok") else "failed", "details": {"ingest": ingest, "search": search}}


def run_profile_checks(
    *,
    profile: profile_utils.SetupProfile,
    base_url: str,
    webui_url: str | None,
    env_path: Path,
    first_value: bool,
    timeout: float = 5.0,
) -> dict[str, Any]:
    env_values = env_utils.load_env(env_path)
    actions: list[dict[str, Any]] = [{"server": {"mode": "existing", "profile": profile.name}}]
    notes: list[str] = []

    endpoint_results = {
        "health": _request("GET", base_url, "/health", timeout=timeout),
        "ready": _request("GET", base_url, "/ready", timeout=timeout),
        "docs": _request("GET", base_url, "/docs", timeout=timeout),
        "quickstart": _request("GET", base_url, "/api/v1/config/quickstart", timeout=timeout),
    }
    actions.append({"endpoints": endpoint_results})

    headers = _headers_for_profile(profile, env_values)
    if profile.auth_mode == "multi_user":
        login = _login_multi_user(base_url, env_values, timeout)
        actions.append({"login": {k: v for k, v in login.items() if k != "token"}})
        token = login.get("token")
        if token:
            headers = {"Authorization": f"Bearer {token}"}
    auth_me = _request("GET", base_url, "/api/v1/auth/me", headers=headers, timeout=timeout)
    actions.append({"auth": {"mode": profile.auth_mode, "me": auth_me}})

    chat = _provider_check(base_url, headers, timeout)
    if chat.get("status") == "provider_missing":
        notes.append("No provider key configured; chat verification skipped.")
    actions.append({"chat": chat})

    if first_value:
        actions.append({"first_value": _first_value_check(base_url, headers, timeout)})

    if profile.includes_webui and webui_url:
        actions.append({"webui": _request("GET", webui_url.rstrip("/"), "/", timeout=timeout)})

    ok = True
    for action in actions:
        if "endpoints" in action:
            ok = ok and all(item.get("ok") for item in action["endpoints"].values())
        if "auth" in action:
            ok = ok and bool(action["auth"]["me"].get("ok"))
        if "first_value" in action:
            ok = ok and action["first_value"]["ingest"] == "ok" and action["first_value"]["search"] == "ok"
    return {"status": "ok" if ok else "error", "actions": actions, "notes": notes}
```

- [ ] **Step 4: Update `verify` command options and flow**

In `tldw_Server_API/cli/wizard/cli.py`, import:

```python
from . import profile_verify
```

Update `verify` signature:

```python
    profile: str | None = typer.Option(None, "--profile", help="Public setup profile"),
    env_file: Path | None = typer.Option(None, "--env-file", help="Explicit env file path"),
    base_url: str | None = typer.Option(None, "--base-url", help="API base URL"),
    webui_url: str | None = typer.Option(None, "--webui-url", help="WebUI base URL"),
    first_value: bool = typer.Option(False, "--first-value", help="Run first ingest/search checks"),
```

At the start of `verify`, after `facts` and `actions` are initialized, add:

```python
    if profile:
        setup_profile = profile_utils.normalize_profile(profile)
        resolved_env = profile_utils.resolve_env_path(
            profile=setup_profile,
            start_dir=Path.cwd(),
            explicit_env_file=env_file,
        )
        resolved_base_url = base_url or setup_profile.default_base_url
        resolved_webui_url = webui_url or setup_profile.default_webui_url
        if dry_run:
            result = {
                "command": "verify",
                "status": "ok",
                "facts": {**facts, "profile": setup_profile.name},
                "actions": [{"server": {"mode": "dry_run", "profile": setup_profile.name}}],
                "notes": ["dry-run only; skipping profile probes."],
                "check_provider": bool(check_provider),
                "dry_run": True,
                "paths": {"env": str(resolved_env)},
            }
            _emit(result, json_out)
            raise typer.Exit(0)
        check_result = profile_verify.run_profile_checks(
            profile=setup_profile,
            base_url=resolved_base_url,
            webui_url=resolved_webui_url,
            env_path=resolved_env,
            first_value=first_value,
            timeout=5.0,
        )
        result = {
            "command": "verify",
            "status": check_result["status"],
            "facts": {**facts, "profile": setup_profile.name},
            "actions": check_result["actions"],
            "notes": check_result["notes"],
            "check_provider": bool(check_provider),
            "dry_run": dry_run,
            "paths": {"env": str(resolved_env)},
        }
        _emit(result, json_out)
        if check_result["status"] != "ok":
            raise typer.Exit(2)
        raise typer.Exit(0)
```

- [ ] **Step 5: Run wizard verify tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/wizard/test_cli_verify.py tldw_Server_API/tests/wizard/test_cli_verify_profiles.py -v
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add tldw_Server_API/cli/wizard/cli.py tldw_Server_API/cli/wizard/profile_verify.py tldw_Server_API/tests/wizard/test_cli_verify.py tldw_Server_API/tests/wizard/test_cli_verify_profiles.py
git commit -m "feat: verify public onboarding profiles"
```

## Task 4: Add Public Make Targets And Safe Aliases

**Files:**
- Modify: `Makefile`
- Modify: `.gitignore`
- Modify: `tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py`
- Create: `tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py`

- [ ] **Step 1: Write failing Makefile contract tests**

Create `tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py`:

```python
from __future__ import annotations

import re
from pathlib import Path


MAKEFILE = Path("Makefile").read_text(encoding="utf-8")


def _target_block(target: str) -> str:
    pattern = rf"^{re.escape(target)}:.*?(?=^[A-Za-z0-9_.-]+:|\Z)"
    match = re.search(pattern, MAKEFILE, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, f"Make target {target} should exist"
    return match.group(0)


def test_public_profile_targets_exist() -> None:
    for target in (
        "setup-wizard-tools",
        "setup-docker-single",
        "start-docker-single",
        "verify-docker-single",
        "setup-docker-multi",
        "start-docker-multi",
        "verify-docker-multi",
        "install-local",
        "setup-local-single",
        "start-local-single",
        "verify-local-single",
    ):
        _target_block(target)


def test_setup_targets_delegate_to_tldw_setup_profiles() -> None:
    assert ".setup-venv" in _target_block("setup-wizard-tools")
    assert "--profile docker-single-webui" in _target_block("setup-docker-single")
    assert "--profile docker-multi-postgres" in _target_block("setup-docker-multi")
    assert "--profile local-single" in _target_block("setup-local-single")


def test_verify_targets_use_first_value_checks() -> None:
    assert "--first-value" in _target_block("verify-docker-single")
    assert "--first-value" in _target_block("verify-docker-multi")
    assert "--first-value" in _target_block("verify-local-single")


def test_quickstart_install_is_install_only() -> None:
    block = _target_block("quickstart-install")
    assert "uvicorn" not in block
    assert "quickstart-local" not in block
    assert "start-local-single" not in block


def test_default_output_does_not_print_full_api_key() -> None:
    for target in ("quickstart", "quickstart-docker-webui", "start-docker-single"):
        assert "grep '^SINGLE_USER_API_KEY='" not in _target_block(target)
    assert "make show-api-key" in MAKEFILE
```

Modify `tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py`:

```python
def test_quickstart_install_is_install_only() -> None:
    text = Path("Makefile").read_text(encoding="utf-8")
    quickstart_install = _target_block(text, "quickstart-install")
    _require(
        "install-local" in quickstart_install,
        "quickstart-install should delegate to install-local",
    )
    _require(
        "quickstart-local" not in quickstart_install,
        "quickstart-install must not start the local server",
    )
```

- [ ] **Step 2: Run Makefile tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py -v
```

Expected:

```text
AssertionError: Make target setup-docker-single should exist
```

- [ ] **Step 3: Ignore the lightweight wizard virtualenv**

Add this line near the existing `.venv` entry in `.gitignore`:

```gitignore
.setup-venv
```

- [ ] **Step 4: Add Makefile variables**

In `Makefile`, add these variables near existing Docker variables:

```make
DOCKER_SINGLE_COMPOSE ?= Dockerfiles/docker-compose.single-user.yml
DOCKER_MULTI_COMPOSE ?= Dockerfiles/docker-compose.multi-user-postgres.yml
SETUP_VENV_DIR ?= .setup-venv
SETUP_VENV_PYTHON ?= $(SETUP_VENV_DIR)/bin/python
TLDW_SETUP ?= $(SETUP_VENV_PYTHON) -m tldw_Server_API.cli.wizard.cli
TLDW_BASE_URL ?= http://127.0.0.1:8000
TLDW_WEBUI_URL ?= http://127.0.0.1:8080
```

- [ ] **Step 5: Add the lightweight wizard bootstrap target**

Add this Makefile target above the public peer targets:

```make
setup-wizard-tools:
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "[setup-wizard-tools] $(PYTHON) not found. Install Python 3.10+ and retry." && exit 1)
	@$(PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || (echo "[setup-wizard-tools] Python 3.10+ is required." && exit 1)
	@if [ ! -x "$(SETUP_VENV_PYTHON)" ]; then \
		echo "[setup-wizard-tools] Creating lightweight wizard virtualenv at $(SETUP_VENV_DIR)"; \
		$(PYTHON) -m venv $(SETUP_VENV_DIR); \
	fi
	@$(SETUP_VENV_PYTHON) -m pip install --upgrade pip setuptools wheel >/dev/null
	@$(SETUP_VENV_PYTHON) -m pip install "typer>=0.12.0" "loguru>=0.7.0" "httpx>=0.24.0" "python-dotenv>=1.0.0" >/dev/null
```

This target installs only the wizard runtime dependencies. It does not run `pip install -e .` and does not replace `make install-local`.

- [ ] **Step 6: Add public peer targets**

Add these Makefile targets above existing quickstart aliases:

```make
setup-docker-single: setup-wizard-tools
	@command -v docker >/dev/null 2>&1 || (echo "[setup-docker-single] docker not found. Install Docker and retry." && exit 1)
	@$(TLDW_SETUP) init --profile docker-single-webui --env-file "$(TLDW_ENV_FILE)" --default --yes
	@echo "[setup-docker-single] Next: make start-docker-single"

start-docker-single:
	@command -v docker >/dev/null 2>&1 || (echo "[start-docker-single] docker not found. Install Docker and retry." && exit 1)
	docker compose --env-file $(TLDW_ENV_FILE) -f $(DOCKER_SINGLE_COMPOSE) -f $(DOCKER_WEBUI_COMPOSE) up -d $(DOCKER_BUILD_FLAG)
	@echo "[start-docker-single] API:   $(TLDW_BASE_URL)"
	@echo "[start-docker-single] WebUI: $(TLDW_WEBUI_URL)"
	@echo "[start-docker-single] Next:  make verify-docker-single"

verify-docker-single: setup-wizard-tools
	@$(TLDW_SETUP) verify --profile docker-single-webui --env-file "$(TLDW_ENV_FILE)" --base-url "$(TLDW_BASE_URL)" --webui-url "$(TLDW_WEBUI_URL)" --first-value

setup-docker-multi: setup-wizard-tools
	@command -v docker >/dev/null 2>&1 || (echo "[setup-docker-multi] docker not found. Install Docker and retry." && exit 1)
	@test -n "$(ADMIN_USERNAME)" || (echo "[setup-docker-multi] Set ADMIN_USERNAME=<admin> ADMIN_PASSWORD=<password> for first admin bootstrap." && exit 1)
	@test -n "$(ADMIN_PASSWORD)" || (echo "[setup-docker-multi] Set ADMIN_PASSWORD=<password> for first admin bootstrap." && exit 1)
	@$(TLDW_SETUP) init --profile docker-multi-postgres --env-file "$(TLDW_ENV_FILE)" --admin-username "$(ADMIN_USERNAME)" --admin-password "$(ADMIN_PASSWORD)" $(if $(ADMIN_EMAIL),--admin-email "$(ADMIN_EMAIL)",) --default --yes
	@echo "[setup-docker-multi] Next: make start-docker-multi"

start-docker-multi:
	@command -v docker >/dev/null 2>&1 || (echo "[start-docker-multi] docker not found. Install Docker and retry." && exit 1)
	docker compose --env-file $(TLDW_ENV_FILE) -f $(DOCKER_MULTI_COMPOSE) up -d $(DOCKER_BUILD_FLAG)
	@echo "[start-docker-multi] API:  $(TLDW_BASE_URL)"
	@echo "[start-docker-multi] Next: make verify-docker-multi"

verify-docker-multi: setup-wizard-tools
	@$(TLDW_SETUP) verify --profile docker-multi-postgres --env-file "$(TLDW_ENV_FILE)" --base-url "$(TLDW_BASE_URL)" --first-value

install-local:
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "[install-local] $(PYTHON) not found. Install Python 3.10+ and retry." && exit 1)
	@$(PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || (echo "[install-local] Python 3.10+ is required." && exit 1)
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		echo "[install-local] Creating virtualenv at $(VENV_DIR)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@echo "[install-local] Installing Python dependencies into $(VENV_DIR)..."
	@$(VENV_PYTHON) -m pip install --upgrade pip setuptools wheel
	@$(VENV_PYTHON) -m pip install -e .
	@echo "[install-local] Next: make setup-local-single"

setup-local-single: setup-wizard-tools
	@$(TLDW_SETUP) init --profile local-single --env-file "$(TLDW_ENV_FILE)" --default --yes
	@echo "[setup-local-single] Next: make start-local-single"

start-local-single:
	@echo "[start-local-single] Starting API at $(TLDW_BASE_URL)"
	$(VENV_PYTHON) -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8000

verify-local-single: setup-wizard-tools
	@$(TLDW_SETUP) verify --profile local-single --env-file "$(TLDW_ENV_FILE)" --base-url "$(TLDW_BASE_URL)" --first-value
```

- [ ] **Step 7: Rewire aliases**

Replace old quickstart alias bodies with:

```make
quickstart: setup-docker-single start-docker-single verify-docker-single

quickstart-docker-webui: quickstart

quickstart-docker: setup-docker-single
	@command -v docker >/dev/null 2>&1 || (echo "[quickstart-docker] docker not found. Install Docker and retry." && exit 1)
	docker compose --env-file $(TLDW_ENV_FILE) -f $(DOCKER_SINGLE_COMPOSE) up -d $(DOCKER_BUILD_FLAG)
	@$(TLDW_SETUP) verify --profile docker-single-webui --env-file "$(TLDW_ENV_FILE)" --base-url "$(TLDW_BASE_URL)"

quickstart-install: install-local

quickstart-local: setup-local-single start-local-single
```

Update the `.PHONY` line to include all new targets.

- [ ] **Step 8: Run Makefile tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py -v
```

Expected:

```text
passed
```

- [ ] **Step 9: Commit Task 4**

Run:

```bash
git add .gitignore Makefile tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py
git commit -m "feat: add public onboarding make targets"
```

## Task 5: Add Docker Single-User Compose Profile

**Files:**
- Create: `Dockerfiles/docker-compose.single-user.yml`
- Modify: `Dockerfiles/docker-compose.webui.yml`
- Create: `tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py`

- [ ] **Step 1: Write failing Docker compose contract tests**

Create `tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml


def _compose(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_single_user_compose_has_no_postgres_service_or_dependency() -> None:
    compose = _compose("Dockerfiles/docker-compose.single-user.yml")
    assert "postgres" not in compose["services"]
    app = compose["services"]["app"]
    depends_on = app.get("depends_on", {})
    assert "postgres" not in depends_on
    env = "\n".join(app["environment"])
    assert "AUTH_MODE=${AUTH_MODE:-single_user}" in env
    assert "DATABASE_URL=${DATABASE_URL:-sqlite:///./Databases/users.db}" in env


def test_single_user_compose_uses_non_overlapping_user_database_volume() -> None:
    app = _compose("Dockerfiles/docker-compose.single-user.yml")["services"]["app"]
    volumes = "\n".join(app["volumes"])
    assert "app-data:/app/Databases" in volumes
    assert "/app/Databases/user_databases" not in volumes
    assert "chroma-data" not in "\n".join(_compose("Dockerfiles/docker-compose.single-user.yml").get("volumes", {}))


def test_webui_overlay_depends_on_app_health() -> None:
    webui = _compose("Dockerfiles/docker-compose.webui.yml")["services"]["webui"]
    assert webui["depends_on"]["app"]["condition"] == "service_healthy"
```

- [ ] **Step 2: Run Docker contract tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py -v
```

Expected:

```text
FileNotFoundError: Dockerfiles/docker-compose.single-user.yml
```

- [ ] **Step 3: Create single-user compose file**

Create `Dockerfiles/docker-compose.single-user.yml`:

```yaml
services:
  app:
    build:
      context: ..
      dockerfile: Dockerfiles/Dockerfile.prod
    image: tldw-server:prod
    container_name: tldw-app
    ports:
      - "8000:8000"
    environment:
      - AUTH_MODE=${AUTH_MODE:-single_user}
      - SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY:-change-me}
      - tldw_production=${tldw_production:-false}
      - DATABASE_URL=${DATABASE_URL:-sqlite:///./Databases/users.db}
      - JOBS_DB_URL=${JOBS_DB_URL:-}
      - UVICORN_WORKERS=${UVICORN_WORKERS:-2}
      - LOG_LEVEL=${LOG_LEVEL:-info}
    volumes:
      - app-data:/app/Databases
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready', timeout=3).status == 200 else 1)"]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 30s

  redis:
    image: redis:7-alpine
    container_name: tldw-redis
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    volumes:
      - redis_data:/data

volumes:
  app-data:
  redis_data:
```

- [ ] **Step 4: Run Docker contract tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py -v
```

Expected:

```text
passed
```

- [ ] **Step 5: Validate compose config syntax**

Run:

```bash
docker compose -f Dockerfiles/docker-compose.single-user.yml -f Dockerfiles/docker-compose.webui.yml config >/tmp/tldw_single_compose_config.yml
```

Expected:

```text
exit code 0
```

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add Dockerfiles/docker-compose.single-user.yml Dockerfiles/docker-compose.webui.yml tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py
git commit -m "feat: add docker single-user onboarding profile"
```

## Task 6: Add Docker Multi-User + Postgres Compose Profile

**Files:**
- Create: `Dockerfiles/docker-compose.multi-user-postgres.yml`
- Modify: `Dockerfiles/entrypoints/tldw-app-first-run.sh`
- Modify: `tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py`

- [ ] **Step 1: Add failing multi-user compose tests**

Append to `tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py`:

```python
def test_multi_user_compose_mounts_postgres_18_volume_at_parent_dir() -> None:
    compose = _compose("Dockerfiles/docker-compose.multi-user-postgres.yml")
    postgres = compose["services"]["postgres"]
    assert postgres["image"] == "postgres:18-bookworm"
    assert "postgres_data:/var/lib/postgresql" in postgres["volumes"]


def test_multi_user_compose_exposes_required_auth_env() -> None:
    app = _compose("Dockerfiles/docker-compose.multi-user-postgres.yml")["services"]["app"]
    env = "\n".join(app["environment"])
    for key in (
        "AUTH_MODE=${AUTH_MODE:-multi_user}",
        "DATABASE_URL=${DATABASE_URL:-postgresql://tldw_user:TestPassword123!@postgres:5432/tldw_users}",
        "JWT_SECRET_KEY=${JWT_SECRET_KEY:?JWT_SECRET_KEY is required}",
        "SESSION_ENCRYPTION_KEY=${SESSION_ENCRYPTION_KEY:?SESSION_ENCRYPTION_KEY is required}",
        "ADMIN_USERNAME=${ADMIN_USERNAME:?ADMIN_USERNAME is required}",
        "ADMIN_PASSWORD=${ADMIN_PASSWORD:?ADMIN_PASSWORD is required}",
    ):
        assert key in env
```

- [ ] **Step 2: Run Docker contract tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py -v
```

Expected:

```text
FileNotFoundError: Dockerfiles/docker-compose.multi-user-postgres.yml
```

- [ ] **Step 3: Create multi-user compose file**

Create `Dockerfiles/docker-compose.multi-user-postgres.yml`:

```yaml
services:
  app:
    build:
      context: ..
      dockerfile: Dockerfiles/Dockerfile.prod
    image: tldw-server:prod
    container_name: tldw-app
    ports:
      - "8000:8000"
    environment:
      - AUTH_MODE=${AUTH_MODE:-multi_user}
      - DATABASE_URL=${DATABASE_URL:-postgresql://tldw_user:TestPassword123!@postgres:5432/tldw_users}
      - JOBS_DB_URL=${JOBS_DB_URL:-postgresql://tldw_user:TestPassword123!@postgres:5432/tldw_users}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY:?JWT_SECRET_KEY is required}
      - SESSION_ENCRYPTION_KEY=${SESSION_ENCRYPTION_KEY:?SESSION_ENCRYPTION_KEY is required}
      - MCP_JWT_SECRET=${MCP_JWT_SECRET:?MCP_JWT_SECRET is required}
      - MCP_API_KEY_SALT=${MCP_API_KEY_SALT:?MCP_API_KEY_SALT is required}
      - BYOK_ENCRYPTION_KEY=${BYOK_ENCRYPTION_KEY:?BYOK_ENCRYPTION_KEY is required}
      - ADMIN_USERNAME=${ADMIN_USERNAME:?ADMIN_USERNAME is required}
      - ADMIN_PASSWORD=${ADMIN_PASSWORD:?ADMIN_PASSWORD is required}
      - ADMIN_EMAIL=${ADMIN_EMAIL:-}
      - tldw_production=${tldw_production:-false}
      - UVICORN_WORKERS=${UVICORN_WORKERS:-2}
      - LOG_LEVEL=${LOG_LEVEL:-info}
    volumes:
      - app-data:/app/Databases
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready', timeout=3).status == 200 else 1)"]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 30s

  postgres:
    image: postgres:18-bookworm
    container_name: tldw-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-tldw_users}
      POSTGRES_USER: ${POSTGRES_USER:-tldw_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-TestPassword123!}
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 12
    volumes:
      - postgres_data:/var/lib/postgresql

  redis:
    image: redis:7-alpine
    container_name: tldw-redis
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    volumes:
      - redis_data:/data

volumes:
  app-data:
  postgres_data:
  redis_data:
```

- [ ] **Step 4: Tighten entrypoint diagnostics**

In `Dockerfiles/entrypoints/tldw-app-first-run.sh`, inside the multi-user `else` branch where no admin vars exist, replace the warning text with an error and exit only when no users exist:

```sh
      if [ "$has_users" = "0" ]; then
        echo "" >&2
        echo "======================================================================" >&2
        echo "  ERROR: Multi-user mode has no admin user and no admin bootstrap env." >&2
        echo "" >&2
        echo "  Set ADMIN_USERNAME and ADMIN_PASSWORD in tldw_Server_API/Config_Files/.env" >&2
        echo "  before starting the public docker-multi-postgres profile." >&2
        echo "======================================================================" >&2
        echo "" >&2
        exit 1
      fi
```

- [ ] **Step 5: Run Docker contract tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py -v
```

Expected:

```text
passed
```

- [ ] **Step 6: Validate compose config syntax**

Run:

```bash
JWT_SECRET_KEY=jwt_secret_key_for_compose_config_32_chars \
SESSION_ENCRYPTION_KEY=session_secret_for_compose_config_32_chars \
MCP_JWT_SECRET=mcp_jwt_secret_for_compose_config_32_chars \
MCP_API_KEY_SALT=mcp_api_salt_for_compose_config_32_chars \
BYOK_ENCRYPTION_KEY=byok_secret_for_compose_config_32_chars \
ADMIN_USERNAME=tldw-admin \
ADMIN_PASSWORD='CorrectHorseBatteryStaple1!' \
docker compose -f Dockerfiles/docker-compose.multi-user-postgres.yml config >/tmp/tldw_multi_compose_config.yml
```

Expected:

```text
exit code 0
```

- [ ] **Step 7: Commit Task 6**

Run:

```bash
git add Dockerfiles/docker-compose.multi-user-postgres.yml Dockerfiles/entrypoints/tldw-app-first-run.sh tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py
git commit -m "feat: add docker multi-user postgres onboarding profile"
```

## Task 7: Finish Local Startup Contract

**Files:**
- Modify: `Makefile`
- Modify: `tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py`
- Modify: `tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py`

- [ ] **Step 1: Add local startup assertions**

Append to `tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py`:

```python
def test_start_local_single_uses_plain_uvicorn_without_reload() -> None:
    block = _target_block("start-local-single")
    assert "uvicorn tldw_Server_API.app.main:app" in block or "-m uvicorn tldw_Server_API.app.main:app" in block
    assert "--reload" not in block


def test_quickstart_local_is_setup_plus_start_alias() -> None:
    block = _target_block("quickstart-local")
    assert "setup-local-single" in block
    assert "start-local-single" in block
```

- [ ] **Step 2: Run local Makefile tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py -v
```

Expected:

```text
passed
```

- [ ] **Step 3: Manually inspect local target output**

Run:

```bash
make -n install-local
make -n setup-local-single
make -n start-local-single
make -n verify-local-single
```

Expected:

```text
install-local output contains pip install commands and no uvicorn command
setup-local-single output contains tldw-setup init --profile local-single
start-local-single output contains uvicorn and no --reload
verify-local-single output contains tldw-setup verify --profile local-single --first-value
```

- [ ] **Step 4: Commit Task 7**

Run:

```bash
git add Makefile tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py
git commit -m "fix: split local install setup start verify targets"
```

## Task 8: Update Peer Profile Docs And Audio Auth Examples

**Files:**
- Modify: `README.md`
- Modify: `Docs/Getting_Started/README.md`
- Modify: `Docs/Getting_Started/Profile_Docker_Single_User.md`
- Modify: `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- Modify: `Docs/Getting_Started/Profile_Local_Single_User.md`
- Modify: `Docs/Getting_Started/QUICKSTART.md`
- Modify: `Docs/Deployment/setup-wizard-guide.md`
- Modify: `Docs/Getting_Started/First_Time_Audio_Setup_CPU.md`
- Modify: `Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md`
- Modify: `Dockerfiles/README.md`
- Modify: `Docs/Website/index.html`
- Modify: `Docs/Published/Getting_Started/README.md`
- Modify: `Docs/Published/Getting_Started/Profile_Docker_Single_User.md`
- Modify: `Docs/Published/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- Modify: `Docs/Published/Getting_Started/Profile_Local_Single_User.md`
- Modify: `Docs/Published/Getting_Started/First_Time_Audio_Setup_CPU.md`
- Modify: `Docs/Published/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md`
- Modify: `tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py`
- Modify: `tldw_Server_API/tests/Docs/test_docker_persistence_docs.py`
- Create: `tldw_Server_API/tests/Docs/test_public_onboarding_profile_parity.py`

- [ ] **Step 1: Write failing docs parity tests**

Create `tldw_Server_API/tests/Docs/test_public_onboarding_profile_parity.py`:

```python
from __future__ import annotations

from pathlib import Path


PROFILE_DOCS = [
    Path("Docs/Getting_Started/Profile_Docker_Single_User.md"),
    Path("Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md"),
    Path("Docs/Getting_Started/Profile_Local_Single_User.md"),
]

LIFECYCLE_HEADINGS = [
    "## Prepare",
    "## Start",
    "## Verify",
    "## First Value",
    "## Audio Path",
    "## Troubleshoot",
]


def test_profile_docs_use_same_lifecycle_headings() -> None:
    for path in PROFILE_DOCS:
        text = path.read_text(encoding="utf-8")
        for heading in LIFECYCLE_HEADINGS:
            assert heading in text, f"{path} missing {heading}"


def test_profile_docs_include_windows_wsl_guidance() -> None:
    for path in PROFILE_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "WSL" in text, f"{path} should include Windows/WSL guidance"


def test_audio_docs_show_single_user_and_multi_user_auth() -> None:
    for path in (
        Path("Docs/Getting_Started/First_Time_Audio_Setup_CPU.md"),
        Path("Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "X-API-KEY" in text, f"{path} missing single-user API key example"
        assert "Authorization: Bearer" in text, f"{path} missing multi-user bearer token example"


def test_public_docs_use_new_profile_commands() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for command in (
        "make setup-docker-single",
        "make start-docker-single",
        "make verify-docker-single",
        "make setup-docker-multi",
        "make start-docker-multi",
        "make verify-docker-multi",
        "make install-local",
        "make setup-local-single",
        "make start-local-single",
        "make verify-local-single",
    ):
        assert command in readme
```

- [ ] **Step 2: Run docs tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_public_onboarding_profile_parity.py tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py -v
```

Expected:

```text
AssertionError: Docs/Getting_Started/Profile_Docker_Single_User.md missing ## Prepare
```

- [ ] **Step 3: Update profile docs with shared lifecycle**

For each profile doc, use the same headings:

```markdown
## Prepare
## Start
## Verify
## First Value
## Audio Path
## Troubleshoot
## Optional Add-ons
```

Use these command sequences:

Docker single-user + WebUI:

```bash
make setup-docker-single
make start-docker-single
make verify-docker-single
```

Docker multi-user + Postgres:

```bash
ADMIN_USERNAME=tldw-admin ADMIN_PASSWORD='replace-with-a-long-password' make setup-docker-multi
make start-docker-multi
make verify-docker-multi
```

Local single-user:

```bash
make install-local
make setup-local-single
make start-local-single
make verify-local-single
```

Add this Windows note to each profile:

```markdown
> **Windows:** Use WSL2 for the documented `make` commands. If you prefer PowerShell, run the equivalent `tldw-setup` command shown under each step and start Docker Desktop before Docker profiles.
```

- [ ] **Step 4: Update Docker persistence docs tests**

Update `tldw_Server_API/tests/Docs/test_docker_persistence_docs.py` so the Docker persistence contract expects the fixed non-overlapping volume model:

```python
def test_dockerfiles_readme_documents_persistence_contract() -> None:
    text = Path("Dockerfiles/README.md").read_text(encoding="utf-8")

    for snippet in (
        "app-data",
        "docker-compose.host-storage.yml",
        "docker compose down -v",
        "tldw_Server_API/Config_Files/.env",
        "No nested named volume is mounted under /app/Databases/user_databases",
    ):
        _require(snippet in text, f"Dockerfiles README should mention {snippet}")


def test_docker_single_user_profile_documents_named_volumes_and_overlay() -> None:
    text = Path("Docs/Getting_Started/Profile_Docker_Single_User.md").read_text(encoding="utf-8")

    for snippet in (
        "Docker named volumes",
        "docker-compose.host-storage.yml",
        "docker compose down -v",
        "app-data",
        "No nested named volume is mounted under /app/Databases/user_databases",
    ):
        _require(snippet in text, f"Docker single-user profile should mention {snippet}")
```

- [ ] **Step 5: Update audio docs with both auth modes**

In both audio setup docs, include:

Single-user:

```bash
curl -sS http://127.0.0.1:8000/api/v1/audio/voices/catalog \
  -H "X-API-KEY: $SINGLE_USER_API_KEY"
```

Multi-user:

```bash
JWT=$(
  curl -sS -X POST http://127.0.0.1:8000/api/v1/auth/login \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=$ADMIN_USERNAME" \
    -d "password=$ADMIN_PASSWORD" | jq -r '.access_token'
)

curl -sS http://127.0.0.1:8000/api/v1/audio/voices/catalog \
  -H "Authorization: Bearer $JWT"
```

State that stock Docker CPU/default audio works with bundled dependencies, while host-side config/model edits require rebuild or a documented host-storage/custom image path.

- [ ] **Step 6: Update README and website entrypoints**

In `README.md` and `Docs/Website/index.html`, present the three peers in this order:

```text
Docker single-user + WebUI
Docker multi-user + Postgres
Local single-user
```

Show prepare/start/verify commands for each. Keep `make quickstart` as the shortest Docker single-user alias but stop using `make quickstart-install` as a start command.

- [ ] **Step 7: Sync published docs**

Copy updated Getting Started content into matching `Docs/Published/Getting_Started/` files by editing those files directly. Preserve published relative links.

- [ ] **Step 8: Run docs tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Docs/test_public_onboarding_profile_parity.py \
  tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py \
  tldw_Server_API/tests/Docs/test_onboarding_entrypoints.py \
  tldw_Server_API/tests/Docs/test_onboarding_default_contract.py \
  tldw_Server_API/tests/Docs/test_published_onboarding_parity.py \
  tldw_Server_API/tests/Docs/test_docker_persistence_docs.py \
  -v
```

Expected:

```text
passed
```

- [ ] **Step 9: Commit Task 8**

Run:

```bash
git add README.md Docs/Getting_Started Docs/Deployment/setup-wizard-guide.md Dockerfiles/README.md Docs/Website/index.html Docs/Published/Getting_Started tldw_Server_API/tests/Docs
git commit -m "docs: align public onboarding profiles"
```

## Task 9: Runtime Validation And Final Hardening

**Files:**
- Create: `Docs/superpowers/reviews/public-onboarding-remediation/2026-04-25-runtime-validation.md`
- Modify: any file needed to fix issues found by runtime validation.

- [x] **Step 1: Run focused unit and docs tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/wizard \
  tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py \
  tldw_Server_API/tests/Utils/test_makefile_quickstart_same_origin.py \
  tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py \
  tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py \
  tldw_Server_API/tests/Docs/test_public_onboarding_profile_parity.py \
  tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py \
  tldw_Server_API/tests/Docs/test_onboarding_entrypoints.py \
  tldw_Server_API/tests/Docs/test_onboarding_default_contract.py \
  tldw_Server_API/tests/Docs/test_published_onboarding_parity.py \
  -v
```

Expected:

```text
passed
```

- [x] **Step 2: Run Docker single-user validation from clean volumes**

Run:

```bash
COMPOSE_PROJECT_NAME=tldw_ftux_single docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.single-user.yml -f Dockerfiles/docker-compose.webui.yml down -v
COMPOSE_PROJECT_NAME=tldw_ftux_single make setup-docker-single
COMPOSE_PROJECT_NAME=tldw_ftux_single make start-docker-single
COMPOSE_PROJECT_NAME=tldw_ftux_single make verify-docker-single
COMPOSE_PROJECT_NAME=tldw_ftux_single docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.single-user.yml -f Dockerfiles/docker-compose.webui.yml ps
```

Expected:

```text
make verify-docker-single exits 0
app and webui containers are running or healthy
```

- [x] **Step 3: Run Docker multi-user validation from clean volumes**

Run:

```bash
COMPOSE_PROJECT_NAME=tldw_ftux_multi docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.multi-user-postgres.yml down -v
COMPOSE_PROJECT_NAME=tldw_ftux_multi ADMIN_USERNAME=tldw-admin ADMIN_PASSWORD='CorrectHorseBatteryStaple1!' ADMIN_EMAIL=tldw-admin@example.com make setup-docker-multi
COMPOSE_PROJECT_NAME=tldw_ftux_multi make start-docker-multi
COMPOSE_PROJECT_NAME=tldw_ftux_multi make verify-docker-multi
COMPOSE_PROJECT_NAME=tldw_ftux_multi docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.multi-user-postgres.yml ps
```

Expected:

```text
make verify-docker-multi exits 0
postgres is healthy
app is healthy
admin login succeeds through tldw-setup verify
```

- [x] **Step 4: Run local single-user validation**

Use a throwaway env path so the user’s real `.env` is not overwritten:

```bash
TLDW_ENV_FILE=/tmp/tldw_local_single.env make install-local
TLDW_ENV_FILE=/tmp/tldw_local_single.env make setup-local-single
TLDW_ENV_FILE=/tmp/tldw_local_single.env make start-local-single
```

In a second shell:

```bash
TLDW_ENV_FILE=/tmp/tldw_local_single.env make verify-local-single
```

Expected:

```text
make verify-local-single exits 0
server was started with plain uvicorn and no --reload
```

- [x] **Step 5: Record validation results**

Create `Docs/superpowers/reviews/public-onboarding-remediation/2026-04-25-runtime-validation.md`:

```markdown
# Public Onboarding Remediation Runtime Validation

Date: 2026-04-25

## Environment

- Host OS:
- Docker version:
- Python version:
- Branch:
- Commit:

## Docker Single-User + WebUI

- Commands:
- Result:
- Notes:

## Docker Multi-User + Postgres

- Commands:
- Result:
- Notes:

## Local Single-User

- Commands:
- Result:
- Notes:

## Remaining Follow-Ups

- None.
```

Fill the environment and result fields with the actual command outputs summarized in one or two lines each.

- [x] **Step 6: Run Bandit on touched Python scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/cli/wizard \
  tldw_Server_API/app/core/AuthNZ/initialize.py \
  tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py \
  tldw_Server_API/app/core/AuthNZ/settings.py \
  tldw_Server_API/app/core/config.py \
  tldw_Server_API/app/api/v1/endpoints/config_info.py \
  -s B105 \
  -f json \
  -o /tmp/bandit_public_onboarding.json
```

Expected:

```text
exit code 0
```

If Bandit reports findings in touched code, fix them before continuing.

- [x] **Step 7: Run final status check**

Run:

```bash
git status --short
```

Expected:

```text
Only intentional files from this plan are modified or untracked.
Unrelated pre-existing dirty files remain untouched.
```

- [x] **Step 8: Commit Task 9**

Run:

```bash
git add Docs/superpowers/reviews/public-onboarding-remediation/2026-04-25-runtime-validation.md
git commit -m "docs: record public onboarding validation"
```

## Final Verification Checklist

- [x] `tldw-setup init --profile docker-single-webui --dry-run --json` masks generated secrets.
- [x] `tldw-setup init --profile docker-multi-postgres --dry-run --json` includes `SESSION_ENCRYPTION_KEY`.
- [x] `tldw-setup verify --profile docker-single-webui --first-value --json` does not spawn an ephemeral local server.
- [x] `make quickstart-install` is install-only.
- [x] `make quickstart` runs setup, start, and verify for Docker single-user + WebUI.
- [x] Docker single-user compose has no Postgres service or dependency.
- [x] Docker multi-user compose mounts Postgres 18 storage at `/var/lib/postgresql`.
- [x] Docker multi-user setup uses env-driven first admin creation.
- [x] Local startup uses plain `uvicorn` without `--reload`.
- [x] Audio docs include `X-API-KEY` and `Authorization: Bearer` examples.
- [x] Profile docs include Windows/WSL guidance.
- [x] Runtime validation transcript exists and reflects clean-state runs.
