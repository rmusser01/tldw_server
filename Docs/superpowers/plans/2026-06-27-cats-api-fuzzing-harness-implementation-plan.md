# CATS API Fuzzing Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first safe, local-only CATS fuzzing harness slice for the tldw_server API.

**Architecture:** Fix the known OpenAPI validation issue first, then add an importable Python harness under `Helper_Scripts/cats_fuzz/`. Keep block definitions, environment isolation, OpenAPI generation, CATS command construction, uvicorn lifecycle, and report summarization in small modules so most behavior is covered by unit tests without launching CATS or the server. The first live path should support `contract` and `public-read`; `auth-read` is scaffolded in the manifest but can remain non-default until the isolated server run is stable.

**Tech Stack:** Python 3.10+, FastAPI OpenAPI generation, pytest, uvicorn, CATS 13.8.0 CLI.

---

## References

- Design spec: `Docs/superpowers/specs/2026-06-27-cats-api-fuzzing-harness-design.md`
- Backlog task: `TASK-2370`
- CATS docs: `https://endava.github.io/cats/docs/intro/`
- Local CATS version: `cats --version` reports `13.8.0`.
- Local CATS help confirms:
  - `--blackbox` ignores all response codes except `5xx`.
  - `--skipReportingForIgnored` suppresses ignored response reports.
  - `--reportFormat` supports HTML/JUnit report families, not JSON.
  - `--path`, `--skipPath`, `-H`, `--refData`, `--urlParams`, and `--dryRun` are available.

## File Structure

- Modify: `tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py`
  - Move query parameter examples off invalid schema-level dicts so CATS strict validation can pass.
- Create: `Helper_Scripts/cats_fuzz/__init__.py`
  - Package marker and public constants.
- Create: `Helper_Scripts/cats_fuzz/__main__.py`
  - Thin `python -m Helper_Scripts.cats_fuzz` entrypoint.
- Create: `Helper_Scripts/cats_fuzz/manifest.py`
  - Built-in block definitions and manifest validation.
- Create: `Helper_Scripts/cats_fuzz/env.py`
  - Local-only child environment construction and credential detection.
- Create: `Helper_Scripts/cats_fuzz/openapi_export.py`
  - Subprocess-safe OpenAPI generation command and generated spec hashing.
- Create: `Helper_Scripts/cats_fuzz/cats_cli.py`
  - CATS argv construction, subprocess execution wrapper, and failure classification.
- Create: `Helper_Scripts/cats_fuzz/server.py`
  - Isolated uvicorn process lifecycle and health wait loop.
- Create: `Helper_Scripts/cats_fuzz/summary.py`
  - Runner summary JSON schema and masked command rendering.
- Create: `Helper_Scripts/cats_fuzz/runner.py`
  - Block orchestration for `contract`, `public-read`, and scaffolded `auth-read`.
- Create: `Helper_Scripts/cats_fuzz/cli.py`
  - argparse CLI that wires manifest, env, server, CATS, and summaries.
- Create: `tldw_Server_API/tests/VectorStores/test_vector_stores_openapi_examples.py`
  - Regression tests for the vector store OpenAPI example shape.
- Create: `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_manifest.py`
  - Unit tests for block manifest validation.
- Create: `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_env.py`
  - Unit tests for credential detection and child env construction.
- Create: `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py`
  - Unit tests for CATS command construction and failure classification.
- Create: `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py`
  - Unit tests for summary JSON output and secret masking.
- Create: `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_runner.py`
  - Unit tests for orchestration with subprocess/server calls mocked.
- Create: `Docs/Development/CATS_Fuzzing.md`
  - Usage, safety defaults, block descriptions, and local verification commands.

## Setup Notes

This plan was written in the clean worktree:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/cats-api-fuzzing-harness
```

The worktree was clean at creation but did not include a `.venv`. Before running Python tests in that worktree, either create a new venv or point to the main repo venv deliberately. Prefer a worktree-local venv for implementation:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

When a command below starts with `source .venv/bin/activate`, run it from the worktree root after creating/installing that venv.

## Task 1: Fix Vector Store OpenAPI Examples

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py:1144-1161`
- Create: `tldw_Server_API/tests/VectorStores/test_vector_stores_openapi_examples.py`

- [ ] **Step 1: Write the failing OpenAPI example-shape test**

Create `tldw_Server_API/tests/VectorStores/test_vector_stores_openapi_examples.py`:

```python
from __future__ import annotations

from typing import Any

import pytest


VECTOR_LIST_PATH = "/api/v1/vector_stores/{store_id}/vectors"


def _openapi_spec(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "THIS-IS-A-SECURE-KEY-123-FAKE-KEY")
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", "THIS-IS-A-SECURE-KEY-123-FAKE-KEY")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("MINIMAL_TEST_INCLUDE_AUDIO", "1")
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")

    from tldw_Server_API.app.main import app

    app.openapi_schema = None
    return app.openapi()


@pytest.mark.unit
def test_vector_list_query_examples_are_cats_validate_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _openapi_spec(monkeypatch)
    params = {
        param["name"]: param
        for param in spec["paths"][VECTOR_LIST_PATH]["get"]["parameters"]
    }

    for name in ("filter", "order_by", "order_dir"):
        schema = params[name].get("schema", {})
        assert not isinstance(schema.get("examples"), dict), name
        assert "examples" in params[name] or isinstance(schema.get("examples"), list), name
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VectorStores/test_vector_stores_openapi_examples.py -q
```

Expected: FAIL because `schema.examples` is currently a dict for `filter`, `order_by`, and `order_dir`.

- [ ] **Step 3: Move the examples to OpenAPI parameter examples**

In `tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py`, replace the three `examples=` arguments with `openapi_examples=`:

```python
filter: str | None = Query(
    None,
    description="Optional JSON metadata filter",
    openapi_examples={
        "simple": {"summary": "Simple equality", "value": "{\"genre\":\"a\"}"},
        "and_numeric": {
            "summary": "AND with numeric",
            "value": "{\"$and\":[{\"genre\":\"a\"},{\"score\":{\"$gte\":0.8}}]}",
        },
    },
),
order_by: str | None = Query(
    "id",
    description="Order field: 'id' or 'metadata.<key>'",
    openapi_examples={
        "metadata": {"summary": "Order by metadata.score desc", "value": "metadata.score"},
    },
),
order_dir: str = Query(
    "asc",
    pattern="^(?i)(asc|desc)$",
    openapi_examples={
        "desc": {"summary": "Descending", "value": "desc"},
    },
),
```

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VectorStores/test_vector_stores_openapi_examples.py -q
```

Expected: PASS.

- [ ] **Step 5: Generate a temporary OpenAPI file and run CATS validation**

Run:

```bash
source .venv/bin/activate
AUTH_MODE=single_user \
SINGLE_USER_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY \
SINGLE_USER_TEST_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY \
MINIMAL_TEST_APP=1 \
MINIMAL_TEST_INCLUDE_AUDIO=1 \
PYTHONWARNINGS=ignore \
LOGURU_LEVEL=ERROR \
python -c 'import json; from pathlib import Path; from tldw_Server_API.app.main import app; app.openapi_schema = None; Path("/tmp/tldw_openapi_cats_fixed.json").write_text(json.dumps(app.openapi()), encoding="utf-8")'
cats validate -c /tmp/tldw_openapi_cats_fixed.json -j
```

Expected: `cats validate` exits `0`. If unrelated validation issues remain, record them in `TASK-2370` and continue with the known-issues path rather than hiding them.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py tldw_Server_API/tests/VectorStores/test_vector_stores_openapi_examples.py
git commit -m "fix: make vector store OpenAPI examples CATS compatible"
```

## Task 2: Add Block Manifest Models

**Files:**
- Create: `Helper_Scripts/cats_fuzz/__init__.py`
- Create: `Helper_Scripts/cats_fuzz/manifest.py`
- Create: `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_manifest.py`

- [ ] **Step 1: Write manifest unit tests**

Create `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_manifest.py`:

```python
from __future__ import annotations

import pytest

from Helper_Scripts.cats_fuzz.manifest import (
    BlockRisk,
    ExpectedGate,
    get_builtin_block,
    get_builtin_manifest,
    validate_block,
)


@pytest.mark.unit
def test_builtin_manifest_contains_initial_blocks() -> None:
    manifest = get_builtin_manifest()

    assert {"contract", "public-read", "auth-read"}.issubset(manifest)
    assert manifest["contract"].risk is BlockRisk.CONTRACT
    assert manifest["public-read"].expected_gate is ExpectedGate.NO_5XX
    assert manifest["public-read"].allows_network is False
    assert "/" in manifest["public-read"].paths


@pytest.mark.unit
def test_mutating_blocks_must_require_seed_or_be_manual() -> None:
    block = get_builtin_block("public-read")
    unsafe = block.__class__(
        **{**block.__dict__, "name": "unsafe", "allows_mutation": True, "requires_seed": False}
    )

    with pytest.raises(ValueError, match="requires_seed"):
        validate_block(unsafe)


@pytest.mark.unit
def test_unknown_builtin_block_fails() -> None:
    with pytest.raises(KeyError):
        get_builtin_block("missing")
```

- [ ] **Step 2: Run tests to verify import failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_manifest.py -q
```

Expected: FAIL because `Helper_Scripts.cats_fuzz` does not exist yet.

- [ ] **Step 3: Implement the manifest module**

Create `Helper_Scripts/cats_fuzz/__init__.py`:

```python
from __future__ import annotations

DEFAULT_TEST_API_KEY = "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
```

Create `Helper_Scripts/cats_fuzz/manifest.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BlockRisk(str, Enum):
    CONTRACT = "contract"
    PUBLIC_READ = "public-read"
    AUTH_READ = "auth-read"
    ISOLATED_MUTATION = "isolated-mutation"
    EXTERNAL_RISK = "external-risk"
    MANUAL = "manual"


class ExpectedGate(str, Enum):
    NO_5XX = "no_5xx"
    CONTRACT_ONLY = "contract_only"


@dataclass(frozen=True)
class CatsBlock:
    name: str
    description: str
    risk: BlockRisk
    paths: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    skip_paths: tuple[str, ...] = ()
    skip_tags: tuple[str, ...] = ()
    skip_methods: tuple[str, ...] = ()
    requires_seed: bool = False
    allows_mutation: bool = False
    allows_network: bool = False
    timeout_seconds: int = 120
    read_timeout: int = 5
    connection_timeout: int = 5
    write_timeout: int = 5
    max_requests_per_minute: int = 120
    expected_gate: ExpectedGate = ExpectedGate.NO_5XX
    skip_reason: str | None = None
    report_formats: tuple[str, ...] = ("HTML_ONLY", "JUNIT")


def validate_block(block: CatsBlock) -> None:
    if not block.name:
        raise ValueError("block name is required")
    if not block.paths and not block.tags and block.risk is not BlockRisk.CONTRACT:
        raise ValueError(f"{block.name}: paths or tags are required")
    if block.allows_mutation and not block.requires_seed and block.risk is not BlockRisk.MANUAL:
        raise ValueError(f"{block.name}: mutating blocks must set requires_seed")
    if block.allows_network and block.risk is not BlockRisk.EXTERNAL_RISK:
        raise ValueError(f"{block.name}: allows_network requires external-risk")


def get_builtin_manifest() -> dict[str, CatsBlock]:
    blocks = {
        "contract": CatsBlock(
            name="contract",
            description="Validate and summarize generated OpenAPI without calling the API.",
            risk=BlockRisk.CONTRACT,
            expected_gate=ExpectedGate.CONTRACT_ONLY,
            timeout_seconds=60,
        ),
        "public-read": CatsBlock(
            name="public-read",
            description="Fuzz public metadata and health endpoints in blackbox mode.",
            risk=BlockRisk.PUBLIC_READ,
            paths=(
                "/",
                "/health",
                "/ready",
                "/health/ready",
                "/api/v1/health",
                "/api/v1/health/live",
                "/api/v1/health/ready",
                "/api/v1/config/docs-info",
                "/api/v1/config/quickstart",
            ),
            skip_methods=("POST", "PUT", "PATCH", "DELETE", "TRACE"),
            max_requests_per_minute=60,
            timeout_seconds=120,
        ),
        "auth-read": CatsBlock(
            name="auth-read",
            description="Authenticated read-only smoke fuzzing with X-API-KEY.",
            risk=BlockRisk.AUTH_READ,
            paths=(
                "/api/v1/llm/providers",
                "/api/v1/mcp/status",
                "/api/v1/rag/health/simple",
            ),
            skip_methods=("POST", "PUT", "PATCH", "DELETE", "TRACE"),
            max_requests_per_minute=60,
            timeout_seconds=180,
        ),
    }
    for block in blocks.values():
        validate_block(block)
    return blocks


def get_builtin_block(name: str) -> CatsBlock:
    return get_builtin_manifest()[name]
```

- [ ] **Step 4: Run manifest tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_manifest.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Helper_Scripts/cats_fuzz/__init__.py Helper_Scripts/cats_fuzz/manifest.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_manifest.py
git commit -m "feat: add CATS fuzz block manifest"
```

## Task 3: Add Local-Only Environment Builder And OpenAPI Export

**Files:**
- Create: `Helper_Scripts/cats_fuzz/env.py`
- Create: `Helper_Scripts/cats_fuzz/openapi_export.py`
- Create: `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_env.py`

- [ ] **Step 1: Write env and OpenAPI command tests**

Create `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_env.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from Helper_Scripts.cats_fuzz import DEFAULT_TEST_API_KEY
from Helper_Scripts.cats_fuzz.env import build_child_env, find_sensitive_values
from Helper_Scripts.cats_fuzz.openapi_export import build_openapi_export_command


@pytest.mark.unit
def test_find_sensitive_values_detects_provider_keys() -> None:
    found = find_sensitive_values({"OPENAI_API_KEY": "sk-real", "SAFE": "value"})

    assert found == {"OPENAI_API_KEY": "set"}


@pytest.mark.unit
def test_build_child_env_rejects_real_credentials_by_default(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="real credentials"):
        build_child_env(tmp_path, parent_env={"OPENAI_API_KEY": "sk-real"})


@pytest.mark.unit
def test_build_child_env_sets_test_paths_and_sentinels(tmp_path: Path) -> None:
    env = build_child_env(tmp_path, parent_env={}, allow_external=False)

    assert env["AUTH_MODE"] == "single_user"
    assert env["SINGLE_USER_API_KEY"] == DEFAULT_TEST_API_KEY
    assert env["SINGLE_USER_TEST_API_KEY"] == DEFAULT_TEST_API_KEY
    assert env["DATABASE_URL"].startswith("sqlite:///")
    assert Path(env["TLDW_ENV_FILE"]).exists()
    assert env["OPENAI_API_KEY"] == ""


@pytest.mark.unit
def test_openapi_export_command_uses_module_and_output_path(tmp_path: Path) -> None:
    output = tmp_path / "openapi.json"
    command = build_openapi_export_command(output)

    assert command[:3] == ["python", "-m", "Helper_Scripts.cats_fuzz.openapi_export"]
    assert str(output) in command
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_env.py -q
```

Expected: FAIL because the modules do not exist yet.

- [ ] **Step 3: Implement `env.py`**

Create `Helper_Scripts/cats_fuzz/env.py`:

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from Helper_Scripts.cats_fuzz import DEFAULT_TEST_API_KEY


SENSITIVE_ENV_NAMES = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "COHERE_API_KEY",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "HUGGINGFACE_API_KEY",
    "MISTRAL_API_KEY",
    "OPENROUTER_API_KEY",
    "QWEN_API_KEY",
    "SLACK_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "WEBHOOK_URL",
}

SENSITIVE_ENV_SUBSTRINGS = ("API_KEY", "TOKEN", "WEBHOOK", "SECRET")


def find_sensitive_values(env: Mapping[str, str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for name, value in env.items():
        if not value or not value.strip():
            continue
        upper_name = name.upper()
        if name in SENSITIVE_ENV_NAMES or any(part in upper_name for part in SENSITIVE_ENV_SUBSTRINGS):
            found[name] = "set"
    return found


def _write_minimal_env_file(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "AUTH_MODE=single_user",
                f"SINGLE_USER_API_KEY={DEFAULT_TEST_API_KEY}",
                f"SINGLE_USER_TEST_API_KEY={DEFAULT_TEST_API_KEY}",
                "MINIMAL_TEST_APP=1",
                "MINIMAL_TEST_INCLUDE_AUDIO=1",
                "TEST_MODE=true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_child_env(
    work_dir: Path,
    *,
    parent_env: Mapping[str, str] | None = None,
    allow_external: bool = False,
) -> dict[str, str]:
    source = dict(parent_env or os.environ)
    sensitive = find_sensitive_values(source)
    if sensitive and not allow_external:
        names = ", ".join(sorted(sensitive))
        raise ValueError(f"Refusing to run with real credentials in environment: {names}")

    runtime_dir = work_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env_file = runtime_dir / "cats-fuzz.env"
    _write_minimal_env_file(env_file)

    child = dict(source)
    for name in SENSITIVE_ENV_NAMES:
        child[name] = ""
    child.update(
        {
            "AUTH_MODE": "single_user",
            "SINGLE_USER_API_KEY": DEFAULT_TEST_API_KEY,
            "SINGLE_USER_TEST_API_KEY": DEFAULT_TEST_API_KEY,
            "DATABASE_URL": f"sqlite:///{runtime_dir / 'users.db'}",
            "USER_DB_BASE_DIR": str(runtime_dir / "user_databases"),
            "TLDW_ENV_FILE": str(env_file),
            "MINIMAL_TEST_APP": "1",
            "MINIMAL_TEST_INCLUDE_AUDIO": "1",
            "TEST_MODE": "true",
            "PYTHONWARNINGS": "ignore",
            "LOGURU_LEVEL": "ERROR",
        }
    )
    return child
```

- [ ] **Step 4: Implement `openapi_export.py`**

Create `Helper_Scripts/cats_fuzz/openapi_export.py`:

```python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_openapi_export_command(output_path: Path) -> list[str]:
    return [
        "python",
        "-m",
        "Helper_Scripts.cats_fuzz.openapi_export",
        "--output",
        str(output_path),
    ]


def export_openapi(output_path: Path) -> str:
    from tldw_Server_API.app.main import app

    app.openapi_schema = None
    payload = json.dumps(app.openapi(), sort_keys=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export tldw_server OpenAPI for CATS")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    digest = export_openapi(args.output)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run env tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_env.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Helper_Scripts/cats_fuzz/env.py Helper_Scripts/cats_fuzz/openapi_export.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_env.py
git commit -m "feat: isolate CATS fuzz runtime environment"
```

## Task 4: Add CATS Command Builder And Summary JSON

**Files:**
- Create: `Helper_Scripts/cats_fuzz/cats_cli.py`
- Create: `Helper_Scripts/cats_fuzz/summary.py`
- Create: `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py`
- Create: `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py`

- [ ] **Step 1: Write command and summary tests**

Create `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from Helper_Scripts.cats_fuzz import DEFAULT_TEST_API_KEY
from Helper_Scripts.cats_fuzz.cats_cli import build_cats_run_command, classify_cats_exit
from Helper_Scripts.cats_fuzz.manifest import get_builtin_block


@pytest.mark.unit
def test_public_read_command_uses_blackbox_and_junit_reports(tmp_path: Path) -> None:
    block = get_builtin_block("public-read")
    command = build_cats_run_command(
        block,
        contract_path=tmp_path / "openapi.json",
        server_url="http://127.0.0.1:8000",
        output_dir=tmp_path / "reports",
        api_key=DEFAULT_TEST_API_KEY,
    )

    assert "--blackbox" in command
    assert "--skipReportingForIgnored" in command
    assert "--reportFormat" in command
    assert command[command.index("--reportFormat") + 1] == "HTML_ONLY,JUNIT"
    assert "-H" in command
    assert f"X-API-KEY={DEFAULT_TEST_API_KEY}" in command
    assert "--path" in command
    assert "/" in command[command.index("--path") + 1].split(",")


@pytest.mark.unit
def test_cats_exit_classification_separates_usage_tool_and_api_failures() -> None:
    assert classify_cats_exit(0, "") == "ok"
    assert classify_cats_exit(2, "Invalid value for option") == "usage"
    assert classify_cats_exit(1, "Internal execution error") == "tool"
    assert classify_cats_exit(1, "Some tests failed with 500") == "api"
```

Create `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from Helper_Scripts.cats_fuzz.summary import CatsRunSummary, mask_command, write_summary


@pytest.mark.unit
def test_mask_command_hides_api_key() -> None:
    masked = mask_command(["cats", "-H", "X-API-KEY=secret-value"])

    assert "secret-value" not in " ".join(masked)
    assert "X-API-KEY=$X-API-KEY" in masked


@pytest.mark.unit
def test_write_summary_persists_expected_shape(tmp_path: Path) -> None:
    summary = CatsRunSummary(
        block="public-read",
        cats_version="13.8.0",
        openapi_sha256="abc",
        command=["cats", "--blackbox"],
        masked_command=["cats", "--blackbox"],
        exit_code=0,
        failure_class="ok",
        stdout_path="stdout.log",
        stderr_path="stderr.log",
        report_dir="report",
    )

    output = write_summary(summary, tmp_path / "summary.json")
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["block"] == "public-read"
    assert data["failure_class"] == "ok"
    assert data["command"] == ["cats", "--blackbox"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py -q
```

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement command builder**

Create `Helper_Scripts/cats_fuzz/cats_cli.py`:

```python
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from Helper_Scripts.cats_fuzz.manifest import CatsBlock


@dataclass(frozen=True)
class CatsProcessResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


def _join(values: tuple[str, ...]) -> str:
    return ",".join(values)


def build_cats_run_command(
    block: CatsBlock,
    *,
    contract_path: Path,
    server_url: str,
    output_dir: Path,
    api_key: str,
    cats_bin: str = "cats",
    dry_run: bool = False,
) -> list[str]:
    command = [
        cats_bin,
        "-c",
        str(contract_path),
        "-s",
        server_url.rstrip("/"),
        "-H",
        f"X-API-KEY={api_key}",
        "--maskHeaders",
        "X-API-KEY,Authorization",
        "--blackbox",
        "--skipReportingForIgnored",
        "--maxRequestsPerMinute",
        str(block.max_requests_per_minute),
        "--connectionTimeout",
        str(block.connection_timeout),
        "--readTimeout",
        str(block.read_timeout),
        "--writeTimeout",
        str(block.write_timeout),
        "--reportFormat",
        _join(block.report_formats),
        "--output",
        str(output_dir),
    ]
    if block.paths:
        command.extend(["--path", _join(block.paths)])
    if block.skip_paths:
        command.extend(["--skipPath", _join(block.skip_paths)])
    if block.skip_methods:
        command.extend(["--skipHttpMethod", _join(block.skip_methods)])
    if dry_run:
        command.append("--dryRun")
    return command


def build_cats_validate_command(contract_path: Path, *, cats_bin: str = "cats", json_output: bool = True) -> list[str]:
    command = [cats_bin, "validate", "-c", str(contract_path)]
    if json_output:
        command.append("-j")
    return command


def build_cats_stats_command(contract_path: Path, *, cats_bin: str = "cats", json_output: bool = True) -> list[str]:
    command = [cats_bin, "stats", "-c", str(contract_path)]
    if json_output:
        command.append("-j")
    return command


def classify_cats_exit(exit_code: int, stderr: str) -> str:
    if exit_code == 0:
        return "ok"
    lowered = stderr.lower()
    if exit_code == 2 or "invalid value for option" in lowered:
        return "usage"
    if "internal execution error" in lowered or "exception occurred" in lowered:
        return "tool"
    return "api"


def run_command(command: list[str], *, timeout_seconds: int, env: dict[str, str] | None = None) -> CatsProcessResult:
    proc = subprocess.run(  # nosec B603 - command is built from trusted harness args
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )
    return CatsProcessResult(command=command, exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
```

- [ ] **Step 4: Implement summary writer**

Create `Helper_Scripts/cats_fuzz/summary.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CatsRunSummary:
    block: str
    cats_version: str
    openapi_sha256: str
    command: list[str]
    masked_command: list[str]
    exit_code: int
    failure_class: str
    stdout_path: str
    stderr_path: str
    report_dir: str
    extra: dict[str, Any] = field(default_factory=dict)


def mask_command(command: list[str]) -> list[str]:
    masked: list[str] = []
    for part in command:
        if part.startswith("X-API-KEY="):
            masked.append("X-API-KEY=$X-API-KEY")
        elif part.startswith("Authorization="):
            masked.append("Authorization=$AUTHORIZATION")
        else:
            masked.append(part)
    return masked


def write_summary(summary: CatsRunSummary, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
```

- [ ] **Step 5: Run command and summary tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Helper_Scripts/cats_fuzz/cats_cli.py Helper_Scripts/cats_fuzz/summary.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py
git commit -m "feat: build CATS commands and run summaries"
```

## Task 5: Add Server Lifecycle And Runner Orchestration

**Files:**
- Create: `Helper_Scripts/cats_fuzz/server.py`
- Create: `Helper_Scripts/cats_fuzz/runner.py`
- Create: `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_runner.py`

- [ ] **Step 1: Write mocked runner tests**

Create `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_runner.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from Helper_Scripts.cats_fuzz.cats_cli import CatsProcessResult
from Helper_Scripts.cats_fuzz.runner import run_contract_block


@pytest.mark.unit
def test_contract_block_writes_summary_for_validate_and_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openapi = tmp_path / "openapi.json"
    openapi.write_text("{}", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(command: list[str], *, timeout_seconds: int, env: dict[str, str] | None = None) -> CatsProcessResult:
        calls.append(command)
        return CatsProcessResult(command=command, exit_code=0, stdout="{}", stderr="")

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.runner.run_command", fake_run)

    result = run_contract_block(
        contract_path=openapi,
        output_dir=tmp_path / "out",
        cats_version="13.8.0",
        openapi_sha256="abc",
    )

    assert result.exit_code == 0
    assert any(command[:2] == ["cats", "validate"] for command in calls)
    assert any(command[:2] == ["cats", "stats"] for command in calls)
    assert (tmp_path / "out" / "contract" / "summary.json").exists()
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_runner.py -q
```

Expected: FAIL because `runner.py` and `server.py` do not exist yet.

- [ ] **Step 3: Implement server lifecycle**

Create `Helper_Scripts/cats_fuzz/server.py`:

```python
from __future__ import annotations

import contextlib
import socket
import subprocess
import time
from dataclasses import dataclass
from urllib.request import urlopen


@dataclass
class UvicornServer:
    process: subprocess.Popen[str]
    url: str


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(env: dict[str, str], *, port: int | None = None) -> UvicornServer:
    resolved_port = port or find_free_port()
    url = f"http://127.0.0.1:{resolved_port}"
    process = subprocess.Popen(  # nosec B603 - fixed module invocation
        [
            "python",
            "-m",
            "uvicorn",
            "tldw_Server_API.app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(resolved_port),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    wait_for_health(url)
    return UvicornServer(process=process, url=url)


def wait_for_health(base_url: str, *, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}/health", timeout=2.0) as response:  # nosec B310
                if int(response.status) < 500:
                    return
        except Exception as exc:  # noqa: BLE001 - health polling should retain last failure
            last_error = exc
        time.sleep(0.25)
    raise TimeoutError(f"Server did not become healthy: {last_error}")


def stop_server(server: UvicornServer) -> None:
    server.process.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        server.process.wait(timeout=10)
    if server.process.poll() is None:
        server.process.kill()
        server.process.wait(timeout=5)
```

- [ ] **Step 4: Implement runner orchestration**

Create `Helper_Scripts/cats_fuzz/runner.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

from Helper_Scripts.cats_fuzz import DEFAULT_TEST_API_KEY
from Helper_Scripts.cats_fuzz.cats_cli import (
    CatsProcessResult,
    build_cats_run_command,
    build_cats_stats_command,
    build_cats_validate_command,
    classify_cats_exit,
    run_command,
)
from Helper_Scripts.cats_fuzz.manifest import CatsBlock, get_builtin_block
from Helper_Scripts.cats_fuzz.summary import CatsRunSummary, mask_command, write_summary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_process_artifacts(result: CatsProcessResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return stdout_path, stderr_path


def _summarize(
    *,
    block: str,
    result: CatsProcessResult,
    output_dir: Path,
    cats_version: str,
    openapi_sha256: str,
    report_dir: Path,
) -> CatsRunSummary:
    stdout_path, stderr_path = _write_process_artifacts(result, output_dir)
    summary = CatsRunSummary(
        block=block,
        cats_version=cats_version,
        openapi_sha256=openapi_sha256,
        command=result.command,
        masked_command=mask_command(result.command),
        exit_code=result.exit_code,
        failure_class=classify_cats_exit(result.exit_code, result.stderr),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        report_dir=str(report_dir),
    )
    write_summary(summary, output_dir / "summary.json")
    return summary


def run_contract_block(
    *,
    contract_path: Path,
    output_dir: Path,
    cats_version: str,
    openapi_sha256: str | None = None,
    cats_bin: str = "cats",
) -> CatsRunSummary:
    block_dir = output_dir / "contract"
    validate_result = run_command(
        build_cats_validate_command(contract_path, cats_bin=cats_bin),
        timeout_seconds=60,
    )
    stats_result = run_command(
        build_cats_stats_command(contract_path, cats_bin=cats_bin),
        timeout_seconds=60,
    )
    merged = CatsProcessResult(
        command=validate_result.command,
        exit_code=validate_result.exit_code or stats_result.exit_code,
        stdout=validate_result.stdout + "\n" + stats_result.stdout,
        stderr=validate_result.stderr + "\n" + stats_result.stderr,
    )
    return _summarize(
        block="contract",
        result=merged,
        output_dir=block_dir,
        cats_version=cats_version,
        openapi_sha256=openapi_sha256 or _sha256(contract_path),
        report_dir=block_dir,
    )


def run_runtime_block(
    block: CatsBlock,
    *,
    contract_path: Path,
    server_url: str,
    output_dir: Path,
    cats_version: str,
    api_key: str = DEFAULT_TEST_API_KEY,
    cats_bin: str = "cats",
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> CatsRunSummary:
    block_dir = output_dir / block.name
    report_dir = block_dir / "cats-report"
    command = build_cats_run_command(
        block,
        contract_path=contract_path,
        server_url=server_url,
        output_dir=report_dir,
        api_key=api_key,
        cats_bin=cats_bin,
        dry_run=dry_run,
    )
    result = run_command(command, timeout_seconds=block.timeout_seconds, env=env)
    return _summarize(
        block=block.name,
        result=result,
        output_dir=block_dir,
        cats_version=cats_version,
        openapi_sha256=_sha256(contract_path),
        report_dir=report_dir,
    )


def get_default_runtime_block() -> CatsBlock:
    return get_builtin_block("public-read")
```

- [ ] **Step 5: Run runner tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_runner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Helper_Scripts/cats_fuzz/server.py Helper_Scripts/cats_fuzz/runner.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_runner.py
git commit -m "feat: orchestrate CATS fuzz runs"
```

## Task 6: Add CLI And Documentation

**Files:**
- Create: `Helper_Scripts/cats_fuzz/cli.py`
- Create: `Helper_Scripts/cats_fuzz/__main__.py`
- Create: `Docs/Development/CATS_Fuzzing.md`
- Modify: `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_runner.py` or create `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cli.py`

- [ ] **Step 1: Write CLI parser tests**

Create `tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cli.py`:

```python
from __future__ import annotations

import pytest

from Helper_Scripts.cats_fuzz.cli import parse_args


@pytest.mark.unit
def test_cli_defaults_to_contract_and_public_read() -> None:
    args = parse_args([])

    assert args.block == ["contract", "public-read"]
    assert args.output == "artifacts/cats-fuzz"
    assert args.start_server is True


@pytest.mark.unit
def test_cli_accepts_existing_server_url() -> None:
    args = parse_args(["--server-url", "http://127.0.0.1:8000", "--no-start-server"])

    assert args.server_url == "http://127.0.0.1:8000"
    assert args.start_server is False
```

- [ ] **Step 2: Run CLI tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cli.py -q
```

Expected: FAIL because `cli.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `Helper_Scripts/cats_fuzz/cli.py`:

```python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from Helper_Scripts.cats_fuzz.env import build_child_env
from Helper_Scripts.cats_fuzz.manifest import get_builtin_block
from Helper_Scripts.cats_fuzz.openapi_export import build_openapi_export_command
from Helper_Scripts.cats_fuzz.runner import run_contract_block, run_runtime_block
from Helper_Scripts.cats_fuzz.server import start_server, stop_server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local CATS fuzzing blocks for tldw_server")
    parser.add_argument("--block", action="append", default=None, help="Block to run; may be repeated")
    parser.add_argument("--output", default="artifacts/cats-fuzz")
    parser.add_argument("--cats-bin", default="cats")
    parser.add_argument("--server-url")
    parser.add_argument("--start-server", dest="start_server", action="store_true", default=True)
    parser.add_argument("--no-start-server", dest="start_server", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-external", action="store_true")
    args = parser.parse_args(argv)
    if args.block is None:
        args.block = ["contract", "public-read"]
    return args


def _cats_version(cats_bin: str) -> str:
    proc = subprocess.run([cats_bin, "--version"], check=False, capture_output=True, text=True)  # nosec B603
    return (proc.stdout or proc.stderr).strip().splitlines()[0] if proc.returncode == 0 else "unknown"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    child_env = build_child_env(output_dir, allow_external=args.allow_external)
    contract_path = output_dir / "openapi.json"

    subprocess.run(build_openapi_export_command(contract_path), check=True, env=child_env)  # nosec B603
    cats_version = _cats_version(args.cats_bin)

    server = None
    server_url = args.server_url
    try:
        if any(block != "contract" for block in args.block) and args.start_server:
            server = start_server(child_env)
            server_url = server.url
        exit_code = 0
        for block_name in args.block:
            if block_name == "contract":
                summary = run_contract_block(
                    contract_path=contract_path,
                    output_dir=output_dir,
                    cats_version=cats_version,
                    cats_bin=args.cats_bin,
                )
            else:
                if not server_url:
                    raise ValueError(f"{block_name} requires --server-url or --start-server")
                summary = run_runtime_block(
                    get_builtin_block(block_name),
                    contract_path=contract_path,
                    server_url=server_url,
                    output_dir=output_dir,
                    cats_version=cats_version,
                    cats_bin=args.cats_bin,
                    dry_run=args.dry_run,
                    env=child_env,
                )
            if summary.exit_code != 0:
                exit_code = summary.exit_code
        return exit_code
    finally:
        if server is not None:
            stop_server(server)


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `Helper_Scripts/cats_fuzz/__main__.py`:

```python
from __future__ import annotations

from Helper_Scripts.cats_fuzz.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Write usage docs**

Create `Docs/Development/CATS_Fuzzing.md`:

````markdown
# CATS API Fuzzing

The CATS harness runs OpenAPI-driven negative fuzzing against an isolated local tldw_server process. Do not run the default harness against production or a server with real provider credentials.

## First Slice

```bash
source .venv/bin/activate
python -m Helper_Scripts.cats_fuzz --block contract
python -m Helper_Scripts.cats_fuzz --block contract --block public-read
```

Artifacts are written under `artifacts/cats-fuzz/` by default:

- `openapi.json`
- per-block `summary.json`
- per-block `stdout.log` and `stderr.log`
- CATS HTML/JUnit report folders for runtime blocks

## Safety Defaults

- Starts uvicorn on `127.0.0.1` by default.
- Uses a deterministic long test API key.
- Writes AuthNZ and user databases under the artifact runtime directory.
- Sets `TLDW_ENV_FILE` to a generated minimal env file.
- Rejects real provider or webhook credentials unless `--allow-external` is passed.
- Uses CATS blackbox mode so only `5xx` responses are gate failures for runtime blocks.
````

- [ ] **Step 5: Run CLI tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Helper_Scripts/cats_fuzz/cli.py Helper_Scripts/cats_fuzz/__main__.py Docs/Development/CATS_Fuzzing.md tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cli.py
git commit -m "feat: add CATS fuzzing CLI"
```

## Task 7: Run End-to-End Harness Verification

**Files:**
- Modify: `backlog/tasks/task-2370 - Plan-CATS-API-fuzzing-harness-implementation.md` only for final notes if this plan task is still open, or the implementation task that owns execution.

- [ ] **Step 1: Run all focused Python tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VectorStores/test_vector_stores_openapi_examples.py \
  tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_manifest.py \
  tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_env.py \
  tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py \
  tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py \
  tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_runner.py \
  tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cli.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run CATS contract block**

Run:

```bash
source .venv/bin/activate
python -m Helper_Scripts.cats_fuzz --block contract --output /tmp/tldw-cats-contract
```

Expected: exit `0`, `/tmp/tldw-cats-contract/contract/summary.json` has `"failure_class": "ok"`.

- [ ] **Step 3: Run CATS public-read live block**

Run:

```bash
source .venv/bin/activate
python -m Helper_Scripts.cats_fuzz --block contract --block public-read --output /tmp/tldw-cats-public-read
```

Expected: exit `0` or a non-zero exit only when `summary.json` clearly classifies an API `5xx` issue worth fixing. If CATS itself throws a tool/internal error, record stderr and either adjust command construction or add a documented fallback.

- [ ] **Step 4: Run Bandit on touched executable harness scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r Helper_Scripts/cats_fuzz tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py -f json -o /tmp/bandit_cats_fuzz.json
```

Expected: no new findings in touched code. If Bandit flags the deliberate local subprocess/urlopen calls, add tight `# nosec` comments with a local-only justification only where needed.

- [ ] **Step 5: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 6: Commit final verification notes if needed**

If verification notes or docs changed:

```bash
git add <changed-files>
git commit -m "docs: record CATS fuzzing verification"
```

## Known Risks And Follow-Ups

- CATS `--dryRun` reached command parsing locally but threw a Java `ClassCastException` against the large tldw OpenAPI shape. Treat dry-run as helpful but not a hard preflight until proven stable with the finalized command.
- The first implementation should not add broad CI yet. Start with local/manual usage and a small contract/public-read command that can later be wired into CI.
- The worktree has no `.venv`; implementation must set up dependencies before running Python tests.
- `auth-read`, `auth-crud-isolated`, `media-light`, and broad/nightly blocks remain follow-up work after the first live public-read path is reliable.
- Subagent plan review was not dispatched while drafting this plan because the current thread policy only permits subagents when the user explicitly requests delegation.

## Success Criteria

- `cats validate` passes against the generated tldw OpenAPI file, or any remaining validation failure is recorded in a known-issues file with an explicit non-blocking contract gate.
- `python -m Helper_Scripts.cats_fuzz --block contract` writes a summary JSON and exits correctly.
- `python -m Helper_Scripts.cats_fuzz --block contract --block public-read` can start an isolated local server and run CATS blackbox fuzzing without real credentials.
- Focused pytest and Bandit commands pass.
- Reports never expose `X-API-KEY` or provider credentials.
