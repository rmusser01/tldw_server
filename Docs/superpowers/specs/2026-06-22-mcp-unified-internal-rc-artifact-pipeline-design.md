# MCP Unified Internal RC Artifact Pipeline Design

## Problem Statement

The `mcp_unified` directory is becoming a standalone package boundary for the
MCP Unified gateway, profile, policy, storage, reporting, smoke, and external
server runtime surfaces. It is not ready for public PyPI publishing yet, but it
does need a reliable internal release-candidate path that proves a built wheel
and source distribution can be installed and operated outside the repository
checkout.

Current repository state makes the boundary easy to confuse:

- the standalone package descriptor lives at `mcp_unified/pyproject.toml`;
- root `pyproject.toml` describes the full `tldw-server` package;
- root `make pypi-check` builds the root package, not the nested standalone
  MCP package;
- `mcp_unified/package_metadata.py` still declares
  `PACKAGE_STATUS = "internal-experimental"` and
  `PUBLISHING_STATUS = "not-published"`;
- package-boundary tests and artifact-gate tests already exist, but they are
  not yet a complete private RC pipeline with installed-artifact UAT evidence.

This design defines the internal artifact pipeline needed before TestPyPI or
PyPI publishing should be considered.

## Goals

- Build private wheel and sdist artifacts from `mcp_unified/pyproject.toml`.
- Keep standalone MCP artifact creation separate from root `tldw-server`
  packaging.
- Validate package metadata, extras, entry points, package data, typed marker,
  README, license metadata, and artifact boundaries.
- Install the built wheel into clean environments and run UAT from installed
  artifacts, not editable source.
- Prove the base package does not depend on root-only or heavyweight
  `tldw-server` dependency stacks.
- Run tiered CLI and smoke-harness UAT for required and optional package
  surfaces.
- Emit one machine-readable evidence report and one human-readable summary.
- Keep this workflow private/internal. Do not publish to TestPyPI or PyPI in
  this slice.

## Non-Goals

- Public PyPI or TestPyPI publishing.
- Adding PyPI trusted-publishing permissions, `id-token: write`, or upload
  credentials to the internal RC workflow.
- Changing package status from `internal-experimental`.
- Changing public install documentation to imply `pip install mcp-unified` is
  available from PyPI.
- Building or validating the root `tldw-server` package.
- Reworking standalone gateway feature behavior unrelated to packaging and UAT.

## Current State Findings

The package-local descriptor already defines:

- package name `mcp-unified`;
- Python support `>=3.10`;
- console scripts `mcp-unified-gateway` and `mcp-unified-smoke`;
- explicit package lists and package-dir mappings;
- package data for `py.typed`, `README.md`, and `USER_GUIDE.md`;
- extras for core, FastAPI, SQLite, federation, gateway, and development
  surfaces.

The repository already has useful validation pieces:

- `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
  contains metadata, artifact, install, and workflow assertions;
- `.github/tests/test_mcp_unified_artifact_gate.py` loads a subset of those
  artifact assertions without importing host pytest fixtures;
- `.github/workflows/pypi-package.yml` runs a package artifact gate for
  `mcp_unified/**` changes.

The missing piece is a cohesive RC harness and workflow that use the nested
package as the only build subject, then install and UAT the produced wheel.

## Internal RC Artifact Contract

An internal MCP Unified RC consists of:

- one wheel built from `mcp_unified/pyproject.toml`;
- one source distribution built from `mcp_unified/pyproject.toml`;
- SHA256 hashes for both artifacts;
- a JSON evidence report;
- a Markdown evidence summary;
- captured command results with redacted logs;
- package metadata snapshot;
- environment summary;
- known limitations and expected skips.

Artifacts must be named with package version and source commit. Example:

```text
mcp-unified-0.1.0-3b4b6a4-wheel-sdist
mcp-unified-0.1.0-3b4b6a4-evidence
```

Local artifacts should be written under a package-specific path such as:

```text
.artifacts/mcp-unified-rc/0.1.0-3b4b6a4/
```

CI artifacts should use the same version and short SHA naming convention.

## RC Harness Design

Use one package RC harness invoked by local Make targets and GitHub Actions.
This avoids YAML/local drift.

Recommended path:

```text
Helper_Scripts/mcp_unified_rc.py
```

The harness should depend only on the Python standard library plus explicitly
installed packaging or dev tools. It must not import `tldw_Server_API`, root
test fixtures, broad repository configuration, or application runtime modules.

The harness should expose subcommands:

- `build`: clean package-local RC output and build wheel plus sdist from
  `mcp_unified/pyproject.toml`;
- `artifact-gate`: validate built artifact metadata and boundaries;
- `install-smoke`: run clean-environment install and import checks;
- `extras-matrix`: install selected extras in clean environments and run
  minimal checks;
- `cli-uat`: run installed `mcp-unified-gateway` CLI workflows;
- `smoke-uat`: run installed `mcp-unified-smoke` scenarios;
- `evidence`: write or refresh the combined JSON and Markdown reports;
- `all`: run the full internal RC sequence.

Every subcommand should write structured result entries into the evidence file.
The harness should continue past optional/degraded checks when configured to do
so, but required-phase failures must mark the RC as failed.

## Local Commands

Add standalone-specific Make targets. They should be thin wrappers around the
RC harness.

Recommended targets:

```text
make mcp-unified-build
make mcp-unified-check
make mcp-unified-uat
make mcp-unified-rc
```

Behavior:

- `mcp-unified-build` runs `build`;
- `mcp-unified-check` runs `build`, `artifact-gate`, `twine check`, and
  `install-smoke`;
- `mcp-unified-uat` runs `cli-uat`, `smoke-uat`, and `extras-matrix` against a
  built wheel;
- `mcp-unified-rc` runs the full internal RC flow and writes the evidence
  bundle.

These targets must not reuse root `pypi-build` or root `pypi-check`.

## CI Workflow

Add a dedicated non-publishing workflow, for example:

```text
.github/workflows/mcp-unified-rc.yml
```

Triggers:

- `workflow_dispatch`;
- pull requests touching `mcp_unified/**`, package-boundary tests, smoke
  harness code, package docs, RC harness code, or the RC workflow;
- optionally pushes to `dev` after the pipeline is stable.

Permissions:

```yaml
permissions:
  contents: read
```

The workflow must not request `id-token: write`, because this slice does not
publish.

Jobs:

1. `build-artifacts`
   - install packaging tools;
   - call the RC harness `build`;
   - upload wheel/sdist artifacts and build metadata.
2. `artifact-gate`
   - validate metadata, README rendering, license metadata, extras, entry
     points, package data, typed marker, and sdist/wheel boundaries.
3. `fresh-install`
   - install the wheel into clean environments;
   - run both `--no-deps` and dependency-resolving install modes.
4. `extras-matrix`
   - test each package extra in a fresh environment with minimal checks.
5. `runtime-uat`
   - run installed CLI workflow UAT;
   - run smoke harness in-process and stdio scenarios;
   - run HTTP/WebSocket smoke only when the job owns server startup and
     teardown robustly.
6. `evidence`
   - collect reports, command summaries, hashes, skip reasons, and environment
     data;
   - upload the evidence bundle;
   - optionally post a concise PR summary if repository policy allows it.

## UAT Matrix

### Phase 1: Artifact Validation

Required:

- wheel and sdist build from nested package only;
- wheel name is `mcp-unified`;
- wheel version matches `mcp_unified.__version__` and
  `mcp_unified/pyproject.toml`;
- entry points include `mcp-unified-gateway` and `mcp-unified-smoke`;
- extras in wheel metadata match `mcp_unified/package_metadata.py`;
- README and user guide are included;
- `py.typed` is included;
- sdist excludes `tldw_Server_API`, WebUI files, media/RAG/STT assets, and
  unrelated repository paths;
- `twine check` passes.

### Phase 2: Fresh Install And Import

Required:

- install wheel with `--no-deps` into an isolated target and import from a
  `python -S` process outside the repository;
- install wheel normally into a clean virtual environment and run:
  - `python -c "import mcp_unified"`;
  - `mcp-unified-gateway package-info`;
  - `mcp-unified-smoke --help`.

Negative checks:

- base install must not import or require `tldw_Server_API`;
- base install must not require heavyweight root stacks such as `torch`,
  `chromadb`, `yt-dlp`, `faster_whisper`, `docling`, or WebUI dependencies.

### Phase 3: Per-Extra Checks

Each extra should be installed in a fresh environment. Testing all extras
together is not enough because it can hide missing declarations.

Required tiers:

- `core`: import package and run package-info;
- `gateway`: import gateway modules and run config validation;
- `sqlite`: import SQLite storage and reporting surfaces, then run a minimal
  store open/close or validation smoke;
- `dev`: import pytest-side artifact-gate dependencies and run a small package
  test selection.

Optional/degraded tiers:

- `lsp`, when present on the active branch: import LSP package surfaces and
  verify tool registration; missing Ruff or pylsp backend execution should be a
  documented degraded pass, not an RC failure.

### Phase 4: CLI Workflow UAT

Run from the installed wheel:

- `mcp-unified-gateway package-info`;
- `mcp-unified-gateway list-presets`;
- `mcp-unified-gateway show-preset project-researcher`;
- `mcp-unified-gateway validate-config`;
- duplicate a preset into a temporary SQLite-backed config;
- set and read the default profile;
- export and import a configuration snapshot;
- create/list/revoke credential grants where the CLI supports them;
- run tool-use report/export/cleanup against a temporary reporting store.

All generated files must live under the harness temporary directory.

### Phase 5: Smoke Harness UAT

Run from the installed wheel:

- `mcp-unified-smoke inprocess --json-report -`;
- `mcp-unified-smoke stdio` against the installed console script or a fixture
  command owned by the harness;
- HTTP smoke against a harness-owned local gateway process;
- WebSocket smoke against a harness-owned local gateway process.

The harness must own process startup and shutdown for HTTP/WebSocket checks.
If a platform forbids loopback binding, the check should report a
`runtime_transport` skip with the underlying reason.

### Phase 6: Cross-Platform Checks

Minimum internal RC matrix:

- Linux, Python 3.10 and 3.12;
- macOS local or CI when available;
- pure Windows path normalization and CLI parsing tests.

Better matrix when CI budget allows:

- Python 3.10, 3.11, 3.12, and 3.13 on Linux;
- macOS on at least one current Python;
- Windows installed CLI smoke once process and path behavior are stable.

### Phase 7: Security And Supply-Chain Checks

Required:

- Bandit on package and harness touched scope;
- `twine check` on generated artifacts;
- artifact path and log redaction checks;
- dependency-boundary checks against forbidden root/heavy packages;
- no raw secrets or absolute local paths in evidence reports.

Optional at first:

- `pip-audit` against the normal dependency-resolving install environment.

The optional audit can become blocking once false-positive policy and
vulnerability triage ownership are defined.

## Evidence Report Schema

The JSON report should include:

```json
{
  "schema_version": "1",
  "ok": true,
  "package": {
    "name": "mcp-unified",
    "version": "0.1.0",
    "commit": "3b4b6a4",
    "status": "internal-experimental",
    "publishing_status": "not-published"
  },
  "artifacts": [
    {
      "kind": "wheel",
      "filename": "mcp_unified-0.1.0-py3-none-any.whl",
      "sha256": "..."
    }
  ],
  "environment": {
    "os": "Linux",
    "python": "3.12.x",
    "runner": "github-actions"
  },
  "results": [
    {
      "phase": "artifact_metadata",
      "name": "wheel_metadata",
      "status": "passed",
      "duration_ms": 123
    }
  ],
  "summary": {
    "passed": 0,
    "failed": 0,
    "skipped": 0
  },
  "known_limitations": []
}
```

Failure categories:

- `artifact_metadata`;
- `dependency_boundary`;
- `fresh_install`;
- `cli_contract`;
- `runtime_transport`;
- `optional_backend`;
- `report_redaction`;
- `security_audit`.

Redaction rules:

- no secret environment variable values;
- no raw admin keys, API keys, credential grants, or access tokens;
- no raw local absolute paths in published summaries;
- use relative paths under the artifact root when possible;
- truncate command output and keep full logs only inside private CI artifacts
  after redaction.

## Publishing Guardrails

The existing root publishing workflow must not be mistaken for standalone MCP
publishing. The implementation should add one of these guardrails:

- rename or document root publishing workflow labels so they clearly refer to
  `tldw-server`; or
- add an assertion/test that fails if standalone MCP docs or metadata imply the
  root `publish-pypi.yml` publishes `mcp-unified`; or
- add a disabled future standalone publishing workflow stub that explicitly
  says publishing is not enabled yet.

The internal RC workflow must not upload to package indexes.

Future public publishing should use PyPI Trusted Publishing/OIDC instead of
long-lived upload tokens. That future workflow should be gated on successful
internal RC evidence, TestPyPI evidence, docs review, and a human-owned change
summary.

## Risks And Mitigations

Risk: wrong package is built or published.

Mitigation: standalone-specific build targets, nested-package artifact
assertions, artifact names that include `mcp-unified`, and root workflow
guardrails.

Risk: editable install hides missing wheel files.

Mitigation: all RC UAT installs from the built wheel.

Risk: standalone package pulls in root dependencies.

Mitigation: wheel metadata checks, clean environment install checks, negative
dependency checks, and import checks outside the repo checkout.

Risk: optional backend availability creates false failures.

Mitigation: tier optional checks separately and classify degraded LSP/backend
status as expected when optional executables are absent.

Risk: local and CI behavior drift.

Mitigation: Make and GitHub Actions both call the same RC harness.

Risk: evidence leaks secrets or host paths.

Mitigation: centralize evidence redaction, avoid raw env dumps, and assert no
secret-like fields or local absolute paths appear in generated reports.

Risk: Windows path bugs remain hidden.

Mitigation: add pure path and CLI parsing tests immediately, then add full
Windows installed-wheel smoke when runtime process behavior is stable.

## Follow-Up Path To TestPyPI And PyPI

After the internal RC pipeline passes consistently:

1. Run the RC pipeline on a clean branch and preserve the evidence bundle.
2. Review package metadata for public readability: classifiers, project URLs,
   author/maintainer, README rendering, license files, and known limitations.
3. Add TestPyPI Trusted Publisher configuration in PyPI for the repository,
   workflow name, environment, and project.
4. Add a TestPyPI-only workflow that consumes the same RC harness artifacts and
   requests `id-token: write` only in the publishing job.
5. Publish to TestPyPI and install from TestPyPI in a clean environment.
6. Only after TestPyPI UAT passes, add the PyPI publishing environment.

Public publishing remains explicitly out of scope for the internal RC slice.

## Acceptance Criteria

- A dedicated internal RC harness design exists and avoids root package imports.
- Local standalone-specific build/check/UAT commands are specified.
- CI workflow responsibilities and permissions are specified.
- Artifact validation and installed-wheel UAT are separated into clear phases.
- Evidence report schema, failure categories, and redaction requirements are
  defined.
- Root-vs-standalone publishing ambiguity is identified as a release blocker
  with guardrail options.
- Public publishing and TestPyPI are documented as follow-up work only.
