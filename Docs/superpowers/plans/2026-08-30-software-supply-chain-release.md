# Software Supply-Chain Release Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public core release candidate dependency-locked, SBOM-complete, digest-addressed, vulnerability-gated, and provenance-verifiable before any production artifact is promoted.

**Architecture:** Establish one repository-owned policy/evidence layer, then feed it deterministic Python and Bun dependency graphs plus exact production image identities. Pull-request workflows validate source and build contracts read-only; trusted release workflows build once, scan exact OCI subject and `linux/amd64` child digests, promote only the approved backend digests, and publish durable evidence.

**Tech Stack:** Python 3.10+ standard library, pytest, PyYAML for existing workflow tests, uv 0.12.7, Bun, Next.js 16.3.3, CycloneDX JSON, cdxgen 13.0.1, CycloneDX CLI 0.33.1, Trivy 0.74.0, Docker Buildx/BuildKit, GitHub Actions, GHCR, GitHub artifact attestations, PyPI trusted publishing.

**Spec:** `Docs/superpowers/specs/2026-08-30-software-supply-chain-release-design.md`

## Global Constraints

- Work only in the isolated `codex/task-13013-7-supply-chain-design` worktree. Do not modify another repository, checkout, or worktree.
- Follow test-driven development: record each intended red failure, implement the smallest complete behavior, rerun focused verification, and commit before moving to the next reviewable unit.
- Keep Next.js exactly `16.3.3` in both apps. Pin WebUI `@sentry/nextjs` to `10.46.0` and align all named Next companion packages to `16.3.3`.
- Keep `apps/bun.lock` as the complete applications-workspace lock and `admin-ui/bun.lock` as the independent Admin UI lock. Every CI/Docker install uses `bun install --frozen-lockfile`.
- Use one committed universal root `uv.lock`. Production installs use `uv sync --locked --no-dev --no-editable` plus only the documented production extra.
- Pin uv as `ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945`.
- Pin cdxgen as `ghcr.io/cdxgen/cdxgen:v13@sha256:0be75639a833b59d1ba29b3c8ac00dfd2e41e7568d56b6c039007caadebebc0d` and assert that `cdxgen --version` reports `13.0.1` before generation.
- Pin CycloneDX CLI as `docker.io/cyclonedx/cyclonedx-cli:0.33.1@sha256:252c2e26f468c25fea1e63ecde1bc3198ad6e9dbb57f5ed3236bddcb2281b3a7`.
- Pin Trivy as `ghcr.io/aquasecurity/trivy:0.74.0@sha256:62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969`.
- Pin every third-party Action changed by this task to a full 40-character commit SHA. Use existing vetted pins when present and let Dependabot own later reviewable updates.
- Generate separate CycloneDX JSON for Python, applications workspace, Admin UI, source aggregate, five project-built images, and six third-party reference images.
- Fail closed for every unexcepted Critical or High production vulnerability, including unfixed findings. Release scanning requires identified vulnerability data no older than 24 hours.
- Begin with `{"schema_version":1,"exceptions":[]}`. Critical exceptions last at most 7 days; High exceptions last at most 30 days.
- Distinguish an OCI subject/index digest from the selected `linux/amd64` child manifest digest in every scan decision and evidence record.
- Certify only `linux/amd64`. Do not emit or document a multi-platform release claim.
- Publish formal tags only for app, worker, and audio-worker. WebUI and Admin UI remain local-OCI build-and-scan artifacts; publishing those images is outside this task.
- Keep release freeze/publication ownership with TASK-13013.3, production
  topology and recovery behavior with TASK-13013.6, downstream handoff with
  TASK-12983, and capacity/soak certification with TASK-13013.9.
- Never claim project provenance for Caddy, PostgreSQL, Redis, Prometheus, Alertmanager, or Grafana. Record only their upstream immutable references, selected platform manifests, SBOMs, and scans.
- Keep pull-request jobs read-only. Registry, attestation, OIDC, and release-write permissions belong only to trusted-ref jobs that need them.
- Never place secrets in Docker build arguments, provenance parameters, labels, SBOMs, reports, manifests, logs, or release assets.
- Refresh `Docs/Published` only through `bash Helper_Scripts/refresh_docs_published.sh`; never edit generated mirrors directly.
- Run focused Bandit on every changed Python supply-chain helper and deployment validator.
- A materially AI-authored PR cannot merge until the requester supplies the repository-required human `Change summary` explaining both what changed and why.

---

## Stage Summary

### Stage 1: Exception and scan-policy core

**Goal:** Provide one deterministic, repository-owned exception schema and exact finding evaluator.

**Success Criteria:** Empty policy passes; invalid, expired, overlong, unmatched, or cross-component exceptions fail; complete and adjusted results remain distinct.

**Tests:** `test_exception_policy.py` fixtures plus focused Bandit.

**Status:** Complete

### Stage 2: Supported framework and deterministic dependency locks

**Goal:** Upgrade both Next.js apps, automate Bun/uv/Docker updates, and commit a universal Python resolution used by production containers.

**Success Criteria:** Exact manifest pins, frozen Bun installs, lock-fresh uv profiles, required regressions, and production import/container smokes pass.

**Tests:** Dependency-lock contracts, WebUI/Admin suites, uv sync/export checks, and container builds.

**Status:** Complete

### Stage 3: Fail-closed source SBOM and vulnerability gate

**Goal:** Replace mutable/fallback SBOM generation with pinned, separately validated component and aggregate evidence.

**Success Criteria:** All expected source SBOMs are nonempty and valid, Trivy reports are complete, missing outputs fail, and required jobs cannot soft-fail or skip.

**Tests:** SBOM workflow contracts, fixture validation, actionlint, and a local pinned-tool smoke.

**Status:** In Progress

### Stage 4: Immutable production image set

**Goal:** Digest-pin all project bases and reference runtime images and require `tag@sha256` through deployment preflight.

**Success Criteria:** Five production Dockerfiles and six reference runtime inputs are immutable; platform manifests are recorded; reference Compose and docs reject tag-only inputs.

**Tests:** Dockerfile/reference-image contracts, production preflight tests, Compose contracts, builds, SBOMs, and scans.

**Status:** In Progress

### Stage 5: Exact-digest candidate admission and promotion

**Goal:** Build once, scan exact artifacts, promote three backend images only after every owned/reference gate passes, and publish durable evidence.

**Success Criteria:** Full-version tags precede floating tags; failed matrices cannot reach promotion; frontend OCI artifacts are never pushed; draft release publication happens last.

**Tests:** Release evidence helpers, workflow contract tests, actionlint, mocked promotion failure cases, and trusted-CI candidate execution.

**Status:** In Progress

### Stage 6: Attestation, operator documentation, and final certification

**Goal:** Make image/PyPI provenance explicit, document verification and exceptions, and complete all repository quality gates.

**Success Criteria:** Maximum BuildKit provenance, OCI SBOMs, GitHub attestations, PEP 740 attestations, release assets, docs, Bandit, regressions, and human Change summary are complete.

**Tests:** PyPI/release/docs contracts, full focused suites, Bandit, secret scan, `git diff --check`, PR checks, and post-review reruns.

**Status:** Not Started

---

## File Responsibility Map

- `.github/supply-chain/vulnerability-exceptions.json` — canonical reviewed exception list, empty initially.
- `.github/supply-chain/vulnerability-exceptions.schema.json` — document exact exception fields, severity values, dates, scope, and maximum durations.
- `.github/supply-chain/reference-images.json` — canonical six-image third-party reference matrix with readable tags, index digests, platform, and selected child digests.
- `Helper_Scripts/Supply_Chain/exception_policy.py` — parse/validate exceptions, emit ephemeral Trivy ignore input, and evaluate complete Trivy JSON without broad suppression.
- `Helper_Scripts/Supply_Chain/release_evidence.py` — validate component records, hashes, platform/subject relationships, and assemble the versioned release digest manifest.
- `tldw_Server_API/tests/Supply_Chain/fixtures/trivy-critical.json` — deterministic complete-report fixture for exact exception matching.
- `tldw_Server_API/tests/Supply_Chain/test_exception_policy.py` — prove fail-closed exception and finding behavior.
- `tldw_Server_API/tests/Supply_Chain/test_release_evidence.py` — prove digest/manifest/file-hash consistency and missing-evidence failures.
- `tldw_Server_API/tests/Supply_Chain/test_dependency_lock_contracts.py` — enforce exact Next pins, two Bun locks, universal uv lock, production sync commands, and Dependabot ecosystems.
- `tldw_Server_API/tests/Supply_Chain/test_image_pinning_contracts.py` — enforce Dockerfile `FROM`, reference matrix, Compose input, and platform identity rules.
- `pyproject.toml` and `uv.lock` — declare and lock root production plus release-tool dependency profiles.
- `apps/tldw-frontend/package.json` and `apps/bun.lock` — own the WebUI framework baseline inside the complete applications workspace.
- `admin-ui/package.json` and `admin-ui/bun.lock` — own the Admin UI framework baseline.
- `.github/dependabot.yml` — update uv, both Bun roots, Dockerfiles, and GitHub Actions.
- `Dockerfiles/Dockerfile.prod` — install exact root production Python resolution into a non-editable virtual environment.
- `Dockerfiles/Dockerfile.worker` — install the locked `multiplayer` production profile.
- `Dockerfiles/Dockerfile.audio_gpu_worker` — install the locked audio-worker production profile.
- `Dockerfiles/Dockerfile.webui` and `Dockerfiles/Dockerfile.admin-ui` — use digest-pinned Bun/Node bases and frozen locks.
- `.github/workflows/sbom.yml` — reusable/read-only source SBOM, validation, scan, and artifact gate.
- `.github/workflows/container-build-check.yml` — build and scan all five production image definitions for pull requests without publishing them.
- `.github/workflows/publish-ghcr-main.yml` — scan the exact main candidate before moving `main`/`sha-*` aliases.
- `.github/workflows/publish-docker.yml` — trusted draft-release candidate build, scan, evidence, promotion, and publication workflow.
- `.github/workflows/publish-pypi.yml` and `.github/workflows/pypi-package.yml` — consume locked build/test inputs and make PEP 740 attestations explicit.
- `Helper_Scripts/Deployment/production_preflight.py` — require digest identity for current, rollback, and all reference images.
- `Dockerfiles/production.env.example` — name every current, rollback, and third-party reference image variable without supplying mutable defaults.
- `Dockerfiles/docker-compose.production.yml` and `Dockerfiles/Monitoring/docker-compose.production.yml` — require immutable external image inputs with accurate messages.
- `tldw_Server_API/tests/CI/test_sbom_workflow_contracts.py` — enforce pinned source tools, complete output matrix, validation, scan, and unconditional result gate.
- `tldw_Server_API/tests/CI/test_release_workflow_contracts.py` — enforce build-once, scan-before-promotion, three published images, two build-only frontends, draft release, and digest agreement.
- `tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py` — enforce locked build inputs, exact artifact reuse, and explicit attestations.
- `tldw_Server_API/tests/Utils/test_production_preflight.py` and `test_docker_production_reference.py` — enforce `tag@sha256` for the reference deployment.
- `Docs/Development/Software_Supply_Chain.md` — canonical dependency, SBOM, scan, exception, attestation, and verification runbook.
- `Docs/Development/Release_Process.md`, `Docs/Development/PyPI_Publishing.md`, and `Docs/Deployment/Production_Reference_Deployment.md` — integrate the new release and operator contract.
- `backlog/tasks/task-13013.7 - Close-dependency-and-software-supply-chain-release-gaps.md` — record plan, files, verification, PR, review, and merge evidence through Backlog MCP only.

---

### Task 1: Build the Fail-Closed Exception Policy Engine

**Files:**
- Create: `.github/supply-chain/vulnerability-exceptions.json`
- Create: `.github/supply-chain/vulnerability-exceptions.schema.json`
- Create: `Helper_Scripts/Supply_Chain/__init__.py`
- Create: `Helper_Scripts/Supply_Chain/exception_policy.py`
- Create: `tldw_Server_API/tests/Supply_Chain/fixtures/trivy-critical.json`
- Create: `tldw_Server_API/tests/Supply_Chain/test_exception_policy.py`

**Interfaces:**
- Produces: `PolicyError(ValueError)` with bounded field/exception context.
- Produces: `VulnerabilityException(id: str, vulnerability_id: str, component: str, purl: str, installed_version: str, severity: str, rationale: str, mitigation: str, owner: str, approval: str, created_on: date, expires_on: date, supersedes: str | None)`.
- Produces: `ExceptionPolicy(schema_version: int, exceptions: tuple[VulnerabilityException, ...])`.
- Produces: `Finding(vulnerability_id: str, purl: str, installed_version: str, severity: str, target: str)`.
- Produces: `ScanDecision(component: str, blocking: tuple[Finding, ...], excepted: tuple[Finding, ...], unmatched_exception_ids: tuple[str, ...])`.
- Produces: `load_policy(path: Path, *, today: date) -> ExceptionPolicy`.
- Produces: `write_trivy_ignore(policy: ExceptionPolicy, *, component: str, output: Path) -> None`.
- Produces: `evaluate_trivy_report(report: Mapping[str, object], *, component: str, policy: ExceptionPolicy, today: date) -> ScanDecision`.

- [x] **Step 1: Create the empty canonical policy and JSON schema**

Write the canonical policy exactly:

```json
{
  "schema_version": 1,
  "exceptions": []
}
```

Define schema-required record fields exactly as the interface above. Set `severity` to `CRITICAL` or `HIGH`, `component` to the enumerated source/image matrix, `additionalProperties` to false, and all dates to ISO `YYYY-MM-DD`.

- [x] **Step 2: Write red tests for valid empty policy and invalid structure**

Add tests using temporary JSON files:

```python
def test_empty_policy_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "exceptions.json"
    path.write_text('{"schema_version":1,"exceptions":[]}\n', encoding="utf-8")

    assert load_policy(path, today=date(2026, 8, 30)).exceptions == ()


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"schema_version": 2, "exceptions": []},
        {"schema_version": 1, "exceptions": "all"},
        {"schema_version": 1, "exceptions": [{"id": "incomplete"}]},
    ),
)
def test_invalid_policy_fails_closed(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PolicyError):
        load_policy(path, today=date(2026, 8, 30))
```

- [x] **Step 3: Write red duration, identity, and scope tests**

Use one valid Critical record with `created_on=2026-08-30` and
`expires_on=2026-09-06`. Assert 8 days fails. Use one valid High record
ending `2026-09-29` and assert 31 days fails. Add failures for duplicate IDs,
invalid PURL, blank owner/rationale/mitigation, non-repository approval URL,
expiry before creation, expiry before `today`, unknown component, and a
`supersedes` value equal to its own ID. For a renewal fixture containing both
records, require a new ID, a `supersedes` link, a different approval
reference, and revised rationale; Git history retains the removed prior entry
after the renewal becomes the only active policy record.

- [x] **Step 4: Write red exact-finding and stale-exception tests**

Build `trivy-critical.json` with one `CRITICAL` finding for component `image-app`, vulnerability `CVE-2026-1000`, PURL `pkg:pypi/example@1.0.0`, and installed version `1.0.0`. Assert:

```python
def test_exception_matches_only_exact_component_package_and_version(
    valid_policy: ExceptionPolicy,
    trivy_report: dict[str, object],
) -> None:
    decision = evaluate_trivy_report(
        trivy_report,
        component="image-app",
        policy=valid_policy,
        today=date(2026, 8, 30),
    )
    assert decision.blocking == ()
    assert [item.vulnerability_id for item in decision.excepted] == ["CVE-2026-1000"]
    assert decision.unmatched_exception_ids == ()
```

Parameterize changed component, vulnerability, PURL, version, and severity; each must remain blocking and mark the declared exception unmatched.

- [x] **Step 5: Run the focused tests and confirm the red state**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/tests/Supply_Chain/test_exception_policy.py -q
```

Expected: collection fails because `Helper_Scripts.Supply_Chain.exception_policy` does not exist.

- [x] **Step 6: Implement strict standard-library parsing and validation**

Use `json`, `dataclasses`, `datetime`, `pathlib`, `re`, and `urllib.parse` only. Reject booleans where integers are expected, unknown keys, non-string scalar values, duplicate IDs, and dates not written canonically. Keep `PolicyError` messages bounded to field names and exception IDs; never echo report bodies.

- [x] **Step 7: Implement ephemeral Trivy policy and complete-report evaluation**

Write JSON syntax to the ephemeral `.trivyignore.yaml` file because JSON is valid YAML 1.2:

```python
payload = {
    "vulnerabilities": [
        {
            "id": item.vulnerability_id,
            "purls": [item.purl],
            "statement": f"{item.id}: {item.rationale}",
            "expired_at": item.expires_on.isoformat(),
        }
        for item in policy.exceptions
        if item.component == component
    ]
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

Independently evaluate the original Trivy JSON so a scanner-ignore parsing difference cannot broaden suppression. A release passes only when `blocking` and `unmatched_exception_ids` are both empty.

- [x] **Step 8: Run tests and Bandit**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/tests/Supply_Chain/test_exception_policy.py -q
python -m bandit -r Helper_Scripts/Supply_Chain/exception_policy.py -f json -o /tmp/bandit_task_13013_7_policy.json
```

Expected: all tests pass; Bandit reports no new findings.

- [x] **Step 9: Commit the policy core**

```bash
git add .github/supply-chain/vulnerability-exceptions.json \
  .github/supply-chain/vulnerability-exceptions.schema.json \
  Helper_Scripts/Supply_Chain \
  tldw_Server_API/tests/Supply_Chain
git commit -m "feat: add fail-closed vulnerability policy (TASK-13013.7)"
```

---

### Task 2: Upgrade Next.js and Put Both Bun Locks Under Update Control

**Files:**
- Create: `tldw_Server_API/tests/Supply_Chain/test_dependency_lock_contracts.py`
- Modify: `apps/tldw-frontend/package.json`
- Modify: `apps/bun.lock`
- Modify: `admin-ui/package.json`
- Modify: `admin-ui/bun.lock`
- Modify: `.github/dependabot.yml`
- Modify only when a named regression proves it necessary: the exact app
  configuration/source file responsible for a 16.3.3 compatibility failure

**Interfaces:**
- Produces: exact WebUI pins `next=16.3.3`, `@sentry/nextjs=10.46.0`, `@next/eslint-plugin-next=16.3.3`.
- Produces: exact Admin pins `next=16.3.3`, `@next/bundle-analyzer=16.3.3`, `eslint-config-next=16.3.3`.
- Produces: Dependabot Bun roots `/apps` and `/admin-ui` plus root uv and Docker entries.

- [x] **Step 1: Write red manifest, lock, and Dependabot contract tests**

Parse both package manifests and `.github/dependabot.yml`:

```python
def test_next_security_baseline_is_exact() -> None:
    web = json.loads(Path("apps/tldw-frontend/package.json").read_text())
    admin = json.loads(Path("admin-ui/package.json").read_text())

    assert web["dependencies"]["next"] == "16.3.3"
    assert web["dependencies"]["@sentry/nextjs"] == "10.46.0"
    assert web["devDependencies"]["@next/eslint-plugin-next"] == "16.3.3"
    assert admin["dependencies"]["next"] == "16.3.3"
    assert admin["devDependencies"]["@next/bundle-analyzer"] == "16.3.3"
    assert admin["devDependencies"]["eslint-config-next"] == "16.3.3"
```

Assert the text locks contain `next@16.3.3` and no locked `@sentry/nextjs@9.`. Assert Dependabot has exactly one `bun` entry for each root, one `uv` root entry, at least one `docker` production-root entry, and the existing GitHub Actions entry.

- [x] **Step 2: Run the contract test and confirm the intended failures**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/tests/Supply_Chain/test_dependency_lock_contracts.py -q
```

Expected: exact Next/Sentry/companion and missing Bun/uv/Docker Dependabot assertions fail.

- [x] **Step 3: Update exact manifests and regenerate each lock**

Edit only the named direct dependencies, then run:

```bash
cd apps
bun install
cd ../admin-ui
bun install
cd ..
```

Review each lock diff for only the expected Next/Sentry/companion graph and necessary transitive changes.

- [x] **Step 4: Add bounded Dependabot entries**

Copy the existing weekly schedule, labels, assignee, commit-message style, and grouped review policy. Use `package-ecosystem: "bun"` for `/apps` and `/admin-ui`, `"uv"` for `/`, and `"docker"` for `/Dockerfiles`. Keep per-ecosystem open-PR limits bounded.

- [x] **Step 5: Prove frozen installs and rerun the contract**

Run:

```bash
apps_lock_before="$(shasum -a 256 apps/bun.lock)"
admin_lock_before="$(shasum -a 256 admin-ui/bun.lock)"
cd apps
bun install --frozen-lockfile
cd ../admin-ui
bun install --frozen-lockfile
cd ..
test "$apps_lock_before" = "$(shasum -a 256 apps/bun.lock)"
test "$admin_lock_before" = "$(shasum -a 256 admin-ui/bun.lock)"
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/tests/Supply_Chain/test_dependency_lock_contracts.py -q
```

Expected: frozen installs and contract tests pass; lockfiles remain unchanged.

- [x] **Step 6: Run WebUI regressions**

Run from `apps/tldw-frontend`:

```bash
bun run lint
bun run typecheck
bun run test:run
bun run build:prod
bun run e2e:critical
```

If a failure is caused by the 16.3.3/Sentry compatibility boundary, add the smallest source/configuration fix and a focused regression. Do not absorb unrelated TASK-12116 cleanup.

- [x] **Step 7: Run Admin UI regressions**

Run from `admin-ui`:

```bash
bun run lint
bun run typecheck
bun run test
bun run build
bun run test:smoke
```

Run the existing real-backend authentication workflow in trusted CI; locally run it only when its backend prerequisites are already available.

- [x] **Step 8: Commit the supported Bun baseline**

```bash
git add .github/dependabot.yml \
  apps/tldw-frontend/package.json apps/bun.lock \
  admin-ui/package.json admin-ui/bun.lock \
  tldw_Server_API/tests/Supply_Chain/test_dependency_lock_contracts.py
git status --short
git commit -m "build: adopt supported Next and Bun locks (TASK-13013.7)"
```

If a regression required a compatibility file, review and stage that exact
path before the commit. `git status --short` must show no required
compatibility change left unstaged.

---

### Task 3: Commit the Universal uv Resolution and Use It in Production Images

**Files:**
- Modify: `pyproject.toml`
- Create: `uv.lock`
- Modify: `Dockerfiles/Dockerfile.prod`
- Modify: `Dockerfiles/Dockerfile.worker`
- Modify: `Dockerfiles/Dockerfile.audio_gpu_worker`
- Modify: `tldw_Server_API/tests/Supply_Chain/test_dependency_lock_contracts.py`
- Modify: `.github/workflows/container-build-check.yml`

**Interfaces:**
- Produces: root default application profile.
- Produces: worker profile `uv sync --locked --no-dev --no-editable --extra multiplayer`.
- Produces: audio-worker profile `uv sync --locked --no-dev --no-editable`.
- Produces: release-tool group `build==1.6.0`, `twine==7.0.0`,
  `setuptools==84.0.0`, and `wheel==0.48.0` in `uv.lock`.
- Produces: `UV_PROJECT_ENVIRONMENT=/opt/tldw-venv` copied unchanged into runtime stages.

- [x] **Step 1: Extend red dependency contracts for uv**

Assert `uv.lock` exists, begins with a supported uv lock version, contains the root `tldw-server` package, and is referenced by all three Python production Dockerfiles. Assert Dockerfiles contain `uv sync --locked`, `--no-dev`, `--no-editable`, no editable `pip install -e`, and no unbounded `pip install --upgrade pip`.

- [x] **Step 2: Run the focused test and confirm the red state**

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/tests/Supply_Chain/test_dependency_lock_contracts.py -q
```

Expected: missing `uv.lock` and mutable Docker install assertions fail.

- [x] **Step 3: Declare the locked release-tool group**

Add one PEP 735 `release` dependency group containing `build==1.6.0`,
`twine==7.0.0`, `setuptools==84.0.0`, and `wheel==0.48.0`. Do not move
application dependencies or optional runtime extras during this task. Pin
`[build-system].requires` to `setuptools==84.0.0` and `wheel==0.48.0` so
isolated PEP 517 builds do not resolve a different backend.

- [x] **Step 4: Generate the universal lock with pinned uv**

Run from the worktree root:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD:/work" \
  --workdir /work \
  ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945 \
  uv lock
```

Review the lock for repository/VCS sources, unexpected prereleases, unsupported Python requirements, and resolution errors before proceeding.

- [x] **Step 5: Prove all production profiles sync without mutation**

Use isolated temporary environments:

```bash
uv_lock_before="$(shasum -a 256 uv.lock)"
UV_PROJECT_ENVIRONMENT=/tmp/tldw-uv-app uv sync --locked --no-dev --no-editable
UV_PROJECT_ENVIRONMENT=/tmp/tldw-uv-worker uv sync --locked --no-dev --no-editable --extra multiplayer
UV_PROJECT_ENVIRONMENT=/tmp/tldw-uv-audio uv sync --locked --no-dev --no-editable
uv lock --check
test "$uv_lock_before" = "$(shasum -a 256 uv.lock)"
```

Run these commands with uv 0.12.7 from the pinned container or a local binary whose `uv --version` is exactly 0.12.7.

- [x] **Step 6: Replace production pip resolution with locked uv sync**

For each Python Dockerfile:

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945 /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/tldw-venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock README.md LICENSE /app/
```

Copy the source required for non-editable installation, run the profile-specific `uv sync`, copy `/opt/tldw-venv` into the runtime image, and prepend `/opt/tldw-venv/bin` to `PATH`. Preserve existing non-root/runtime/health behavior.

- [x] **Step 7: Run contract, import, and container build smokes**

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/tests/Supply_Chain/test_dependency_lock_contracts.py -q
docker build --platform linux/amd64 -f Dockerfiles/Dockerfile.prod -t tldw-task-13013-7-app .
docker build --platform linux/amd64 -f Dockerfiles/Dockerfile.worker -t tldw-task-13013-7-worker .
docker build --platform linux/amd64 -f Dockerfiles/Dockerfile.audio_gpu_worker -t tldw-task-13013-7-audio .
docker run --rm --entrypoint python tldw-task-13013-7-app -c "import tldw_Server_API"
docker run --rm --entrypoint python tldw-task-13013-7-worker -c "import tldw_Server_API"
docker run --rm --entrypoint python tldw-task-13013-7-audio -c "import tldw_Server_API"
```

Expected: all builds and imports pass without dependency resolution warnings.

- [x] **Step 8: Commit deterministic Python production resolution**

```bash
git add pyproject.toml uv.lock \
  Dockerfiles/Dockerfile.prod Dockerfiles/Dockerfile.worker \
  Dockerfiles/Dockerfile.audio_gpu_worker \
  .github/workflows/container-build-check.yml \
  tldw_Server_API/tests/Supply_Chain/test_dependency_lock_contracts.py
git commit -m "build: lock Python production resolution with uv (TASK-13013.7)"
```

---

### Task 4: Replace the SBOM Workflow With a Pinned Fail-Closed Source Gate

**Files:**
- Rewrite: `.github/workflows/sbom.yml`
- Rewrite: `tldw_Server_API/tests/CI/test_sbom_workflow_contracts.py`
- Modify: `tldw_Server_API/tests/Supply_Chain/test_exception_policy.py`

**Interfaces:**
- Produces: `sbom-python-root.cdx.json`.
- Produces: `sbom-apps-workspace.cdx.json`.
- Produces: `sbom-admin-ui.cdx.json`.
- Produces: `sbom-source-aggregate.cdx.json`.
- Produces: `trivy-source-python-root.json`,
  `trivy-source-apps-workspace.json`, and
  `trivy-source-admin-ui.json` plus matching `scan-decision-source-*` files.
- Consumes: `exception_policy.py` and canonical empty/approved policy.
- Produces: reusable `workflow_call` plus existing pull-request/protected-branch entry points.

- [x] **Step 1: Replace obsolete workflow tests with red fail-closed contracts**

Assert:

```python
def test_sbom_workflow_has_exact_component_outputs() -> None:
    workflow = _load(".github/workflows/sbom.yml")
    text = Path(".github/workflows/sbom.yml").read_text(encoding="utf-8")
    for name in (
        "sbom-python-root.cdx.json",
        "sbom-apps-workspace.cdx.json",
        "sbom-admin-ui.cdx.json",
        "sbom-source-aggregate.cdx.json",
    ):
        assert name in text
    assert "package-lock.json" not in text
    assert "continue-on-error: true" not in text
    assert "npx -y" not in text
```

Also assert all external `uses` values match `@[0-9a-f]{40}`; all four pinned tool refs appear literally; uploads use `if-no-files-found: error`; a final gate needs every producer; and the job has no package-write permission on pull requests.

- [x] **Step 2: Run the SBOM contracts and confirm the red state**

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/tests/CI/test_sbom_workflow_contracts.py -q
```

Expected: old fallback, npm-only, mutable tool, soft validation, and missing-output assertions fail.

- [x] **Step 3: Generate the Python CycloneDX file with pinned uv**

Use:

```bash
uv export \
  --locked \
  --no-dev \
  --no-editable \
  --format cyclonedx1.5 \
  --output-file sbom-python-root.cdx.json
```

Immediately parse JSON and assert `bomFormat == "CycloneDX"`,
`specVersion == "1.5"`, a valid `serialNumber`, a nonempty `components`
list, and root metadata identifying `tldw-server`.

- [x] **Step 4: Generate the two Bun SBOMs with pinned cdxgen**

Run the pinned container separately with read-only source and a writable
output directory. Before generation run `--version` and require `13.0.1`.
Use JavaScript/Bun project type, `--required-only`, multi-project/workspace
mode for `apps`, and no registry submission or license fetching. Assert a
known root-only development package is absent while the WebUI, extension, and
shared runtime roots remain represented. Move only the produced JSON into the
evidence directory.

- [x] **Step 5: Validate and merge with pinned CycloneDX CLI**

Validate each component before merge, merge exactly the three expected inputs
into `sbom-source-aggregate.cdx.json`, validate the aggregate, then run a
Python sanity check for the three named metadata roots. After the first
successful pinned generation, commit literal per-root minimum component
counts at 90 percent of the observed totals and maximum counts at 150 percent;
later unexpected shrinkage or explosion fails review.

- [x] **Step 6: Scan each source SBOM with pinned Trivy**

Update the database once, record `trivy --version` and database metadata, then scan each component with:

```bash
trivy sbom \
  --scanners vuln \
  --ignore-unfixed=false \
  --format json \
  --output "trivy-source-$component.json" \
  "sbom-$component.cdx.json"
```

Run `exception_policy.py` against the all-severity complete JSON. The policy
evaluator selects Critical/High as blockers while retaining Low, Medium,
Unknown, and unscored findings in the raw report and summary. Upload both raw
and adjusted decisions; never overwrite raw scanner output.

- [x] **Step 7: Add an unconditional result gate**

The final job uses `if: always()`, needs `generate-python`,
`generate-apps-workspace`, `generate-admin-ui`, `merge-source`, and
`scan-source`, checks each named result is `success`, downloads all named
evidence, verifies every required filename and checksum, and exits nonzero on
any missing, skipped, or cancelled producer.

- [ ] **Step 8: Run local pinned-tool and workflow verification**

```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/CI/test_sbom_workflow_contracts.py \
  tldw_Server_API/tests/Supply_Chain/test_exception_policy.py -q
actionlint .github/workflows/sbom.yml
```

Run one local generation/validation smoke for each source root. Expected: four valid SBOMs, three complete Trivy reports, three passing decisions, and no changed lock.

- [x] **Step 9: Commit the source supply-chain gate**

```bash
git add .github/workflows/sbom.yml \
  tldw_Server_API/tests/CI/test_sbom_workflow_contracts.py \
  tldw_Server_API/tests/Supply_Chain/test_exception_policy.py
git commit -m "ci: make source SBOM admission fail closed (TASK-13013.7)"
```

---

### Task 4A: Remediate Source Vulnerabilities Exposed by the Gate

**Files:**
- Modify: `uv.lock`
- Modify: `apps/bun.lock`
- Modify: `admin-ui/bun.lock`
- Modify only if existing ranges cannot select a fix: `pyproject.toml`,
  `apps/package.json`, workspace manifests, and `admin-ui/package.json`
- Modify only after explicit approval: `.github/supply-chain/vulnerability-exceptions.json`

**Interfaces:**
- Preserves the universal Python and two-root Bun lock ownership contracts.
- Produces three source scan decisions with no unexcepted Critical or High finding.
- Requires a repository approval URL before any narrowly scoped, expiring exception.

- [x] **Step 1: Map every blocker to its direct dependency owners**

Use the complete Task 4 Trivy JSON plus `uv tree --invert` and `bun pm why`.
Separate packages with an available fixed version from findings for which the
current vulnerability database identifies no upstream fix.

- [x] **Step 2: Refresh fixable packages within existing constraints**

Use Bun 1.3.2 at both lock roots and uv 0.12.7 at the universal Python root.
Do not widen a manifest or add an override until an unchanged-range refresh is
proven insufficient.

- [x] **Step 3: Regenerate SBOMs and rescan with the pinned Task 4 tools**

Require unchanged lock hashes during generation. Record the new component
counts and exact residual Critical/High finding identities.

- [x] **Step 4: Apply the smallest necessary direct constraint or override**

For a patched transitive dependency that remains trapped below its safe
version, prefer upgrading its direct owner. Use a root override only when the
patched release satisfies the owner contract and focused regressions prove
compatibility.

- [x] **Step 5: Run affected Python, WebUI, extension, shared UI, and Admin gates**

Run the frozen-install, typecheck, unit, build, and critical journey subsets
owned by each changed dependency graph. Rerun the full source gate afterward.

- [ ] **Step 6: Handle genuinely unfixed findings without weakening policy**

Remove or replace the dependency where practical. Otherwise pause for an
explicit human-approved, exact package/component/version exception with the
required repository issue or PR URL, owner, mitigation, and expiry.

Evidence recorded 2026-09-04 with uv 0.12.7, cdxgen 13.0.1, and Trivy
0.74.0 against a fresh database:

- applications workspace: 995 components, 11 total findings, zero
  Critical/High blockers;
- Admin UI: 325 components, one total finding, zero Critical/High blockers;
- Python root: 307 components with seven residual blockers (two Critical,
  five High): four unfixed ChromaDB findings, unfixed python-ecdsa and NLTK
  findings, and one Darwin-only transformers fork while every production
  Linux profile resolves the fixed 5.16.1 release;
- issue https://github.com/rmusser01/tldw_server/issues/2866 contains the
  exact-match, short-lived exception proposal and awaits explicit repository
  owner approval; the canonical policy remains empty until approval;
- frontend typecheck, lint, focused documentation test, and production build
  passed; Chrome/Firefox/Edge extension builds passed; Admin lint, typecheck,
  production build, and dependency-sensitive tests passed on CI-compatible
  Node 20;
- the full Admin suite completed with 766/807 tests passing and 41 inherited
  source/test expectation failures outside this task's source diff; the first
  Node 26 run was discarded because repository CI is pinned to Node 20.

- [x] **Step 7: Review, record evidence, and commit remediation**

Keep source admission fail closed and record both remediated and residual
findings in TASK-13013.7 before committing.

---

### Task 5: Pin the Complete Production Image and Reference Runtime Set

**Files:**
- Create: `.github/supply-chain/reference-images.json`
- Create: `tldw_Server_API/tests/Supply_Chain/test_image_pinning_contracts.py`
- Modify: `Dockerfiles/Dockerfile.prod`
- Modify: `Dockerfiles/Dockerfile.worker`
- Modify: `Dockerfiles/Dockerfile.audio_gpu_worker`
- Modify: `Dockerfiles/Dockerfile.webui`
- Modify: `Dockerfiles/Dockerfile.admin-ui`
- Modify: `Helper_Scripts/Deployment/production_preflight.py`
- Modify: `Dockerfiles/production.env.example`
- Modify: `Dockerfiles/docker-compose.production.yml`
- Modify: `Dockerfiles/Monitoring/docker-compose.production.yml`
- Modify: `tldw_Server_API/tests/Utils/test_production_preflight.py`
- Modify: `tldw_Server_API/tests/Utils/test_docker_production_reference.py`
- Modify: `tldw_Server_API/tests/Utils/test_production_probe_contract.py`
- Modify: `tldw_Server_API/tests/Utils/test_production_deploy.py`

**Interfaces:**
- Produces: `is_digest_pinned_image(value: str) -> bool` requiring the
  `@sha256:[0-9a-f]{64}` suffix and a readable non-`latest` tag.
- Produces: reference-image records `name`, `reference`, `platform`, `index_digest`, and `platform_manifest_digest`.
- Produces: exact five-image project matrix and six-image reference matrix.

- [x] **Step 1: Write red Dockerfile and reference-image contracts**

Parse every `FROM` after stripping an optional `--platform` segment and stage alias. Require:

```python
PRODUCTION_DOCKERFILES = (
    Path("Dockerfiles/Dockerfile.prod"),
    Path("Dockerfiles/Dockerfile.worker"),
    Path("Dockerfiles/Dockerfile.audio_gpu_worker"),
    Path("Dockerfiles/Dockerfile.webui"),
    Path("Dockerfiles/Dockerfile.admin-ui"),
)
IMAGE_REF = re.compile(r"^[^@\s]+:[^@\s]+@sha256:[0-9a-f]{64}$")


@pytest.mark.parametrize("path", PRODUCTION_DOCKERFILES)
def test_every_production_from_is_digest_pinned(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("FROM "):
            reference = line.split()[1]
            assert IMAGE_REF.fullmatch(reference), (path, line)
```

Load `reference-images.json` and require exact names `caddy`, `postgres`, `redis`, `prometheus`, `alertmanager`, `grafana`; platform `linux/amd64`; tag-plus-digest reference; and distinct valid index/platform digests where the registry reports an index.

- [x] **Step 2: Extend production preflight red tests**

Change valid fixtures so `TLDW_APP_IMAGE`, `TLDW_ROLLBACK_IMAGE`, and all six
reference-image fields use tag-plus-digest values. Add
`PROMETHEUS_IMAGE`, `ALERTMANAGER_IMAGE`, and `GRAFANA_IMAGE` as names-only
production environment entries. Parameterize tag-only, digest-only, uppercase
digest, short digest, `latest@digest`, registry-with-port, whitespace, and
mismatched rendered image cases. Expect one stable
`immutable_image_required` code without echoing the rejected value.

- [x] **Step 3: Run focused tests and confirm the red state**

```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Supply_Chain/test_image_pinning_contracts.py \
  tldw_Server_API/tests/Utils/test_production_preflight.py \
  tldw_Server_API/tests/Utils/test_docker_production_reference.py -q
```

Expected: mutable `FROM` lines and tag-only reference fixtures fail.

- [x] **Step 4: Resolve and record project base digests**

For each current readable base tag (`python:3.12-slim`, `python:3.11-slim`, `oven/bun:1.3.2-debian`, `node:20-bookworm-slim`), run:

```bash
docker buildx imagetools inspect "$reference"
```

Record the manifest-list digest directly in every `FROM` while preserving the readable tag. Re-run inspect against `tag@digest` and require the same digest before committing.

- [x] **Step 5: Resolve, inventory, SBOM, and scan six reference images**

Start from the exact supported tags in the current production runbook. For each image:

1. resolve its immutable index digest;
2. resolve the `linux/amd64` child manifest;
3. generate CycloneDX from that platform;
4. scan with pinned Trivy and fresh database;
5. reject unexcepted Critical/High findings;
6. if blocked, move to the smallest supported patched tag and repeat;
7. record the passing literal `tag@digest` plus both digest identities in `reference-images.json`.

Do not add an exception during this task unless the requester has supplied the required owner, rationale, approval reference, and expiry.

- [x] **Step 6: Enforce one digest predicate in deployment preflight**

Replace `_is_immutable_app_image` and `_is_exact_third_party_image` with one strict predicate:

```python
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NAME_TAG = re.compile(r"^[A-Za-z0-9._:/-]+$")


def is_digest_pinned_image(value: str) -> bool:
    name_tag, separator, digest = value.rpartition("@sha256:")
    if separator != "@sha256:" or not _DIGEST.fullmatch(digest):
        return False
    if not _NAME_TAG.fullmatch(name_tag):
        return False
    last_segment = name_tag.rsplit("/", 1)[-1]
    if ":" not in last_segment:
        return False
    tag = last_segment.rsplit(":", 1)[-1]
    return bool(tag) and tag.lower() != "latest"
```

Use this predicate for target, rollback, Caddy, PostgreSQL, Redis,
Prometheus, Alertmanager, and Grafana inputs. The `rpartition` permits a
registry port while still requiring a tag on the final path segment.

- [x] **Step 7: Update Compose messages and reference contracts**

Keep external required variables, but change all messages from “exact version or digest” to “version tag plus @sha256 digest.” Preserve the names-only environment file and TASK-13013.6 topology/secret behavior.

- [ ] **Step 8: Build and scan all five project images**

```bash
docker buildx build --platform linux/amd64 -f Dockerfiles/Dockerfile.prod --load -t tldw-task-13013-7-app .
docker buildx build --platform linux/amd64 -f Dockerfiles/Dockerfile.worker --load -t tldw-task-13013-7-worker .
docker buildx build --platform linux/amd64 -f Dockerfiles/Dockerfile.audio_gpu_worker --load -t tldw-task-13013-7-audio .
docker buildx build --platform linux/amd64 -f Dockerfiles/Dockerfile.webui --load -t tldw-task-13013-7-webui .
docker buildx build --platform linux/amd64 -f Dockerfiles/Dockerfile.admin-ui --load -t tldw-task-13013-7-admin .
```

Generate an image SBOM and complete Trivy JSON for each. The task remains red until the exception evaluator returns no blockers and no unmatched exception IDs for all five.

Local `linux/amd64` builds and runtime smokes passed for app, worker,
audio-worker, and Admin UI. The WebUI exhausted the three-attempt local limit
after reaching all 153 static pages and then exceeding the 8 GB Docker Desktop
VM during finalization; the pull-request container matrix remains responsible
for its clean-run build. Validated CycloneDX SBOMs and complete Trivy reports
were produced for the four local images and all six reference images. The
immutable set still has upstream/locked Critical and High findings, so the
zero-high admission policy remains correctly blocked pending exact human-
approved exceptions or upstream remediation.

- [x] **Step 9: Run focused tests and Bandit**

```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Supply_Chain/test_image_pinning_contracts.py \
  tldw_Server_API/tests/Utils/test_production_preflight.py \
  tldw_Server_API/tests/Utils/test_docker_production_reference.py -q
python -m bandit -r Helper_Scripts/Deployment/production_preflight.py -f json -o /tmp/bandit_task_13013_7_images.json
```

Evidence recorded 2026-09-04: 418 focused Python deployment/image contracts
passed, five focused frontend contracts passed, Bandit reported zero findings,
and every one of the ten locally generated image SBOMs validated with the
digest-pinned CycloneDX CLI.

- [x] **Step 10: Commit immutable image identity**

```bash
git add .github/supply-chain/reference-images.json \
  Dockerfiles/Dockerfile.prod Dockerfiles/Dockerfile.worker \
  Dockerfiles/Dockerfile.audio_gpu_worker Dockerfiles/Dockerfile.webui \
  Dockerfiles/Dockerfile.admin-ui Dockerfiles/docker-compose.production.yml \
  Dockerfiles/Monitoring/docker-compose.production.yml \
  Dockerfiles/production.env.example \
  Helper_Scripts/Deployment/production_preflight.py \
  tldw_Server_API/tests/Supply_Chain/test_image_pinning_contracts.py \
  tldw_Server_API/tests/Utils/test_production_preflight.py \
  tldw_Server_API/tests/Utils/test_docker_production_reference.py
git commit -m "build: pin the production image set by digest (TASK-13013.7)"
```

Committed as `23189edcf1` after the focused image/deployment and security gates.

---

### Task 6: Build the Release Evidence Manifest and Digest Cross-Checks

**Files:**
- Create: `.github/supply-chain/release-evidence.schema.json`
- Create: `Helper_Scripts/Supply_Chain/release_evidence.py`
- Create: `tldw_Server_API/tests/Supply_Chain/test_release_evidence.py`

**Interfaces:**
- Produces: `EvidenceError(ValueError)` with bounded component/file context.
- Produces: `ImageEvidence(name: str, ownership: str, platform: str, subject_digest: str, platform_manifest_digest: str, reference: str, dockerfile: str | None, publication: str, sbom_file: str, scan_file: str, decision_file: str, provenance_ref: str | None)`.
- Produces: `ReleaseManifest(schema_version: int, repository: str, source_commit: str, release_tag: str, workflow_run: str, platform: str, policy_sha256: str, scanner: Mapping[str, str], project_images: tuple[ImageEvidence, ...], reference_images: tuple[ImageEvidence, ...], files: Mapping[str, str], decision: str)`.
- Produces: `load_image_evidence(path: Path) -> ImageEvidence`.
- Produces: `build_release_manifest(evidence_dir: Path, metadata: Mapping[str, str]) -> ReleaseManifest`.
- Produces: `verify_release_manifest(manifest: ReleaseManifest, evidence_dir: Path) -> None`.
- Produces CLI commands `assemble` and `verify`.
- Raw image records additionally carry `subject_media_type`,
  `scan_subject_digest`, and `scan_platform_manifest_digest`; these fields
  prove whether subject/child equality is valid and bind scan orchestration to
  the inspected OCI identities without claiming they are Trivy-native fields.

- [x] **Step 1: Write the versioned JSON schema**

Require exact root keys, `schema_version=1`, `platform=linux/amd64`, five unique project image names, six unique reference image names, SHA-256 file hashes, `pass` decision, and ownership values `project-built` or `third-party-reference`. Require `publication` values `promoted` or `build-and-scan-only`.

- [x] **Step 2: Write red manifest completeness tests**

Build temporary evidence files and assert:

```python
def test_release_manifest_requires_exact_image_sets(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    (tmp_path / "image-admin-ui.json").unlink()

    with pytest.raises(EvidenceError, match="admin-ui"):
        build_release_manifest(tmp_path, _metadata())
```

Add failures for duplicate name, wrong platform, missing file, incorrect file checksum, malformed digest, subject/scan mismatch, stale scanner DB, frontend publication set to `promoted`, third-party provenance reference, and a non-pass decision.

- [x] **Step 3: Write red OCI index/child identity tests**

Require each component record to carry both `subject_digest` and `platform_manifest_digest`. Allow equality only when registry inspection proves the subject is already a single-platform manifest. Assert scan metadata targets the subject and resolves the recorded child.

- [x] **Step 4: Run tests and confirm the red state**

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/tests/Supply_Chain/test_release_evidence.py -q
```

Expected: collection fails because `release_evidence.py` does not exist.

- [x] **Step 5: Implement strict record and hash validation**

Use only `argparse`, `dataclasses`, `datetime`, `hashlib`, `json`, `pathlib`, and `re`. Open evidence below one explicit root, reject symlinks and traversal, cap input file sizes, compute hashes by streaming, sort all output arrays, and serialize with stable indentation/key ordering.

- [x] **Step 6: Implement assemble and verify commands**

```bash
python Helper_Scripts/Supply_Chain/release_evidence.py assemble \
  --evidence-dir .artifacts/release-evidence \
  --metadata .artifacts/release-metadata.json \
  --output .artifacts/release-manifest.json
python Helper_Scripts/Supply_Chain/release_evidence.py verify \
  --manifest .artifacts/release-manifest.json \
  --evidence-dir .artifacts/release-evidence
```

`assemble` fails unless all 11 images and their raw/adjusted/SBOM files exist. `verify` recomputes every checksum and relationship independently.

- [x] **Step 7: Run tests and Bandit**

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/tests/Supply_Chain/test_release_evidence.py -q
python -m bandit -r Helper_Scripts/Supply_Chain/release_evidence.py -f json -o /tmp/bandit_task_13013_7_evidence.json
```

Evidence recorded 2026-09-04: the initial test collection failed because the
module did not exist; the final suite passes 22/22 and the combined exception
policy/evidence regression passes 72/72. Ruff, JSON parsing, `git diff
--check`, and Bandit pass; Bandit reports zero findings across 657 lines.

- [x] **Step 8: Commit the evidence core**

```bash
git add .github/supply-chain/release-evidence.schema.json \
  Helper_Scripts/Supply_Chain/release_evidence.py \
  tldw_Server_API/tests/Supply_Chain/test_release_evidence.py
git commit -m "feat: validate digest-bound release evidence (TASK-13013.7)"
```

---

### Task 7: Stage Candidate Builds, Scans, Promotion, and Attestations

**Files:**
- Rewrite: `.github/workflows/publish-docker.yml`
- Modify: `.github/workflows/publish-ghcr-main.yml`
- Modify: `.github/workflows/container-build-check.yml`
- Modify: `.github/workflows/publish-pypi.yml`
- Modify: `.github/workflows/pypi-package.yml`
- Modify: `Makefile`
- Rewrite: `tldw_Server_API/tests/CI/test_release_workflow_contracts.py`
- Modify: `tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py`

**Interfaces:**
- Consumes reusable `sbom.yml` source gate.
- Produces backend candidate records for app, worker, and audio-worker.
- Produces local OCI records for WebUI and Admin UI with `build-and-scan-only`.
- Produces six third-party reference records.
- Produces full-version then floating tag promotion for the exact three backend subject digests.
- Produces maximum BuildKit provenance, OCI SBOM, GitHub build attestation, release assets, and final draft-release publication.

- [x] **Step 1: Rewrite red release workflow contracts around admission order**

Assert `publish-docker.yml`:

- has only trusted `workflow_dispatch` admission with `release_tag` and exact confirmation input;
- rejects the old `release: published` trigger;
- verifies an existing draft GitHub release and protected tag/commit;
- invokes or depends on the source SBOM gate;
- names five project candidates and six reference images;
- publishes only app/worker/audio-worker;
- sets `platforms: linux/amd64`, `provenance: mode=max`, and `sbom: true`;
- scans before any metadata/tag promotion step;
- verifies full-version tags before floating tags;
- uploads and verifies release evidence before changing draft state;
- has no Docker Hub credentials.

- [x] **Step 2: Add red main-publish, container-check, and PyPI contracts**

Require `publish-ghcr-main.yml` to build a run-unique/commit candidate, scan its exact digest, then move `main` and `sha-*` aliases. Require `container-build-check.yml` to cover all five production Dockerfiles and scan local OCI outputs without package-write permissions. Require both PyPI publish jobs to set `attestations: true` and consume the exact checked distribution artifact.

- [x] **Step 3: Run workflow contracts and confirm the red state**

```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/CI/test_release_workflow_contracts.py \
  tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py -q
```

Expected: old post-publication, pre-scan tagging, three-image build-only, and implicit PyPI attestation assertions fail.

- [x] **Step 4: Build three backend candidates without formal tags**

Use a matrix with explicit name, Dockerfile, GHCR suffix, and safe build
arguments. Push only
`candidate-${{ github.run_id }}-${{ github.run_attempt }}`, capture Buildx
`digest`, inspect the subject/index and `linux/amd64` child, enable maximum
provenance and OCI SBOM, create GitHub provenance for the subject digest, and
upload one signed/checksummed component record.

- [x] **Step 5: Build two frontend OCI artifacts without registry publication**

Build WebUI and Admin UI once with `outputs: type=oci` into runner-local archives. Capture their OCI subject and child digests, generate/retain provenance and SBOM evidence, scan with Trivy `--input`, and upload records marked `build-and-scan-only`. Do not run Docker login in these jobs and do not create a GHCR tag.

- [x] **Step 6: Scan the backend and reference matrices fail closed**

For backend candidates, pull and scan `name@subject_digest` with explicit `--platform linux/amd64`. For third-party records, use the literal `tag@digest` from `reference-images.json`. For each matrix item upload:

- raw CycloneDX image SBOM;
- raw complete Trivy JSON;
- policy-adjusted decision;
- subject/index and platform manifest digests;
- scanner version/database metadata;
- file checksums.

Each job fails on scanner/database/tool/exception error or any blocker.

- [x] **Step 7: Gate and promote immutable full-version aliases**

The promotion job needs all source, owned-image, and reference-image jobs and
has `if` conditions that cannot run after skipped, cancelled, or failed
producers. For each backend image run:

```bash
docker buildx imagetools create \
  --tag "${image_name}:${release_version}" \
  "${candidate_name}@${subject_digest}"
```

Inspect the resulting full-version tag and require exact subject digest
equality.

- [x] **Step 8: Promote floating aliases only after the full set verifies**

After all three full-version aliases match, apply the repository's major/minor and `latest` aliases to those same digests. Verify every alias again. Never rebuild or select by candidate tag during promotion.

- [x] **Step 9: Assemble assets and publish the draft release last**

Download all source and image evidence, run `release_evidence.py assemble` and `verify`, generate SHA-256 checksums, upload every file to the draft release, query the release asset list to prove completeness, then PATCH `draft=false`. If publication fails, leave the release draft and floating tags pointing only to already verified digests.

- [x] **Step 10: Harden main publication with the same exact-digest rule**

Keep the backend-only main publication boundary. Build/push one unique app candidate, scan and evaluate it, then move `main` and `sha-*` aliases. Upload Actions evidence; do not attach it to a formal GitHub release.

- [x] **Step 11: Make PyPI attestation and locked artifact reuse explicit**

Make the PyPI build and both publish jobs depend on the reusable source
supply-chain gate for the same commit. Use the uv release group for
build/check tooling. Hash distributions after `make pypi-check`, upload with
`if-no-files-found: error`, download into publish jobs, verify hashes, and set:

```bash
uv sync --locked --no-dev --no-editable --group release
```

```yaml
with:
  attestations: true
```

Keep OIDC trusted publishing and do not add a second attestor.

- [x] **Step 12: Run workflow tests and actionlint**

```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/CI/test_sbom_workflow_contracts.py \
  tldw_Server_API/tests/CI/test_release_workflow_contracts.py \
  tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py \
  tldw_Server_API/tests/Supply_Chain/test_release_evidence.py -q
actionlint .github/workflows/sbom.yml \
  .github/workflows/container-build-check.yml \
  .github/workflows/publish-ghcr-main.yml \
  .github/workflows/publish-docker.yml \
  .github/workflows/publish-pypi.yml \
  .github/workflows/pypi-package.yml
```

Evidence recorded 2026-09-04: the initial release/PyPI contracts failed 20 of
26 assertions against the legacy workflows. A review-added signed-evidence
contract then failed before the exact attestation URLs and bundles were wired
into all five project-built release records. The final combined matrix passes
61/61 tests; Ruff, actionlint v1.7.12 across all six workflows, and `git diff
--check` pass. An isolated Python 3.12 environment also built one wheel and one
sdist with the exact locked release group and `--no-isolation`; Twine, package
content checks, and the generated SHA-256 checksum verification all passed.

- [x] **Step 13: Commit release admission and attestations**

```bash
git add .github/workflows/sbom.yml \
  .github/workflows/container-build-check.yml \
  .github/workflows/publish-ghcr-main.yml \
  .github/workflows/publish-docker.yml \
  .github/workflows/publish-pypi.yml \
  .github/workflows/pypi-package.yml \
  Makefile \
  tldw_Server_API/tests/CI/test_release_workflow_contracts.py \
  tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py \
  Docs/superpowers/plans/2026-08-30-software-supply-chain-release.md \
  'backlog/tasks/task-13013.7 - Close-dependency-and-software-supply-chain-release-gaps.md'
git commit -m "ci: scan exact candidates before release promotion (TASK-13013.7)"
```

---

### Task 8: Publish the Operator Contract and Complete Release Certification

**Files:**
- Create: `Docs/Development/Software_Supply_Chain.md`
- Modify: `Docs/Development/Release_Process.md`
- Modify: `Docs/Development/PyPI_Publishing.md`
- Modify: `Docs/Deployment/Production_Reference_Deployment.md`
- Modify: `Dockerfiles/README.md`
- Modify: `Docs/mkdocs.yml` (disable git-date multiprocessing after reproducible host semaphore exhaustion)
- Modify: `Helper_Scripts/release.py` and `tldw_Server_API/tests/Utils/test_release_helper.py` (draft creation and offline root-version lock refresh before commit)
- Modify: `.github/workflows/publish-docker.yml` and `tldw_Server_API/tests/CI/test_release_workflow_contracts.py` (reject version-tag replacement and ambiguous registry lookup failures)
- Modify: `tldw_Server_API/tests/Supply_Chain/test_dependency_lock_contracts.py` (follow the renamed build-and-scan job)
- Modify: `tldw_Server_API/tests/Docs/test_release_docs_contract.py`
- Modify: `tldw_Server_API/tests/Docs/test_production_reference_deployment_docs.py`
- Modify: `backlog/tasks/task-13013.7 - Close-dependency-and-software-supply-chain-release-gaps.md` through Backlog MCP

**Interfaces:**
- Produces: one public runbook for lock updates, SBOM generation, scan policy, exception review, digest refresh, release evidence, image attestation verification, and PyPI attestation verification.
- Produces: final TASK-13013.7 acceptance/verification/PR evidence.

- [x] **Step 1: Write red documentation contracts**

Assert the canonical runbook contains:

- exact four tool versions and immutable references;
- `uv lock --check` and frozen Bun install commands;
- all four source SBOM names and 11 image classes;
- zero-unexcepted Critical/High and `ignore-unfixed=false`;
- 7-day Critical and 30-day High exception limits;
- owner/rationale/mitigation/approval/creation/expiry fields;
- `tag@sha256` and subject/index versus child-manifest explanation;
- `linux/amd64` limitation;
- build-and-scan-only frontend boundary;
- GitHub `gh attestation verify` and PyPI attestation verification instructions;
- no claim of third-party provenance.

- [x] **Step 2: Run docs tests and confirm the red state**

```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Docs/test_release_docs_contract.py \
  tldw_Server_API/tests/Docs/test_production_reference_deployment_docs.py -q
```

Expected: supply-chain runbook and tag-plus-digest/attestation assertions fail.

- [x] **Step 3: Write the supply-chain and release runbooks**

Document exact operator commands, expected evidence filenames, failure recovery, digest refresh, scanner DB freshness, exception lifecycle, draft-release admission, promotion ordering, and verification. State explicitly that a past clean scan is time-bound evidence rather than a future-safety guarantee.

- [x] **Step 4: Update production reference and PyPI documentation**

Replace tag-only examples with the committed candidate literals from `reference-images.json`, explicitly retaining the pending-certification warning while their scans block. Explain that operators may refresh them only through the same SBOM/scan gate. Document `attestations: true`, trusted publishing, distribution hash verification, and how to inspect PyPI's PEP 740 material.

- [x] **Step 5: Refresh generated documentation and run docs tests**

```bash
bash Helper_Scripts/refresh_docs_published.sh
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Docs/test_release_docs_contract.py \
  tldw_Server_API/tests/Docs/test_production_reference_deployment_docs.py -q
```

- [x] **Step 6: Run the full focused verification matrix**

```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Supply_Chain \
  tldw_Server_API/tests/CI/test_sbom_workflow_contracts.py \
  tldw_Server_API/tests/CI/test_release_workflow_contracts.py \
  tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py \
  tldw_Server_API/tests/Utils/test_production_preflight.py \
  tldw_Server_API/tests/Utils/test_docker_production_reference.py \
  tldw_Server_API/tests/Docs/test_release_docs_contract.py \
  tldw_Server_API/tests/Docs/test_production_reference_deployment_docs.py -q
python -m bandit -r \
  Helper_Scripts/Supply_Chain \
  Helper_Scripts/Deployment/production_preflight.py \
  -f json -o /tmp/bandit_task_13013_7_final.json
git diff --check
```

- [ ] **Step 7: Re-run locks, framework suites, builds, SBOMs, and scans**

Repeat Tasks 2–5 frozen lock, uv, WebUI, Admin UI, five-image build, four-source-SBOM, five-owned-image, and six-reference-image commands. Record exact passes, skips, scanner/database metadata, and any environment-limited live gate honestly in Backlog notes.

2026-09-04 checkpoint: fresh uv 0.12.7 lock check and both Bun 1.3.2 frozen installs pass with all three lock hashes unchanged. The focused matrix passes 543 tests, including strict MkDocs; Bandit has zero findings, scoped Ruff and actionlint pass, and public/private boundary plus detect-private-key checks pass. A live isolated release-version bump changes only the root package version in the 672-package lock. Independent review and re-review accept the draft/lock/promotion fixes with no remaining Important or Critical findings. Full live certification remains open: the documented source/image vulnerability blockers have no approved exceptions, and the WebUI image requires CI after the three-attempt local OOM limit. Do not interpret the documentation or these local tests as passing release admission.

- [x] **Step 8: Scan the final diff for secrets and scope leakage**

Run repository secret tooling plus:

```bash
rg -n -i \
  'BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|password\s*[:=]\s*[^${]|api[_-]?key\s*[:=]\s*[^${]' \
  .github/supply-chain Helper_Scripts/Supply_Chain Docs/Development Dockerfiles
rg -n -i 'non-public repository URL|downstream infrastructure identifier' \
  .github/supply-chain Docs/Development Docs/Deployment
```

Review every match; the final diff must contain no credential or non-public environment data.

- [x] **Step 9: Update Backlog through MCP and commit documentation**

Record the implementation plan, touched files, verification commands/results, Bandit evidence, known limitations, and documentation paths through `backlog.task_edit`. Then:

```bash
git add Docs/Development/Software_Supply_Chain.md \
  Docs/Development/Release_Process.md \
  Docs/Development/PyPI_Publishing.md \
  Docs/Deployment/Production_Reference_Deployment.md \
  Docs/Published Dockerfiles/README.md \
  tldw_Server_API/tests/Docs/test_release_docs_contract.py \
  tldw_Server_API/tests/Docs/test_production_reference_deployment_docs.py \
  'backlog/tasks/task-13013.7 - Close-dependency-and-software-supply-chain-release-gaps.md'
git commit -m "docs: publish supply-chain release verification (TASK-13013.7)"
```

- [x] **Step 10: Request code review and remediate findings**

Use the requesting-code-review skill. Treat review feedback through the receiving-code-review skill, verify each finding against source behavior, add a red regression for every accepted defect, commit focused fixes, and rerun the affected plus final focused matrix.

- [x] **Step 11: Rebase on current `origin/dev` and verify again**

```bash
git fetch origin dev
git rebase origin/dev
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/tests/Supply_Chain tldw_Server_API/tests/CI/test_sbom_workflow_contracts.py tldw_Server_API/tests/CI/test_release_workflow_contracts.py -q
git diff --check origin/dev...HEAD
```

Resolve only this task's conflicts. Do not overwrite concurrent work.

2026-09-04 rebase checkpoint: rebased all 36 commits onto `c5dfe0ff73d17e177380551c946c109008d0c2cd`, preserving every upstream monitoring-rotation test in the sole textual conflict. A recovery branch retains the pre-rebase checkpoint. Added the newly upstreamed public `packages/tldw_profile_core` to all three locked Python builder contexts, without changing that package. An isolated wheel contains its modules, schemas, and fixtures. Fixed an existing preflight test's reproducible logging-capture dependency by owning and removing only its own sink. The same failing-order seed now passes all 549 focused tests; independent re-review reports no Important/Critical defects. Upstream also raised the Python minimum to 3.11 and added `rfc8785==0.1.4`: regenerated the universal lock with uv 0.12.7 while retaining the prior Docling versions, adding only rfc8785 and dropping obsolete package variants. The resulting 654-package lock passes offline checks; both Bun locks remain unchanged; all 106 supply-chain tests pass after regeneration. The regenerated 297-component Python SBOM still reports 2 Critical and 5 High findings using the pinned scanner and the database updated 2026-09-04T13:08:55Z. Certification remains blocked; these findings are not approved exceptions.

- [x] **Step 12: Open the PR with the requester-authored Change summary**

Before opening or merging, obtain a human-written `Change summary` explaining what changed and why these lock, scan, digest, promotion, and exception choices were made. Add the task/spec/plan, verification, evidence, platform, frontend-publication boundary, and known limitations to the PR body.

2026-09-04: requester supplied the Change summary and authorized publication against `dev`. Opened [PR #2869](https://github.com/rmusser01/tldw_server/pull/2869) with that summary verbatim, technical scope, verification, and explicit remaining blockers. Publication uses sanitized commit `843dc9e8cc`; original development history is retained locally. CI/review and live release certification remain pending.

- [ ] **Step 13: Address automated and human review, then merge only when green**

2026-09-04 Qodo follow-up: all five findings have local corrections. Evidence
verification now invokes `gh` for retained bundles and exact OCI subject bytes,
with canonical image-role, repository, source/signer digest, tag, and workflow
constraints. An independently supplied trusted root enables disconnected
verification; a URL alone is no longer admission evidence. Independent review
and re-review found and closed a cross-role substitution gap. Final combined
validation reports 601 passed and one existing shard-coverage failure; the
578-test classification audit, scoped Ruff, actionlint, Bandit (zero production
findings), and diff checks pass. The requester approved CI remediation on
2026-09-04 for packaging, scanner mount permissions, frontend failures, and the
missing shard assignment. Implementation and regression verification are in progress.
Do not commit, publish these corrections, or merge until required validation is
satisfied. Existing vulnerability blockers and the empty exception policy remain.

2026-09-04 approved CI setup checkpoint: release-only packaging, scanner UID/GID,
and all five Character integration shard matrices are corrected. The touched
Python suite passes 583 tests, with exactly one accepted marker per test.
Actual isolated wheel/sdist builds, Twine and artifact-content checks, pinned
cdxgen generation, Trivy cache download, scoped Ruff, actionlint, lock freshness,
and Bandit pass. Independent CI review found no issues. The wider CI-contract
suite reports 344 passed and six failures: four workflow-admission regressions
introduced by the PR, a routing inventory mismatch partly inherited from dev,
and a schema text expectation already failing on dev. Preserve and restore the
existing workflow gate; do not bypass it. The WebUI webpack bundle-budget
decision remains pending, with no threshold change made. No commit or push at
this checkpoint; full CI and release admission remain open.

Frontend follow-up is verified: both full Characters harness entry points pass
110 tests each under concurrent execution, Watchlists help passes 13 tests, and
the page-object guard passes 10 tests. Independent review found no Critical or
Important issue in the three test-only corrections.

### Approved follow-up: workflow admission and unchanged WebUI budgets

2026-09-04: requester approved this follow-up. Keep all existing vulnerability
and bundle-size limits. Do not change the two unrelated failures present on dev.

1. **Restore workflow admission (Complete locally).** Update the existing graph
   contracts for the three source producers, merge, scan, and result gate; run
   `python -m pytest -q tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`
   and the PyPI lock routing regression to observe failures. Restore the existing
   admission job/condition and immutable checkout refs in `sbom.yml`, retain
   caller-scoped concurrency, and permit only the required read scopes in the
   three reusable publishing callers. Add `uv.lock` to the PyPI route inventory.
   Rerun the admission, SBOM, PyPI, and release contracts plus actionlint.
2. **Investigate WebUI build (Complete; resource decision pending).** Compare the real Node 24/Turbopack
   path with the prior Bun/Turbopack OOM evidence using a bounded isolated
   experiment. Preserve the 600/900 KB bundle budgets and all build checks.
   Change tracked build configuration only after evidence supports the fix.
3. **Review and publish (Reviewed; publication blocked).** Independently review the new diff,
   rerun touched tests, security checks, and lock freshness, and classify any
   wider-suite failures against dev. Commit and push only validated corrections;
   then reply to the five review findings and recheck exact-head CI. Merge only
   when all required review, CI, and vulnerability gates are satisfied.

Gate repair results: 70 focused tests pass after the red regressions; independent
review found no issues. The wider matrix reports 840 passed and only the two
known dev failures. Ruff, actionlint, lock freshness, Bandit, and diff checks pass.
The bounded real Node 24/Turbopack experiment failed on the 7.65 GiB Docker VM
with a memory allocation error after 48.35 seconds. No tracked build changes or
budget increases were made, and no further retry was attempted. The same-head
Turbopack command meets the unchanged budgets on the 15.61 GiB CI runner. Choosing
a larger build host or expanding into shared-bundle optimization requires the
requester's direction. No commit, push, or merge at this checkpoint.

2026-09-04 approved final CI repairs: restored the canonical Node 24/Turbopack
Docker production command and documented a 16 GiB-class build host, keeping the
600/900 KB budgets unchanged. Synchronized the Watchlists admission inventory
with the existing workflow filters and corrected the Postgres test to assert
the existing schema-qualified tables; no trigger policy or SQL changed. All
seven red regression cases now pass, and the full focused matrix passes 846
tests. Independent review found no issues. Actionlint, pinned offline lock
freshness, diff checks, and Bandit pass (zero production-scope findings).
Touched Python lint passes apart from the confirmed unchanged broad-exception
warning in the Watchlists fixture; its import block was formatted. The exact
Docker image still requires CI verification. Latest dev remains a5aa0c8e6751.
Commit and publish the reviewed fixes, then respond to the five Qodo threads.
Vulnerability admission remains blocked; no exception or bypass was approved.

2026-09-05 integration checkpoint: rebased four commits cleanly onto dev
3bc8c6a98ccd; range-diff preserves each patch. Added the missing Supply_Chain
directory to all five existing tooling shards and their core coverage inventory,
with five real coverage-matcher regressions. Updated the stale frontend Docker
command assertion. The shard guard now reports zero newly uncovered files;
851 focused Python tests and four frontend contract tests pass. The previous
head's exact WebUI image built within budget (563.1/779.4 KB gzip) but failed
vulnerability admission. New SBOM metadata and two real-backend journey failures
are diagnosed separately and await a scoped repair decision; no security policy,
timeouts, or assertions were relaxed. All five original Qodo threads are resolved.

2026-09-05 approved SBOM and journey follow-up: clean manifest-and-lock fixtures
confirmed that pinned cdxgen omits child metadata without child locks or local
node_modules, including in JavaScript mode. Keep the canonical required-only Bun
dependency scan and derive only the three workspace identity records from the
checked-in manifests. The actual workflow now passes a clean-fixture Docker
regression and pinned CycloneDX schema validation. All 852 focused Python tests
pass; workflow syntax, touched lint, and Bandit checks pass. Test-only Bandit
annotations document fixed subprocess invocations and a bounded container tmpfs;
pytest assertions are excluded from that test-file scan. No scanner/admission
policy or vulnerability exception changed. Real-backend journey investigation
continues separately; merge remains blocked on exact-head review and admission.

Journey checkpoint: both real Watchlist tests pass after allowing its tracked
local feed in the isolated critical CI job. The existing localhost suffix
allowlist and default port checks remain active; private-address blocking is
disabled only for that fixture job, not production. Six policy regressions clear
pytest's port bypass and verify fixture access plus rejected unrelated hosts,
literal private/link-local addresses, and a non-default port. This is not an
exact-host or resolved-IP restriction. Pinning the Character Chat model removes
nondeterministic provider selection, but the journey still fails: complete-v2
normalizes the intended custom provider/model, then raises ChatBadRequestError
before sending a request to the mock. Direct mock, adapter, credential, and
perform_chat_api_call probes succeed. Stop further retries at this checkpoint;
the next diagnostic boundary is complete-v2 request/payload validation. No
backend implementation, assertions, or timeouts changed. Independent review
found no material risk in the isolated fixture setup after its wording was
corrected. Bandit reports only the unchanged test-file `--cov` false positive.

2026-09-05 approved PDF shard follow-up: reproduced the upstream PDF test's
missing assignment with the existing real coverage matcher and a strengthened
exact shard expectation. Added that test to `media-core-documents` in all five
full-suite matrices; no application code or PDF assertions changed. All 858
focused tests pass, including pinned Docker SBOM generation/schema validation.
Actionlint and touched Ruff pass; Bandit reports only the unchanged `--cov`
test-string false positive. Independent review found no issues and confirmed
exactly one primary media-shard assignment on each platform. Rebased cleanly
onto dev `2742468a19fd`, preserving all eight patches. The post-rebase combined
matrix, including all ten upstream PDF integration tests, passes 868 tests;
workflow syntax, touched lint, offline lock freshness and diff checks pass.
Other source-SBOM/E2E and vulnerability admission failures still block merge.

2026-09-05 approved Source SBOM condition repair — Complete locally: latest run
33972378502 again generated all three inventories successfully but skipped merge
and scan after the intentionally skipped direct-PR admission ancestor. Add
explicit cancellation-aware conditions requiring every direct prerequisite to
succeed. First reproduce with conjunction-guard regressions for successful,
failed, skipped and cancelled prerequisites, then repair both jobs and run the
focused matrix, workflow lint, Bandit and independent review. Admission rules,
source-result checks and vulnerability policy remain unchanged. Character Chat
diagnostics remain paused.

Condition repair verification checkpoint: both RED cases reproduced, all 16
new guards pass, and the final SBOM/admission contract suite passes 49 tests.
Updated only the two intermediate-job expectations in the existing admission
contract; root admission and immutable-checkout assertions are unchanged.
Independent re-review has no remaining findings. Ruff, actionlint, offline
pinned uv lock freshness and diff checks pass. Bandit reports test assertions
and seven non-assertion findings on verified unchanged baseline lines, with no
new security findings.

Earlier commit/push pause: the full matrix at seed 1189791922 reported 883 passed
and one failure in the untouched schema warm-up guard. Its module passes all
three tests in isolation, but warm-up spy calls are empty after earlier tests
in both full runs. The likely boundary is a pre-warmed schema memo shared via
the real app/user database; the guard installs its spy after fixture startup.
No backend, PDF or schema-test edits were made. Request approval for a bounded
test-only cache-isolation repair that preserves both cold-path and warm-path
assertions before publishing. Current published HEAD remains `ffe625d078e13`;
`dev` is still `2742468a19fd`. No vulnerability exception is approved.

2026-09-05 approved schema cache-isolation repair — Complete: parameterized the
existing HTTP guard with cold and deliberately pre-warmed starting states. The
pre-warmed case reproduced the exact missing cold-verification failure before
the fix. One call to the existing cache reset after spy installation and before
measurement restores isolation; there is no reset between measured cold and
warm requests. Both assertions and all HTTP checks remain intact, with no
backend changes. All four module cases pass. Runtime-only mutation of the memo
key to disable caching fails both variants at the warm-request assertion,
confirming regression sensitivity. The full focused matrix at the previously
failing seed now passes all 885 tests (26 warnings), including pinned Docker
SBOM generation/validation and PDF integration. Independent review has no
findings; touched Ruff, actionlint, pinned offline lock freshness and diff checks
pass. Bandit has only test assertions and seven verified unchanged baseline
findings. The combined approved CI fixes are ready to commit/push; live final-
head admission and other existing PR blockers remain required before merge.

2026-09-05 approved Admin Webhooks CI repair — Locally verified: backend status now
includes a sanitized `delivery` diagnostics object. The Admin client still
requires exactly the older seven fields, so a valid response is rejected before
the page can show operational status. Keep the existing strict readiness
validator and project out only this documented, object-valued extension; do not
retain unused diagnostics in page state or accept arbitrary response fields.
The new client regression reproduces the rejection before implementation; the
fixed client/page suites pass 58 tests, including malformed extension/readiness
and unexpected-field rejection. ESLint, typecheck and the production build pass.
The unchanged real-backend test reproduced the CI failure before the fix; the
fixed full JWT suite passes 26 tests with one existing auth-project skip. The
focused release matrix passes 885 tests, including pinned Docker SBOM coverage.
Independent review reports no findings. Bandit cannot parse TypeScript and is
not applicable to this TS-only executable change; security validation uses
strict malformed-response regression cases, ESLint and independent review.
No page, backend or E2E assertions are changed. Browser prerequisites were
restored without changing the shared Python environment. The final browser
invocation uses canonical `/private/tmp` paths and CI minimal-startup settings.
Publish after checking latest-dev integration; other PR merge gates remain open.

Latest-dev integration checkpoint: rebasing onto `63358431d7` preserved all 11
commits exactly. The existing inventory regression caught the new upstream EPUB
test missing from the explicit shard lists. Added its exact path to documents
on all five platforms and to the expected set without weakening exhaustive or
disjoint coverage. Independent review is clean. Rebased verification: 898 focused
tests pass (including EPUB and pinned Docker SBOM), 58 client/page tests pass,
and real-backend JWT passes 26 with one expected skip. Ruff and actionlint pass.
Bandit on the changed Python contract reports assertions plus the unchanged
false positive on `--cov` at line 193; no new security finding. The next fetched
dev `dc0b7455f2` changes only unrelated design notes and its task record; include
it unchanged, verify patch preservation, and publish with an exact remote lease.

Wait for all required checks and reviewer comments. Resolve every actionable thread with evidence, rerun affected tests, rebase again if `dev` moved, and merge only after:

- all required checks pass;
- every review thread is resolved;
- the final diff has no unexpected files;
- the requester-authored Change summary is present;
- TASK-13013.7 Backlog acceptance criteria and final summary are current.

After merge, record the PR number, merge commit, final check state, and any deferred lower-severity findings in TASK-13013.7.

---

## Final Acceptance Checklist

- [ ] Both apps lock Next.js 16.3.3 and pass their required regression suites.
- [ ] WebUI Sentry and all named Next companion packages are on the approved exact versions.
- [ ] Dependabot owns uv, both Bun roots, Docker, and GitHub Actions updates.
- [ ] `uv.lock` is committed and every production Python profile syncs locked, non-dev, and non-editable.
- [ ] Python, applications workspace, Admin UI, and aggregate source SBOMs are separately generated and validated.
- [ ] All five project images and six reference images have separate SBOMs and complete Trivy evidence.
- [ ] Every required source/tool/image identity is immutable and human-readable.
- [ ] Every production reference value requires `tag@sha256`.
- [ ] Scanner/database metadata is present and release data is no older than 24 hours.
- [ ] No unexcepted Critical or High production finding remains.
- [ ] The repository exception list is empty or every entry is exact, owned, approved, justified, and within 7/30-day limits.
- [ ] OCI subject/index and selected `linux/amd64` child manifest digests agree across scans, SBOMs, provenance, promotion, and manifest.
- [ ] App, worker, and audio-worker are promoted only after the full gate.
- [ ] WebUI and Admin UI are build-and-scan-only and receive no public release tags.
- [ ] BuildKit maximum provenance, OCI SBOMs, GitHub image attestations, and PyPI PEP 740 attestations are verifiable.
- [ ] Formal release evidence is attached durably with checksums and a verified manifest.
- [ ] Focused tests, regressions, builds, actionlint, Bandit, secret scan, and `git diff --check` pass.
- [ ] The human requester has supplied the required Change summary.
- [ ] Backlog, PR, review, merge, and final evidence are recorded.
