# Dependency And Static Analysis Risk Specialist Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: Dependency and static-analysis risk
- In scope: Bandit summary, dependency manifests, frontend/backend supply-chain risk, noisy-tool triage, scan follow-up recommendations, and dependency/static-analysis-relevant domain findings.
- Out of scope: remediation implementation, package installation, and networked package audits unless coordinator-approved.
- Review mode: local static review only. No dependency installation, networked audit, Docker build, service startup, package-manager audit command, production code edit, index edit, Backlog task edit, staging, or commit was performed.

## Findings Table

| ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUDIT-2026-06-27-DEPS-001 | likely_risk | static_confirmed | medium | high | dependency | Python runtime and release installs lack a committed lockfile or constraints | open | validated |
| AUDIT-2026-06-27-DEPS-002 | likely_risk | static_confirmed | medium | high | dependency | Static-analysis and CI gates bootstrap mutable external tooling | open | validated |
| AUDIT-2026-06-27-DEPS-003 | improvement_opportunity | static_confirmed | low | high | operations | Bandit app baseline mixes production code with in-package tests | open | validated |

## Index Mapping

Use finding IDs like `AUDIT-2026-06-27-DEPS-001`. Set `evidence_tier` from the report section bucket (`confirmed_issue`, `likely_risk`, or `improvement_opportunity`) and `evidence_strength` from the schema allowed values. Set `source_report` to `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/dependency-static-analysis.md`, set `owner_domain` to this report owner, and include `affected_paths`, `recommendation`, `status`, and `validation_status` in each detailed finding.

New specialist finding details for index ingestion:

- `AUDIT-2026-06-27-DEPS-001`
  - `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/dependency-static-analysis.md`
  - `owner_domain`: `Dependency and static-analysis risk`
  - `affected_paths`: `pyproject.toml`, `Dockerfiles/Dockerfile.prod`, `Dockerfiles/Dockerfile.worker`, `Dockerfiles/Dockerfile.audio_gpu_worker`, `.github/actions/setup-python-deps/action.yml`, `.github/workflows/backend-required.yml`, `.github/workflows/security-required.yml`, `.github/workflows/pypi-package.yml`, `.github/workflows/publish-pypi.yml`, `.github/workflows/mcp-unified-rc.yml`, `.github/workflows/mcp-unified-publish.yml`
  - `recommendation`: Add a committed lock or constraints workflow for supported Python install profiles, then make production Docker builds, required backend/security gates, and publish/RC jobs install with locked or constrained resolution. Keep separate runtime and dev/test profiles if needed, and document any intentionally floating optional extras.
  - `status`: `open`
  - `validation_status`: `validated`
- `AUDIT-2026-06-27-DEPS-002`
  - `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/dependency-static-analysis.md`
  - `owner_domain`: `Dependency and static-analysis risk`
  - `affected_paths`: `.github/workflows/actionlint.yml`, `.github/workflows/security-required.yml`, `.github/workflows/pre-commit.yml`, `.github/workflows/sbom.yml`, `.github/actions/setup-python-deps/action.yml`, `.github/workflows/backend-required.yml`, `.github/workflows/frontend-required.yml`
  - `recommendation`: Replace branch-head remote installers with pinned releases and checksums or pinned action SHAs, pin static-analysis tool versions through the same constraints process as application dependencies, and pin third-party actions in required/static-analysis workflows with scheduled update automation.
  - `status`: `open`
  - `validation_status`: `validated`
- `AUDIT-2026-06-27-DEPS-003`
  - `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/dependency-static-analysis.md`
  - `owner_domain`: `Dependency and static-analysis risk`
  - `affected_paths`: `.bandit`, `tldw_Server_API/app/core/MCP_unified/tests`, `tldw_Server_API/app/core/DB_Management/ACP_Audit_DB.py`, `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py`, `tldw_Server_API/app/services/admin_e2e_support_service.py`, `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
  - `recommendation`: Split Bandit execution into a production profile that excludes in-package tests and a separate test-profile scan. Keep the reviewed B608 dynamic-SQL patterns under explicit allowlist or refactor them to helper builders so future Bandit deltas are easier to triage.
  - `status`: `open`
  - `validation_status`: `validated`

Existing normalized findings confirmed or recommended for dependency/static-analysis follow-up:

- Confirmed: `AUDIT-2026-06-27-OPS-001`, `AUDIT-2026-06-27-OPS-002`, `AUDIT-2026-06-27-OPS-003`, `AUDIT-2026-06-27-OPS-004`, `AUDIT-2026-06-27-OPS-005`, `AUDIT-2026-06-27-OPS-006`.
- Confirmed for static-analysis follow-up, but not duplicated here: `AUDIT-2026-06-27-INTEGRATIONS-001`, `AUDIT-2026-06-27-INTEGRATIONS-002`, `AUDIT-2026-06-27-INTEGRATIONS-003`, and `AUDIT-2026-06-27-CHAT-002`.
- No existing normalized finding was refuted by this pass.

## Confirmed Issues

No new specialist-specific confirmed issue was added. The dependency/static-analysis issues with enough current evidence for confirmed status are already represented in `findings-index.json`:

- `AUDIT-2026-06-27-OPS-004`: Confirmed. The SBOM workflow can emit a Python-only SBOM even though the repository ships Bun-managed frontend/admin/client workspaces. This pass rechecked that `apps/bun.lock` and `admin-ui/bun.lock` are the tracked frontend lockfiles, while `.github/workflows/sbom.yml` only looks for `package-lock.json` paths and then skips Node SBOM generation.
- `AUDIT-2026-06-27-OPS-005`: Confirmed. Dependency update automation covers only root `pip` and GitHub Actions. It does not cover Bun workspaces, nested Python packages, or the Go agent module listed in `dependency-manifest-inventory.txt`.
- `AUDIT-2026-06-27-OPS-001`: Confirmed. The published worker images remain dependency/static-analysis relevant because unbuilt worker Dockerfiles can carry dependency and packaging drift past the PR container gate.
- `AUDIT-2026-06-27-OPS-003`: Confirmed. The actionlint workflow only lints selected workflows, which weakens static-analysis coverage for required and release-sensitive GitHub Actions files.

Bandit medium finding triage did not produce a new confirmed security issue. The `B608` SQL records in `ACP_Audit_DB.py` and `ACP_Sessions_DB.py` were source-reviewed: their dynamic `WHERE` clauses are assembled from fixed local condition strings with user values kept in parameter lists, and dynamic `UPDATE` column lists are assembled from fixed allowlists before parameter binding. The `B108` record in `admin_e2e_support_service.py` is an E2E support cleanup path that resolves the configured backup root and only allows deletion under `tempfile.gettempdir()` or `/tmp`. The remaining medium `B108`/`B103` records are test-fixture paths under `tldw_Server_API/app/core/MCP_unified/tests`.

## Likely Risks

### AUDIT-2026-06-27-DEPS-001 - Python runtime and release installs lack a committed lockfile or constraints

- `severity`: `medium`
- `confidence`: `high`
- `category`: `dependency`
- `evidence_tier`: `likely_risk`
- `evidence_strength`: `static_confirmed`
- `status`: `open`
- `validation_status`: `validated`
- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/dependency-static-analysis.md`
- `owner_domain`: `Dependency and static-analysis risk`
- `affected_paths`:
  - `pyproject.toml`
  - `Dockerfiles/Dockerfile.prod`
  - `Dockerfiles/Dockerfile.worker`
  - `Dockerfiles/Dockerfile.audio_gpu_worker`
  - `.github/actions/setup-python-deps/action.yml`
  - `.github/workflows/backend-required.yml`
  - `.github/workflows/security-required.yml`
  - `.github/workflows/pypi-package.yml`
  - `.github/workflows/publish-pypi.yml`
  - `.github/workflows/mcp-unified-rc.yml`
  - `.github/workflows/mcp-unified-publish.yml`
- `evidence`:
  - The tracked lock-like files are `apps/bun.lock`, `admin-ui/bun.lock`, `mock_openai_server/requirements.txt`, and `tools/tldw-agent/go.sum`; no tracked root `uv.lock`, Python constraints file, Poetry lock, Pipfile lock, or backend requirements file was found.
  - `pyproject.toml` carries the backend dependency graph and includes many open lower-bound ranges plus unbounded package names such as `aiohttp`, `aioresponses`, `aiosqlite`, `asyncpg`, `hypothesis`, `prometheus-client`, `pyotp`, and `matplotlib`.
  - `Dockerfiles/Dockerfile.prod` installs the root project with `pip install --prefix=/install .`; `Dockerfiles/Dockerfile.worker` installs `-e ".[multiplayer]"`; `Dockerfiles/Dockerfile.audio_gpu_worker` installs `-e .`. None passes a constraints or lock file.
  - `.github/actions/setup-python-deps/action.yml` supports `uv` but installs the editable project through `uv pip install --system -e ".[...]"` or `pip install -e ".[...]"` without `--locked` or constraints. `.github/workflows/backend-required.yml` includes `uv.lock` in cache paths, but no such file is tracked.
  - Publish, RC, security, and pre-commit workflows install packaging/security/test tools directly from package indexes, often without version pins.
- `impact`: Identical commits can resolve different Python dependency graphs in production images, worker images, required backend/security gates, and release packaging jobs. That weakens reproducibility, complicates incident rollback, and can let a compromised or incompatible newest dependency affect builds before dependency review or SBOM evidence reflects a stable graph.
- `recommendation`: Add a committed lock or constraints workflow for supported Python install profiles, then use locked/constrained installs in production Docker builds, worker Docker builds, required CI gates, and publish/RC jobs. Treat optional heavy extras separately if one universal lock is impractical, but make runtime and CI resolution explicit.

### AUDIT-2026-06-27-DEPS-002 - Static-analysis and CI gates bootstrap mutable external tooling

- `severity`: `medium`
- `confidence`: `high`
- `category`: `dependency`
- `evidence_tier`: `likely_risk`
- `evidence_strength`: `static_confirmed`
- `status`: `open`
- `validation_status`: `validated`
- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/dependency-static-analysis.md`
- `owner_domain`: `Dependency and static-analysis risk`
- `affected_paths`:
  - `.github/workflows/actionlint.yml`
  - `.github/workflows/security-required.yml`
  - `.github/workflows/pre-commit.yml`
  - `.github/workflows/sbom.yml`
  - `.github/actions/setup-python-deps/action.yml`
  - `.github/workflows/backend-required.yml`
  - `.github/workflows/frontend-required.yml`
- `evidence`:
  - `.github/workflows/actionlint.yml` executes `bash <(curl -sSfL https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash)`, so a required workflow-lint gate runs a branch-head remote installer.
  - `.github/workflows/security-required.yml` installs `bandit` and `pyyaml` without version pins immediately before the Bandit gate.
  - `.github/workflows/pre-commit.yml` installs `pre-commit` and `pytest` without version pins.
  - `.github/workflows/sbom.yml` installs `cyclonedx-bom` without a version pin and uses `npx -y @cyclonedx/cyclonedx-npm` for Node SBOM generation when a package-lock path exists.
  - `.github/actions/setup-python-deps/action.yml` uses `astral-sh/setup-uv@v3`; sampled required frontend/backend workflows use moving action tags such as `actions/setup-python@v6`, `actions/setup-node@v6`, and `oven-sh/setup-bun@v2`. By contrast, publish workflows already pin several high-risk publish and Docker actions by SHA, showing an available local pattern.
- `impact`: Static-analysis and supply-chain gates can change behavior between identical commits because their own installer scripts, action tags, and tool versions float. The branch-head actionlint installer is the sharpest case: a compromised upstream branch or unexpected installer change would execute arbitrary code in the PR/push workflow runner before linting. Read-only workflow permissions reduce blast radius, but the gate result and checked-out source are still exposed to mutable tooling.
- `recommendation`: Pin actionlint to a release and checksum or a pinned action SHA; remove branch-head shell execution. Pin Bandit, PyYAML, pre-commit, CycloneDX, and other gate tooling through the Python constraints process from `AUDIT-2026-06-27-DEPS-001`. Pin third-party setup actions in required/static-analysis workflows to immutable SHAs and rely on Dependabot or Renovate for controlled updates.

## Improvement Opportunities

### AUDIT-2026-06-27-DEPS-003 - Bandit app baseline mixes production code with in-package tests

- `severity`: `low`
- `confidence`: `high`
- `category`: `operations`
- `evidence_tier`: `improvement_opportunity`
- `evidence_strength`: `static_confirmed`
- `status`: `open`
- `validation_status`: `validated`
- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/dependency-static-analysis.md`
- `owner_domain`: `Dependency and static-analysis risk`
- `affected_paths`:
  - `.bandit`
  - `tldw_Server_API/app/core/MCP_unified/tests`
  - `tldw_Server_API/app/core/DB_Management/ACP_Audit_DB.py`
  - `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py`
  - `tldw_Server_API/app/services/admin_e2e_support_service.py`
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
- `evidence`:
  - The existing Bandit summary scanned `tldw_Server_API/app` and reported 4,818 results: 4,792 low, 26 medium, and 0 high.
  - Direct review of `/tmp/tldw_repo_audit_bandit_app.json` grouped the medium records into 17 `B108`, 8 `B608`, and 1 `B103` records.
  - Most medium records are test-only paths under `tldw_Server_API/app/core/MCP_unified/tests`, which lives inside the app tree and is also listed in `pyproject.toml` pytest `testpaths`.
  - `.bandit` excludes virtualenvs and `node_modules`, but it does not exclude in-package tests under `tldw_Server_API/app/core/MCP_unified/tests`.
  - The production-looking `B608` records and admin E2E temp-dir record were source-reviewed and not promoted to new security findings in this pass.
- `impact`: Medium baseline counts are dominated by test fixture patterns, which makes the audit baseline noisy and can hide meaningful production deltas. Because security-required currently fails only on high/critical Bandit findings, the medium baseline is mostly informational today, but it still weakens reviewer attention and static-analysis trend quality.
- `recommendation`: Run a production Bandit profile that excludes test directories and a separate test-code Bandit profile with test-appropriate expectations. Keep a short reviewed allowlist for source-reviewed `B608` helper-builder patterns, or refactor dynamic SQL construction into reusable fixed-clause helpers so future static-analysis output points at novel risk.

Additional improvement notes without new IDs:

- `AUDIT-2026-06-27-OPS-004` should be fixed before relying on release SBOM artifacts for frontend or admin dependency visibility.
- `AUDIT-2026-06-27-OPS-005` should be expanded to include a clear policy for whether nested packages are independently maintained, published, or intentionally excluded from automation.
- `AUDIT-2026-06-27-INTEGRATIONS-001`, `AUDIT-2026-06-27-INTEGRATIONS-002`, and `AUDIT-2026-06-27-INTEGRATIONS-003` should get central-HTTP-client tests. They are not dependency findings, but they are static-analysis-relevant because direct HTTP clients bypass central egress/proxy defaults.
- `AUDIT-2026-06-27-CHAT-002` is a good candidate for a targeted logging/static-analysis guard that detects raw query text in info-level logs.

## Coverage And Evidence

### Files Inspected

- Audit context:
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
  - all reports under `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/`
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/security-boundaries.md`
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/reliability-lifecycle.md`
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/api-webui-contracts.md`
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/dependency-manifest-inventory.txt`
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/ci-deployment-operations-release-candidates.txt`
- Dependency manifests and lockfiles:
  - `pyproject.toml`
  - `apps/package.json`
  - `apps/tldw-frontend/package.json`
  - `apps/packages/ui/package.json`
  - `apps/packages/voice-assistant-sdk/package.json`
  - `apps/extension/package.json`
  - `admin-ui/package.json`
  - `apps/bun.lock`
  - `admin-ui/bun.lock`
  - `apps/mcp-unified/pyproject.toml`
  - `mock_openai_server/pyproject.toml`
  - `mock_openai_server/requirements.txt`
  - `sdks/python/pyproject.toml`
  - `tools/backlog-py/pyproject.toml`
  - `tools/tldw-agent/go.mod`
  - `tools/tldw-agent/go.sum`
- CI, dependency, release, and Docker surfaces:
  - `.github/dependabot.yml`
  - `.github/workflows/actionlint.yml`
  - `.github/workflows/backend-required.yml`
  - `.github/workflows/security-required.yml`
  - `.github/workflows/codeql.yml`
  - `.github/workflows/sbom.yml`
  - `.github/workflows/pre-commit.yml`
  - `.github/workflows/frontend-required.yml`
  - `.github/workflows/ci.yml`
  - `.github/workflows/pypi-package.yml`
  - `.github/workflows/publish-pypi.yml`
  - `.github/workflows/mcp-unified-rc.yml`
  - `.github/workflows/mcp-unified-publish.yml`
  - `.github/actions/setup-python-deps/action.yml`
  - `Dockerfiles/Dockerfile.prod`
  - `Dockerfiles/Dockerfile.worker`
  - `Dockerfiles/Dockerfile.audio_gpu_worker`
  - `Dockerfiles/Dockerfile.webui`
  - `Dockerfiles/Dockerfile.admin-ui`
  - `.bandit`
- Bandit medium source-review samples:
  - `/tmp/tldw_repo_audit_bandit_app.json`
  - `tldw_Server_API/app/core/DB_Management/ACP_Audit_DB.py`
  - `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py`
  - `tldw_Server_API/app/services/admin_e2e_support_service.py`
  - representative files under `tldw_Server_API/app/core/MCP_unified/tests`

### Tests Or Scans Run

- No runtime tests, dependency installs, package-manager audit commands, Docker commands, or networked scans were run for this specialist pass.
- Static/local inspection commands included:
  - `git status --short`
  - `find Docs/superpowers/reviews/2026-06-27-repo-audit -maxdepth 3 -type f | sort`
  - `jq -r '.findings[] | [.id,.severity,.confidence,.category,.evidence_tier,.evidence_strength,.status,.validation_status,.owner_domain,.title] | @tsv' Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
  - `git ls-files '*uv.lock' '*poetry.lock' '*Pipfile.lock' '*constraints*.txt' '*requirements*.txt' '*bun.lock' '*go.sum' '*package-lock.json' '*pnpm-lock.yaml' '*yarn.lock' | sort`
  - `find . -maxdepth 4 \( -name 'package-lock.json' -o -name 'bun.lock' -o -name 'bun.lockb' -o -name 'pnpm-lock.yaml' -o -name 'yarn.lock' -o -name 'requirements*.txt' -o -name 'go.sum' -o -name 'poetry.lock' -o -name 'uv.lock' -o -name 'Pipfile.lock' \) -not -path './.git/*' -not -path './**/node_modules/*' | sort`
  - `rg --pcre2 -n "curl .*raw\.githubusercontent|bash <\(curl|wget .*raw\.githubusercontent|npx -y|\bpython -m pip install\b|\bpip install\b|\buv pip install\b|\bbun install\b" .github/workflows .github/actions Dockerfiles Makefile -g '!**/node_modules/**'`
  - `rg -n "uses: [^#]+@[A-Za-z0-9._-]+($|[[:space:]#])" .github/workflows .github/actions`
  - `jq -r '.results[] | select(.issue_severity=="MEDIUM") | [.test_id,.test_name,.issue_confidence,.filename,(.line_number|tostring),.issue_text] | @tsv' /tmp/tldw_repo_audit_bandit_app.json | sort`
  - `jq -r '.results[] | select(.issue_severity=="MEDIUM") | .test_id + " " + .test_name' /tmp/tldw_repo_audit_bandit_app.json | sort | uniq -c`
  - targeted `sed`, `nl -ba`, and `rg` reads over the files listed above.
- Scoped evidence file created:
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/dependency-static-analysis-evidence.txt`

### Blocked Or Unverified Areas

- No networked package audit was performed, so this pass did not identify or rule out current CVEs, malware advisories, yanked packages, typosquatting, license policy issues, or maintainer compromise.
- No dependency install or lock generation was performed, so transitive Python/Bun/Go dependency graphs were not resolved or compared.
- No Docker image build or runtime inspection was performed; Dockerfile conclusions are static.
- No GitHub Actions workflow was executed; CI conclusions are based on workflow source.
- No full Bandit rerun was performed because this pass changed only report/evidence files and the audit already had a baseline summary.
- `/tmp/tldw_repo_audit_bandit_app.json` existed locally and was used for medium-record triage. If that temp artifact is absent in another checkout, regenerate it from the command recorded in `bandit-app-summary.txt` before repeating the exact medium-record review.
- The root worktree did not have its own `.venv`; this pass avoided Python execution and used `jq` for JSON processing.
- Full frontend/API/DB/security behavior was not reassessed beyond dependency/static-analysis relevance. Domain owners remain authoritative for their normalized findings.

### Evidence Notes

- This report intentionally does not duplicate `AUDIT-2026-06-27-OPS-004` or `AUDIT-2026-06-27-OPS-005`. `AUDIT-2026-06-27-DEPS-001` is narrower and different: it covers Python reproducibility/lock coverage for runtime, CI, Docker, and release jobs, not update automation coverage.
- `AUDIT-2026-06-27-DEPS-002` is also distinct from `AUDIT-2026-06-27-OPS-003`: OPS-003 covers how many workflow files actionlint checks; DEPS-002 covers mutable installer/action/tool inputs used by the static-analysis and CI gates themselves.
- Bandit evidence was treated as a starting point only. Medium records were promoted only after source review; none warranted a new source-backed security finding in this pass.
- Positive supply-chain controls observed: publish workflows pin many release-critical actions by SHA; WebUI/admin production Dockerfiles use `bun install --frozen-lockfile`; the Go agent has a committed `go.sum`; GitHub dependency-review is present for pull requests with `fail-on-severity: high`; CodeQL runs for Python.
- Existing normalized findings with the strongest dependency/static-analysis follow-up value are `AUDIT-2026-06-27-OPS-004`, `AUDIT-2026-06-27-OPS-005`, and `AUDIT-2026-06-27-OPS-003`.
