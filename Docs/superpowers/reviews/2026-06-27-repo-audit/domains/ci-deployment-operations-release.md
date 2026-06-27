# CI, Deployment, Operations, And Release Surfaces Domain Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: CI, Deployment, Operations, and Release Surfaces
- In scope: GitHub workflows, Dockerfiles, deployment samples, operations docs, monitoring configs, release surfaces, scripts, and CI/deployment reliability and security posture.
- Out of scope: remediation implementation and release execution.

## Findings Table

| ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CANDIDATE-ci-deployment-operations-release-001 | confirmed_issue | static_confirmed | medium | high | test_gap | Published worker images are not built by the PR container gate | open | validated |
| CANDIDATE-ci-deployment-operations-release-002 | likely_risk | static_confirmed | medium | high | security | Published worker images run as root and keep build tooling in runtime layers | open | validated |
| CANDIDATE-ci-deployment-operations-release-003 | confirmed_issue | static_confirmed | medium | high | test_gap | actionlint gate covers only a small subset of workflow files | open | validated |
| CANDIDATE-ci-deployment-operations-release-004 | confirmed_issue | static_confirmed | medium | high | dependency | SBOM workflow skips Bun-based frontend dependencies | open | validated |
| CANDIDATE-ci-deployment-operations-release-005 | improvement_opportunity | source_linked | low | high | dependency | Dependency update automation omits nested JS, Python, and Go package roots | open | validated |
| CANDIDATE-ci-deployment-operations-release-006 | confirmed_issue | static_confirmed | medium | high | operations | Kubernetes sample Secret ships an invalid DATABASE_URL and default password | open | validated |

## Index Mapping

Candidate IDs use `CANDIDATE-ci-deployment-operations-release-NNN` per coordinator instruction. If promoted into `findings-index.json`, map them to stable audit IDs like `AUDIT-2026-06-27-OPS-NNN`. For every promoted finding, set `source_report` to `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md`, set `owner_domain` to CI, Deployment, Operations, and Release Surfaces, and preserve the `evidence_tier`, `evidence_strength`, `affected_paths`, `recommendation`, `status`, and `validation_status` fields recorded below.

## Confirmed Issues

### CANDIDATE-ci-deployment-operations-release-001: Published worker images are not built by the PR container gate

- `severity`: medium
- `confidence`: high
- `category`: test_gap
- `evidence_tier`: confirmed_issue
- `evidence_strength`: static_confirmed
- `owner_domain`: CI, Deployment, Operations, and Release Surfaces
- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md`
- `status`: open
- `validation_status`: validated
- `affected_paths`: `.github/workflows/publish-docker.yml`, `.github/workflows/container-build-check.yml`, `Dockerfiles/Dockerfile.worker`, `Dockerfiles/Dockerfile.audio_gpu_worker`
- `evidence`: `publish-docker.yml` publishes `Dockerfiles/Dockerfile.worker` and `Dockerfiles/Dockerfile.audio_gpu_worker` in addition to the app image, but `container-build-check.yml` only builds `Dockerfiles/Dockerfile.prod`, `Dockerfiles/Dockerfile.webui`, and `Dockerfiles/Dockerfile.admin-ui`.
- `impact`: Worker and audio-worker Dockerfile regressions can merge without the PR container gate catching build failures, dependency breakage, or packaging drift; the first deterministic failure may occur at release publication time.
- `recommendation`: Add `worker` and `audio-worker` entries to `container-build-check.yml`, or remove them from `publish-docker.yml` if they are no longer supported published artifacts. Prefer adding a minimal runtime smoke or image inspection step that verifies the expected command, non-root user, and core imports for every published image.

### CANDIDATE-ci-deployment-operations-release-003: actionlint gate covers only a small subset of workflow files

- `severity`: medium
- `confidence`: high
- `category`: test_gap
- `evidence_tier`: confirmed_issue
- `evidence_strength`: static_confirmed
- `owner_domain`: CI, Deployment, Operations, and Release Surfaces
- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md`
- `status`: open
- `validation_status`: validated
- `affected_paths`: `.github/workflows/actionlint.yml`, `.github/workflows/*.yml`, `.github/actions/**`
- `evidence`: The workflow triggers on `.github/workflows/**` and `.github/actions/**`, and the repository has 38 workflow files, but the `Run actionlint on targeted workflows` step enumerates only 8 workflow files. Required and release-sensitive workflows such as `backend-required.yml`, `security-required.yml`, `frontend-required.yml`, `e2e-required.yml`, `publish-pypi.yml`, `mcp-unified-publish.yml`, `mcp-unified-rc.yml`, `pypi-package.yml`, `jobs-suite.yml`, and the large `ci.yml` workflow are outside the lint command.
- `impact`: YAML/action expression mistakes in most required and release workflows can pass PR review without the dedicated workflow linter, including mistakes in path filters, conditions, permissions, and publishing jobs.
- `recommendation`: Run actionlint against all workflows and composite actions, or generate the target list from the filesystem in CI. If a workflow must be excluded, record the explicit exclusion and reason in the workflow.

### CANDIDATE-ci-deployment-operations-release-004: SBOM workflow skips Bun-based frontend dependencies

- `severity`: medium
- `confidence`: high
- `category`: dependency
- `evidence_tier`: confirmed_issue
- `evidence_strength`: static_confirmed
- `owner_domain`: CI, Deployment, Operations, and Release Surfaces
- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md`
- `status`: open
- `validation_status`: validated
- `affected_paths`: `.github/workflows/sbom.yml`, `apps/bun.lock`, `admin-ui/bun.lock`, `apps/package.json`, `admin-ui/package.json`, `apps/tldw-frontend/package.json`, `apps/packages/ui/package.json`, `apps/extension/package.json`
- `evidence`: The repository has `apps/bun.lock` and `admin-ui/bun.lock`, with no `package-lock.json` found. `sbom.yml` only checks `apps/package-lock.json` and `apps/tldw-frontend/package-lock.json`; when neither exists, it prints that no package lock was found and skips Node SBOM generation. The workflow can still merge/upload a Python-only SBOM, and the final SBOM validation is both gated on the presence of Python and Node SBOMs and marked `continue-on-error`.
- `impact`: Release SBOM artifacts can omit the WebUI, Admin UI, extension, and shared package dependencies even though those dependencies are part of the shipped container and client surfaces.
- `recommendation`: Add Bun-aware SBOM generation for `apps/bun.lock` and `admin-ui/bun.lock`, include all frontend workspaces, and fail the SBOM job when an expected ecosystem cannot be represented. Make validation required once the generator coverage is complete.

### CANDIDATE-ci-deployment-operations-release-006: Kubernetes sample Secret ships an invalid DATABASE_URL and default password

- `severity`: medium
- `confidence`: high
- `category`: operations
- `evidence_tier`: confirmed_issue
- `evidence_strength`: static_confirmed
- `owner_domain`: CI, Deployment, Operations, and Release Surfaces
- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md`
- `status`: open
- `validation_status`: validated
- `affected_paths`: `Helper_Scripts/Samples/Kubernetes/app-secret.yaml`, `Helper_Scripts/Samples/Kubernetes/tldw-app-deployment.yaml`, `Helper_Scripts/Samples/Kubernetes/postgres-statefulset.yaml`
- `evidence`: `app-secret.yaml` sets `DATABASE_URL` to `postgresql://tldw_user:${POSTGRES_PASSWORD}@postgres:5432/tldw_users` and `POSTGRES_PASSWORD` to `TestPassword123!`. The app deployment imports the whole Secret with `envFrom`, while the Postgres StatefulSet reads `POSTGRES_PASSWORD` from that Secret. Kubernetes does not interpolate `${POSTGRES_PASSWORD}` inside `stringData`, so the app receives a literal password in `DATABASE_URL` that does not match Postgres. The sample also includes a concrete default database password.
- `impact`: Applying the sample as-is can produce a non-working Kubernetes deployment, and operators may copy a weak default database password into live manifests.
- `recommendation`: Replace the sample with non-runnable placeholders plus explicit generation instructions, or derive `DATABASE_URL` from separate env vars at runtime. If keeping a complete sample, duplicate a clearly generated example value consistently and add comments requiring rotation before deployment.

## Likely Risks

### CANDIDATE-ci-deployment-operations-release-002: Published worker images run as root and keep build tooling in runtime layers

- `severity`: medium
- `confidence`: high
- `category`: security
- `evidence_tier`: likely_risk
- `evidence_strength`: static_confirmed
- `owner_domain`: CI, Deployment, Operations, and Release Surfaces
- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md`
- `status`: open
- `validation_status`: validated
- `affected_paths`: `Dockerfiles/Dockerfile.worker`, `Dockerfiles/Dockerfile.audio_gpu_worker`, `Dockerfiles/Dockerfile.prod`, `.github/workflows/publish-docker.yml`, `Dockerfiles/docker-compose.embeddings.yml`
- `evidence`: `Dockerfile.worker` installs compiler/build and network tooling including `gcc`, `g++`, `make`, `curl`, `git`, `ffmpeg`, `portaudio19-dev`, and `python3-dev` in the final image, then runs the worker command without a `USER` directive. `Dockerfile.audio_gpu_worker` likewise keeps build tooling and has no `USER` directive. By contrast, `Dockerfile.prod` creates `appuser`, copies files with appuser ownership, and switches to `USER appuser`. The release workflow publishes the worker and audio-worker images.
- `impact`: If a worker path is compromised through media, document, model, or queue input handling, the container process has root privileges inside the container and a larger runtime toolset than necessary. This increases the blast radius for mounted volumes and lateral movement attempts.
- `recommendation`: Align worker images with `Dockerfile.prod`: use multi-stage builds, remove build-only packages from final stages, create a fixed non-root runtime user, chown only required writable paths, and add image checks to CI. For GPU workers, document any required device permissions separately from Linux user identity.

## Improvement Opportunities

### CANDIDATE-ci-deployment-operations-release-005: Dependency update automation omits nested JS, Python, and Go package roots

- `severity`: low
- `confidence`: high
- `category`: dependency
- `evidence_tier`: improvement_opportunity
- `evidence_strength`: source_linked
- `owner_domain`: CI, Deployment, Operations, and Release Surfaces
- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md`
- `status`: open
- `validation_status`: validated
- `affected_paths`: `.github/dependabot.yml`, `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/dependency-manifest-inventory.txt`, `apps/bun.lock`, `admin-ui/bun.lock`, `apps/mcp-unified/pyproject.toml`, `sdks/python/pyproject.toml`, `tools/backlog-py/pyproject.toml`, `tools/tldw-agent/go.mod`
- `evidence`: `dependabot.yml` configures only root `pip` updates and GitHub Actions updates. The dependency-manifest inventory lists additional release-relevant roots: multiple frontend/admin/package manifests, the standalone MCP Python package, Python SDK/tool packages, and the Go agent module.
- `impact`: Dependency drift and vulnerable dependency updates for nested packages depend on manual review or incidental PRs rather than scheduled automation, which is especially risky for separately shipped frontend/admin images, the MCP package, SDKs, and the Go agent.
- `recommendation`: Add update automation for every maintained package root. If Dependabot cannot handle the Bun lockfiles reliably, use Renovate or another Bun-aware updater for `apps/` and `admin-ui/`, add `gomod` coverage for `tools/tldw-agent`, and add nested Python package coverage or an explicit documented exclusion for packages not intended to be maintained independently.

## Coverage And Evidence

### Files Inspected

- Audit context: `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`, `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`, `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/ci-deploy-ops-inventory.txt`, `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/dependency-manifest-inventory.txt`, `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
- CI and release: `.github/workflows/actionlint.yml`, `.github/workflows/backend-required.yml`, `.github/workflows/container-build-check.yml`, `.github/workflows/codeql.yml`, `.github/workflows/mcp-unified-publish.yml`, `.github/workflows/mcp-unified-rc.yml`, `.github/workflows/publish-docker.yml`, `.github/workflows/publish-ghcr-main.yml`, `.github/workflows/publish-pypi.yml`, `.github/workflows/pypi-package.yml`, `.github/workflows/sbom.yml`, `.github/workflows/security-required.yml`, `.github/actions/detect-required-gate-changes/action.yml`, `.github/dependabot.yml`, `.github/security/ci-allowlist.yml`
- Docker and compose: `Dockerfiles/Dockerfile.prod`, `Dockerfiles/Dockerfile.worker`, `Dockerfiles/Dockerfile.audio_gpu_worker`, `Dockerfiles/Dockerfile.webui`, `Dockerfiles/Dockerfile.admin-ui`, `Dockerfiles/docker-compose.yml`, `Dockerfiles/docker-compose.single-user.yml`, `Dockerfiles/docker-compose.multi-user-postgres.yml`, `Dockerfiles/docker-compose.webui.yml`, `Dockerfiles/docker-compose.workers.yml`, `Dockerfiles/docker-compose.embeddings.yml`, `Dockerfiles/README.md`
- Deployment and samples: `Docs/Development/Container_Image_Lifecycle.md` was located through `Dockerfiles/README.md`; targeted evidence was read from `Helper_Scripts/Samples/Kubernetes/app-secret.yaml`, `Helper_Scripts/Samples/Kubernetes/app-configmap.yaml`, `Helper_Scripts/Samples/Kubernetes/tldw-app-deployment.yaml`, `Helper_Scripts/Samples/Kubernetes/postgres-statefulset.yaml`, and `Helper_Scripts/Samples/Kubernetes/ingress.yaml`
- Dependency surfaces: `apps/package.json`, `admin-ui/package.json`, `apps/bun.lock`, `admin-ui/bun.lock`, `apps/mcp-unified/pyproject.toml`

### Tests Or Scans Run

- `git status -sb`
- `git diff --stat`
- `sed -n '1,260p' Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
- `sed -n '1,260p' Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `sed -n '1,260p' Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/ci-deploy-ops-inventory.txt`
- `sed -n '1,220p' Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/dependency-manifest-inventory.txt`
- `sed -n '1,260p' Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
- `sed -n '1,260p' Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md`
- `rg -n "^(name:|on:|permissions:|  contents:|  packages:|  id-token:|  pull-requests:|  actions:|  security-events:)|pull_request_target|workflow_run|secrets:|uses:|docker/(login|metadata|build-push)|pypa/gh-action-pypi-publish|softprops|actions/upload-artifact|actions/download-artifact|npm publish|twine|docker push|gh release" .github/workflows .github/actions .github/codeql .github/security`
- `find .github/workflows -maxdepth 1 -type f -name '*.yml' -o -name '*.yaml'`
- Searched workflow and action files for task-marker, CI-bypass, continue-on-error, unconditional-success, and fail-fast patterns.
- `rg -n "(password|passwd|secret|token|api[_-]?key|jwt|SINGLE_USER_API_KEY|DATABASE_URL|POSTGRES_PASSWORD|AUTH_SECRET|SESSION_SECRET|OPENAI_API_KEY|ANTHROPIC_API_KEY)" .github Dockerfiles Docs/Operations Docs/Deployment Helper_Scripts/Samples`
- `nl -ba` targeted reads for release workflows, Dockerfiles, compose files, actionlint, security-required, backend-required, SBOM, Dependabot, and Kubernetes samples
- `rg -n "Dockerfile.worker|Dockerfile.audio_gpu_worker|audio-worker|worker" .github/workflows Dockerfiles Docs/Deployment Docs/Operations Helper_Scripts/Samples`
- `rg -n "^USER\\b|user:|read_only:|cap_drop:|security_opt:|no-new-privileges|privileged:|network_mode:|ports:" Dockerfiles`
- `rg -n "apt-get install|apk add|pip install|npm install|bun install|curl |wget |git clone|USER\\b" Dockerfiles/Dockerfile.* Dockerfiles/ACP/Dockerfile`
- `command -v actionlint` (no local binary found)
- `find .github/workflows -maxdepth 1 -type f \( -name '*.yml' -o -name '*.yaml' \) | wc -l` (38 workflows)
- `find . -name 'package-lock.json' -o -name 'bun.lock' -o -name 'bun.lockb' -o -name 'pnpm-lock.yaml' -o -name 'yarn.lock' | sort`
- `rg -n "package-lock|bun.lock|bun.lockb|pnpm-lock|yarn.lock" .github/workflows/sbom.yml apps admin-ui -g '!node_modules'`
- `rg -n "dependabot|renovate|npm audit|bun audit|pip-audit|safety|uv pip compile|dependency-review|Dependabot" .github Docs Dockerfiles Helper_Scripts apps admin-ui pyproject.toml`
- `find . -maxdepth 4 -iname '*renovate*' -o -iname '.snyk' -o -iname '*dependabot*'`

### Blocked Or Unverified Areas

- Did not install dependencies, use network access, start services, run Docker, or execute GitHub Actions, per coordinator rules.
- Did not run actionlint locally because no local `actionlint` binary was installed and downloading it would require network access.
- Did not build or inspect container images at runtime; Dockerfile/image findings are static review findings.
- Did not apply Kubernetes manifests; Kubernetes sample finding is based on static manifest behavior.
- Did not rerun Bandit because no production/source code was changed in this pass. The existing audit Bandit baseline was reviewed.
- Local `python` was not on `PATH` in this worktree during one attempted helper command; follow-up evidence used shell-only commands.
- Existing dirty worktree state before this pass: `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/command-log.md` had uncommitted changes that were not touched.

### Evidence Notes

- Additional scoped evidence file: `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/ci-deployment-operations-release-candidates.txt`
- The public quickstart compose files `docker-compose.single-user.yml` and `docker-compose.multi-user-postgres.yml` bind the API to `127.0.0.1` and do not publish Redis/Postgres ports; no finding was recorded for those current public quickstart defaults.
- `Dockerfiles/docker-compose.embeddings.yml` exposes Redis and optional monitoring/debug ports with development-style defaults. This was not promoted to a candidate finding because the stack is presented as an embeddings worker/monitoring helper, but it remains residual hardening risk if reused outside local/dev environments.
