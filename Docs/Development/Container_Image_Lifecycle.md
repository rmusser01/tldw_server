# Container Image Lifecycle

This document describes how container images are built, validated, and published across CI workflows. It is aimed at contributors and CI maintainers.

For operational guidance on running containers locally, see [Dockerfiles/README.md](../../Dockerfiles/README.md). For the required PR gate contract, see [CI_REQUIRED_GATES.md](CI_REQUIRED_GATES.md).

## Overview

The project produces five container images across two tracks:

- **User-facing track** (app, webui, admin-ui): validated on every PR. During
  the pre-counsel frontend licensing freeze, only the GPL backend `app` image
  is published as a GHCR snapshot on merges to `main`.
- **Infrastructure track** (worker, audio-worker): published to GHCR on tagged
  releases only.

The `app` image is the only image that appears in all three workflows.

Three GitHub Actions workflows manage the lifecycle:

| Workflow | Trigger | Registry | Purpose |
|----------|---------|----------|---------|
| `container-build-check` | PR to `main`/`dev`, manual dispatch | None (build-only) | Validate Dockerfiles build |
| `publish-ghcr-main` | Push to `main` | GHCR | Publish the rolling GPL backend snapshot |
| `publish-docker` | Release (published), manual dispatch | GHCR | Publish versioned releases |

## Coverage Matrix

| Image | Dockerfile | Port | `container-build-check` | `publish-ghcr-main` | `publish-docker` |
|-------|-----------|------|------------------------|---------------------|-----------------|
| app | `Dockerfiles/Dockerfile.prod` | 8000 | Yes | Yes | Yes |
| webui | `Dockerfiles/Dockerfile.webui` | 3000 | Yes | No (licensing freeze) | -- |
| admin-ui | `Dockerfiles/Dockerfile.admin-ui` | 3001 | Yes | No (licensing freeze) | -- |
| worker | `Dockerfiles/Dockerfile.worker` | -- | -- | -- | Yes |
| audio-worker | `Dockerfiles/Dockerfile.audio_gpu_worker` | -- | -- | -- | Yes |

## Workflows

### `container-build-check`

**File:** `.github/workflows/container-build-check.yml`

**Trigger:** Pull requests targeting `main` or `dev`, plus `workflow_dispatch` (manual).

**What it does:** Builds the `app`, `webui`, and `admin-ui` images without pushing. This validates that the Dockerfiles and their build contexts are healthy before merge.

**Key details:**
- Matrix: `app`, `webui`, `admin-ui` with `fail-fast: false` (all three run even if one fails).
- Build-only: `push: false`. No registry login or image push.
- No GHA cache: each run is a cold build, ensuring Dockerfiles build from scratch.
- Timeout: 30 minutes.
- Passes `build-args` for webui and admin-ui (Next.js environment variables like `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE`). The `app` entry has no build args.
- Concurrency: grouped per PR number with `cancel-in-progress: true` (a new push to the same PR cancels the previous run).

**Branch protection:** A summary job rolls up the matrix results into a single `container-build-check` status. Branch protection only needs to require this one check name. See [CI_REQUIRED_GATES.md](CI_REQUIRED_GATES.md) for rollout status.

---

### `publish-ghcr-main`

**File:** `.github/workflows/publish-ghcr-main.yml`

**Trigger:** Push to `main` (fires on every merge).

**What it does:** Builds and pushes only the GPL backend `app` snapshot to
GHCR. WebUI and Admin UI remain build-checked on pull requests but are not
published by this workflow during the frontend licensing freeze.

**Key details:**
- Registry: GHCR only (not Docker Hub).
- Tags: `main` (rolling, always points to latest merge) and `sha-<shortsha>` (immutable, pinnable).
- Image name: `ghcr.io/<owner>/<repo>`.
- No frontend build arguments are used because the publishing matrix contains
  only the backend image.
- Uses GHA build cache (`cache-from: type=gha`, `cache-to: type=gha,mode=max`) for faster rebuilds.
- Generates SLSA provenance attestations for each image via `actions/attest-build-provenance` and pushes them to the registry.
- Timeout: 30 minutes.
- Concurrency: grouped by ref with `cancel-in-progress: true`.

Protected image publishing resumes only through a release-specific licensing
workflow after artifact notices and image separation pass their later gate.

---

### `publish-docker`

**File:** `.github/workflows/publish-docker.yml`

**Trigger:** GitHub Release publication (`release.published`), plus `workflow_dispatch` with an optional `manual_tag` input for ad-hoc builds.

**What it does:** Builds and pushes release images for `app`, `worker`, and
`audio-worker` to GHCR. GitHub Release publication is the release-driving event
for this workflow in v1.

**Key details:**
- Registry: GHCR only.
- Matrix: `app`, `worker`, `audio-worker`. Does **not** include webui or admin-ui.
- Image suffixes: none for app, `-worker`, `-audio-worker`.
- No build-args: unlike the snapshot workflows, no Next.js environment variables are passed (no frontend images in this workflow).
- Uses GHA build cache.
- Generates SLSA provenance attestations on GHCR.
- No concurrency group: two simultaneous releases would race. This is acceptable given releases are infrequent and serialized by convention.

**Tags by trigger:**

| Trigger | Tags produced |
|---------|---------------|
| Release (published) | `<version>` (e.g., `1.2.3`), `<major>.<minor>` (e.g., `1.2`), `latest` |
| Manual dispatch | `<manual_tag>` (user-supplied), `sha-<shortsha>` |

Note: `latest` is only set on release events, never on manual dispatch. `sha-*` tags only appear on manual dispatch, not on releases.

## Tagging Convention

| Tag | Source | Mutable | Use case |
|-----|--------|---------|----------|
| `main` | `publish-ghcr-main` | Yes (rolling) | Track latest `main` in dev/staging |
| `sha-<shortsha>` | `publish-ghcr-main` / `publish-docker` (dispatch) | No | Pin to exact commit |
| `<version>` (e.g., `1.2.3`) | `publish-docker` (release) | No | Production deployments |
| `<major>.<minor>` (e.g., `1.2`) | `publish-docker` (release) | Yes (within patch) | Track latest patch |
| `latest` | `publish-docker` (release) | Yes (rolling) | Convenience; not recommended for production |

## Attestation

Both publish workflows generate SLSA provenance attestations via
`actions/attest-build-provenance` and push them to GHCR alongside the image.

Attestations allow consumers to verify the image was built by this repository's CI and trace it back to the source commit.

## Adding a New Image

1. **Decide which track** the image belongs to:
   - Backend user-facing (PR gate + GHCR snapshots): add to
     `container-build-check` and `publish-ghcr-main`.
   - Protected frontend during the licensing freeze: add only to
     `container-build-check` until a release-specific licensing workflow is
     approved.
   - Infrastructure (release only): add to `publish-docker`.
   - Both: add to all three workflows.

2. **Create the Dockerfile** in `Dockerfiles/`. Follow the existing multi-stage pattern (builder + runtime). Use a non-root user with a unique UID (existing: `appuser` 10001, `webui` 10002, `adminui` 10003).

3. **Add a matrix entry** to each relevant workflow:
   - Include the `dockerfile` path and `image_suffix` (or `ghcr_suffix` for
     `publish-docker`).
   - Add `build_args` if the image requires build-time configuration (e.g., Next.js environment variables).

4. **Update documentation:**
   - Add a row to the Coverage Matrix in this document.
   - Update `Dockerfiles/README.md` with the new image in the Images section.
   - If the image should be a PR gate, update [CI_REQUIRED_GATES.md](CI_REQUIRED_GATES.md) and branch protection settings.

## Design Asymmetries

The coverage matrix is intentionally asymmetric. This section explains why.

**Worker and audio-worker are not in the PR gate or snapshot workflows.**

- These are infrastructure-tier images for the embeddings scale-out pipeline, not part of the core user-facing stack.
- `Dockerfile.audio_gpu_worker` has CUDA/GPU dependencies that would fail or be slow on standard `ubuntu-latest` runners.
- PR #996 originally scoped snapshot publishing to user-facing images (`app`
  + UIs); the current licensing freeze narrows publishing to `app`. Extending
  build coverage to workers remains a future option if GPU runners become
  available.

**WebUI and admin-ui are not in either publishing workflow during the licensing freeze.**

- These images are compose-first in v1: they are designed to run alongside the
  `app` service via Docker Compose, not as standalone registry pulls.
- They remain build-checked on pull requests, but neither rolling nor release
  workflows push them while protected artifact grants and notices are pending.
- Publishing resumes only through the later release-specific licensing gate.

**`app` is the only image in all three workflows.** It serves as the integration point: if `app` builds and publishes correctly, the core server is healthy across all pipeline stages.

## Branch Protection

`container-build-check` is pending addition to the branch protection required status checks. See [CI_REQUIRED_GATES.md](CI_REQUIRED_GATES.md) rollout phase 5.

The workflow includes a summary job that rolls up the matrix results into a single `container-build-check` status. Branch protection only needs to require this one check name, and it remains stable if the matrix changes.
