# Container Image Lifecycle

This document describes how container images are built, validated, and published across CI workflows. It is aimed at contributors and CI maintainers.

For operational guidance on running containers locally, see [Dockerfiles/README.md](../../Dockerfiles/README.md). For the required PR gate contract, see [CI_REQUIRED_GATES.md](CI_REQUIRED_GATES.md).

## Overview

The project produces five container images across two tracks:

- **User-facing track** (app, webui, admin-ui): validated on every PR, published as GHCR snapshots on every merge to `main`.
- **Infrastructure track** (worker, audio-worker): published to Docker Hub and GHCR on tagged releases only.

The `app` image is the only image that appears in all three workflows.

Three GitHub Actions workflows manage the lifecycle:

| Workflow | Trigger | Registry | Purpose |
|----------|---------|----------|---------|
| `container-build-check` | PR to `main`/`dev`, manual dispatch | None (build-only) | Validate Dockerfiles build |
| `publish-ghcr-main` | Push to `main` | GHCR | Publish rolling snapshots |
| `publish-docker` | Release (published), manual dispatch | Docker Hub + GHCR | Publish versioned releases |

## Coverage Matrix

| Image | Dockerfile | Port | `container-build-check` | `publish-ghcr-main` | `publish-docker` |
|-------|-----------|------|------------------------|---------------------|-----------------|
| app | `Dockerfiles/Dockerfile.prod` | 8000 | Yes | Yes | Yes |
| webui | `Dockerfiles/Dockerfile.webui` | 3000 | Yes | Yes | -- |
| admin-ui | `Dockerfiles/Dockerfile.admin-ui` | 3001 | Yes | Yes | -- |
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

**What it does:** Builds and pushes snapshot images for `app`, `webui`, and `admin-ui` to GHCR. These snapshots let operators and CI test the latest `main` without waiting for a release, and they continue to republish on every `main` push independently of GitHub Release publication.

**Key details:**
- Registry: GHCR only (not Docker Hub).
- Tags: `main` (rolling, always points to latest merge) and `sha-<shortsha>` (immutable, pinnable).
- The `sha-<shortsha>` tag is consistent across all three images for a given commit, so operators can pin all three to the same revision.
- Image names: `ghcr.io/<owner>/<repo>` (app), `ghcr.io/<owner>/<repo>-webui`, `ghcr.io/<owner>/<repo>-admin-ui`.
- Passes the same `build-args` as `container-build-check` for webui and admin-ui.
- Uses GHA build cache (`cache-from: type=gha`, `cache-to: type=gha,mode=max`) for faster rebuilds.
- Generates SLSA provenance attestations for each image via `actions/attest-build-provenance` and pushes them to the registry.
- Timeout: 30 minutes.
- Concurrency: grouped by ref with `cancel-in-progress: true`.

---

### `publish-docker`

**File:** `.github/workflows/publish-docker.yml`

**Trigger:** GitHub Release publication (`release.published`), plus `workflow_dispatch` with an optional `manual_tag` input for ad-hoc builds.

**What it does:** Builds and pushes release images for `app`, `worker`, and `audio-worker` to both Docker Hub and GHCR. GitHub Release publication is the release-driving event for this workflow in v1.

**Key details:**
- Registries: both Docker Hub and GHCR.
- Matrix: `app`, `worker`, `audio-worker`. Does **not** include webui or admin-ui.
- Image suffixes: none for app, `-worker`, `-audio-worker`.
- No build-args: unlike the snapshot workflows, no Next.js environment variables are passed (no frontend images in this workflow).
- Uses GHA build cache.
- Generates SLSA provenance attestations on GHCR only (not Docker Hub).
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

Both publish workflows generate SLSA provenance attestations via `actions/attest-build-provenance` and push them to the GHCR registry alongside the image. Docker Hub does not receive attestations.

Attestations allow consumers to verify the image was built by this repository's CI and trace it back to the source commit.

## Adding a New Image

1. **Decide which track** the image belongs to:
   - User-facing (PR gate + GHCR snapshots): add to `container-build-check` and `publish-ghcr-main`.
   - Infrastructure (release only): add to `publish-docker`.
   - Both: add to all three workflows.

2. **Create the Dockerfile** in `Dockerfiles/`. Follow the existing multi-stage pattern (builder + runtime). Use a non-root user with a unique UID (existing: `appuser` 10001, `webui` 10002, `adminui` 10003).

3. **Add a matrix entry** to each relevant workflow:
   - Include the `dockerfile` path and `image_suffix` (or `ghcr_suffix`/`dockerhub_suffix` for `publish-docker`).
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
- PR #996 scoped the snapshot publishing to user-facing images (`app` + UIs). Extending coverage to workers is a future option if GPU runners become available.

**WebUI and admin-ui are not in the release workflow.**

- These images are compose-first in v1: they are designed to run alongside the `app` service via Docker Compose, not as standalone Docker Hub pulls.
- They are published to GHCR only, with `main` and `sha-*` tags, which is sufficient for the current deployment model.
- Adding them to the release workflow with semver tags is a future option when standalone deployment is supported.

**`app` is the only image in all three workflows.** It serves as the integration point: if `app` builds and publishes correctly, the core server is healthy across all pipeline stages.

## Branch Protection

`container-build-check` is pending addition to the branch protection required status checks. See [CI_REQUIRED_GATES.md](CI_REQUIRED_GATES.md) rollout phase 5.

The workflow includes a summary job that rolls up the matrix results into a single `container-build-check` status. Branch protection only needs to require this one check name, and it remains stable if the matrix changes.
