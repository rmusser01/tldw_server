# GHCR Main Snapshot Publishing Design

Date: 2026-04-03
Status: Approved for planning
Owner: Codex brainstorming session

## Summary

Add a dedicated GitHub Actions workflow that publishes rolling GHCR snapshot images when commits land on `main`. This snapshot flow covers the primary API image, the WebUI image, and the Admin UI image. Release publishing remains a separate concern: semver and `latest` tags continue to come only from the release-driven publish workflow.

The design intentionally separates stable release tags from rolling `main` tags. That avoids turning a release workflow into a mixed policy engine and makes it clear which images are safe for production pinning versus which images represent the newest post-merge state. It also treats `main` as a convenience tag, not as the authoritative cross-image release identifier. Exact multi-image consistency comes from the shared `sha-<shortsha>` tags.

## Problem

The repository already has a container publish workflow in [`.github/workflows/publish-docker.yml`](../../../.github/workflows/publish-docker.yml), but it only runs on release publication and manual dispatch.

Today:

- container publishing is release-centric
- there is no automatic GHCR snapshot publication on merge to `main`
- operators who want the latest merged container build must either build locally or wait for a formal release
- the current publish matrix focuses on `app`, `worker`, and `audio-worker`, which does not match the requested first snapshot scope
- release-tagging behavior and snapshot-tagging behavior are currently not separated

This leaves a gap between merged code and distributable container artifacts.

## Goals

- Publish rolling GHCR images automatically on `push` to `main`
- Cover exactly these images in the first iteration:
  - API image from `Dockerfiles/Dockerfile.prod`
  - WebUI image from `Dockerfiles/Dockerfile.webui`
  - Admin UI image from `Dockerfiles/Dockerfile.admin-ui`
- Publish snapshot tags suitable for “latest main” consumption:
  - `main`
  - `sha-<shortsha>`
- Keep release publishing responsible for semver tags and `latest`
- Keep GHCR authentication and provenance attestation aligned with current repository practice
- Add pre-merge validation so the same images are built on PRs before changes can land on `main`
- Make the PR container build check a standalone required status for protected branches
- Document the runtime contract for published UI snapshots so operators understand the difference between compose-first defaults and direct `docker run` expectations

## Non-Goals

- Replace the existing release publish workflow
- Introduce Docker Hub publishing for `main` snapshots
- Publish worker or audio-worker snapshots in this first change
- Redesign all existing CI workflows into reusable workflow primitives
- Create a full release-management system for multi-channel container promotion

## Current State

### Existing publish workflow

[`.github/workflows/publish-docker.yml`](../../../.github/workflows/publish-docker.yml):

- triggers on `release.published` and `workflow_dispatch`
- logs into both GHCR and Docker Hub
- publishes a matrix of:
  - `Dockerfiles/Dockerfile.prod`
  - `Dockerfiles/Dockerfile.worker`
  - `Dockerfiles/Dockerfile.audio_gpu_worker`
- generates provenance attestations

This is a solid base for release publishing, but it does not model rolling snapshots on `main`.

### Existing image definitions

- API image: [`Dockerfiles/Dockerfile.prod`](../../../Dockerfiles/Dockerfile.prod)
- WebUI image: [`Dockerfiles/Dockerfile.webui`](../../../Dockerfiles/Dockerfile.webui)
- Admin UI image: [`Dockerfiles/Dockerfile.admin-ui`](../../../Dockerfiles/Dockerfile.admin-ui)

The WebUI and Admin UI images already have Dockerfiles and compose overlays, but they are not included in the current publish matrix.

### Current CI structure

[`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml) already runs on `push` to `main` and on pull requests, but it is not responsible for container publishing. The repository also uses focused required-check workflows rather than routing every concern through one monolithic pipeline.

That existing structure favors a separate snapshot-publish workflow rather than overloading the release workflow with event-specific branching.

## Proposed Design

### 1. Keep release publishing separate

Retain [`.github/workflows/publish-docker.yml`](../../../.github/workflows/publish-docker.yml) as the release-oriented workflow.

Responsibilities that stay there:

- semver tag publication
- `latest` tag publication
- release/manual event handling
- Docker Hub publication if still desired for release artifacts

This preserves the existing release contract and avoids mixing stable and rolling tags in one job.

### 2. Add a dedicated `main` snapshot publish workflow

Add a new workflow, for example:

- `.github/workflows/publish-ghcr-main.yml`

Trigger:

- `push` on branch `main`

Permissions:

- `contents: read`
- `packages: write`
- `attestations: write`
- `id-token: write`

Registry target:

- GHCR only

This workflow should not log into Docker Hub and should not emit semver or `latest` tags.

### 3. Publish exactly three images in the first snapshot matrix

Snapshot matrix:

- `app`
  - dockerfile: `Dockerfiles/Dockerfile.prod`
  - image: `ghcr.io/<owner>/<repo>`
- `webui`
  - dockerfile: `Dockerfiles/Dockerfile.webui`
  - image: `ghcr.io/<owner>/<repo>-webui`
- `admin-ui`
  - dockerfile: `Dockerfiles/Dockerfile.admin-ui`
  - image: `ghcr.io/<owner>/<repo>-admin-ui`

This first version deliberately excludes:

- `Dockerfiles/Dockerfile.worker`
- `Dockerfiles/Dockerfile.audio_gpu_worker`

Reason:

- requested scope is API + WebUI + Admin UI
- the worker Dockerfiles currently appear to reference a non-existent repo-root `Config_Files` path in this checkout, so including them would add unrelated failure risk to the first snapshot rollout

### 4. Use snapshot-only tags

Each published image gets:

- `main`
- `sha-<shortsha>`

Examples:

- `ghcr.io/<owner>/<repo>:main`
- `ghcr.io/<owner>/<repo>:sha-abc1234`
- `ghcr.io/<owner>/<repo>-webui:main`
- `ghcr.io/<owner>/<repo>-admin-ui:sha-abc1234`

Rules:

- no `latest` from `push` to `main`
- no semver tags from `push` to `main`
- no branch-name explosion beyond `main` in this first version
- snapshot publication is per-image, not atomic across the full matrix

Operational meaning:

- `main` is a floating convenience tag for each image independently
- `sha-<shortsha>` is the authoritative identifier for a coherent code revision across API, WebUI, and Admin UI
- deployments that need strict cross-image consistency should pin all images to the same `sha-<shortsha>` tag, not to `main`

This keeps snapshot tags easy to understand and easy to consume in staging or internal environments without pretending that a matrix push is transactionally atomic.

### 5. Preserve current build metadata and provenance patterns

The new workflow should reuse the same action family and pinned-action style already present in the repo:

- `actions/checkout`
- `docker/setup-buildx-action`
- `docker/login-action`
- `docker/metadata-action`
- `docker/build-push-action`
- `actions/attest-build-provenance`

That keeps the supply-chain story consistent with current release publishing and avoids unnecessary workflow divergence.

### 6. Mirror documented container defaults for WebUI and Admin UI

The snapshot workflow must pass build arguments that match the documented compose defaults so published artifacts behave like the containerized quickstart/setup paths.

Expected defaults:

- WebUI should follow the same default build args documented in [`Dockerfiles/docker-compose.webui.yml`](../../../Dockerfiles/docker-compose.webui.yml)
- Admin UI should follow the same default build args documented in [`Dockerfiles/docker-compose.admin-ui.yml`](../../../Dockerfiles/docker-compose.admin-ui.yml)

This matters because both Next.js images bake public environment values into the client bundle at build time. Snapshot images should reflect the intended default runtime story instead of relying on unspecified CI defaults.

Runtime contract for v1:

- API snapshots should remain straightforward to run directly with `docker run`
- WebUI and Admin UI snapshots are compose-first artifacts in v1 because their baked defaults assume the project’s standard container topology
- direct `docker run` usage of the UI images is only expected to work with minimal wiring when the operator deliberately provides a compatible backend origin/network arrangement
- truly portable UI snapshots with minimal direct-run wiring are a follow-up concern and likely require a more runtime-configurable image strategy

### 7. Add PR build validation for the same three images

Snapshot publishing should not be the first time these images are built.

Add a separate pull-request validation workflow, for example:

- `.github/workflows/container-build-check.yml`

Trigger:

- `pull_request` targeting `main` and `dev`
- optional `workflow_dispatch`

Behavior:

- build, but do not push, the same three images
- use the same Dockerfiles and build args as the `main` snapshot workflow
- fail the PR if one of those images no longer builds
- expose this workflow as its own named status check, `container-build-check`
- configure repository branch protection so `container-build-check` is a required status on protected target branches

This provides a clean guardrail:

- PRs prove the images build
- merges to `main` publish the validated snapshot images

Important operational note:

- the required-check policy itself lives in repository settings, not in workflow YAML
- implementation must therefore include both the workflow and the corresponding branch-protection update/runbook note

### 8. Update documentation and release checklists to reflect the split contract

The repository should explicitly document the difference between release-published images and `main` snapshot images.

Expected documentation updates:

- [`Docs/Release_Checklist.md`](../../../Docs/Release_Checklist.md) should keep release verification focused on release-published artifacts
- Docker or deployment docs should explain that `main` snapshots exist for API, WebUI, and Admin UI
- docs should state clearly that semver and `latest` remain release-only unless a later change expands that contract

Without this, operators are likely to assume that every image published on `main` is also release-tagged or vice versa.

## Architecture

### Release path

1. Maintainer publishes a GitHub release.
2. Existing release workflow runs.
3. Release workflow pushes semver and `latest` images according to current release policy.

### Main snapshot path

1. A PR merges into `main`.
2. GitHub emits a `push` event for `main`.
3. New snapshot workflow runs a matrix for API, WebUI, and Admin UI.
4. Workflow logs into GHCR with `GITHUB_TOKEN`.
5. Metadata step computes `main` and `sha-<shortsha>` tags.
6. Buildx builds and pushes each image.
7. Provenance attestation is generated for each pushed GHCR image.

Note:

- these matrix entries publish independently
- one image may advance its `main` tag even if another image fails later in the workflow
- consumers that need all three artifacts from the exact same revision should deploy the shared `sha-<shortsha>` tags instead of relying on the floating `main` tags

### PR validation path

1. A PR targets `main` or `dev`.
2. Container build-check workflow runs.
3. Workflow builds API, WebUI, and Admin UI images without pushing.
4. The workflow reports the standalone `container-build-check` status.
5. Because that status is required in branch protection, failures block merge before the `main` publish path is reached.

## Components

### Snapshot publish workflow

Responsibilities:

- run on `push` to `main`
- build and push only the three approved snapshot images
- emit only rolling snapshot tags
- attest pushed artifacts

### Release publish workflow

Responsibilities:

- continue handling release publication
- remain the only source of semver and `latest` tags
- avoid taking on branch-snapshot logic

### Container build-check workflow

Responsibilities:

- pre-merge build verification
- keep snapshot publish failures out of the default merge path
- ensure build args stay aligned with published images

### Image naming contract

Responsibilities:

- keep API on the repository root package name
- use explicit suffixed names for sibling UIs
- avoid collisions between API, WebUI, and Admin UI artifacts

## Error Handling

### Build failures on PRs

If any of the three images fails to build in PR validation:

- the check fails
- merge is blocked
- no publish attempt happens on `main`

### Build failures on `main`

If snapshot publishing fails after merge:

- the workflow fails visibly in Actions
- any image that already pushed may have advanced its own floating `main` tag
- images that did not reach push remain on their previous floating `main` tag
- the shared `sha-<shortsha>` tags make it obvious which images published successfully for that revision
- release publishing remains unaffected because it is isolated in a separate workflow

### Registry/auth failures

If GHCR login or push fails:

- fail the job immediately
- do not try to treat the build as partially successful
- rely on Actions logs for diagnosis rather than hiding the failure behind best-effort behavior

## Testing Strategy

### Workflow validation

- run `actionlint` over the new workflow files
- ensure pinned actions and permissions syntax are valid

### Build validation

- add PR build-only workflow for API, WebUI, and Admin UI
- confirm all three build from repo root using current Dockerfiles
- verify the workflow name and reported status match the branch-protection requirement exactly

### Policy validation

Manual checks after implementation:

- merge to `main` publishes:
  - `ghcr.io/<owner>/<repo>:main`
  - `ghcr.io/<owner>/<repo>:sha-<shortsha>`
  - `ghcr.io/<owner>/<repo>-webui:main`
  - `ghcr.io/<owner>/<repo>-admin-ui:main`
- release publication still publishes semver and `latest`
- no `latest` tag is emitted by the `main` snapshot workflow
- repository settings show `container-build-check` as a required status for the intended protected branches
- docs and checklists describe the release-vs-snapshot split accurately

## Trade-Offs

### Chosen trade-off: separate workflows over one highly-conditional workflow

Pros:

- simpler event logic
- clearer operator expectations
- safer release path
- easier future maintenance

Cost:

- some duplicated workflow YAML

This duplication is acceptable in the first version because the policy split is more important than eliminating repetition immediately.

### Chosen trade-off: exclude workers in v1

Pros:

- meets the requested scope exactly
- avoids unrelated worker-Dockerfile issues
- gets the highest-value images published first

Cost:

- worker-style snapshot artifacts remain unavailable until a follow-up change

This is acceptable because the initial requirement was API + WebUI + Admin UI, not full image parity.

## Open Follow-Ups

- Decide later whether release publishing should also expand to WebUI and Admin UI semver tags
- Decide later whether worker and audio-worker Dockerfiles should be repaired and brought into snapshot publishing
- Consider refactoring release and snapshot workflows into a reusable workflow only after the first version is stable
- Decide later whether the UI images should move from compose-first baked defaults to a stronger runtime-configurable model for better direct `docker run` ergonomics

## Recommendation

Implement the smallest reliable version:

- new GHCR-only snapshot workflow on `push` to `main`
- new PR build-check workflow for the same images
- no change to the release-tagging contract beyond leaving it isolated

This delivers immediately useful container artifacts without entangling release semantics, and it keeps the first rollout aligned with the project’s existing CI structure.
