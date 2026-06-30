# Release Process Automation Design

Date: 2026-04-22
Status: Approved for planning
Owner: Codex brainstorming session

## Summary

Add a repo-owned, local release entrypoint that cuts a release from `main` with a single command and then hands off artifact publication to existing GitHub Actions workflows. The first iteration should automate:

- version bumping from the canonical package version in [`pyproject.toml`](../../../pyproject.toml)
- changelog promotion from [`CHANGELOG.md`](../../../CHANGELOG.md)
- release commit creation
- annotated tag creation
- push of `main` and the release tag
- GitHub Release publication via `gh`

Publishing of release artifacts remains GitHub-native. A published GitHub Release should trigger Docker publishing for the release image set already defined in [`.github/workflows/publish-docker.yml`](../../../.github/workflows/publish-docker.yml). Rolling snapshot publishing for `main` remains separate and continues to be handled by [`.github/workflows/publish-ghcr-main.yml`](../../../.github/workflows/publish-ghcr-main.yml).

This design intentionally does not try to automate everything in one pass. The goal is to create one authoritative maintainer path that is accurate, safe, resumable, and aligned with the repository’s current release topology.

## Problem

The repository currently has useful release-related building blocks, but they are not composed into one authoritative release flow.

Today:

- package versioning is defined in [`pyproject.toml`](../../../pyproject.toml), but human-facing version references already drift in places such as [`README.md`](../../../README.md)
- release notes exist in [`CHANGELOG.md`](../../../CHANGELOG.md), but there is no repo-owned release command that turns changelog state into a published release
- Docker release publishing already exists in [`.github/workflows/publish-docker.yml`](../../../.github/workflows/publish-docker.yml), but it only runs after a GitHub Release is published
- `main` snapshot image publishing already exists in [`.github/workflows/publish-ghcr-main.yml`](../../../.github/workflows/publish-ghcr-main.yml), which means the repo already has two distinct publication modes that must remain clearly separated
- PyPI publishing is also currently tied to `release.published` in [`.github/workflows/publish-pypi.yml`](../../../.github/workflows/publish-pypi.yml), which conflicts with the desired first-iteration scope of “Docker + GitHub Release only”
- release readiness guidance exists in [`Docs/Release_Checklist.md`](../../../Docs/Release_Checklist.md), but that document is a broad checklist rather than a concise authoritative operator workflow

This leaves maintainers without a safe, consistent, local “cut a release” command and makes it too easy for documentation to describe a release process that the repo does not actually enforce.

## Goals

- Provide one authoritative local maintainer entrypoint for cutting releases
- Support these user-facing commands in the first iteration:
  - `make release-patch`
  - `make release-minor`
  - `make release VERSION=X.Y.Z`
- Restrict automated releases to `main`
- Require the release command to fail closed on unsafe repo state
- Keep release publication separate from `main` snapshot publication
- Treat `main` snapshot publication as an explicit release side effect when the release commit is pushed
- Gate release initiation on green required checks for the pre-bump `origin/main` commit
- Use [`pyproject.toml`](../../../pyproject.toml) as the canonical release version source
- Use [`CHANGELOG.md`](../../../CHANGELOG.md) as the canonical GitHub Release notes source
- Make the release helper resumable across partial-failure states
- Add a short maintainer-facing release-process document that matches the actual repo behavior
- Keep the first automated release scope to Docker publishing plus GitHub Release publication
- Keep source docs and maintainer release/publishing docs aligned with the release workflow

## Non-Goals

- Automate merging or syncing `dev` into `main`
- Redesign all existing CI workflows into reusable workflow primitives
- Expand the release image set beyond what [`.github/workflows/publish-docker.yml`](../../../.github/workflows/publish-docker.yml) already publishes
- Include PyPI in the first automated release path
- Invent a full semantic-release or release-train system
- Replace [`Docs/Release_Checklist.md`](../../../Docs/Release_Checklist.md) with a smaller document

## Current State

### Existing release artifact publishing

[`.github/workflows/publish-docker.yml`](../../../.github/workflows/publish-docker.yml):

- triggers on `release.published` and `workflow_dispatch`
- publishes release images to GHCR and Docker Hub
- currently publishes exactly these release image variants:
  - `app`
  - `worker`
  - `audio-worker`

This workflow already represents the remote “publish release images” half of the desired system.

### Existing snapshot publishing

[`.github/workflows/publish-ghcr-main.yml`](../../../.github/workflows/publish-ghcr-main.yml):

- triggers on `push` to `main`
- publishes rolling GHCR snapshots
- currently covers:
  - `app`
  - `webui`
  - `admin-ui`

This workflow is already the source of truth for `main` snapshots and must remain distinct from formal releases.

### Existing PyPI publish coupling

[`.github/workflows/publish-pypi.yml`](../../../.github/workflows/publish-pypi.yml):

- triggers on `release.published`
- also supports `workflow_dispatch`

That means any automation that publishes a GitHub Release today will also publish to PyPI. This is incompatible with the desired first release automation boundary unless the workflow trigger is changed.

### Existing version drift

The canonical package version is currently defined in [`pyproject.toml`](../../../pyproject.toml), but not every visible version reference follows it.

Examples:

- [`pyproject.toml`](../../../pyproject.toml) currently reports `0.1.30`
- [`README.md`](../../../README.md) currently describes the release line as `0.3.1`
- some CLI modules still contain hard-coded version strings

The release design must treat version drift as a first-class problem, not an incidental cleanup.

Additional tracked documentation drift already exists in the docs toolchain:

- [`Docs/mkdocs.yml`](../../../Docs/mkdocs.yml) still contains an older site version string
- maintainer publishing docs such as [`Docs/Development/PyPI_Publishing.md`](../../../Docs/Development/PyPI_Publishing.md) still describe release-triggered PyPI publication

Generated documentation output under `Docs/site/` is not release source material and must remain ignored. The release design therefore needs concrete rules for source docs and maintainer docs, not generated-site cleanup.

### Existing release notes source

[`CHANGELOG.md`](../../../CHANGELOG.md) already has an `Unreleased` section and versioned sections, which makes it the best source for GitHub Release notes. However, existing sections already show signs of duplication, so the release helper cannot assume that changelog input is always clean.

## Proposed Design

### 1. Define the release boundary explicitly

The release command will support only one release source branch:

- `main`

This means:

- the helper must fail if the current branch is not `main`
- the helper must fail if the local branch is behind or diverged from `origin/main`
- the helper does not perform merges or decide what from `dev` should be released

Maintainer workflow:

1. prepare or merge releasable work onto `main`
2. run the local release command from `main`
3. let GitHub Actions publish artifacts after the GitHub Release is published

This keeps release scope crisp and prevents the release command from becoming a policy engine for cross-branch promotion.

### 2. Decouple PyPI from the first automated release path

Before the new release command is documented as authoritative, [`.github/workflows/publish-pypi.yml`](../../../.github/workflows/publish-pypi.yml) should be adjusted so PyPI publication is not triggered by `release.published`.

First-iteration recommendation:

- keep PyPI publish available via `workflow_dispatch`
- remove `release.published` as an automatic trigger

Reason:

- the user-requested first boundary is “Docker + GitHub Release”
- leaving PyPI tied to release publication would make the operator doc false on day one

This is a small but required precondition for the automation to match the intended scope.

### 3. Keep published release images and `main` snapshots distinct

The release-process doc must explicitly describe the two different image contracts.

Release artifacts:

- produced by [`.github/workflows/publish-docker.yml`](../../../.github/workflows/publish-docker.yml)
- image set in v1:
  - `app`
  - `worker`
  - `audio-worker`

Rolling `main` snapshots:

- produced by [`.github/workflows/publish-ghcr-main.yml`](../../../.github/workflows/publish-ghcr-main.yml)
- image set:
  - `app`
  - `webui`
  - `admin-ui`

This matters because a vague “publishes Docker images” statement would cause maintainers to assume the release path also publishes the UI images, which it does not today.

It also means that pushing the release commit to `main` will intentionally republish the rolling `main` snapshot tags before the GitHub Release is published. That snapshot republish is an expected, first-class side effect of the release flow in v1, not an incidental background event.

### 4. Standardize the release version source

Canonical source of release version:

- [`pyproject.toml`](../../../pyproject.toml)

The release helper must:

- read the current version from [`pyproject.toml`](../../../pyproject.toml)
- compute the target version from:
  - patch bump
  - minor bump
  - explicit version input
- write the new version back to [`pyproject.toml`](../../../pyproject.toml)

The release helper should also update known human-facing release-line references that are meant to track the current release version.

First-pass cleanup scope:

- [`README.md`](../../../README.md) current release line
- [`Docs/mkdocs.yml`](../../../Docs/mkdocs.yml) docs-site version/copyright metadata

For v1, that means:

- `Docs/site/` remains generated output and is excluded from version control
- the helper must not stage or mutate `Docs/site/`
- maintainers can build the documentation site separately when publishing docs, but generated-site churn is not part of the release commit

This keeps release diffs bounded and avoids rewriting large portions of the generated site for unrelated reasons such as time-relative rendering or other build-time noise.

Related maintainer docs that describe release and publishing behavior are also in scope for alignment in this project, but they are workflow-alignment tasks rather than per-release version-bump targets. Minimum scope:

- [`Docs/Development/PyPI_Publishing.md`](../../../Docs/Development/PyPI_Publishing.md)
- [`Docs/Published/RELEASE_NOTES.md`](../../../Docs/Published/RELEASE_NOTES.md)

Hard-coded CLI versions need an explicit policy. The safest first-iteration rule is:

- update human-facing release-line references that claim to reflect the overall project release
- do not claim universal version unification until module-specific hard-coded CLI versions are either wired to package metadata or deliberately declared independent

This avoids silently overpromising what “version automation” covers.

### 5. Use `CHANGELOG.md` as the canonical release-notes source

The release helper will derive the GitHub Release body from the finalized release section in [`CHANGELOG.md`](../../../CHANGELOG.md).

Expected changelog flow:

1. read the `Unreleased` section
2. validate that it contains meaningful content
3. create a new release section for `X.Y.Z` with the current date
4. move the `Unreleased` contents into that new versioned section
5. reset `Unreleased` to empty section headings

The helper should fail closed if:

- `Unreleased` is empty
- the changelog layout is malformed
- the extracted release slice cannot be unambiguously identified
- exact duplicate bullets are detected within the same changelog subsection after simple normalization

For v1, simple normalization means:

- trim leading/trailing whitespace
- collapse internal runs of whitespace to a single space
- compare bullet text only within the same subsection (`Added`, `Changed`, `Fixed`, `Removed`)

Potential near-duplicates across different subsections should be reported as warnings in the dry-run/release output, not treated as hard failures.

This protects the GitHub Release body from being generated from junk input and acknowledges that the existing changelog already contains duplicate content in at least one versioned section.

### 6. Add a repo-owned local release helper

Add a repo-owned helper script:

- `Helper_Scripts/release.py`

Responsibilities:

- validate local and remote git state
- validate `gh` authentication
- compute target version
- update release metadata
- run focused preflight checks
- create the release commit
- create the annotated tag
- push `main` and the tag
- create and publish the GitHub Release
- print the resulting release URL and expected workflow handoff

Suggested command surface:

- `make release-patch`
- `make release-minor`
- `make release VERSION=X.Y.Z`

Optional but strongly recommended:

- `make release-dry-run VERSION=X.Y.Z`
- or a `DRY_RUN=1` flag on the main entrypoint

The dry-run mode should show:

- current version
- target version
- planned file edits
- derived changelog slice
- commit message
- tag name
- proposed release title/body

### 7. Fail closed on unsafe preconditions

The helper should stop before any irreversible action if any of these checks fail:

- current branch is not `main`
- working tree is dirty
- `origin/main` cannot be fetched
- local branch is behind or diverged from `origin/main`
- required files are missing
- `gh auth status` is not healthy
- the changelog cannot be promoted safely

Recommended first-iteration preflight checks after metadata edits but before commit/tag/push:

- build a candidate release body from the changelog
- run a focused packaging/doc sanity check appropriate to touched release metadata
- verify that the current pre-bump `origin/main` commit already has green required checks before proceeding, using the stable gate names documented in [`Docs/Development/CI_REQUIRED_GATES.md`](../../../Docs/Development/CI_REQUIRED_GATES.md) as the authoritative required-check contract for v1

The helper should allow a narrow explicit bypass only where justified, not a broad “ignore everything” mode by default.

The v1 gate is intentionally on the pre-bump `origin/main` commit, not on the generated release commit. The release commit is treated as a derived administrative change on top of already-green code, and the helper does not wait for a second CI cycle before publishing the GitHub Release in the first iteration.

This choice is deliberate:

- the helper validates the exact upstream code revision being released before any local release metadata is committed
- the generated release commit is not itself required to go green before `gh release create`
- the helper should query GitHub for the documented stable check names rather than discover branch-protection settings dynamically, which would add permission and configuration coupling

### 8. Make the release helper resumable by state detection

The release command will perform several irreversible steps. It must therefore be resumable by inspecting current state rather than assuming every rerun starts cleanly.

Minimum state cases to support:

- release commit exists locally but tag does not
- tag exists locally but has not been pushed
- remote tag exists but GitHub Release does not
- GitHub Release already exists for the tag

Desired behavior:

- continue from the earliest missing step
- avoid creating duplicate commits or tags
- exit cleanly and report success if the release is already fully published

Because the push step republishes `main` snapshots as a first-class side effect, reruns after a successful push but before GitHub Release creation should detect the already-pushed release commit and continue from GitHub Release creation rather than trying to create another commit or re-push new release state.

If `main` advances after the helper validates `origin/main` but before the push succeeds, the helper must hard-abort on non-fast-forward push failure. It must not auto-rebase, auto-merge, or retry against the new head inside the same release attempt. The maintainer must rerun the release command from a freshly fetched checkout so the required-check gate is re-evaluated against the new `origin/main` commit.

This makes the command robust against a failed network call, interrupted shell session, or a partial `gh` failure after push.

### 9. Keep the operator doc concise and authoritative

Add a concise maintainer-facing document:

- `Docs/Development/Release_Process.md`

This document should cover:

- what triggers publishing
- which artifacts are release artifacts versus `main` snapshots
- the authoritative local release commands
- what the command does
- recovery and retry behavior
- explicit boundary note that PyPI is outside the first automated release path

[`Docs/Release_Checklist.md`](../../../Docs/Release_Checklist.md) should remain the broad release-readiness checklist and be referenced as supporting material, not replaced.

## Command Flow

Nominal flow:

1. verify branch, cleanliness, and remote state
2. verify `gh` authentication
3. compute target version
4. update [`pyproject.toml`](../../../pyproject.toml)
5. update human-facing release-line references in scope
6. promote changelog `Unreleased` into `X.Y.Z`
7. run focused preflight checks
8. create release commit with a standard message such as `release: vX.Y.Z`
9. create annotated tag `vX.Y.Z`
10. push `main` and `vX.Y.Z`
11. create and publish the GitHub Release using the promoted changelog section as the body
12. print the release URL and expected Docker workflow handoff

Remote handoff:

- pushing the release commit to `main` triggers [`.github/workflows/publish-ghcr-main.yml`](../../../.github/workflows/publish-ghcr-main.yml), which republishes the rolling `main` snapshot images
- GitHub Release publication then triggers [`.github/workflows/publish-docker.yml`](../../../.github/workflows/publish-docker.yml), which publishes the formal release image set

## Verification Strategy

The implementation should be verified at three levels.

### Unit-level

- target-version parsing and bump logic
- changelog section detection and promotion
- duplicate-bullet detection for release-body candidates
- resumable state detection

### Integration-level

- dry-run against a fixture repo state
- successful release path against a local temporary repo with mocked `gh`
- partial-state rerun behavior:
  - tag exists, no release
  - release exists already

### Manual validation

- verify the new local command from a clean `main` checkout
- confirm that the helper blocks when required checks on the pre-bump `origin/main` commit are not green
- confirm that pushing the release commit republishes the expected `main` snapshots before the GitHub Release exists
- confirm that GitHub Release publication triggers Docker publishing only
- confirm that PyPI does not publish automatically once its workflow trigger is narrowed
- confirm that docs source updates stay in sync with the release version and `Docs/site/` remains untracked generated output
- confirm that the release-process doc matches actual repository behavior after the code lands

## Risks

- Hidden version references may still drift if the first iteration updates too small a set of files
- Existing changelog quality issues may cause the helper to fail frequently until changelog discipline improves
- Release automation that commits and pushes automatically can be dangerous if preflight checks are too weak
- GitHub CLI auth or API behavior may differ between maintainer environments unless preflight is explicit

## Mitigations

- keep the first release scope narrow and explicit
- fail closed on ambiguity instead of trying to auto-correct everything
- provide a dry-run mode
- make the helper resumable
- document artifact boundaries precisely
- keep PyPI outside the automated release path until intentionally added

## Open Questions

- Should module-specific CLI version strings be unified now or deferred to a separate cleanup task?

## Recommendation

Proceed in two tightly scoped phases:

1. Align the repo’s release boundaries with the intended workflow:
   - narrow PyPI publish triggers
   - define `main` as the only supported release branch
   - decide the first-pass version-reference scope
2. Implement the repo-owned release helper and the concise maintainer release-process doc

This yields a release workflow that is materially more automated without pretending the repository already has a unified release-management system.
