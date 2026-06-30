# Release Process

This is the authoritative local operator path for cutting a release. Use [Docs/Release_Checklist.md](../Release_Checklist.md) for the broad readiness checklist before you start, but run the release from this document.

## Supported Commands

- `make release` runs the default patch flow.
- `make release-patch` cuts the next patch release.
- `make release-minor` cuts the next minor release.

All three commands call `Helper_Scripts/release.py`, which:

1. fetches `origin/main`
2. requires the local branch to be `main` and exactly aligned with pre-bump `origin/main`
3. reads required checks from `Docs/Development/CI_REQUIRED_GATES.md`
4. verifies those required checks are green on the pre-bump `origin/main` commit
5. updates release metadata and source docs, creates the release commit, tags it, pushes `main`, then creates the GitHub Release

## Branch Rule

`main` is the only supported source branch for this rollout. The helper aborts on any other branch because the required-check gate is defined against `origin/main`, the release commit is meant to land directly on `main`, and pushing that release commit republishes `main` snapshots before the GitHub Release drives formal publication.

## Artifact Boundary

Treat main snapshots republish as a first-class side effect of the release command.

- Formal release artifacts are the `app`, `worker`, and `audio-worker` images published by the GitHub Release workflow with versioned release tags.
- `main` snapshots are the rolling GHCR `app`, `webui`, and `admin-ui` images republished by pushes to `main`.
- Pushing the release commit republishes `main` snapshots before GitHub Release publication triggers the formal Docker release artifacts.
- `Docs/site/` is generated documentation-site output, not release source material. It remains ignored and is not staged by the release helper.

## Retry And Recovery

The helper is intentionally resumable for narrow failure states.

- If the push to `origin/main` fails with non-fast-forward, stop and rerun from a fresh fetch; do not force-push around it.
- If the release commit exists locally or the tag already exists locally, rerun the same command after confirming the worktree is clean.
- If the remote tag exists but the GitHub Release does not, rerunning recovers by creating the missing GitHub Release instead of cutting a second release.
- If the GitHub Release already exists, treat the release as already published and verify artifacts rather than retrying the cut.

## PyPI Boundary

PyPI is outside the automatic GitHub Release path in this rollout. The release command only handles the local release commit, tag push, snapshot republish, and GitHub Release publication that drives Docker release images. PyPI remains a narrow manual boundary: run `publish-pypi.yml` manually only after the release is published and you are ready to upload the matching package.
