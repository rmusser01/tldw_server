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
5. updates release metadata and source docs, creates the release commit, tags it,
   pushes `main`, then creates an existing draft GitHub Release

Install the reviewed uv 0.12.7 before starting. Before staging the release, the
helper refreshes `uv.lock` offline for the bumped root package version, checks
it, and includes it in the release commit. An unavailable or incorrect uv
version, incomplete local cache, or lock validation failure stops before
commit/tag/push; inspect the prepared metadata and lock diff before retrying.

The helper deliberately stops at a draft. Review the notes and follow
[`Software_Supply_Chain.md`](Software_Supply_Chain.md) before publication.

## Branch Rule

`main` is the only supported source branch for this rollout. The helper aborts on any other branch because the required-check gate is defined against `origin/main`, the release commit is meant to land directly on `main`, and pushing that release commit triggers scan-gated `main` snapshots independently of manually dispatched formal publication.

## Artifact Boundary

Treat main snapshots republish as a first-class side effect of the release command.

- Formal release artifacts are the `app`, `worker`, and `audio-worker` images
  published by the manually dispatched `publish-docker.yml` workflow with
  versioned release tags.
- During the frontend licensing freeze, the only `main` snapshot is the rolling
  GHCR GPL backend `app` image. WebUI and Admin UI remain build-checked but are
  not published.
- Pushing the release commit republishes `main` snapshots. It does not publish
  the draft release or trigger formal Docker artifacts.
- `Docs/_site/` is generated documentation-site output, not release source material. It remains ignored and is not staged by the release helper.

## Verified Container Publication

After `make release*` succeeds, confirm the tag is a stable semantic version,
the GitHub Release is still a draft, and its notes are correct. Dispatch on the
tag itself; the confirmation must be the literal `publish <tag>`:

```bash
RELEASE_TAG=v0.1.35
gh release view "$RELEASE_TAG" --json isDraft,tagName
gh workflow run publish-docker.yml \
  --ref "$RELEASE_TAG" \
  -f release_tag="$RELEASE_TAG" \
  -f confirmation="publish $RELEASE_TAG"
```

The workflow admits source dependencies, builds and scans unique candidate
digests, verifies the full release-evidence set, and creates full-version
aliases for all three backend images. Only after every version
alias matches does it move the major, minor, and `latest` floating aliases.
Release evidence is uploaded and verified before the workflow changes the
draft to public. WebUI and Admin UI are build-and-scan-only and are not
published.

## Retry And Recovery

The helper is intentionally resumable for narrow failure states.

- If the push to `origin/main` fails with non-fast-forward, stop and rerun from a fresh fetch; do not force-push around it.
- If the release commit exists locally or the tag already exists locally, rerun the same command after confirming the worktree is clean.
- If the remote tag exists but the GitHub Release does not, rerunning recovers
  by creating the missing draft instead of cutting a second release.
- If a draft already exists, do not rerun the release helper. Inspect the
  draft, then dispatch or resume `publish-docker.yml` for that exact tag.
- If the GitHub Release is already public, treat it as already published and
  verify its assets, image digests, and attestations rather than retrying the
  cut.
- If `publish-docker.yml` fails, keep the release as a draft. Do not move tags
  or publish it by hand; retain the evidence, correct the failing gate, and
  resume failed jobs against the same admitted candidates. Full-version
  promotion rejects a different existing digest. If a rebuild changes the
  digest after partial promotion, stop and investigate rather than overwriting
  or deleting the version tag.

## PyPI Boundary

PyPI is outside the container-release path. The release command handles the
local release commit, tag push, snapshot republish, and draft creation. PyPI
publishing is handled by `publish-pypi.yml`: manual dispatch remains available,
and pushes to `main` that change `pyproject.toml` may publish only when the
version is missing from PyPI and the workflow source, test, build, checksum,
and trusted-publishing gates pass.
