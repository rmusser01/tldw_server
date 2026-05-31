# Phase 4.2 Deployment Docs Refresh Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. This is a Phase 4 docs plan. Do not rewrite deployment docs until Phase 2/3 closeout is stable or maintainers explicitly approve docs-only work.

**Goal:** Refresh deployment and getting-started docs with a clear canonical source, publishing flow, validation gate, and owner review path.

**Architecture:** Treat `Docs/Getting_Started/` and `Docs/Deployment/` as source docs. Treat `Docs/Published/` as generated/curated output refreshed by `Helper_Scripts/refresh_docs_published.sh`. Edit source docs first, regenerate published docs, then run the onboarding docs gate locally.

**Tech Stack:** Markdown, MkDocs, pytest docs tests, onboarding docs scripts

---

## Current Signals

- Inventory exists: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-deployment-docs-inventory.md`.
- CI workflow exists: `.github/workflows/onboarding-docs-gate.yml`.
- Published docs refresh script exists: `Helper_Scripts/refresh_docs_published.sh`.
- Docs tests auto-refresh `Docs/Published` in `tldw_Server_API/tests/Docs/conftest.py` when required published mirrors are absent.
- `Docs/Deployment/Monitoring` is promoted to `Docs/Published/Monitoring` by the refresh script, not copied under `Docs/Published/Deployment/Monitoring`.

## Stage 1: Owner Decisions And Scope Lock

**Goal:** Confirm what is allowed to change before prose edits.
**Success Criteria:** Maintainers accept the source/published flow and the first deployment-mode slice.
**Tests:** None.
**Status:** Complete

- [ ] Confirm `Docs/Getting_Started/` and `Docs/Deployment/` are canonical source docs.
- [ ] Confirm `Docs/Published/` should be regenerated with `Helper_Scripts/refresh_docs_published.sh`, not edited by hand except for preserved `index.md` files.
- [ ] Decide whether `Docs/Getting_Started/Getting-Started-with-HA-Guide.md` is canonical, draft, or onboarding-only.
- [ ] Decide whether monitoring docs should remain published as top-level `Docs/Published/Monitoring`.
- [ ] Pick the first source-doc slice:
  - recommended: `Docs/Getting_Started/README.md` deployment-mode matrix
  - alternate: `Docs/Deployment/First_Time_Production_Setup.md` production checklist
  - alternate: `Docs/Deployment/horizontal-scaling.md` plus HA guide alignment

Status note: maintainer continuation accepted the docs-only Phase 4.2 slice. This tranche uses `Docs/Getting_Started/README.md` as the source edit and preserves the existing HA and Monitoring publishing decisions without changing them.

## Stage 2: Source-Only Refresh

**Goal:** Make one small source-doc improvement without touching runtime behavior.
**Success Criteria:** One deployment-mode slice is clearer and still matches current commands.
**Tests:** Static command/path checks from Stage 4.
**Status:** Complete

Recommended first edit:

- Add or refine a deployment-mode matrix in `Docs/Getting_Started/README.md`.

Minimum coverage:

- local single-user
- Docker single-user + WebUI
- Docker multi-user + Postgres
- production/horizontal scaling
- offline/air-gapped
- sidecar workers
- audio/GPU setup
- monitoring and operations

Implementation constraints:

- Do not rewrite all deployment docs in one PR.
- Do not alter setup commands without checking corresponding Makefile, Docker, or workflow references.
- Do not add hosted/commercial docs to `Docs/Published`.
- Do not edit service files under `Docs/Deployment/systemd` or `Docs/Deployment/launchd` in the first slice.

## Stage 3: Regenerate Published Docs

**Goal:** Refresh generated/curated published docs from source after source edits.
**Success Criteria:** Generated published changes are intentional and limited to the selected docs slice.
**Tests:** Refresh script exits 0.
**Status:** Complete

Run:

```bash
bash Helper_Scripts/refresh_docs_published.sh
```

Review:

- `git diff -- Docs/Published`
- confirm `index.md` files remain preserved
- confirm Monitoring is under `Docs/Published/Monitoring`
- confirm no private or draft-only docs were promoted accidentally

Status note: `Helper_Scripts/refresh_docs_published.sh` exits 0, but the full refresh currently surfaces unrelated source/published drift. This PR keeps only the matching `Docs/Published/Getting_Started/README.md` mirror for the selected source slice; the broader refresh drift remains out of scope.

## Stage 4: Local Docs Gate

**Goal:** Reproduce the required onboarding docs gate locally.
**Success Criteria:** Docs refresh, boundary checks, docs tests, and MkDocs build pass.
**Tests:** Commands below.
**Status:** Complete

Run:

```bash
source .venv/bin/activate
bash Helper_Scripts/refresh_docs_published.sh
python Helper_Scripts/docs/check_onboarding_command_boundaries.py
python Helper_Scripts/docs/check_onboarding_endpoint_drift.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/Docs
mkdocs build -f Docs/mkdocs.yml
```

Optional hygiene checks:

```bash
source .venv/bin/activate
python Helper_Scripts/docs/check_public_private_boundary.py
python Helper_Scripts/docs/check_top_guides_docs_path_hygiene.py
python Helper_Scripts/docs/check_readme_docs_path_hygiene.py
python Helper_Scripts/docs/check_docs_index_path_hygiene.py
```

Status note: required docs checks pass when the repo-root virtual environment is on `PATH`; direct venv Python without `PATH` fails two docs tests that spawn `python`. `mkdocs build -f Docs/mkdocs.yml` exits 0 with existing nav/link warnings. Optional public/private and path-hygiene checks pass.

## Stage 5: Owner Review Packet

**Goal:** Make docs review easy for the owner.
**Success Criteria:** PR description explains source edits, generated published changes, validation, and open owner decisions.
**Tests:** None.
**Status:** In Progress

Include in handoff:

- source docs changed
- generated `Docs/Published` files changed
- whether HA guide status changed
- whether monitoring publishing status changed
- exact docs-gate commands and results
- any commands that were inspected but intentionally not changed

## Out Of Scope

- Runtime deployment behavior changes.
- Docker Compose or Makefile changes.
- Service unit changes.
- Broad docs rewrites.
- Manual edits to generated published docs outside preserved landing pages.
- Fixing unrelated onboarding-docs CI failures from active PRs.

## Handoff Checklist

- [ ] Owner decisions are recorded.
- [ ] Source-doc slice is accepted.
- [ ] Published docs are refreshed through the script.
- [ ] Docs gate passes locally.
- [ ] Generated published diffs are reviewed before PR handoff.
