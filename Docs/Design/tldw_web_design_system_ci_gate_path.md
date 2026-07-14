# tldw Web Design-System CI Gate Path

Date: 2026-07-04

## Purpose

Move the product-state guard from local/report-mode usage into CI without
blocking unrelated work on legacy baseline debt.

The guard command is:

```bash
cd apps/packages/ui
bun run verify:design-system-state
```

## Current Behavior

`verify:design-system-state` fails on unbaselined product-state findings and
invalid baseline rows. It reports active baseline exceptions and stale baseline
rows so migration PRs can shrink debt deliberately.

## Gate Stages

1. **Report-only PR signal**
   - Owner: `.github/workflows/frontend-required.yml`
   - Trigger: `tldw_frontend_changed` from
     [`Helper_Scripts/ci/path_classifier.py`](../../Helper_Scripts/ci/path_classifier.py),
     currently covering `apps/tldw-frontend/**`, `apps/packages/ui/**`,
     `apps/extension/**`, `apps/bun.lock`, and
     `apps/tldw-frontend/package-lock.json`. Docs-only changes do not trigger
     this signal unless the classifier is intentionally extended.
   - Behavior: install `apps` dependencies, run `bun run verify:design-system-state`
     from `apps/packages/ui`, and keep `continue-on-error: true`.
   - Exit criteria: at least one week of stable CI runtime and no false positives.

2. **Required new-finding gate**
   - Owner: `frontend-required`.
   - Behavior: remove `continue-on-error`; the existing verifier blocks new
     unbaselined findings and invalid baseline rows while allowing current
     baseline exceptions.
   - Entry criteria: latest `dev` passes the verifier with only accepted
     baseline exceptions and no blocked findings.

3. **Required stale-baseline cleanup**
   - Owner:
     [`apps/packages/ui/scripts/verify-design-system-product-state.mjs`](../../apps/packages/ui/scripts/verify-design-system-product-state.mjs)
     plus `frontend-required`.
   - Behavior: make stale baseline rows fail CI.
   - Entry criteria: all open migration PRs use the stale-row cleanup workflow
     in
     [tldw_web_design_system_baseline_reporting.md](tldw_web_design_system_baseline_reporting.md).

4. **Area-zero enforcement**
   - Owner: tracker issues under TASK-45.44 / GitHub epic #1655.
   - Behavior: once a product-area issue reaches zero exceptions, any new
     baseline row for that area requires a new owner issue, replacement target,
     and migration queue.

## PR Requirements

Design-system migration PRs must record:

- verifier command and result,
- before/after baseline count for the touched area,
- stale baseline cleanup result,
- PR link in the GitHub tracker issue,
- Backlog task verification notes.

## Non-Goals

- Do not fail all CI on the current legacy baseline.
- Do not add a new workflow when `frontend-required` can own the gate.
- Do not add new dependencies for reporting.
