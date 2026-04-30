# Phase 3 Readiness Gate

**Date:** 2026-04-25

**Status:** Gate definition complete; implementation remains blocked.

## Purpose

Define the exact gate for starting Phase 3.1, Phase 3.2, and Phase 3.4 runtime implementation. This prevents broad API contract work from starting on unstable PR bases or while sanitized error behavior is still changing.

## Current Read-Only PR Refresh

Checked with `gh pr view` on 2026-04-25.

| PR | Topic | State | Draft | Merge state | Gate status |
| --- | --- | --- | --- | --- | --- |
| `#1115` | Phase 2.3 ChaChaNotes | Open | No | Unknown | Not stable. `build-sbom` failed; full-suite jobs were cancelled. |
| `#1121` | Phase 2.4 Config sections | Open | No | Unknown | Not stable. `build-sbom` failed; full-suite jobs were cancelled. |
| `#1122` | Phase 2.2 Router groups | Open | No | Unknown | Not stable. `build-sbom` failed; full-suite jobs were cancelled. |
| `#1123` | Phase 2.1 Lifespan extraction | Open | No | Unknown | Not stable. `build-sbom` and `UX Smoke Gate` failed; full-suite jobs were cancelled. |
| `#1120` | Phase 2.5 Unified errors | Open | No | Unknown | Not stable. `build-sbom` and `UX Smoke Gate` failed; full-suite jobs were cancelled. |
| `#1125` | Phase 3.3 Error-handler adoption | Open | Yes | Unstable | Not stable. `onboarding-docs-gate`, `run-pre-commit`, `UX Smoke Gate`, and `Jobs (PostgreSQL)` failed; full-suite jobs were still in progress. |

Note: `build-sbom` is now green on PR `#1125`, which differs from the older tracker note. It is still failing on the older Phase 2 PRs listed above.

## Required Green Conditions

Before starting Phase 3 runtime code:

- PR `#1125` is merged, or maintainers explicitly accept it as the stable error-handling base.
- Phase 2 PRs `#1115`, `#1120`, `#1121`, `#1122`, and `#1123` are merged, or maintainers explicitly accept their current heads as stable bases.
- The Phase 3.1 response-envelope rollout decision is accepted:
  - default legacy payloads
  - header opt-in `X-TLDW-Response-Envelope: v1`
  - no default breaking envelope
  - explicit exemptions for streaming, files, `204`, webhooks, and provider-compatible routes
- The Phase 3.2 pagination compatibility decision is accepted:
  - canonical first-party `limit`/`offset`
  - legacy aliases accepted during migration
  - `skills` list as first offset pilot
- The Phase 3.4 auth alias decision is accepted:
  - `CurrentPrincipal` or equivalent principal-returning alias
  - existing lower-case role/permission/scope factories preserved unless maintainers choose explicit aliases
  - `require_token_scope(...)` remains a guard returning `None`
- Frontend owner accepts the temporary `skills` client opt-in and unwrap approach.

## Start Criteria By Phase

### Phase 3.1

Can start when:

- PR `#1125` is stable enough that sanitized error details are no longer moving.
- The envelope helper contract spec is accepted.
- Maintainers accept the header opt-in rollout.

First implementation PR:

- shared response envelope schemas/builders only
- no endpoint behavior changes
- helper unit tests

Do not start with:

- repo-wide endpoint wrapping
- default envelope response changes
- provider-compatible route shape changes

### Phase 3.2

Can start when:

- Phase 3.1 has a stable place for `meta.pagination` or maintainers accept that pagination helpers ship independently first.
- The pagination helper contract spec is accepted.
- `skills` remains the selected first offset pilot.

First implementation PR:

- shared pagination schemas/helpers
- backwards-compatible `build_link_header(...)`
- helper unit tests

Do not start with:

- route-wide alias rejection
- provider pagination changes
- page-style route rewrites

### Phase 3.4

Can start when:

- Phase 2 AuthNZ-adjacent PRs are stable.
- The auth alias/helper contract spec is accepted.
- Contract tests prove request-state behavior before endpoint migration.

First implementation PR:

- alias/helper exports and contract tests only
- no endpoint migration unless the helper PR is already green

Do not start with:

- removing `get_current_user`
- removing `get_current_active_user`
- changing TEST_MODE override behavior
- changing `require_token_scope(...)` return behavior

## Recommended Implementation Order

1. Phase 3.1 shared response-envelope helpers.
2. Phase 3.2 shared pagination helpers.
3. Phase 3.4 auth aliases and contract tests.
4. `skills` response-envelope and pagination pilot.
5. `skills` auth cleanup pilot.
6. `slides` or `data_tables` as the second route-family pilot.

This keeps shared primitives separate from endpoint behavior changes and avoids combining response-shape, pagination, and auth behavior in one broad PR.

## Explicit Blockers To Recheck

Before implementation starts, rerun read-only PR checks for:

```bash
gh pr view 1115 --repo rmusser01/tldw_server --json state,isDraft,mergeStateStatus,statusCheckRollup
gh pr view 1120 --repo rmusser01/tldw_server --json state,isDraft,mergeStateStatus,statusCheckRollup
gh pr view 1121 --repo rmusser01/tldw_server --json state,isDraft,mergeStateStatus,statusCheckRollup
gh pr view 1122 --repo rmusser01/tldw_server --json state,isDraft,mergeStateStatus,statusCheckRollup
gh pr view 1123 --repo rmusser01/tldw_server --json state,isDraft,mergeStateStatus,statusCheckRollup
gh pr view 1125 --repo rmusser01/tldw_server --json state,isDraft,mergeStateStatus,statusCheckRollup
```

Do not rely on the 2026-04-25 snapshot after new pushes land.

## Handoff Checklist

- [x] Consolidated Phase 3 and Phase 4 remaining-work handoff exists: `Docs/superpowers/reviews/phase3-pilots/2026-04-25-phase3-phase4-remaining-work-handoff.md`.
- [ ] PR base status refreshed after latest pushes.
- [ ] Maintainer accepts the response-envelope rollout switch.
- [ ] Maintainer accepts pagination alias precedence and first pilot.
- [ ] Maintainer accepts auth alias naming and `require_token_scope(...)` guard behavior.
- [ ] Frontend owner accepts `skills` client opt-in and unwrap strategy.
- [ ] A clean implementation worktree is created from the accepted base.
- [ ] Shared-helper PR starts before endpoint migrations.
- [ ] Runtime tests and Bandit commands are run in the implementation PR, not inferred from this planning artifact.
