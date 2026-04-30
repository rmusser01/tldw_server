# Phase 4 Readiness Gate

**Date:** 2026-04-25

**Status:** Gate defined; Phase 4 implementation remains blocked.

## Purpose

Summarize what must be true before Phase 4 moves from planning to implementation. This prevents later roadmap work from racing ahead of Phase 2/3 closeout or mixing unrelated changes into active PR stabilization.

Consolidated handoff:

- `Docs/superpowers/reviews/phase3-pilots/2026-04-25-phase3-phase4-remaining-work-handoff.md`

## Global Blockers

Phase 4 implementation should not start until one of these is true:

- Phase 2 closeout PRs are merged; or
- maintainers explicitly accept the open Phase 2 PR heads as stable implementation bases.

Phase 4 implementation should also wait until one of these is true:

- PR `#1125` is merged; or
- maintainers explicitly accept PR `#1125` as a stable base for dependent API cleanup.

Phase 3 dependency gate:

- Phase 3.1 response-envelope helper contract accepted.
- Phase 3.2 pagination helper contract accepted.
- Phase 3.4 auth dependency helper contract accepted.
- The `skills` pilot is complete or intentionally paused with documented blockers.

## Phase-Specific Entry Criteria

### Phase 4.1 Coverage Ratchet

Required before implementation:

- Clean accepted base chosen.
- Backend baseline measured with `--cov-fail-under=0`.
- Frontend baseline measured separately.
- Maintainers accept first threshold bump.

Ready artifact:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-coverage-ratchet-measurement-packet.md`

### Phase 4.2 Deployment Docs

Required before implementation:

- Docs owner confirms source/published flow.
- First source-doc slice accepted.
- HA guide publishing status decided.
- Monitoring publishing shape decided.

Ready artifact:

- `Docs/superpowers/plans/2026-04-25-phase4-2-deployment-docs-refresh-plan.md`

### Phase 4.3 DB Decomposition

Required before implementation:

- Maintainers accept `Prompts_DB.py` or choose an alternate first DB file.
- Clean accepted base chosen.
- Baseline prompt DB tests pass.
- Transaction boundaries stay in place for the first extraction.

Ready artifact:

- `Docs/superpowers/plans/2026-04-25-phase4-3-prompts-db-decomposition-plan.md`

### Phase 4.4 Endpoint Decomposition

Required before implementation:

- Maintainers accept `storage.py` user-owned JSON routes or choose an alternate first endpoint family.
- Clean accepted base chosen.
- OpenAPI path baseline captured.
- File download and admin quota routes remain excluded from the first split.

Ready artifact:

- `Docs/superpowers/plans/2026-04-25-phase4-4-storage-endpoint-decomposition-plan.md`

### Phase 4.5 API Versioning

Required before implementation:

- Maintainers accept or amend the policy decision packet.
- OpenAPI owner decides how opt-in envelope variants should appear in `v1` OpenAPI.
- Frontend owner decides whether any pilot client sends the envelope opt-in header.

Ready artifact:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-policy-decision-packet.md`

### Phase 4.6 OpenAPI Contract Testing

Required before implementation:

- Phase 3.1 helper schema names are stable.
- Phase 3.2 pagination schema names are stable.
- Maintainers decide whether generated `app.openapi()` or checked-in `apps/extension/openapi.json` is canonical for CI.
- Existing `bun run verify:openapi` remains green.

Ready artifact:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-openapi-contract-testing-plan.md`

## Recommended Execution Order After Gate Opens

1. Accept Phase 4.5 versioning policy.
2. Stabilize Phase 4.6 OpenAPI contract decisions around Phase 3 helper schemas.
3. Measure Phase 4.1 coverage baseline and make at most a small floor bump.
4. Execute Phase 4.2 deployment docs refresh.
5. Start Phase 4.3 `Prompts_DB.py` decomposition.
6. Start Phase 4.4 `storage.py` route split.

## Do Not Do

- Do not start DB or endpoint decomposition from this dirty workspace.
- Do not raise coverage thresholds without a fresh baseline.
- Do not make standard envelopes default in `v1` before policy acceptance.
- Do not force OpenAPI strict mode while reviewed exceptions remain.
- Do not combine docs refresh, coverage ratchet, DB decomposition, and endpoint decomposition in one PR.

## Handoff Checklist

- [ ] Phase 2 base accepted.
- [ ] PR `#1125` base accepted.
- [ ] Phase 3 helper contracts accepted.
- [ ] `skills` pilot status accepted.
- [ ] Maintainers choose first Phase 4 implementation item.
- [ ] Clean worktree is created for the chosen implementation item.
