# Maintainer Decision Checklist For Phase 3 And Phase 4

**Date:** 2026-04-25

**Status:** Checklist ready for maintainer review; no runtime work approved by this document.

## Purpose

Collect the remaining yes/no decisions that block Phase 3 and Phase 4 implementation. This is the short review surface for maintainers who do not need to read every supporting inventory before choosing the next implementation step.

## How To Use This Checklist

For each decision:

- accept the recommended default;
- amend it with a concrete replacement; or
- reject it and mark the dependent work blocked.

Do not start implementation until the relevant decision row is resolved and the PR base is refreshed.

## Base Stability Decisions

| ID | Decision | Recommended default | Unlocks | Source |
| --- | --- | --- | --- | --- |
| `BASE-1` | Are Phase 2 PR heads stable enough for dependent work? | No, wait for merge or explicit maintainer acceptance after fresh `gh pr view` checks. | Phase 3 shared helpers, Phase 4 implementation. | `2026-04-25-phase3-readiness-gate.md` |
| `BASE-2` | Is PR `#1125` stable enough as the sanitized-error base? | No, leave PR fixes with the PR-fixing fork unless redirected. | Phase 3.1 response envelopes, Phase 3.3-dependent error tests. | `2026-04-25-phase3-phase4-remaining-work-handoff.md` |
| `BASE-3` | Should this dirty workspace be used for runtime implementation? | No, use clean worktrees per implementation item. | Any code PR. | `2026-04-25-phase3-phase4-remaining-work-handoff.md` |

## Phase 3 Contract Decisions

| ID | Decision | Recommended default | Unlocks | Source |
| --- | --- | --- | --- | --- |
| `P3-1` | Should `/api/v1/` remain legacy-default? | Yes. Do not make envelopes default in `v1`. | Response-envelope helper PR and pilot policy. | `2026-04-25-api-versioning-policy-decision-packet.md` |
| `P3-2` | What is the public response-envelope opt-in? | Header: `X-TLDW-Response-Envelope: v1`. Query flag stays debug/test-only unless approved. | Phase 3.1 helper and `skills` pilot. | `2026-04-25-skills-pilot-execution-packet.md` |
| `P3-3` | Which routes are envelope-exempt by default? | Streaming, file downloads, webhooks, WebSockets, `204 No Content`, OpenAI/provider-compatible payloads. | Exemption tests and OpenAPI contract plan. | `2026-04-25-api-versioning-policy-decision-packet.md` |
| `P3-4` | Where does canonical pagination metadata live during `v1` pilot? | `meta.pagination` only for opt-in envelope responses. | Phase 3.2 helpers and `skills` list pilot. | `2026-04-25-helper-contract-spec.md` in api-pagination reviews. |
| `P3-5` | Are legacy pagination aliases still accepted? | Yes. Keep route-family aliases during `v1`. | Backwards-compatible pagination helpers. | `2026-04-25-pagination-inventory.md` |
| `P3-6` | What is the first route-family pilot? | `skills`. | Phase 3.1, 3.2, and 3.4 pilot work. | `2026-04-25-skills-pilot-execution-packet.md` |
| `P3-7` | Should frontend callers become envelope-aware? | No. Client domain unwraps envelope and components keep legacy-shaped data. | `skills` frontend pilot tests. | `2026-04-25-skills-pilot-execution-packet.md` |
| `P3-8` | What auth alias behavior is accepted? | Add principal-returning aliases; preserve existing lower-case guard factories. | Phase 3.4 helper PR. | `2026-04-25-helper-contract-spec.md` in auth-dependencies reviews. |
| `P3-9` | Should `require_token_scope(...)` return a principal? | No. Preserve guard-returning-`None` behavior. | Auth dependency cleanup without behavior change. | `2026-04-25-phase3-readiness-gate.md` |

## Phase 4 Decisions

| ID | Decision | Recommended default | Unlocks | Source |
| --- | --- | --- | --- | --- |
| `P4-1` | Which API versioning policy should be adopted? | Accept `v1` legacy-default and reserve defaults for future `/api/v2/`. | `Docs/API/api-versioning-strategy.md` update. | `2026-04-25-api-versioning-policy-decision-packet.md` |
| `P4-2` | What is the canonical OpenAPI source for CI? | Generated `app.openapi()` unless maintainers require a checked-in snapshot refresh. | Phase 4.6 contract tests. | `2026-04-25-openapi-contract-testing-plan.md` |
| `P4-3` | Should strict OpenAPI mode become required now? | No. Trial strict mode only after known exceptions are resolved or documented. | Future strict OpenAPI gate. | `2026-04-25-openapi-contract-testing-plan.md` |
| `P4-4` | Should coverage be raised before measuring? | No. Measure clean-base backend coverage with `--cov-fail-under=0` first. | Phase 4.1 ratchet PR. | `2026-04-25-coverage-ratchet-measurement-packet.md` |
| `P4-5` | Are backend and frontend coverage thresholds combined? | No. Keep separate baselines and policies. | Coverage baseline note and ratchet design. | `2026-04-25-coverage-ratchet-measurement-packet.md` |
| `P4-6` | What is the docs publishing flow? | Edit source docs, refresh `Docs/Published` with `Helper_Scripts/refresh_docs_published.sh`. | Phase 4.2 deployment docs refresh. | `2026-04-25-phase4-2-deployment-docs-refresh-plan.md` |
| `P4-7` | Is the HA guide canonical or draft? | Needs owner decision. Do not publish-status-change it until decided. | Phase 4.2 HA docs work. | `2026-04-25-deployment-docs-inventory.md` |
| `P4-8` | Is top-level published Monitoring intentional? | Needs owner decision. Current script promotes it to `Docs/Published/Monitoring`. | Phase 4.2 monitoring docs work. | `2026-04-25-deployment-docs-inventory.md` |
| `P4-9` | What is the first DB decomposition target? | `Prompts_DB.py`. | Phase 4.3 DB decomposition PR. | `2026-04-25-phase4-3-prompts-db-decomposition-plan.md` |
| `P4-10` | What is the first endpoint decomposition target? | `storage.py` user-owned JSON routes. | Phase 4.4 endpoint decomposition PR. | `2026-04-25-phase4-4-storage-endpoint-decomposition-plan.md` |

## Recommended Approval Bundle

If maintainers want the lowest-risk path, approve these together:

- `BASE-1`: wait for Phase 2 merge or explicit stable-base acceptance after fresh PR checks.
- `BASE-2`: wait for PR `#1125` merge or explicit stable-base acceptance after fresh PR checks.
- `P3-1` through `P3-9`: accept recommended Phase 3 contract defaults.
- `P4-1` through `P4-5`: accept API versioning/OpenAPI/coverage guardrails.
- `P4-9` and `P4-10`: accept first DB and endpoint targets, but only after Phase 3 stabilizes.

Hold for owner review:

- `P4-6`, `P4-7`, and `P4-8`, because docs publishing and monitoring/HA status need docs-owner confirmation.

## First Implementation After Approval

After the base gates are accepted, start with:

1. Phase 3.1 shared response-envelope helpers.
2. Phase 3.2 shared pagination helpers.
3. Phase 3.4 auth aliases and contract tests.
4. `skills` response-envelope and pagination pilot.

Do not start Phase 4 implementation before the Phase 3 helper contracts and `skills` pilot status are accepted.

## Handoff Checklist

- [ ] Fresh PR status refresh completed.
- [ ] Base stability decisions resolved.
- [ ] Phase 3 contract decisions resolved.
- [ ] Phase 4 policy decisions resolved.
- [ ] Docs-owner decisions resolved for deployment docs.
- [ ] Clean worktree created for the selected implementation item.
