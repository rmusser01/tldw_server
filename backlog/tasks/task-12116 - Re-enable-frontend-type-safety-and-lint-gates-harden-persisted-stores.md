---
id: TASK-12116
title: 'Re-enable frontend type-safety and lint gates, harden persisted stores'
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-09-10 19:27'
labels:
  - tech-debt
  - high
  - frontend
  - ci
  - packages-ui
dependencies: []
documentation:
  - apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: High (safety nets disabled — the reason bug classes ship silently).** From the 2026-07-02 frontend audit (§9). All verified by direct read.

- `apps/tldw-frontend/tsconfig.json:11` and `apps/extension/tsconfig.json:9` — **`"strict": false`** in both apps. No null-safety on a ~1.2M-LOC shared surface.
- `apps/tldw-frontend/next.config.mjs:59` — **`typescript.ignoreBuildErrors: true`**. TS errors never fail the build or CI.
- `apps/tldw-frontend/eslint.config.mjs:78-84` — the newer **react-compiler-era `react-hooks` rules are disabled** (`immutability`, `purity`, `preserve-manual-memoization`, `refs`, `set-state-in-effect`, `static-components`, `use-memo`); `set-state-in-effect` in particular would flag effect-race bugs. **The classic `react-hooks/rules-of-hooks` is NOT globally disabled** — the `off` at `:118` is scoped to `e2e/**` only; the rule is active everywhere else via the `reactHooksRules` preset. `@typescript-eslint/no-explicit-any` is only `warn`.
- **Persisted stores lack `version`/`migrate`** (8 of 9): `playground-session`, `persona-buddy-shell`, `notes-dock`, `ui-mode`, `actor`, `quick-ingest-session`, `folder`, `feedback`, `acp-sessions`. The day someone adds `version:1` to reshape a store without a `migrate`, all users' persisted state is discarded; a field rename before then ships `undefined` into consumers.
- **Shared-code dependency skew**: frontend vs extension pin different majors of libraries that both feed `packages/ui` — `zustand ^5`/`^4`, `dexie-react-hooks ^4`/`^1.1.7`, `marked 17`/`15`, `d3-dsv 3`/`2`, `react ^18.3`/pinned `18.2`, TS `5.6`/`5.9`. (Zustand specifically is currently safe, but it is a standing hazard.)

This is a phased hardening ticket; land incrementally so each step keeps CI green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A TypeScript typecheck runs in CI and gates merges (either remove `ignoreBuildErrors` once `packages/ui` typechecks, or add a separate `tsc --noEmit` gate).
- [ ] #2 `strict` is turned on incrementally (start with `noImplicitAny`, then `strictNullChecks`), with a tracked path to `strict: true`.
- [ ] #3 `react-hooks/rules-of-hooks` is re-enabled and violations fixed; the remaining `react-hooks` rules are re-enabled or individually justified.
- [x] #4 Every persisted Zustand store declares a `version` + `migrate` (or a documented reason it needs neither).
- [x] #5 Shared-code dependency majors are aligned between frontend and extension (or hoisted to one workspace-level version), with a note on the reconciliation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

PR2761 refresh: repairing the real nonincremental WebUI TypeScript baseline, then adding an explicit failing typecheck step to frontend-required. No tsconfig relaxation; remaining strictness/hooks/dependency-major work stays open. Persisted version/migration work from PR2575 and661b is present in frozen candidate and should be verified before closing criterion4.

Verified all9 named persisted Zustand stores already declare version1 plus identity migrate in this candidate (playground-session, persona-buddy-shell, notes-dock, ui-mode, actor, quick-ingest-session, folder, feedback, acp-sessions). Existing five-suite persistence/store selection passed17tests. Criterion4 closed for the current unchanged schema; this does not claim forward-schema migrations. WebUI nonincremental tsc also passes; CI gate added but criterion1 awaits new-head CI evidence. Strictness/additional hook rules/dependency-major alignment remain open.

30338follow-up: typecheck remains skipped because frontend aggregate stops after shard5 failure; criterion1 stays open. Measured separate flags: noImplicitAny948diagnostics/252files; strictNullChecks664/177. Seven disabled compiler-hook rules add25diagnostics/14files in328file sample. Five runtime major mismatches remain (zustand,dexie-react-hooks,marked,d3-dsv,property-information). Baseline logs under/tmp/pr2761-*baseline.log; plan records counts. No broad strictness/dependency change made.

Continuing requester-authorized release blocker closure. Investigating shared dependency major alignment and individually justified React compiler hook rules while current-head typecheck awaits frontend test repair; no gate suppression or wholesale strictness claim.

Aligned five shared runtime dependency majors and peer ranges; frozen install resolves identical zustand5.0.10/dexie-react-hooks4.2.0/marked17.0.1/d3-dsv3.0.1/property-information7.1.0 in extension/WebUI/UI. 28 persistence-Markdown plus10Dexie-TTS tests pass; full WebUI and existing extension strict compile pass. Added required strict project for four shared security/error utilities; actual compiler rejects implicit-any and null probes. Re-enabled use-memo after fixing two Timeline dependency expressions. Other strictness/hooks criteria remain open; detailed scope and full baseline in Docs/Evidence/PR2761-frontend-hardening.md.
<!-- SECTION:NOTES:END -->
