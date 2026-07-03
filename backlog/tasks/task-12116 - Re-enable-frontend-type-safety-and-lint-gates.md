---
id: TASK-12102
title: Re-enable frontend type-safety and lint gates, harden persisted stores
status: In Progress
labels:
- tech-debt
- high
- frontend
- ci
- packages-ui
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
- [ ] #4 Every persisted Zustand store declares a `version` + `migrate` (or a documented reason it needs neither).
- [ ] #5 Shared-code dependency majors are aligned between frontend and extension (or hoisted to one workspace-level version), with a note on the reconciliation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
