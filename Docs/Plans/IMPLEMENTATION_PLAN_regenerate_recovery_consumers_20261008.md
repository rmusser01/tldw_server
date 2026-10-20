# Regenerate settlement-loss and mounted recovery consumers

Task: TASK-13260.278.18.83.45 (parent owns the Backlog record; separate review-found fix unit recorded before edits).

ADR check: ADR required: no. [ADR-049](../ADR/049-chat-history-selection-ownership.md) and [Proposed ADR-067](../ADR/067-selected-existing-input-chat-regeneration.md) already govern owner-bound capture, stable exact replay and retained recovery. This fixes implementation gaps without changing those rules or either existing design artifact.

Scope: normal Chat's pre-settlement recovery record; the mounted Keep hook and shared recovery review; Playground and Sidepanel consumers; their focused local/native regression and existing affected tests. No backend, Prompt/Character, dependencies, provider, build/runtime, Git/index/ref or network actions. Preserve owner, lease, profile, operation, view, capability and exact-payload checks.

## Stage 1: Causal regressions
**Goal**: Reproduce reload while settlement is unresolved and mounted stale-native recovery without invoking Keep directly.
**Success Criteria**: New tests fail on missing completion outcome and mounted handling/visibility; real controller/storage/service code remains in the path, with only table engine and transport boundaries doubled.
**Tests**: Hold pre-write settlement; snapshot durable record, unmount/reset live registry/reopen and retry exact payload without generating. Mount Keep and shared review on bound stale projection; test successful replay and refused replay visibility. Add actual Sidepanel consumer coverage using its existing route harness.
**Status**: Complete

## Stage 2: Minimal shared correction
**Goal**: Preserve complete Regenerate payloads across process loss and consume recoveries in mounted chat surfaces.
**Success Criteria**: Complete outcome is durable before settlement; only existing-input recoveries can run while stale; refused completed records stay reviewable; Sidepanel uses the same hook; reviewed Keep delegates to the existing guarded service. No production source edit while root build freeze is active.
**Tests**: New causal tests pass, including zero additional model calls and no duplicate user/assistant write; affected existing controller/settlement/caller tests pass.
**Status**: Complete

## Stage 3: Configured client qualification
**Goal**: Verify both installed UI/Web configurations, configured types and changed lint scope.
**Success Criteria**: Fresh natural-exit results and exact scopes reported to parent; no assertion, guard or plugin bypass. Root retains whole build, hooks, live Chat/43-workflow acceptance, final review and integration ownership.
**Tests**: Both configured Vitest scopes; UI and Web tsc --noEmit; changed TS lint and project-venv Bandit touched production scope (TS-only applicability recorded).
**Status**: In Progress

## Attempt boundary

At most three distinct causal correction attempts for these two newly found integration failures. Do not replay the previous exhausted setup/control diagnostics. No real IndexedDB/browser/backend/native workflow acceptance is claimed by these component tests.

## Focused results

- New causal UI red: 3 failed / 34 filter-excluded, natural exit 1, 4.44 seconds. The held settlement's complete payload was downgraded to `interrupted` by a pending checkpoint; mounted stale-native saved/refused cases made zero settlement calls and hid the recovery review. Stopping checkpoints at complete staging plus the outcome/consumer correction is one production candidate.
- Focused UI green: 4 passed / 116 filter-excluded, natural exit 0, 3.47 seconds. Covers unresolved local settlement → new mounted page → exact selected-input replay, real mounted native controller/Keep/review successful and refused replay, and mounted shared Sidepanel delegation.
- Sidepanel consumer red with its hook omitted: 1 failed / 82 filter-excluded, natural exit 1, 2.69 seconds; actual review was mounted and settlement was absent. First restored-hook run reached the full scoped three-argument API request; its test expected a two-element argument array. The corrected fixture preserves every expected payload field and adds the scoped third argument. No extra production candidate.
- Full configured UI: 12 files / 382 passed, zero skips/filter-exclusions, natural exit 0, 68.59 seconds. Includes all 19 existing mounted Playground interrupted/Stop/navigation/retry cases, controller and main/sibling callers, shared review, settlement, Sidepanel ownership/freshness/open-tab consumers and ordinary overlays/provider recovery.
- Full configured Web: 11 files and 380 individual cases passed, 2 cases failed, natural exit 1, 69.49 seconds. Both failures were the new test's injected stale controller leaking into following shared Sidepanel cases because Web does not auto-restore spies. Its test now explicitly unmounts/restores its three owned spies in `finally`; no production or expectation change. The changed complete Sidepanel file then passes all 83 cases in UI (4.69 seconds) and Web (4.67 seconds), each natural exit 0. Other 299 Web cases remain their separately executed green scope; no repeated unchanged suite.
- Configured UI typecheck passed, including the final run after spy cleanup (natural exit 0). Normal Web typecheck failed only writing the incremental cache at the sandbox-restricted worktree path (TS5033/EPERM); the same configured check with a fresh `/private/tmp` cache output passes, natural exit 0.
- Six changed TypeScript files lint with zero errors. Production warnings remain 22 at existing sites; both edited test files and the Keep hook have zero warnings. In-memory reconstruction of the four production files before this correction verifies baseline/current diagnostic parity. No warning suppression or source/config change.
- Normal project-venv Bandit on the four changed directories exits 0 with no errors or findings and **zero Python LOC**. It does not validate TypeScript security; owner/capability/view/operation fences are preserved in unchanged production services.

## Limits

Table-engine doubles preserve real local transactions/records but do not certify native IndexedDB durability. The native mounted controller/service tests double remote transport; the real Sidepanel route test injects a controller to isolate its consumption, while native capability parsing, binding and guarded Keep remain real. Backend/browser/model/native Chat and all 43 workflow acceptance stay with root. The 12-file UI run's verbose inherited harness log display was truncated, while its complete terminal summary and natural exit were recovered; no log/proof file was saved. Root owns final independent review, build/hooks/integration and the parent task. Existing design/spec/ADR documents were preserved.
