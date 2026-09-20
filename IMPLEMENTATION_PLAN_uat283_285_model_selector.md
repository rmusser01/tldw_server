# UAT283 and UAT285: Chat model selector

Backlog: TASK13260.220 and TASK13260.222. The user authorized repairing identified UAT defects before another full matrix. Preserve provider-qualified routing and configured model identity; reuse the existing selection parser for display lookup. Suppress the current-model tooltip while its dropdown is open, preserving closed-menu hover help. No model capability or provider configuration override.

## Stage 1: Causal regression
**Goal**: Reproduce setup-qualified model display and tooltip overlap.
**Success Criteria**: New tests fail for the observed causes; existing controls pass.
**Tests**: Selection identity/alias/collision/loading controls; real Ant tooltip and dropdown interaction.
**Status**: Complete

## Stage 2: Minimal repairs and review
**Goal**: Resolve display metadata through existing parsing; disable tooltip during selection.
**Success Criteria**: Correct provider/model labels with unchanged raw IDs and pointer selection; reviewed diff.
**Tests**: Focused and adjacent tests, ESLint, TypeScript baseline comparison. TypeScript-only scope: Bandit not applicable.
**Status**: Complete

## Stage 3: Committed native acceptance
**Goal**: Recheck original fresh setup and desktop model-menu flow.
**Success Criteria**: Correct settled/reloaded label and ordinary pointer selection without force or injected events.
**Tests**: Fresh PostgreSQL single-user setup with real provider; attach image and retain evidence; source audit/owned cleanup. Related UAT284/286 remain separately tracked.
**Status**: Not Started

Final146 affected tests pass. Reviewer alias/native-title findings reproduced and fixed; final review clear. ESLint36 unchanged baseline warnings, TypeScript93 unchanged baseline errors. Native acceptance remains pending.
