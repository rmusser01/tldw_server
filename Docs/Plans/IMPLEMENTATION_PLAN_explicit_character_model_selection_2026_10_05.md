# Explicit character model selection — TASK-13454.14

Approved program: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. The live automatic tldw model was displayed as Current/Healthy although native character preflight required an explicit provider. No backend authority or ordinary fallback dispatch is changed.

## Stage 1: Trace identity and settings ownership
**Goal**: Trace all full/Research/compact selector callers, model settings Save, compact scope effects, and character readiness before edits.
**Success Criteria**: A catalog match alone cannot define an explicit provider; raw unqualified settings remain owned by their raw key. Known aliases, model-ID case, declared Research defaults, and exact opaque catalog selections remain supported.
**Tests**: Existing shared hook/dropdown baseline: 23 passed. Source trace includes both selectors, all three shared hook callers, modal Save, compact ControlRow, readiness callers and native preflight.
**Status**: Complete

## Stage 2: Correct the shared lookup and readiness
**Goal**: Reuse one small existing-utility lookup for selected metadata and settings scope. Commit canonical provider/model keys only on explicit menu action. Native readiness checks the actual selection and provider; the selector offers a concrete provider action.
**Success Criteria**: Catalog arrival/reordering cannot create Current, borrow provider capabilities, or change a raw prompt scope. Explicit provider overrides and qualified selections match exactly. Compact repeated selection retains its choice. Native character readiness names an unqualified/auto selection and checks the selected provider; ordinary fallback remains intact.
**Tests**: Qualified red/green regressions for hook/dropdown, real scoped store with catalog changes, actual settings Save, compact ControlRow and ModelSelect, canonical local alias, readiness/native preflight.
**Status**: Complete

## Stage 3: Verify and review the concrete result
**Goal**: Validate both clients, independent reviews, bounded live evidence, and a local reviewable commit.
**Success Criteria**: Relevant tests/type gates pass; no new lint diagnostics; two independent reviewers have no unresolved blockers. Browser evidence distinguishes source-tested negative cases from live selected-provider proof. Task and approved program evidence are current.
**Tests**: Targeted/bounded frontend suites, WebUI typecheck, extension compile, touched lint baseline comparison, diff check, browser screenshot. No Python source is planned, so Bandit is not applicable unless scope changes.
**Status**: In Progress

Verification notes: two independent final source reviews are clear after the scoped-provider, opaque-catalog/native-route, cockpit A→B Save, and LM Studio/llamafile compatibility corrections. Qualified native-route tests: 95 passed across 3 suites. The real settings visibility assertion remains; its fixture disables CSS motion because the diagnostic proved jsdom retained AntD zoom-appear opacity 0. Live UI showed providerless Choose provider with no Current, then explicit Custom OpenAI and Healthy while preserving the exact two-line draft and trailing spaces. Screenshots: /private/tmp/chat-model-provider-required.jpg, /private/tmp/chat-model-unqualified-menu.jpg, /private/tmp/chat-model-explicit-choice-draft-retained.jpg. These are browser/fixture checks, not packaged extension, assistive-technology or real-provider generation proof. Latest fetched dev is 587cd8e9fe3b42eba451b83c1ba690607035eace; integration and post-integration checks remain.
