# Character export, speech readiness and repository output repair

Task: TASK13260.281.4.

## Stage 1: Reproduce
**Goal:** Demonstrate double character envelopes, enabled synthesis after missing provider data, and missing themed output colors.
**Success Criteria:** JSON/PNG, the actual Speech page and readiness regressions fail before correction. The minimal output-color change uses the original UAT contrast failure and source inspection.
**Tests:** Vitest literal card/PNG inputs and the existing Speech rendering harness; no live generation.
**Status:** Complete

## Stage 2: Correct
**Goal:** Normalize character exports once, gate dispatch on actual provider/voice readiness, and apply existing theme colors.
**Success Criteria:** Retain character fields/avatar security, typed speech drafts and available-provider behavior; explicit unavailable/loading guidance.
**Tests:** New regressions plus existing character SSRF, Speech/readiness and repository component suites.
**Status:** Complete

## Stage 3: Review
**Goal:** Verify affected source and publish in the umbrella PR.
**Success Criteria:** Formatting/lint/types, independent review, truthful Bandit TypeScript applicability; live UAT remains pending.
**Tests:** Actual changed source checks; no screenshot/proof/manifest bundles.
**Status:** In Progress

Reviewed final source passes Character envelope/PNG5, SSRF10, Speech32, readiness11, VoicePreview4, Inspector4 and existing repository3 checks. Stop remains available while playback is active even if provider capability disappears. Touched lint0errors; final consolidated types have0touched diagnostics but inherited unrelated failures. Python Bandit applies to the backend scope; it cannot validate TypeScript. Publication is the remaining source-plan step; live synthesis/visual acceptance is separate.
