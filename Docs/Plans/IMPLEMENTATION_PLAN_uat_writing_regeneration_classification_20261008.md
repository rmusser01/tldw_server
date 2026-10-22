# Writing regeneration classification repair

Associated task: TASK-13260.278.18.83.45 (root-owned; separate unit recorded before edits).

**Goal:** Keep a usable pending revision when regeneration returns an unusable result, and retain that result under its parser-assigned classification for review.
**Spec:** [Document-first Writing revisions](../superpowers/specs/2026-05-22-writing-playground-document-first-revisions-design.md), especially Model Output Cannot Be Parsed.
**Architecture:** Reuse the existing proposal parser, session-payload revisions, and queue. No output repair, provider request change, new classifier, or automatic admission. A new parsed pending text replacement supersedes the original; raw output remains copy-only.
**ADR check:** ADR required: no. The [ADR index](../ADR/README.md) has no governing Writing revision decision. This restores the existing spec's parser authority and copy-only fallback; no API, persistence, ownership, provider, or dependency rule changes.

## Stage 1: Establish causal regressions
**Goal:** Demonstrate that regeneration currently rejects a usable original and promotes raw output to pending.
**Success Criteria:** Real parser-to-hook and mounted queue controls fail for the stated classification/data-preservation cause before production edits. Existing meaningful assertions remain.
**Tests:** Plain, missing, and non-string replacement responses; valid parsed pending and deliberate empty-string replacement; held malformed responses after session replacement, list epoch change, or source rejection; mounted copy-only fallback while the original remains Apply eligible.
**Status:** Complete

Causal red: configured UI runner, default case timeout / 600-second outer cap, natural exit 1 in 6.439 seconds. Five expected failures (three hook and two mounted classification/preservation cases); five passing controls (parsed replacement/deletion and three held-response fences). The remaining 96 cases were deselected by the focused filter. No setup failure or changed assertion was used to obtain the red.

## Stage 2: Preserve parsed status and usable original
**Goal:** Change only useWritingRevisions regeneration mutation.
**Success Criteria:** Keep the parser's result status; retire the original only for a pending text-changing result with string replacement text. Preserve source linkage and existing session/list-version/current-source guards. An empty string remains a deliberate deletion.
**Tests:** New causal controls green; existing valid regeneration/persistence and concurrent-response controls unchanged. The old positive fixture uses actual parser output rather than an invented already-applied callback status, retaining its target/instruction/preset/supersession assertions.
**Status:** Complete

Minimal correction: use the existing pending/non-advisory/string-replacement rules inside regeneration; preserve the parsed status and reject the original only when these rules permit a replacement. All post-await session, revision-list version, source identity, and pending-source checks remain unchanged. There is no exported usable predicate to reuse; no new classifier or queue/apply behavior was introduced.

## Stage 3: Qualify and freeze
**Goal:** Qualify affected shared UI/Web checks and scoped lint, then provide exact freeze paths and coverage limits for independent review.
**Success Criteria:** Node 20.19.5/Vitest 4.0.18 normal configured tests, 600-second outer/default case limits; no new touched lint findings; root/peer independent source review. No runtime/model/provider/browser/auth/Git actions. Maximum three distinct attempts. Root owns combined types/build/live UAT.
**Tests:** useWritingRevisions and existing queue/parser controls plus genuinely affected mounted regeneration cases in UI and Web; baseline/current scoped ESLint including syntax. Bandit N/A for TypeScript-only changes.
**Status:** In Progress

Current evidence: all ten new causal cases passed in the configured UI runner (natural exit 0, 5.051 seconds). After formatting only the new test bodies, affected configured UI and Web each passed 50 cases across four files (natural exit 0, 34.162 / 33.415 seconds): full hook/queue/parser suites and sixteen connected mounted cases. Seventy-one unrelated mounted cases were deselected in each runner. Node 20.19.5 / Vitest 4.0.18, default case timeout / 600-second outer cap; no setup failure, timeout, skipped relevant case, or weakened assertion. Scoped ESLint before production and after the correction both returned zero errors, fatal errors, and warnings; its apps-base Next pages-directory advisory is unchanged. Bandit is not applicable to this TypeScript-only scope. One causal red and one minimal production correction; no alternative attempts were needed. Independent source review and root-owned combined typing/build/live validation remain pending.

Frozen source/test SHA256:
- hook: `5ca5dce9c2758062c5bef285b92d6c98a7f2f63297b5d41b4fd785d2148f4b1d`
- hook regression: `e867711a2867984d8b43bbaa8ed3b64aa87bed2bcdec92aa1c4017f9234ddd6b`
- mounted regression: `cb1ae9a97424fba743b56cea773969342c11e89a6db29904c54f3a0c54786a05`

Coverage limits: tests use existing transport/owner/editor abstractions and real parser/hook/queue; they establish classification, preservation, apply behavior, payload persistence, and stale-response refusal. They do not establish live model output quality, server persistence, full browser reload, rich-text automatic apply, or all43 acceptance. The separate Writing target component remains unchanged at `1a6512b2475f83e03751f49cd028a6a07a7dc4066c850160548ff31ca72afe92`.
