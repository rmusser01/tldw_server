---
id: TASK-2430
title: Design Moderation PolicyCompiler refactor
status: Done
created_date: 2026-06-24 18:26
labels:
- moderation
- design
- refactor
priority: medium
documentation:
- Docs/superpowers/specs/2026-06-24-moderation-policy-compiler-refactor-design.md
modified_files:
- Docs/superpowers/specs/2026-06-24-moderation-policy-compiler-refactor-design.md
- backlog/tasks/task-2430 - Design-Moderation-PolicyCompiler-refactor.md
updated_date: 2026-06-24 18:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an implementation-ready design spec for a compiler-first refactor of the Moderation module. The design should preserve ModerationService as the public facade, keep existing ModerationPolicy compatibility, extract deterministic policy assembly into a PolicyCompiler, and document compatibility, error handling, testing, and non-goals before implementation planning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec is written under Docs/superpowers/specs with the approved compiler-first scope.
- [x] #2 Spec explicitly addresses compatibility with ModerationPolicy/PatternRule, sanitized reports, I/O boundaries, precedence, strict vs forgiving override handling, PII provider boundaries, and supervised-policy non-goals.
- [x] #3 Spec is self-reviewed for placeholders, contradictions, ambiguity, and scope.
- [x] #4 Spec commit is ready for user review before implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrote the compiler-first Moderation refactor spec and self-reviewed it for scope, contradictions, ambiguity, and draft markers. Design-only verification: `git diff --check` passed for the spec/task scope; draft-marker scan found no TODO/TBD/FIXME/unresolved markers after wording cleanup. Bandit is not applicable to this design-only branch because no code is touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec written for the compiler-first Moderation refactor. The approved direction keeps ModerationService as the public facade, preserves ModerationPolicy/PatternRule compatibility, moves deterministic policy assembly into PolicyCompiler, keeps I/O/persistence/logging in the service, preserves strict-vs-forgiving validation boundaries, keeps supervised policy out of scope, and defines focused compiler/service/supervised regression tests. Verification for this design-only branch: git diff --check passed; draft-marker scan found no TODO/TBD/FIXME/unresolved markers. Bandit is not applicable because no code is touched.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
