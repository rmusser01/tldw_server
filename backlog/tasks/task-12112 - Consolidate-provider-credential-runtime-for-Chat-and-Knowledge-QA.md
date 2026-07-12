---
id: TASK-12112
title: Consolidate provider credential runtime for Chat and Knowledge QA
status: In Progress
documentation:
- Docs/superpowers/specs/2026-07-12-shared-provider-credential-runtime-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement a shared execution-scoped provider credential runtime used by Chat and every LLM-backed RAG stage. Preserve user/team/org/server precedence, fail closed on invalid or unavailable BYOK, keep credentials server-side and non-serializable, make RAG semantic caching retrieval-only, and prevent terminal streaming credential errors from triggering non-stream fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Approved design specification is written, reviewed, and linked from this task.
- [ ] #2 Chat and all RAG LLM call paths use the shared credential runtime with explicit no-fallback semantics.
- [ ] #3 Invalid credentials, credential-store failures, and revoked background scope fail closed.
- [ ] #4 RAG semantic cache reuses documents but never cached generated answers.
- [ ] #5 Streaming credential errors use sanitized structured codes and do not trigger non-stream fallback.
- [ ] #6 Focused backend/frontend tests and Bandit pass for touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-12: Brainstorming completed and the approved design was written. Spec review is pending before implementation planning.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
