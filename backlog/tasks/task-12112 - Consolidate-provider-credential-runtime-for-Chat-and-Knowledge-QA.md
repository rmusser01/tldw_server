---
id: TASK-12112
title: Consolidate provider credential runtime for Chat and Knowledge QA
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-12 22:24'
labels: []
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-07-12-shared-provider-credential-runtime-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement a shared execution-scoped provider credential runtime used by Chat and every provider-backed RAG stage, including query-time hosted embeddings. Preserve user/team/org/server precedence, fail closed on invalid or unavailable BYOK, keep credentials server-side and non-serializable, make RAG semantic caching retrieval-only, and prevent terminal streaming credential errors from triggering non-stream fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Approved design specification is written, reviewed, and linked from this task.
- [ ] #2 Invalid credentials, credential-store failures, and revoked background scope fail closed.
- [ ] #3 RAG semantic cache reuses documents but never cached generated answers.
- [ ] #4 Streaming credential errors use sanitized structured codes and do not trigger non-stream fallback.
- [ ] #5 Focused backend/frontend tests and Bandit pass for touched scope.
- [ ] #6 Chat and all provider-backed RAG call paths, including query-time hosted embeddings, use the shared credential runtime with explicit no-fallback semantics.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-12: Brainstorming completed and the approved design was written at Docs/superpowers/specs/2026-07-12-shared-provider-credential-runtime-design.md.
2026-07-12: Independent spec review approved the complete design with no remaining issues or recommendations.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-07-12: Adversarial review found six hardening gaps. The design now covers query-time hosted embeddings, typed legacy-adapter failures, explicit empty-stream completion and fail-closed replay fields, concrete serialization barriers, allowlisted shared transport policy, and phased integration gates. Independent re-review pending.

2026-07-12: Final independent re-review approved the amended design with no blocking issues. Planning must include a shared backend/frontend contract fixture for schema-version handling and malformed terminal stream-event combinations.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
