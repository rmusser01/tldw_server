---
id: TASK-12112
title: Consolidate provider credential runtime for Chat and Knowledge QA
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-13 00:33'
labels: []
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-07-12-shared-provider-credential-runtime-design.md
  - >-
    Docs/superpowers/plans/2026-07-12-shared-provider-credential-runtime-implementation-plan.md
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-12-shared-provider-credential-runtime-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-12: Brainstorming completed and the approved design was written at Docs/superpowers/specs/2026-07-12-shared-provider-credential-runtime-design.md.
2026-07-12: Independent spec review approved the complete design with no remaining issues or recommendations.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-07-12: Adversarial review found six hardening gaps. The design now covers query-time hosted embeddings, typed legacy-adapter failures, explicit empty-stream completion and fail-closed replay fields, concrete serialization barriers, allowlisted shared transport policy, and phased integration gates. Independent re-review pending.

2026-07-12: Final independent re-review approved the amended design with no blocking issues. Planning must include a shared backend/frontend contract fixture for schema-version handling and malformed terminal stream-event combinations.

2026-07-12: Written-spec approval received. Added one implementation plan with five dependency-ordered stages: shared runtime/boundaries, Chat migration, RAG plus query embeddings, persistence/client contract, and integration/security gate.

2026-07-12: Independent plan-document review approved the complete implementation plan with no remaining issues or recommendations after adding explicit revalidation of base-URL override authority on resume and exact auxiliary-stage test paths.

2026-07-13: Task 1 complete. Commits 552a3f1c5e, 9af6a502ed, 66251817de, and c4350a520d implement typed fail-closed BYOK outcomes, scoped/redacted config, concrete operational-store classification, and the OAuth revocation-race fix. Final focused result: 73 passed, 1 capability-based skip, 3 pre-existing warnings; Black clean; Bandit 0 findings; git diff check clean. Independent specification and code-quality reviews approved with no open issues.

2026-07-13: Task 2 complete. Commits 6187ba4ffd, 8cae33be3f, and 4a73349cfd add the execution-scoped runtime, close-cancellation cleanup, and explicit credential-handle persistence guards. Final focused results: runtime/cache 18 passed; BYOK regressions 73 passed, 1 capability skip; Black clean; Bandit 0 findings; diff clean. Independent specification and code-quality reviews approved with no open issues.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-13: Task 3 complete. Commits 67ee17d8e5, b6fedc208d, 3a6666e1bc, 6cf99c752c, and b9254fdc16 enforce exact execution-scoped credentials at Chat, summarization, and async/sync embedding boundaries; close downstream config/model/endpoint fallback gaps; preserve sanitized auth/config/rate taxonomy; and prevent upstream provider bodies from reaching explicit errors or logs. Independent specification and code-quality reviews approved with no open Critical/Important issues. Reviewer verification: 60 focused/compatibility tests passed; Bandit 0 findings; py_compile, scoped Black, and git diff checks clean.

2026-07-13: Task 4 SSE-control blocker hit the repository's three-attempt stop rule. Root-cause reassessment showed the endpoint workaround could not be reliable because StreamingResponseHandler treated standard SSE id:/retry: fields as assistant text. User approved expanding Task 4 to streaming_utils.py and its unit tests so the shared SSE parser handles control fields correctly; the implementation plan file list was updated before parser edits.

2026-07-13: Task 4 complete. Commits 3d6f788c89, cff6fbaa71, 224319a1df, be41e42663, 2b70933476, and 738c307e04 migrate Chat routing, selected-provider execution, permitted health fallback, streaming, and one pre-output OpenAI OAuth refresh to a single execution-scoped credential runtime. The final correction records use at the first valid provider text/tool/function output before moderation holdback, preserves raw non-SSE id:/retry: text while ignoring framed controls, and maps post-output provider failures to one sanitized provider_unavailable terminal event without replay or failover. The approved scope expanded by one chat_service.py wiring line. Committed-tree verification: 122 passed, 2 skipped; Black and py_compile clean; Bandit 0 findings over 12,411 LOC; diff checks clean. Independent specification and code-quality reviews approved with no Critical or Important findings.

2026-07-13: Task 5 complete. Commits 5384f959d4 and eb6aa24529 create one trusted execution-scoped runtime for authenticated standard, agentic, streaming, batch, and resume RAG paths; propagate it outside serialized request/checkpoint/response state; preserve legacy no-runtime callers; and map typed provider failures to bounded 400/502/503 responses or terminal stream events. Review corrections added the real native-agentic parameter, lazy stream allocation for never-consumed responses, and cancellation-safe draining of checkpointed batch children before runtime cleanup. Final scope is six files, including the one-line agentic_chunker.py propagation seam; Task 6 retains credential consumption. Verification: Task 5 40 passed; adjacent stream/agentic/checkpoint/resume 22 passed; broader RAG 47 passed, 1 skipped; agentic chunker 11 passed; py_compile clean; Bandit 0 findings; diff checks clean. Independent specification and code-quality reviews approved with no Critical or Important findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
