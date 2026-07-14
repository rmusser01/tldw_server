---
id: TASK-12963
title: Consolidate provider credential runtime for Chat and Knowledge QA
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-14 02:15'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2727'
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
- [x] #2 Invalid credentials, credential-store failures, and revoked background scope fail closed.
- [x] #3 RAG semantic cache reuses documents but never cached generated answers.
- [x] #4 Streaming credential errors use sanitized structured codes and do not trigger non-stream fallback.
- [x] #5 Focused backend/frontend tests and Bandit pass for touched scope.
- [x] #6 Chat and all provider-backed RAG call paths, including query-time hosted embeddings, use the shared credential runtime with explicit no-fallback semantics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Completed; integration plan removed after final verification. Design: Docs/superpowers/specs/2026-07-12-shared-provider-credential-runtime-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-12: Brainstorming completed and the approved design was written at Docs/superpowers/specs/2026-07-12-shared-provider-credential-runtime-design.md.
2026-07-12: Independent spec review approved the complete design with no remaining issues or recommendations.

2026-07-12: Adversarial review found six hardening gaps. The design now covers query-time hosted embeddings, typed legacy-adapter failures, explicit empty-stream completion and fail-closed replay fields, concrete serialization barriers, allowlisted shared transport policy, and phased integration gates. Independent re-review pending.

2026-07-12: Final independent re-review approved the amended design with no blocking issues. Planning must include a shared backend/frontend contract fixture for schema-version handling and malformed terminal stream-event combinations.

2026-07-12: Written-spec approval received. Added one implementation plan with five dependency-ordered stages: shared runtime/boundaries, Chat migration, RAG plus query embeddings, persistence/client contract, and integration/security gate.

2026-07-12: Independent plan-document review approved the complete implementation plan with no remaining issues or recommendations after adding explicit revalidation of base-URL override authority on resume and exact auxiliary-stage test paths.

2026-07-13: Task 1 complete. Commits 552a3f1c5e, 9af6a502ed, 66251817de, and c4350a520d implement typed fail-closed BYOK outcomes, scoped/redacted config, concrete operational-store classification, and the OAuth revocation-race fix. Final focused result: 73 passed, 1 capability-based skip, 3 pre-existing warnings; Black clean; Bandit 0 findings; git diff check clean. Independent specification and code-quality reviews approved with no open issues.

2026-07-13: Task 2 complete. Commits 6187ba4ffd, 8cae33be3f, and 4a73349cfd add the execution-scoped runtime, close-cancellation cleanup, and explicit credential-handle persistence guards. Final focused results: runtime/cache 18 passed; BYOK regressions 73 passed, 1 capability skip; Black clean; Bandit 0 findings; diff clean. Independent specification and code-quality reviews approved with no open issues.

2026-07-13: Task 3 complete. Commits 67ee17d8e5, b6fedc208d, 3a6666e1bc, 6cf99c752c, and b9254fdc16 enforce exact execution-scoped credentials at Chat, summarization, and async/sync embedding boundaries; close downstream config/model/endpoint fallback gaps; preserve sanitized auth/config/rate taxonomy; and prevent upstream provider bodies from reaching explicit errors or logs. Independent specification and code-quality reviews approved with no open Critical/Important issues. Reviewer verification: 60 focused/compatibility tests passed; Bandit 0 findings; py_compile, scoped Black, and git diff checks clean.

2026-07-13: Task 4 SSE-control blocker hit the repository's three-attempt stop rule. Root-cause reassessment showed the endpoint workaround could not be reliable because StreamingResponseHandler treated standard SSE id:/retry: fields as assistant text. User approved expanding Task 4 to streaming_utils.py and its unit tests so the shared SSE parser handles control fields correctly; the implementation plan file list was updated before parser edits.

2026-07-13: Task 4 complete. Commits 3d6f788c89, cff6fbaa71, 224319a1df, be41e42663, 2b70933476, and 738c307e04 migrate Chat routing, selected-provider execution, permitted health fallback, streaming, and one pre-output OpenAI OAuth refresh to a single execution-scoped credential runtime. The final correction records use at the first valid provider text/tool/function output before moderation holdback, preserves raw non-SSE id:/retry: text while ignoring framed controls, and maps post-output provider failures to one sanitized provider_unavailable terminal event without replay or failover. The approved scope expanded by one chat_service.py wiring line. Committed-tree verification: 122 passed, 2 skipped; Black and py_compile clean; Bandit 0 findings over 12,411 LOC; diff checks clean. Independent specification and code-quality reviews approved with no Critical or Important findings.

2026-07-13: Task 5 complete. Commits 5384f959d4 and eb6aa24529 create one trusted execution-scoped runtime for authenticated standard, agentic, streaming, batch, and resume RAG paths; propagate it outside serialized request/checkpoint/response state; preserve legacy no-runtime callers; and map typed provider failures to bounded 400/502/503 responses or terminal stream events. Review corrections added the real native-agentic parameter, lazy stream allocation for never-consumed responses, and cancellation-safe draining of checkpointed batch children before runtime cleanup. Final scope is six files, including the one-line agentic_chunker.py propagation seam; Task 6 retains credential consumption. Verification: Task 5 40 passed; adjacent stream/agentic/checkpoint/resume 22 passed; broader RAG 47 passed, 1 skipped; agentic chunker 11 passed; py_compile clean; Bandit 0 findings; diff checks clean. Independent specification and code-quality reviews approved with no Critical or Important findings.
2026-07-13: Task 6 complete. Commits e733170a7d, 57c968de9c, a87dac4890, ac27831ab0, 2e559eb018, and 1ca940031a migrate final RAG generation plus legacy SGL-backed grading, reranking, faithfulness, claims, repair, and streaming lifecycle paths to the shared execution-scoped credential runtime. Review corrections enforce nonempty real SGL dispatch, fail-closed runtime-only Error handling with no-runtime compatibility, first-valid-content and clean-empty exactly-once usage, transport-control versus parsed-model-content separation, cancellation/prior-success marking, bounded unavailable trust metadata, and sanitized auxiliary logs/errors. Final verification: semantic stream matrix 36 passed; exact Task 6 175 passed; targeted 170 passed; adjacent 77 passed; reviewer focused suites 159 and 164 passed; py_compile clean; Bandit 0 findings; full diff checks clean. Independent specification and code-quality reviews approved with no open Critical or Important findings. Black was not rerun for the final narrow correction; the previously recorded touched-scope Black baseline remained clean.
2026-07-13: Task 7 complete. Commits fcadb65745, 89028920c8, and 4fd6ea4316 migrate direct async auxiliary provider calls across query classification/reformulation, research and media actions, suggestions, Knowledge Strips, evidence accumulation/chains, citations, and unified-pipeline wiring to the shared execution-scoped credential runtime. Optional stages preserve legacy no-runtime callers while runtime-bound failures prohibit config fallback, degrade through native local behavior, and expose only bounded trust metadata; failure trust is monotonic, citation shortcuts reuse the same runtime, research preserves independent full-stack skip decisions, direct Knowledge processors resolve and safely retry missing handles, and successful zero-result media actions still contribute allowlisted trust. Final verification: exact Task 7 suite 199 passed; adjacent suite 99 passed; direct failure-to-retry probe passed; py_compile clean; Bandit 0 findings and 0 errors across all nine production modules; secret/sentinel scan and full-range git diff checks clean. Independent specification and code-quality reviews approved with no open Critical or Important findings.
2026-07-13: Task 8 complete. Commits d5603d610a, 5796abd40b, 49341d61b6, c0cfd73b57, c0c3e0730c, 66b1dd19c6, and a99ed2cf58 credentialize hosted/local query-time embeddings across unified, agentic, HyDE, evidence, claim, and verifier paths. Runtime-explicit calls use one-attempt sanitized boundaries, hashed endpoint cache identities, sensitive HTTP/egress observability, cancellation-safe and retryable durable use accounting, atomic scrubbed agentic config snapshots, deadline/failure latches, bounded provider taxonomy, and fail-closed required retrieval. Optional retrieval stops after typed provider failure and preserves completed answers/base evidence; configured local_api dispatches its effective endpoint without global-batcher ambiguity. Final verification: exact Task 8 suite 182 passed; full RAG_NEW/unit 787 passed, 3 skipped; full http_client suite 86 passed; Ruff/compile/diff/sentinel checks clean (two unchanged http_client TRY203 baseline findings were independently reproduced where applicable); Bandit 0 findings. Independent specification/security and code-quality/simplicity reviews approved with no open findings. Accepted residual: a synchronous within-document agentic embedding already in flight can overrun the wall-clock budget and delay cancellation, but no secret-bearing worker is orphaned.
2026-07-13: Task 8A complete (Stage 3 complete). Residual runtime callers now use the shared provider runtime across async HyDE, agentic planning, research retrieval, unified generation, and adaptive post-generation verification.
2026-07-13: Implementation commits 861b2d207e, 1fc7b851ad, ab01dfbb23, 86969c8167, and 2dd1019009 add real SGL and MultiDatabaseRetriever bindings, typed bounded failures, trusted server scope injection, immediate schema-allowlisted action normalization, fail-closed source handling, bounded numeric inputs, provider/model propagation, and semantic image/video dedup keys.
2026-07-13: Final verification: Task 8A 109 passed; Task 8 matrix 197 passed; full RAG unit suite 827 passed, 3 skipped; Ruff, py_compile, and diff checks clean; Bandit 0 findings; secret sentinel 18 passed with no injected-secret matches.
2026-07-13: Independent final spec/security and quality/simplicity reviews approved Task 8A at 2dd1019009 with no unresolved findings.

2026-07-13: Task 9 complete. Commits 2a418f0728, 693056059f, acb0092068, and 70085147ca make semantic caching retrieval-only and tenant-safe: legacy/generated answers and generation-only document metadata are removed; persisted payloads are strict finite JSON; direct/fake cache payloads use the same sanitizer; namespace identities and filenames are collision-resistant; semantic matches are bounded and never expose raw cached queries; cache identity covers the effective base retrieval configuration; immutable trusted base-retrieval snapshots prevent double processing; advanced secondary retrieval, auto-temporal, explicit include-ID, research-loop, and classification external-prefetch paths fail closed from caching; failed base retrieval fallback evidence is never stored. TDD included regressions for punctuation/case/safe-mimic collisions, non-finite values, malformed matches, FTS/date/late-chunk identity, raw pre-transform snapshots, bypass provenance, and failed-execution fallback. Final root verification: exact Task 9 suite 132 passed; full RAG_NEW/unit 866 passed, 3 skipped; Ruff and py_compile clean; Bandit 0 findings/0 errors across 8,538 production LOC; diff and production sentinel scans clean. Independent specification and quality/security reviews approved 70085147ca with no open findings.

2026-07-13: Task 10 complete. Commits e36c3538dd, 55d81d3b3f, and c4df083458 bind new RAG batch checkpoints to server-derived owner/team/org metadata, authorize owner or explicit admin before runtime creation and early completion, reload strict current memberships, recompute base-URL authority, keep legacy ownerless checkpoints on server credentials, and persist only bounded result error codes. Review hardening aligns owner credentials, media/notes/prompts/Kanban paths, request/cache identity, and the ambient content ScopeContext for both self-resume and delegated-admin resume; delegated admin bypass is removed, caller scope is restored after runtime/media cleanup, malformed membership rows and orphan ownerless scopes fail closed, and ownerless server checkpoints omit credential metadata. Fresh root verification: checkpoint/resume suite 85 passed; adjacent provider-credential suite 19 passed; Ruff and py_compile clean; Bandit 0 findings/0 errors over 2,345 production LOC; diff check clean. Independent final specification and quality/security reviews approved c4df083458 with no open findings.

2026-07-13: Task 11 complete. Commit 5c7d8a11ce adds a shared versioned terminal RAG stream contract, explicit clean completion, sanitized provider errors, exact EOF-before-replay validation, and fail-closed frontend/background transport behavior. Unsafe stream POSTs never direct-fetch replay after background handoff; standard RAG no longer retries arbitrary HTTP 500 responses; malformed/unknown terminal events fail closed; optional provider status_code compatibility is retained. Final root verification: backend 79 passed; frontend 80 passed; Ruff and py_compile clean; Bandit 0 findings/0 errors across 3,014 production LOC; diff checks clean. Independent specification and quality/security re-reviews approved with no open findings.

2026-07-13: Tasks 12 and 13 complete. Commits ffbc73dfc1, 1399ffef70, and 7905df629b add the cross-surface no-fallback/secret-leak gate, close authenticated RAG ablate/simple/advanced runtime gaps, reject credential handles before checkpoint persistence, and protect credential-derived embedding endpoints across egress, transport, logs, metrics, manual tracing, third-party logging, and OpenTelemetry auto-instrumentation. Public redirect observability remains accurate. Final committed-tree verification: backend union 240 passed, 1 documented TestClient streaming skip; frontend 35 passed; Chromium credential workflow 1 passed; full HTTP client 95 passed. All changed Python files py_compile clean; git diff check clean. Broad Bandit artifact /tmp/bandit_TASK-12112_final.json matches base exactly: 34 existing Low findings, 0 new, 0 Medium/High, 0 scan errors. Ruff E9/F821 final sweep found only two unchanged ChatBadRequestError F821 baseline findings present at 4f88741711; Task 13 touched scope is clean. Independent whole-feature review ended with no findings and explicit SPEC APPROVED / QUALITY APPROVED.

2026-07-14: Integration cleanup approved. Preserve the completed source branch, replay only provider-credential-runtime commits onto current origin/dev, renumber this task because TASK-12112 collides on the target branch, resolve target overlap, and rerun the full verification gate.

2026-07-13: Current-dev integration complete on codex/provider-credential-runtime-dev. Replayed the 70 provider-credential commits onto origin/dev at 8dbeb383ac, preserving current-dev RAG diagnostic sanitization, response-acquisition stream semantics, origin-bound frontend credential handling, and safe/idempotent-only replay after background handoff. Corrected one stale Retry-After test fixture to include the current origin-bound manual credential metadata; production fail-closed credential behavior was unchanged. Final integrated verification: backend gate 240 passed and 1 documented TestClient streaming skip; affected frontend matrix 95 passed; full HTTP client suite 112 passed; Chromium Knowledge QA credential/no-fallback workflow 1 passed; frontend TypeScript typecheck passed; all changed Python files py_compile clean; git diff check clean. Ruff E9/F821 found only two ChatBadRequestError findings reproduced identically on origin/dev. Bandit scanned the changed production Python scope with 0 findings on both integrated (50,914 LOC) and origin/dev baseline (45,056 LOC), 0 scan errors. Final range-diff/conflict review found no unresolved Critical or Important issues. Latest origin/dev ancestry rechecked at completion: 0 behind. The two unrelated untracked watchlist templates remain untouched.

2026-07-13: Draft PR #2727 opened: https://github.com/rmusser01/tldw_server/pull/2727. The PR remains draft and must receive a requester-authored Change summary explaining what changed and why before it is marked ready or merged.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated the reviewed server-only provider credential runtime onto the latest origin/dev while preserving current development behavior at overlapping RAG and frontend stream boundaries. Chat and all provider-backed RAG paths, including hosted query embeddings, share one execution-scoped credential policy with user/team/org/server precedence; invalid credentials, store failures, and revoked scopes fail closed. Credentials remain non-serializable and excluded from client/cache/checkpoint state, semantic cache remains retrieval-only, and terminal stream failures cannot trigger unsafe replay. Final current-dev verification passed: backend 240 passed/1 documented skip, frontend 95 passed, HTTP client 112 passed, Chromium workflow 1 passed, TypeScript and Python compilation passed, Bandit found 0 issues with no baseline delta, and final integration review found no unresolved issues. The credential task was renumbered to TASK-12963 without modifying the existing TASK-12112 microphone task. The two unrelated untracked watchlist templates were not touched. A human-authored PR Change summary is still required before merge.
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
