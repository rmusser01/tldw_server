# Claims integrity confirmation: server, WebUI, and extension

The Claims module is an existing foundation for evidence integrity. It extracts, verifies, persists, clusters, reviews, and monitors claims, and already provides artifact grounding gates. The remaining work is to make its verdicts dependable and connect them to the final answer users see.

This investigation confirmed gaps in verdict rules, selected-source enforcement, answer coverage, streaming, and report persistence. It also identified safeguards and corrected two overly broad assumptions: post-verification claim retrieval currently falls back after a callback signature mismatch, and the FVA service has constructor incompatibilities before its suspected retrieval-scope gap can execute.

## Scope and evidence

- Date: October 2, 2026.
- Tracking: [TASK-13422](../../backlog/tasks/task-13422%20-%20Confirm-claims-integrity-across-server-WebUI-and-extension.md).
- Saved progress: [draft PR #3090](https://github.com/rmusser01/tldw_server/pull/3090), targeting dev.
- Initial reviewed/executed product revision: `8140e493f2d0a79e2039084930151eba6565df82`.
- Latest remote dev checked: `9958110df2a9011e19f48b0eae821353e19d4af8`.
- The diff between those revisions is empty across Claims_Extraction, core RAG, unified RAG endpoints/schemas, shared UI, WebUI, and extension. Links below use latest checked dev.
- This supplements the [NotebookLM capability review](NotebookLM_Thread_Capability_Review_2026_10_02.md). Subsequent brainstorming accepted strict evidence as the default for source-based research, supported partial answers with explicit gaps, and verification before display with progress during generation/checks. The ongoing design is tracked in [TASK-13423](../../backlog/tasks/task-13423%20-%20Design-strict-source-grounded-answers-before-display.md).

**Imported-module probes** use the real Claims implementation with injected provider/NLI responses. **Isolated control-flow probes** select code from its syntax tree and stub dependencies. **Static traces** follow source and existing tests without executing the full flow. These are distinct evidence levels; none certifies a deployed API incident or live model accuracy.

No product code, dependencies, or behavior changed. This saves investigation evidence, not an approved feature design or implementation plan.

## What is already supported

| Capability | Existing implementation | Qualification |
| --- | --- | --- |
| Claims engine and operations | Extraction, parsing controls, NLI/LLM verification, numeric/date checks, evidence/citations, persistence, indexes, clustering, review, monitoring, notifications, and rebuild helpers. [Module map][module] | Reuse this foundation; verification labels still need stronger evidence semantics. |
| Selected-source retrieval | Initial unified RAG honors collection/Media filters. Research Workspace sends selected Media IDs. [Server scope][initial_scope] [Workspace request][workspace_scope] | Later verification and repair do not consistently preserve those restrictions. |
| Missing-source protection | Workspace refuses general-chat fallback when selected-source retrieval is empty/fails and avoids server mirroring of that grounding diagnostic. [Workspace guards][workspace_guards] | This does not verify every statement generated after successful retrieval. |
| Expert controls | Knowledge QA exposes claims, extractor/verifier choices, confidence, post-verification, hard citations, and numeric checking. Thorough mode enables claims/post-verification. [Controls][expert] [Defaults/presets][defaults] | Default claims/post-verification are off; document-only/report fields have transport support without dedicated controls found here. |
| Artifact gates | Artifact verification rejects empty extraction, verifies supplied documents, and rejects non-grounded results before selected artifacts are saved. [Verifier][artifact_verifier] [Generation gate][artifact_gate] | These gates share the verdict issues below; presence is not certification of every artifact. |
| Client evidence | Source cards, citation/trust displays, local source metadata, and artifact verifier identity already exist. [Answer panel][answer_panel] [Artifact detail][artifact_detail] | Detailed claim evidence is not consistently displayed or restored. |

## Server confirmation

### Verdict rules can contradict evidence checks

Eight observations were confirmed against the imported Claims engine. The [saved JSON](artifacts/claims-integrity-13422/claims-verdict-results.json) records actual labels, confidence, document IDs, and citation ranges.

The main fixture was document `selected`, text `The study enrolled 42 participants.`, retrieval score `0.8`. Provider/configuration/usage logging were stubbed to prevent external work. The verifier, decision rules, numeric guardrails, parser, and citation construction remained real code. [Decision rules][decision] [Numeric/quote checks][guardrail_call]

| Input/condition | Observed result | Implication |
| --- | --- | --- |
| NLI neutral 0.97, entailment 0.02, contradiction 0.01 for a claim about Mars | `refuted` at 0.97; rationale `NLI-nei 0.97` | Insufficient evidence is conflated with contradiction. |
| Claim says 999 participants; academic numeric check rejects it; judge says supported at 0.99 | `verified` at 0.99 | Confident model judgment can override a failed numeric check. |
| Claim quotes `unicorns live here`, absent from the document; judge says supported at 0.99 | `verified` at 0.99 | Confident model judgment can override a failed quotation check. |
| Claim says Mars; judge says not-enough-information at 0.6; document score 0.8 | `verified` at approximately 0.72, rationale `Moderate evidence support` | Retrieval relevance can become claim support. |
| No documents, document-only mode, judge says supported at 0.99 | `verified` with no evidence/citations | Document-only mode is not an unconditional evidence requirement inside this verifier. Artifact source prechecks separately reject empty sources. |
| Supplied selected document says 42; callback supplies outside-selection document saying 999; document-only mode | `verified` using `outside-selection` | The mode does not enforce membership in the original corpus. |
| Document/claim/snippet share no text | Citation helper returns prefix range `(0, 8)` | Nonempty offsets can be a fallback rather than a located supporting passage. |
| Inspect combined `ClaimsEngine.run` signature | Missing `doc_only_mode` and `numeric_precision_mode`, present on `verify_claims_only` | Combined answer extraction cannot receive those controls through its current signature. |

These isolate reachable decision behavior, not its real-model frequency. Outside-selection evidence demonstrates corpus membership failure, not cross-user database access. [Offsets][offsets] [Combined run][run] [Two-stage verification][verify_only]

### The answer can be checked against source claims instead

When retrieved media has stored ingestion claims, unified RAG verifies those source claims and bypasses extracting claims from the newly generated answer. It attaches the result to the answer. Correct source claims do not prove that the answer says the same thing. **Static trace**, with an existing test explicitly expecting extraction to be bypassed. [Branch][preclaims] [Existing test][preclaims_test]

Document-only/numeric-precision options are forwarded in that stored-claim branch. The normal answer-extraction branch calls `run` without them; the imported signature probe confirms why. [Answer extraction][answer_extract] [Combined run][run]

### Selected-source scope does not survive every step

Initial allowlists are real. Unified RAG's per-claim callback later invokes database retrieval without those allowlists. **Static trace**, reinforced by the imported verifier probe accepting replacement evidence. [Initial scope][initial_scope] [Per-claim callback][claim_callback]

Post-generation verification needs a narrower finding: its callback accepts `top`, while the engine passes `top_k`. The caught TypeError causes fallback to base documents. An isolated signature-binding probe confirmed this mismatch; this callback is therefore not a successful out-of-scope search through the real engine. A fake-engine test calls it without that keyword and misses the incompatibility. [Callback][post_callback] [Invocation][callback_call] [Test][callback_test]

Adaptive repair independently queries the Media retriever without selected IDs or source-family scope, then regenerates using that context. When repair runs with a Media DB path, it can include unselected media or Media evidence in a notes-only workflow. An **isolated control-flow probe** recorded the unfiltered retrieval arguments and replacement context; this was not a full API reproduction. [Repair retrieval][repair] [Pipeline invocation][verify_invocation]

### Empty verification and repaired-answer reports are ambiguous

An empty initial verification summary produces zero unsupported ratio and passes the threshold. An empty repair recheck can be marked fixed. Isolated probes confirmed both, plus legacy exception paths accepting repair after failed rechecks. Existing tests expect those legacy outcomes. Runtime-bound verification failures have explicit unavailable handling; do not generalize legacy behavior to every credential/runtime path. [Threshold][empty_check] [Recheck][repair_recheck]

Simple repair retains the old claims/summary and returns a replacement answer. Unified RAG adopts it after earlier citation/report construction. An isolated probe retained an old refuted claim while returning the new answer. Evidence must describe the final answer. A separate full-pipeline adaptive rerun has different handling. [Repair return][repair_return] [Answer replacement][answer_replace]

### Streaming overlays do not certify the final answer

Server streaming forwards limited claims settings, checks recent text under sentence/length/debounce conditions, and carries the latest overlay to its final event. No complete final-answer verification/repair/report step was found in this path; short answers can finish without an overlay. **Static trace**, not a streaming acceptance run. [Executor][stream_executor] [Overlay generation][overlay]

### FVA has earlier constructor failures

Falsification/anti-context retrieval exists, but its service constructs `MultiDatabaseRetriever` without required `db_paths` and supplies unsupported `top_k`/`search_mode` arguments to `RetrievalConfig`. Isolated signature-binding and dataclass-construction checks confirmed those incompatible calls. Full retriever import was blocked by the local FastAPI mismatch below. [Service][fva_service] [Constructor][retriever_constructor] [Configuration][retrieval_config]

Anti-context selected-item scope is also absent in this service path, but that finding is **conditional on fixing the constructor failures**. It was not reproduced as a successful unpatched HTTP request. Inspected FVA tests cover schemas/settings and mocked components rather than this service invocation. [Anti-context retrieval][anti_scope]

## WebUI and extension confirmation

### Research Workspace generates a separate final answer

The WebUI workspace loads the shared route; extension options aliases the same UI and exposes workspace/Knowledge QA routes. This reuse does not imply every extension surface has those screens. The sidepanel route registry does not expose those two pages. [WebUI page][webui_page] [Extension aliases][extension_aliases] [Sidepanel routes][sidepanel_routes]

Workspace sends selected IDs and forces Media-only retrieval. Its first preparation stage retains the raw RAG response, but final prompt preparation returns chat history, prompt, and sources, dropping the raw answer/claims/report. The chat pipeline then generates the displayed answer through a separate model stream. Enabling server RAG verification alone therefore cannot certify the displayed workspace answer. **Static trace.** [Request/preparation][workspace_scope] [Final preparation return][workspace_return] [Chat generation][chat_generation]

No dedicated workspace claims toggle was found. Ordinary chat supports an arbitrary `extra_body` object, but its adapter does not turn claim/report events into a durable detailed verification record. Transport capability is not a completed verification experience. [Chat adapter][chat_adapter]

### Knowledge QA has controls but incomplete inspection/reload

Knowledge QA is closer to the server RAG answer path. Its expert controls already exist; claims/post-verification default off and document-only/report default false. Explicit false overrides survive request construction, so changing defaults needs settings/preset precedence decisions. [Controls][expert] [Defaults/builder][defaults]

Nonstream responses can retain report/faithfulness temporarily. “Claim check” displays an aggregate count rather than opening detailed evidence; search details also show aggregates. Streaming summaries leave verification metrics null and `verificationReportAvailable: false`. Saved RAG context omits report/faithfulness, and reload restores citation/trust information rather than the detailed verification result. **Static trace**; focused client test collection was blocked. [Answer panel][answer_panel] [Details][search_details] [Streaming metadata][stream_metadata] [Saved context][saved_context]

WebUI `claims-review` redirects to Content Review, which reviews ingestion drafts rather than Claims-engine verdicts on answers. [Redirect][claims_redirect]

### Artifacts already carry useful verification metadata

Requests forward verifier overrides; successful artifact records retain `claimVerification` alongside verifier identity/mode/verdict metadata. Artifact detail displays identity/mode without complete per-claim review. Some failure paths reduce nested verification reports to generic error text. Reuse this metadata when designing inspection. [Generation hook][artifact_hook] [Detail][artifact_detail]

### Extension-specific paths differ

Request/proxy layers preserve source filters and serialized bodies; no extra verification stage was found there. **Static trace.** [Transport][rag_transport] [Proxy][proxy]

Sidepanel “Chat with current page” uses a separate helper. It ingests the page, then requests `media_db` with that Media ID. **Four temporary Bun probes passed** using injected clients and a stubbed credential-scope hash dependency: outgoing restriction; inline-excerpt fallback when no persisted ID exists; propagation of scope-change errors; and acceptance of injected response metadata bearing a different Media ID. The last shows the helper does not validate returned membership, not that a live server returned the wrong page. Its fallback uses inline page text instead of persisted-source RAG. [Website context helper][website_context]

Copilot popup sends selected text to ordinary chat and displays/replaces the response. No library allowlist or claims-report path was found. This is a separate surface, not evidence that Knowledge QA controls are absent. [Copilot][copilot]

Current quick ingestion has no dedicated claims control found in its curated advanced settings. Background ingestion can forward settings if supplied. A legacy schema-driven modal advertises a string field, which does not establish a reachable current boolean claims toggle. [Settings][quick_ingest] [Forwarding][ingest_forward]

## Verification results and limits

| Check | Result |
| --- | --- |
| Existing server tests | **35 passed**, zero failures/errors/skips; 202 warnings. Files: `test_claims_engine_modes.py`, `test_claims_engine_verification_status_fallback.py`, `test_artifact_verification.py`, `test_artifact_verification_properties.py`. |
| Imported Claims engine probes | **8 observations confirmed** with deterministic injected responses; JSON linked above. |
| Isolated server probes | Confirmed empty checks/repair, legacy failure behavior, stale repair summary, repair scope, callback mismatch, and FVA constructors. Dependency-stubbed checks, not full service execution. |
| Existing extension Bun tests | **5 passed**: one copilot lazy-import test and four unified-RAG builder tests. |
| Temporary extension helper probes | **4 passed** using the actual helper with injected clients. |
| WebUI focused Vitest checks | **Not collected** due local dependencies; no assertions passed/failed. |
| Latest-dev comparison | Relevant source diff between executed snapshot and latest checked dev is empty. |
| Browser, extension build, full API, external models | Not executed/certified. |

After activating the primary project virtual environment, pytest ran the four files under `tldw_Server_API/tests/Claims/` with `-o addopts='' -p no:cacheprovider`, temporary `--basetemp`/`--junitxml`, and `-q`. Model downloads, Docker-backed fixtures for this scope, and bytecode writes were disabled; a temporary test database was used. JUnit recorded 35 tests, zero failures/errors/skips, about 27.28 seconds. Existing plugin/configuration and deprecation warnings were not hidden failures.

Environment limits:

- Installed FastAPI `0.128.2` differs from required `>=0.142.1,<0.143.0`. Full retriever/app import fails because installed `fastapi.routing` lacks `iter_route_contexts`. No packages changed. [Requirement][fastapi_dependency] [Import][fastapi_import]
- WebUI attempts encountered missing `vite/internal` in the React/Vite plugin combination, then missing `react-router-dom`. The worktree has no own installed client dependency tree; temporary configuration reused primary-checkout dependencies.
- Existing/helper extension Bun tests succeeded with temporary configuration; a separate website-RAG Vitest test hit the missing router dependency.
- Imported Claims source SHA-256: `9965961bc431fb271d1686e4f13912bbbfe04dd8ff35208c67b3cfb976aa6743`.
- No full-suite, live-provider, browser, or deployment correctness claim follows.

## Implications for design

A complete first slice must verify the **final displayed answer**, restrict every retrieval/repair step to the selected corpus, distinguish unsupported/refuted/unavailable states, and retain evidence through streaming and reload.

Candidate discussion order:

1. Reliable verdict semantics and hard evidence requirements, including numeric/quotation failures and missing passage anchors.
2. One selected-corpus contract through retrieval, per-claim searches, repair, and anti-context checks.
3. Extract from the final answer and refresh verification/citations after repair; define zero-claim/provider-failure outcomes.
4. Choose first workflow: Workspace needs verification after its separate generation; Knowledge QA already displays the server RAG answer.
5. Inspectable evidence tied to answer/source identity and retained through terminal streaming state and reload.

These are findings and proposed discussion order, not approved implementation changes. The requester accepted supported facts with explicit gaps, declining when nothing useful is supported, and withholding draft answer text until verification completes while showing progress. Architecture, rollout scope, and the written spec remain under design review.

## ADR and change validation

ADR required: **no** for investigation/documentation. Future evidence contracts, API/persistence changes, or default behavior decisions need their own assessment.

Bandit is inapplicable to touched Markdown/JSON/tracking files; no Python/application code changed. Documentation verification passed for 53 immutable source paths/line anchors, four relative links, all eight saved observations, the imported-source checksum, JUnit counts, and current-dev source equivalence. The working diff passed whitespace checks; staged verification is recorded in the tracking task. The draft PR still requires the requester's human-written Change summary before merge.

[module]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/Claims_Extraction/README.md#L1
[decision]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/Claims_Extraction/claims_engine.py#L447
[guardrail_call]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/Claims_Extraction/claims_engine.py#L1307
[offsets]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/Claims_Extraction/claims_engine.py#L546
[run]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/Claims_Extraction/claims_engine.py#L1580
[verify_only]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/Claims_Extraction/claims_engine.py#L1784
[initial_scope]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py#L4394
[claim_callback]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py#L7763
[preclaims]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py#L7827
[preclaims_test]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_focused.py#L1973
[answer_extract]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py#L7934
[post_callback]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py#L428
[callback_call]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/Claims_Extraction/claims_engine.py#L1036
[callback_test]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/tests/RAG_NEW/unit/test_post_verifier.py#L968
[repair]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py#L554
[verify_invocation]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py#L8208
[empty_check]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py#L502
[repair_recheck]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py#L702
[repair_return]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py#L759
[answer_replace]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py#L8267
[stream_executor]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py#L703
[overlay]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/generation.py#L859
[fva_service]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/Claims_Extraction/claims_service.py#L4882
[retriever_constructor]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py#L4748
[retrieval_config]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py#L479
[anti_scope]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/Claims_Extraction/anti_context_retriever.py#L143
[artifact_verifier]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/Claims_Extraction/artifact_verification.py#L408
[artifact_gate]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/Research_Workspace/artifact_generation.py#L211
[webui_page]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/tldw-frontend/pages/research-workspace.tsx#L3
[extension_aliases]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/extension/wxt.config.ts#L9
[sidepanel_routes]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/routes/sidepanel-route-registry.tsx#L20
[workspace_scope]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/hooks/chat-modes/ragMode.ts#L470
[workspace_guards]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/hooks/chat-modes/ragMode.ts#L551
[workspace_return]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/hooks/chat-modes/ragMode.ts#L645
[chat_generation]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts#L808
[chat_adapter]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/models/ChatTldw.ts#L197
[expert]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/components/Option/KnowledgeQA/SettingsPanel/ExpertSettings.tsx#L1275
[defaults]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/services/rag/unified-rag.ts#L350
[answer_panel]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/components/Option/KnowledgeQA/AnswerPanel.tsx#L816
[search_details]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/components/Option/KnowledgeQA/SearchDetailsPanel.tsx#L233
[stream_metadata]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx#L1364
[saved_context]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx#L2219
[claims_redirect]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/tldw-frontend/pages/claims-review.tsx#L3
[artifact_hook]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx#L1745
[artifact_detail]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/TraceableArtifactDetail.tsx#L274
[rag_transport]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/services/tldw/domains/chat-rag.ts#L124
[proxy]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/entries/background.ts#L1699
[website_context]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/hooks/useMessage.website-context.ts#L76
[copilot]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/entries/copilot-popup.content.tsx#L625
[quick_ingest]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/components/Common/QuickIngest/WizardConfigureStep.tsx#L873
[ingest_forward]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/apps/packages/ui/src/entries/background.ts#L2951
[fastapi_dependency]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/pyproject.toml#L56
[fastapi_import]: https://github.com/rmusser01/tldw_server/blob/9958110df2a9011e19f48b0eae821353e19d4af8/tldw_Server_API/app/core/Utils/fastapi_routes.py#L30
