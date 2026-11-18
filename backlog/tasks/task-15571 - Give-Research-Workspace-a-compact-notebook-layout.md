---
id: TASK-15571
title: Give Research Workspace a compact notebook layout
status: Done
labels:
- webui
- ux
- research-workspace
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved notebook layout in the existing shared owner workspace so research sources, conversations and saved outputs have priority on desktop, tablet and phone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Three panes fit available desktop space and reachable tabs cover constrained widths with accessible pane handoffs.
- [x] #2 Compact header, source organization and contextual chat controls reclaim reading space while preserving existing actions and recovery.
- [x] #3 Studio leads with saved outputs, exposes generation on demand and avoids fixed empty sections.
- [x] #4 Verify owner workspace behavior, shared-client contracts and desktop/tablet/phone rendering.
- [x] #5 Verify real selected-source retrieval, cited passage preview, and generated Studio output with the local model.
- [x] #6 With queued execution disabled, healthy chat reports ready and the workspace shows no degradation warning. Preserve warnings for an expected missing queue and unhealthy providers. Rerun the five local-model workflows.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR check: no new ADR. Docs/ADR/007-research-workspace-canonical-first-slice-shell.md governs the existing shared shell. No API or data ownership change. Plan stages: shell/navigation, sources/chat/studio, cross-client verification and review.
Approved mockup reviewed 2026-11-17. Implementation in codex/research-notebook-layout. Verify existing linked WebUI/extension contract. Stages and objectives in IMPLEMENTATION_PLAN_research_notebook_layout.md during development.
Implemented the shell and pane changes. Browser checks cover source intake, keyboard RAG, tablet citation preview with draft preservation, Quick Notes keyboard focus, Review/Download, generation cancellation/reload Retry, desktop restore rails and Chrome extension parity. Independent review found first-visit Studio discussion, cross-workspace draft isolation, and constrained-width tour targets; fixing and verifying all three before delivery.
Independent review fixes verified. Chat receives Studio discussions on first entry, drafts are scoped to the active chat, and every workspace tour pane has a visible tab fallback. The shared workspace and tutorial suite passed 1,530 tests across 83 files. Chromium passed all 18 WebUI browser workflows together, including desktop/tablet/phone navigation and light/dark composer readability. The production Chrome extension build and its shared parity workflow passed. Frontend TypeScript, scoped ESLint, and git diff --check passed. Bandit ran on the touched scope, with zero Python lines and zero findings. Live provider and real-backend generation were not exercised. User-facing guidance and tour targets were updated in the UI.
Follow-up requested live-backend verification using isolated test data and a real local inference server.
Live follow-up passed all five selected Chromium workflows in one run (2.0 minutes, zero skips). Used the real FastAPI service, Qwen2.5-0.5B-Instruct via local llama.cpp, and installed all-MiniLM-L6-v2 embeddings. Grounded chat returned 37 blue herons from the selected document, omitted the distractor from retrieval, and opened the correct source and quoted passage. Workspace search found the live conversation. The saved Studio comparison matched the real completion. Quiz questions and flashcards were persisted and their workspace/general visibility and record IDs checked. Fixed an early pass on streaming status and an obsolete flashcard modal selector in the live tests. TypeScript, scoped ESLint and diff checks passed. Bandit has no Python scope in this follow-up. No production code changed.
Queue-off follow-up requested after the rendered notebook showed a false LLM warning. Reproduce the backend health result, align it with queued execution configuration, then verify the API, rendered composer and five live browser workflows.
Shared queued-execution configuration across startup, chat routing, the LLM health endpoint and Research Workspace readiness. A missing queue affects health only when enabled. Red/green confirmed the queue-off regression. Six health regression scenarios cover default off, config-only enablement, environment overrides, unhealthy providers and open provider circuits. All 122 targeted Python tests passed. Ruff reported the same pre-existing findings as HEAD. Bandit found zero issues in the touched production code and flagged the two added pytest asserts in test code. Python compilation and git diff --check passed.
Live checks with CHAT_QUEUED_EXECUTION=off: startup logged the disabled queue, LLM health returned healthy with initialized=false, and workspace chat returned ready/allow. Chromium verified the banner was absent before and after a real cited answer and saved Studio comparison. Grounded chat with cited passage, conversation search, Studio comparison, and flashcard generation and persistence passed. Quiz did not pass this run: local Qwen2.5-0.5B repeated the prompt example on two runs and claim verification rejected it (HTTP 422). An unchanged quiz run using the locally cached Qwen3.8-27B reached the provider read timeout (HTTP 502). No validators or tests were relaxed. Preserved render and test evidence in the task visualization directory and /private/tmp/rw-queue-off.HPsSbw.
The queue-off fix is present in the notebook worktree. This follow-up remains uncommitted because the required live quiz test is red. The last larger-model attempt returned HTTP 502 after the provider read timed out at 90 seconds.
Luna-only verification: OpenAI confirmed access to gpt-6-luna. The unmodified notebook branch returned HTTP 400 on its Luna chat preflight. The verified existing GPT-6 request-format change in 2355113b80 was applied solely in a temporary test copy of this worktree; all 47 GPT-6 adapter regression cases passed there. With queued execution off, the real Chromium suite passed source-scoped chat and citation preview, Studio comparison, conversation search and quiz generation/persistence. Flashcard generation twice returned HTTP 422 because the claims verifier accepted five of six claims. An additional factual survey source pair also received needs_revision (five of six). Traces confirm the disputed answer is present in the supplied text. Verifier behavior and the actual notebook branch adapter were left intact. The readiness bar trace showed unconfigured MCP policy, blocked ACP agent setup, and a third no_queryable_sources item while the selected-source chat succeeded. The workspace status response still listed zero selected sources. Evidence: /private/tmp/rw-luna-queue-off.9nGlqP.
Repeated live verification with the OpenAI key specifically designated by the user in the Chatbook project. The provider model lookup returned HTTP 200 for gpt-6-luna. With queued execution off, this branch reported ready chat but received HTTP 400 on a Luna completion. In the isolated copy with the existing 2355113b80 Luna adapter change, Chromium completed grounded chat and cited passage preview, Studio source comparison, conversation search, and quiz generation/persistence. The flashcards endpoint returned HTTP 422: five of six claims verified, and the remaining answer was verbatim in the submitted source text. No credentials were copied into the repository. Browser test evidence in /private/tmp/rw-luna-chatbook.T0pb5s/five.log.
User requested root-cause repairs for the Luna HTTP 400, false flashcard-verification rejection, and misleading Workspace Readiness source state. Keep provider handling in the existing adapter boundary and retain evidence validation and workspace capability contracts.
Root causes confirmed in browser traces and failing regression tests. Luna inherited token and sampling fields rejected by GPT-6 reasoning. Reused the native OpenAI request normalization from 2355113b80 with streaming, non-streaming, legacy model and explicit reasoning tests. The flashcard judge received a fixed 480-character prefix that cut off the supporting passage. Semantic claims now select overlapping, relevant passages while still requiring the judge to verify the question and answer together. Concurrent source reconciliations let an older empty selection overwrite the newer selection after readiness fetched. Writes now run in order and the page refreshes readiness after saving. An empty selection no longer counts as a source failure. Existing decisions ADR-007 and ADR-025 govern the same workspace and adapter boundaries.
Live verification on the notebook branch with the Chatbook OpenAI key, gpt-6-luna and queued execution off covered all five workflows across runs: selected-source chat and cited passage, comparison, search, quiz persistence and flashcard persistence. The grounded trace recorded one selected source and grounded readiness after the write. Flashcards passed with the original source pair. A local per-excerpt NLI contradiction on a cross-source artifact now goes to the full-evidence judge, with ambiguity remaining blocked. The quiz verification fixture now contains two factual survey notes after the old placeholder fixture elicited an unsupported explanation that the guard correctly blocked. Targeted final backend suite: 145 passed. Targeted final WebUI suite: 71 passed. Bandit found zero issues on changed lines. Evidence: /private/tmp/rw-root-fix.gU9icp.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Luna request compatibility, flashcard evidence truncation, cross-source claims verification, and stale source selection and readiness. Verified the real notebook workflows with Luna and the queue disabled. The existing owner workspace, provider adapter boundary and grounding gate remain in use.
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
