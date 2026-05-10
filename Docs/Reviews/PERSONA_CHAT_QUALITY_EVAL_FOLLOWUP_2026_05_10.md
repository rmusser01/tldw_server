# Persona Chat Quality And Evaluation Follow-Up

Date checked: 2026-05-10

## Summary

Persona-backed ordinary chat already exists. The current work is not to invent a new chat runtime, but to define how to measure and improve quality for the existing path without folding it back into Persona Live, Buddy shell diagnostics, or VN/CYOA work.

The current implementation has three related but distinct strands:

- Ordinary persona-backed chat uses explicit assistant identity fields: `assistant_kind=persona`, `assistant_id`, and `persona_memory_mode`.
- Persona-owned exemplars and shared prompt assembly are used by ordinary persona chat and Persona Live.
- The older character role-play exemplar stack still owns several telemetry/evaluation primitives that are useful but partly character-shaped.

Stage 2 should therefore start with a repo-grounded error-analysis and fixture pass. Add judges only after failure modes are named, sampled traces exist, and deterministic checks cover what can be checked without a model.

## Tracker Context

- Parent epic: [#1510](https://github.com/rmusser01/tldw_server/issues/1510)
- Stage 2 issue: [#1543](https://github.com/rmusser01/tldw_server/issues/1543)
- Superseded legacy tracker: [#635](https://github.com/rmusser01/tldw_server/issues/635)
- Separate VN/CYOA tracker: [#1391](https://github.com/rmusser01/tldw_server/issues/1391)

The Stage 0 audit explicitly kept Persona Chat quality out of Stage 1 Buddy/Live reliability. Stage 1 reliability issues are now closed or tracked separately; this document defines the next Persona Chat quality/eval workstream.

## Preserved Legacy Inputs

The useful #635 references should be treated as inspiration for quality axes and evaluation design, not as direct implementation requirements:

| Reference | Recheck result | Stage 2 use |
| --- | --- | --- |
| StickToYourRoleLeaderboard | Current persona-backed chat has identity, exemplar, memory-mode, and telemetry hooks, but no ordinary-chat role-adherence trace taxonomy or sampled label set. Existing tests prove plumbing rather than role consistency across varied prompts. | Use as inspiration for Slice 1 failure labels and later human review rubrics; do not adopt its benchmark format as a dependency. |
| UGI-Leaderboard | Current tests cover assistant identity submission and memory mode boundaries, but do not yet distinguish instruction-following failure from persona-specific drift in ordinary chat. | Use to split general instruction-following errors from persona-role errors in the taxonomy. |
| NovelChallenge | The repo has character role-play exemplar tests and dialogue-tree robustness recipes, but ordinary persona-backed chat lacks long-horizon continuity fixtures after reopen, memory-mode changes, and exemplar changes. | Use as inspiration for deterministic continuity fixtures in Slice 2 after the trace taxonomy is defined. |
| Guided Generations | Persona exemplar assembly already injects boundary and style guidance through shared prompt assembly; there is no need to change runtime architecture just to support guided responses. | Use to define prompt-assembly and constraint-presence checks, not a new generation engine. |
| semperai/amica | The Buddy/Persona Live shell and visual/avatar work is tracked separately from ordinary Persona Chat quality. No Stage 2 chat-quality gap requires adopting avatar product behavior. | Keep as product inspiration only; do not make avatar/live interaction a prerequisite for Persona Chat evaluation. |
| LlamaTale | Story/VN style interaction remains separate in #1391. Current Persona Chat quality work should not inherit branching-story runtime requirements. | Keep VN/CYOA out of this Stage 2 scope; only borrow role-consistency scenario ideas if they fit ordinary chat traces. |

Recheck summary: the #635 references mostly point to evaluation categories rather than missing runtime infrastructure. The current repo already has persona-backed chat identity, persona exemplar prompt assembly, memory-mode controls, frontend create/restore paths, and some telemetry/evaluation primitives. The material gap is evidence quality: ordinary persona chat does not yet have trace-backed failure labels, deterministic scenario fixtures, or user-visible effective-context checks. That directly drives the Slice 1 to Slice 5 sequence below.

## Current Contract Inventory

| Area | Evidence | Current behavior | Stage 2 implication |
| --- | --- | --- | --- |
| Persona chat projection | `tldw_Server_API/app/core/Chat/chat_service.py:450` and `:467` | A persona profile is projected into the minimal assistant card needed by ordinary chat. Missing persona ids raise conflict; missing profiles raise 404. | Quality work can rely on persona as first-class assistant identity instead of source-character fallback. |
| Conversation identity | `tldw_Server_API/app/api/v1/endpoints/chat.py:4673` | Conversation responses include `character_id`, `assistant_kind`, `assistant_id`, and `persona_memory_mode`. | Eval fixtures should record assistant identity fields, not only character ids. |
| Frontend creation | `apps/packages/ui/src/hooks/chat/personaServerChat.ts:96` | Selecting a persona creates or reuses a server chat with `assistant_kind=persona` and default `persona_memory_mode=read_only`. | UX/eval slices can assume normal chat is the entry point; no Persona Live websocket is required. |
| Frontend restore | `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts:261` and `:775` | Server chat metadata resolves assistant identity and reloads persona profile presentation when reopening chats. | Stage 2 should smoke restore behavior before adding subjective quality tests. |
| Persona picker | `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx:1575` | The existing picker has `Characters` and `Personas` tabs and selects personas as assistant identities. | Follow-up UX should focus on effective-context visibility, not creating another picker. |
| Memory mode control | `apps/packages/ui/src/components/Common/Settings/tabs/ConversationTab.tsx:1064` | Persona memory mode is visible only for persona-backed chats; `read_only` is the default and `read_write` is explicit. | Quality tests should distinguish read-only continuity from durable writeback expectations. |
| Runtime exemplar assembly | `tldw_Server_API/app/api/v1/endpoints/chat.py:1058` and `tldw_Server_API/app/core/Persona/exemplar_prompt_assembly.py:111` | Shared persona exemplar sections append boundary and style guidance when persona-backed chat has enabled exemplars. | Deterministic tests can verify selection, section presence, and prompt-injection boundary guidance. |
| Exemplar lookup | `tldw_Server_API/app/api/v1/endpoints/chat.py:3365` | Persona-backed chat lists persona-owned exemplars off the event loop and assembles runtime guidance from the current turn text. | Error analysis should include selection misses, wrong exemplar class, and over-selection. |
| Memory writeback | `tldw_Server_API/app/api/v1/endpoints/chat.py:1234` and `:1244` | Durable persona memory writes only happen for `persona_memory_mode=read_write`; current persistence stores assistant reply text as a persona turn. | Stage 2 should test and explain what is read, what is written, and what is not written. |
| Telemetry primitives | `tldw_Server_API/app/api/v1/endpoints/chat.py:1286` and `tldw_Server_API/tests/Evaluations/test_persona_telemetry_metrics_summary.py:52` | IOO, IOR, LCS, and safety counters exist for persona-style telemetry. Labels remain character-shaped in places. | First quality slice should normalize persona-backed labels and avoid misleading character-only summaries. |
| Existing backend tests | `tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py:187` and `:448` | Tests cover persona identity, exemplar guidance, prompt preview parity, current-turn classification, async lookup, memory read-only, and read-write behavior. | The gap is not basic coverage; it is scenario coverage and quality failure taxonomy. |
| Existing frontend tests | `apps/packages/ui/src/hooks/chat/__tests__/personaServerChat.test.ts:24` and `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx:230` | Tests cover persona chat creation/reuse and action submission with assistant identity. | Add restore/effective-context tests before adding new UI. |
| Character role-play eval stack | `Docs/Product/Completed/Persona_Roleplay_PRD.md:170`, `tldw_Server_API/tests/Chat_NEW/integration/test_chat_persona_exemplars_integration.py:64`, and `tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_selector.py:35` | Character-scoped exemplar selection, debug metadata, telemetry, and performance tests exist. | Reuse ideas carefully; do not conflate character exemplar metrics with persona-backed chat identity. |
| Dialogue-tree robustness | `tldw_Server_API/app/core/Evaluations/README.md:58` and `tldw_Server_API/tests/Evaluations/test_persona_dialogue_tree_recipe.py:35` | Defensive persona dialogue-tree recipe exists for robustness, policy, privacy, and grounded refusal behavior. | Treat as safety/robustness substrate, not as a direct Persona Chat role-adherence score. |

## Quality Axes

Stage 2 quality work should use these axes as the initial review rubric:

1. Role adherence
   - The response follows persona profile instructions, persona-owned boundary exemplars, and style exemplars without copying exemplars verbatim.
   - It does not drift into the source character or generic assistant voice when persona identity is available.

2. Boundary and policy behavior
   - Prompt-reveal, tool-pressure, unsafe, or impossible requests are refused or redirected while staying consistent with persona boundaries.
   - The persona does not claim capabilities it does not have.

3. Memory semantics
   - `read_only` may read available persona memory/state but must not write durable persona memory.
   - `read_write` can write durable persona memory only through explicit mode selection and should record enough provenance to audit why a memory exists.

4. Grounding and source visibility
   - Users can tell whether a response was shaped by persona profile, exemplars, memory, chat history, RAG/media context, or no extra context.
   - Quality evaluation should separate persona consistency from factual/RAG grounding.

5. Conversation continuity
   - Reopened persona-backed chats preserve assistant identity, memory mode, and visible persona presentation.
   - Session-scoped behavior should not imply durable personality evolution unless memory writeback is enabled.

6. Over-copy and under-use
   - High overlap with exemplars should be flagged as over-copy risk.
   - Low or irrelevant exemplar use should be flagged as retrieval/selection miss, not automatically as poor generation.

7. UX confidence
   - Effective persona, memory mode, and context sources are visible enough for a user to understand why the reply behaved the way it did.

## Gaps And Risks

| Gap | Evidence | Risk | Recommended handling |
| --- | --- | --- | --- |
| No trace-backed error taxonomy for ordinary persona chat quality | Current tests assert prompt wiring and mocked provider calls, not human-labeled output failures. | Judges or metrics could become vanity checks. | Start with representative trace capture and human error analysis before adding judges. |
| Character-shaped metrics remain mixed with persona-backed chat | Telemetry labels include `character_id` while persona-backed chat may have no character id. | Persona Chat dashboards can undercount or misattribute results. | Normalize labels to include assistant kind and assistant id in a focused metrics follow-up. |
| Effective context is distributed | Persona profile, exemplars, memory mode, chat history, RAG, and source context are inspected in different places. | Users and reviewers cannot explain why a persona answered a certain way. | Add a compact effective-context summary before broad quality UI. |
| Memory writeback semantics are underexplained | UI exposes read-only/read-write, and backend enforces writeback gates, but user-visible provenance is limited. | Users may assume ordinary chat is always training or never remembering. | Add explicit copy/tests for read-only vs read-write and provenance of writes. |
| Existing dialogue-tree recipe is not the normal chat eval harness | It is defensive robustness infrastructure, not a direct ordinary-chat quality score. | Stage 2 could accidentally overfit to robustness while ignoring normal chat coherence. | Keep dialogue-tree as one robustness dimension; add normal persona-chat fixtures separately. |
| External #635 references are broad | Links span roleplay, guided generation, avatars, and story systems. | Scope drift into VN/CYOA or avatar/rendering work. | Use them only to inspire quality axes; keep runtime ownership in ordinary chat and separate #1391 for VN/CYOA. |

## PR-Sized Follow-Up Slices

### Slice 1: Persona Chat Trace And Error Taxonomy

Goal: create the evidence base for Stage 2 without changing runtime behavior.

Scope:

- Add or document a small trace sampling workflow for ordinary persona-backed chat.
- Define a human labeling rubric using the quality axes above.
- Produce fixture examples for role drift, prompt-reveal pressure, memory-mode confusion, exemplar over-copying, exemplar miss, and restore/context mismatch.

Acceptance:

- A review artifact records at least 20 representative persona-chat cases or explicitly documents why fixtures are synthetic.
- Each failure label maps to a deterministic check, an optional judge candidate, or a human-only review note.
- No new LLM judge is trusted until calibration criteria are specified.

### Slice 2: Deterministic Persona Chat Quality Fixtures

Goal: turn the first error-taxonomy cases into stable regression tests.

Scope:

- Extend backend tests around `test_persona_backed_chat_conversations.py` and prompt assembly.
- Cover persona identity preservation, profile/source-character independence, memory mode, prompt-reveal boundary exemplars, and exemplar selection misses.
- Add frontend restore/effective-assistant identity smoke coverage if gaps remain.

Acceptance:

- Tests fail for at least one known missing or brittle behavior before implementation.
- Existing ordinary character chat behavior remains unchanged.
- Tests name persona-backed chat explicitly instead of overloading character-only paths.

### Slice 3: Persona Telemetry Label Normalization

Goal: make existing metrics usable for persona-backed chat.

Scope:

- Add assistant identity labels such as `assistant_kind` and `assistant_id` where safe.
- Keep backward-compatible character labels for character-backed role-play metrics.
- Update `persona_telemetry_metrics` summaries so persona-backed chat is not collapsed into `character_id=none`.

Acceptance:

- Metrics tests cover persona-backed and character-backed label shapes.
- No secrets or raw prompt/exemplar text enter labels.
- Existing evaluation metrics summary keeps old totals while exposing assistant-kind splits.

### Slice 4: Effective Persona Context Preview

Goal: make quality failures diagnosable by users and reviewers.

Scope:

- Add or extend a backend/frontend preview that reports effective persona profile, memory mode, exemplar section ids/counts, chat/RAG context status, and policy/tool constraints.
- Prefer existing prompt-preview and settings surfaces over adding a new product area.
- Keep raw exemplar and memory content bounded/redacted.

Acceptance:

- Persona-backed chat has a visible effective-context summary before or during a turn.
- The summary distinguishes profile, exemplars, memory, conversation history, and RAG/media sources.
- Tests prove read-only vs read-write copy and no-persona/missing-profile states.

### Slice 5: Optional LLM-As-Judge Evaluation After Calibration

Goal: add subjective scoring only after deterministic and human-labeled foundations exist.

Scope:

- Create a judge prompt for role adherence, boundary behavior, and memory-grounding only after labeled examples exist.
- Validate the judge against held-out human labels.
- Keep deterministic checks as gates before judge scoring.

Acceptance:

- Judge validation records TPR/TNR or equivalent agreement measures.
- Known bias cases are documented, especially generic-assistant bias and over-rewarding theatrical style.
- The judge is advisory and does not replace deterministic safety or policy checks.

## Non-Goals

- Do not reopen Stage 1 Buddy/Live diagnostics as a dependency for Persona Chat quality.
- Do not add native/background wake behavior.
- Do not add renderer/provider/Live2D work.
- Do not route this through VN/CYOA runtime ownership; keep #1391 separate.
- Do not create a parallel eval/run/status system outside the existing Evaluations and Jobs paths.
- Do not rely on external roleplay benchmarks without local traces, fixtures, and calibration.

## Immediate Recommendation

Open the next implementation task as Slice 1: Persona Chat Trace And Error Taxonomy. Keep it docs/test-fixture focused and require a review step before any runtime or judge implementation. Slice 2 can follow only after the taxonomy names concrete failure modes and expected deterministic coverage.

## Verification

This document is a planning/audit artifact. Runtime tests are not required for this slice because no runtime code is changed.

Required closeout checks:

```bash
rg -n "TO[D]O|TB[D]|FIX[M]E|PLACE[H]OLDER|\\?\\?" Docs/Reviews/PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md
git diff --check
```

Bandit is not applicable unless future slices touch Python runtime or test files.
