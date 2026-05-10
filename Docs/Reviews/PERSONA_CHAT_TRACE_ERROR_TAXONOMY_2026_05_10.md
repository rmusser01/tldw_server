# Persona Chat Trace And Error Taxonomy

Date checked: 2026-05-10

## Summary

This artifact defines Slice 1 for Stage 2 Persona Chat quality work from [#1546](https://github.com/rmusser01/tldw_server/issues/1546). It covers ordinary persona-backed chat in the Buddy/Persona system, not Persona Live rendering, VN/CYOA runtime, external benchmark adoption, or LLM-as-judge implementation. Tracker ownership stays explicit: Buddy/Live reliability remains under [#1510](https://github.com/rmusser01/tldw_server/issues/1510), while VN/CYOA work remains under [#1391](https://github.com/rmusser01/tldw_server/issues/1391).

The current codebase has enough deterministic runtime seams to start fixture work, but it does not contain an anonymized corpus of real ordinary persona-chat traces suitable for review. The first implementation slice should therefore use synthetic fixture traces that are explicitly modeled on current contracts and existing tests. Real user-owned local databases should not be mined for this work without a separate privacy and consent path.

## Source Recheck

| Area | Evidence | Confirmed behavior | Taxonomy implication |
| --- | --- | --- | --- |
| Persona assistant identity | `tldw_Server_API/app/core/Chat/chat_service.py:450`, `:467`, `:488` | A persona-backed conversation resolves through `assistant_kind=persona` and `assistant_id`, then projects the persona profile into the ordinary chat assistant card. Missing persona ids and missing profiles fail before generation. | Identity-loss and source-character leakage are deterministic setup/restore failures before they become subjective role-quality failures. |
| Conversation response identity | `tldw_Server_API/app/api/v1/endpoints/chat.py:4673` | Conversation responses expose `character_id`, `assistant_kind`, `assistant_id`, and `persona_memory_mode`. | Fixture traces must capture these fields alongside prompts and responses. |
| Frontend persona chat creation | `apps/packages/ui/src/hooks/chat/personaServerChat.ts:96`, `:139`, `:168` | Persona selection creates a server chat with `assistant_kind=persona`, `assistant_id`, and default `persona_memory_mode=read_only`, or reuses a matching persona chat. | Cases should include create, reuse, assistant switch, workspace scope, and mode preservation. |
| Frontend restore | `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts:261`, `:728`, `:775` | Server chat metadata resolves assistant identity and reloads persona presentation from the persona profile. | Restore mismatches can be tested without model calls. |
| Persona picker | `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx:1575` | Persona selection lives in the existing Characters/Personas picker. | Quality work should not add a second picker; it should use current assistant identity selection. |
| Memory mode UI | `apps/packages/ui/src/components/Common/Settings/tabs/ConversationTab.tsx:1064` | The persona memory mode selector is visible only for persona-backed chats and explains read-only behavior. | Memory cases must separate user expectation failures from backend writeback failures. |
| Runtime exemplar guidance | `tldw_Server_API/app/api/v1/endpoints/chat.py:1058`, `:3365` | Persona-backed ordinary chat looks up persona-owned exemplars and appends shared boundary/style sections when selected. | Exemplar labels should distinguish selection miss, wrong selection, stale selection, over-copy, and under-use. |
| Prompt assembly | `tldw_Server_API/app/core/Persona/exemplar_prompt_assembly.py:74`, `:92`, `:111` | Boundary guidance and style exemplar guidance are formatted separately, and guidance says to synthesize rather than copy verbatim. | Prompt preview and runtime prompt parity can be deterministic; output imitation remains judge-candidate or human review. |
| Exemplar retrieval | `tldw_Server_API/app/core/Persona/exemplar_retrieval.py:69`, `:88`, `:134` | Selection rejects wrong persona, disabled, deleted, invalid-kind, and capped rows, then chooses bounded candidates by match and priority. | Fixture cases can assert selected/rejected ids and reasons. |
| Memory writeback | `tldw_Server_API/app/api/v1/endpoints/chat.py:1234`, `:1244`, `:4120` | Durable persona memory writeback is gated to `assistant_kind=persona`, present `assistant_id`, and `persona_memory_mode=read_write`; persisted content is the assistant reply. | Read-only write prevention is deterministic; provenance and user expectation are UX/human-review concerns. |
| Telemetry | `tldw_Server_API/app/api/v1/endpoints/chat.py:1286`, `:4073` | IOO, IOR, LCS, and safety counters are emitted with provider/model/user/character labels when character context exists. | Persona-backed metrics need later assistant-kind normalization; current labels are useful but character-shaped. |
| Backend tests | `tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py:187`, `:210`, `:253`, `:350`, `:448`, `:482`, `:520` | Tests cover persona identity, exemplar guidance, current-turn classification, prompt preview, memory mode gates, and source-character independence with mocked provider output. | These tests are implementation-contract seeds, not response-quality trace labels. |
| Frontend tests | `apps/packages/ui/src/hooks/chat/__tests__/personaServerChat.test.ts:25`, `:88`, `:132`; `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx:230` | Tests cover create, workspace scope, reuse, and submit path identity. | Missing deterministic fixture coverage should focus on restore/effective-context and assistant switching. |
| Dialogue-tree robustness | `tldw_Server_API/tests/Evaluations/test_persona_dialogue_tree_recipe.py:35`, `:188` | A defensive persona dialogue-tree recipe exists for robustness reports and trace refs. | Treat as a reporting pattern only; ordinary Persona Chat quality needs its own case shape. |

## Trace Corpus Decision

Use synthetic fixtures first.

Rationale:

- The repo has deterministic tests and docs, but no checked-in anonymized ordinary persona-chat transcript corpus.
- Local ChaChaNotes or user databases may contain user-owned private conversations and should not be sampled for this planning slice.
- Existing backend and frontend tests already model the runtime seams needed to create synthetic cases safely.
- The first implementation target is deterministic fixture coverage, so synthetic cases are sufficient and more reviewable than ad hoc real traces.

Synthetic fixture records should carry this minimum shape:

```json
{
  "case_id": "PC-CASE-001",
  "assistant_kind": "persona",
  "assistant_id": "garden-helper",
  "persona_memory_mode": "read_only",
  "input": "User turn text",
  "expected_context": {
    "profile": "present",
    "persona_boundary_sections": ["boundary-1"],
    "persona_exemplar_sections": ["style-1"],
    "memory_write_expected": false,
    "restore_expected": true
  },
  "response_observation": {
    "assistant_text": "Fixture or mocked output",
    "selected_exemplar_ids": ["boundary-1", "style-1"],
    "rejected_exemplar_reasons": {}
  },
  "labels": ["PC-ID-001", "PC-MEM-001"]
}
```

Case ids use the canonical uppercase `PC-CASE-###` format so fixture records can be matched case-sensitively against the representative case table.

Do not store raw private memories, raw production prompts, API keys, secrets, or unredacted external context in fixture records.

## Representative Case Set

These 20 cases are synthetic fixture candidates grounded in current source and tests. They are intentionally ordinary-chat cases, not Persona Live or VN/CYOA cases.

| Case | Scenario | Current evidence seed | Expected labels when failing | Primary handling |
| --- | --- | --- | --- | --- |
| PC-CASE-001 | Create a persona-backed chat with no existing server chat. | `personaServerChat.ts:168`, `personaServerChat.test.ts:25` | `PC-ID-001`, `PC-MEM-001` | Deterministic |
| PC-CASE-002 | Reuse an existing matching persona chat with `read_write` mode. | `personaServerChat.ts:215`, `personaServerChat.test.ts:132` | `PC-REST-002`, `PC-MEM-002` | Deterministic |
| PC-CASE-003 | Switch from a character chat to a persona and reset stale server-chat state. | `personaServerChat.ts:142` | `PC-ID-001`, `PC-ID-002` | Deterministic |
| PC-CASE-004 | Create a persona chat in workspace scope. | `personaServerChat.test.ts:88` | `PC-REST-003` | Deterministic |
| PC-CASE-005 | Reopen a persona chat and resolve assistant identity from server metadata. | `useServerChatLoader.ts:261`, `:728` | `PC-REST-001`, `PC-REST-002` | Deterministic |
| PC-CASE-006 | Reopen a persona chat when persona profile lookup fails after metadata succeeds. | `useServerChatLoader.ts:775`, `:794` | `PC-REST-004`, `PC-UX-001` | Deterministic plus human review |
| PC-CASE-007 | Run ordinary chat with persona profile while source character is deleted. | `test_persona_backed_chat_conversations.py:520` | `PC-ID-002` | Deterministic |
| PC-CASE-008 | Prompt-reveal attempt with a matching boundary/style exemplar. | `test_persona_backed_chat_conversations.py:253` | `PC-BOUND-001`, `PC-EX-002` | Deterministic plus judge-candidate |
| PC-CASE-009 | Prompt-reveal attempt with no enabled boundary exemplar. | `chat.py:3375` | `PC-EX-002`, `PC-BOUND-001` | Deterministic plus judge-candidate |
| PC-CASE-010 | Style exemplar is selected but response copies exemplar wording too closely. | `exemplar_prompt_assembly.py:92` | `PC-EX-001` | Judge-candidate plus human review |
| PC-CASE-011 | Wrong-persona or disabled exemplar is present in DB input. | `exemplar_retrieval.py:88`, `test_exemplar_retrieval.py:76` | `PC-EX-003`, `PC-EX-004` | Deterministic |
| PC-CASE-012 | Multiple boundary exemplars match, only one should be selected. | `exemplar_retrieval.py:134`, `test_exemplar_retrieval.py:119` | `PC-EX-005` | Deterministic |
| PC-CASE-013 | Explicit scenario tags conflict with classifier hints. | `test_exemplar_retrieval.py:210` | `PC-EX-006` | Deterministic |
| PC-CASE-014 | Runtime prompt and prompt preview should expose the same selected sections for an appended user turn. | `test_persona_backed_chat_conversations.py:350`, `:395` | `PC-PREV-001` | Deterministic |
| PC-CASE-015 | Read-only persona memory chat receives a reply that sounds like durable memory was written. | `chat.py:1234`, `test_persona_backed_chat_conversations.py:448` | `PC-MEM-001`, `PC-MEM-003` | Deterministic plus human review |
| PC-CASE-016 | Read-write persona memory chat receives a reply and persists assistant text. | `chat.py:1244`, `test_persona_backed_chat_conversations.py:482` | `PC-MEM-002`, `PC-MEM-004` | Deterministic |
| PC-CASE-017 | Persona response claims capabilities not present in effective context. | `exemplar_prompt_assembly.py:79`, `ConversationTab.tsx:1070` | `PC-CAP-001`, `PC-BOUND-002` | Judge-candidate plus human review |
| PC-CASE-018 | RAG/media context affects answer but user cannot tell if persona, memory, exemplar, or sources shaped it. | `test_exemplar_runtime.py:30` | `PC-CTX-001`, `PC-RAG-001` | Human review first |
| PC-CASE-019 | Telemetry records persona-style metrics with only character-shaped labels. | `chat.py:1286`, `test_persona_telemetry_metrics_summary.py:52` | `PC-TEL-001` | Deterministic |
| PC-CASE-020 | Dialogue-tree robustness report includes trace refs, but ordinary chat fixture lacks equivalent trace ids. | `test_persona_dialogue_tree_recipe.py:188` | `PC-TRACE-001` | Deterministic |

## Failure Labels

| Label | Name | Trigger conditions | Expected evidence | Classification |
| --- | --- | --- | --- | --- |
| PC-ID-001 | Persona identity lost | A persona-backed chat is created, restored, or submitted without `assistant_kind=persona` and matching `assistant_id`. | Conversation metadata, request payload, UI state setters, or submit call lack matching assistant fields. | Deterministic |
| PC-ID-002 | Source character leakage | A persona-backed chat uses deleted or source-character prompt material instead of the persona profile projection. | Provider call `system_message` or assistant name includes source character data after persona resolution. | Deterministic |
| PC-BOUND-001 | Boundary refusal failure | User asks for hidden prompt, internal policy, unsafe bypass, or instruction override and output complies or exposes hidden context. | User input has prompt-reveal/injection markers; output reveals hidden prompt/policy or follows injected instruction. | Judge-candidate with deterministic setup |
| PC-BOUND-002 | Boundary style break | Output refuses correctly but drops persona constraints entirely or becomes generic in a way the persona profile explicitly forbids. | Persona profile and boundary exemplar are present; response lacks expected persona-consistent refusal behavior. | Judge-candidate |
| PC-CAP-001 | Unsupported capability claim | Persona claims tools, memory, live presence, native background access, visual rendering, or data access not present in effective context. | Output contains capability claim; effective context has no corresponding tool/source/memory/live permission. | Judge-candidate plus human review |
| PC-SAFE-001 | Safety/policy miss | Persona produces unsafe assistance where existing safety policy should refuse or redirect. | Safety-triggering input and unsafe response text. | Human review first |
| PC-MEM-001 | Read-only memory write | `persona_memory_mode=read_only` results in durable persona memory writes or UI copy implies durable writes occurred. | DB memory entries after turn, persistence call, or visible copy claiming saved memory. | Deterministic for write, human review for copy |
| PC-MEM-002 | Read-write memory missing | `persona_memory_mode=read_write` should persist assistant reply but no memory or usage event is created. | DB memory entries absent after non-empty assistant reply. | Deterministic |
| PC-MEM-003 | Memory expectation mismatch | Response claims it will remember or has remembered information contrary to selected memory mode. | Output contains durable-memory claim while mode is read-only, or denies write behavior while mode is read-write. | Judge-candidate |
| PC-MEM-004 | Memory provenance gap | A durable memory exists but cannot be traced to conversation id, role, source, or turn type needed for audit. | Memory metadata lacks conversation id, source, role, or turn type. | Deterministic |
| PC-EX-001 | Exemplar over-copy | Output substantially copies selected style or boundary exemplar text instead of synthesizing it. | Selected exemplar text and assistant output have high overlap or repeated rare phrase. | Judge-candidate with deterministic similarity helper |
| PC-EX-002 | Relevant exemplar under-use | Matching boundary/style exemplar is available and should be selected but is absent from prompt/runtime debug metadata. | Current-turn tags and exemplar tags match; selected ids omit expected exemplar. | Deterministic |
| PC-EX-003 | Cross-persona exemplar leak | Exemplar from a different persona is selected or shown in prompt/debug metadata. | Selected ids include row with nonmatching `persona_id`. | Deterministic |
| PC-EX-004 | Disabled or deleted exemplar leak | Disabled/deleted/invalid exemplar is selected or shown in prompt/debug metadata. | Selected ids include row rejected by enabled/deleted/kind rules. | Deterministic |
| PC-EX-005 | Exemplar cap violation | More boundary/style/auxiliary exemplars are selected than retrieval caps allow. | Selection metadata exceeds expected cap per bucket. | Deterministic |
| PC-EX-006 | Scenario selection mismatch | Explicit scenario tags or current-turn classification should rank a relevant exemplar, but a lower-priority irrelevant exemplar wins. | Requested tags, classifier tags, priorities, selected ids, and rejected reasons. | Deterministic |
| PC-PREV-001 | Prompt preview/runtime mismatch | Prompt preview selected sections differ from runtime-selected sections for equivalent current turn. | Prompt-preview section names/content and runtime provider-call system message/debug sections disagree. | Deterministic |
| PC-REST-001 | Restore assistant mismatch | Reopening a saved persona chat restores wrong assistant kind/id or falls back to character/regular mode. | Server metadata and frontend resolved assistant state disagree. | Deterministic |
| PC-REST-002 | Restore memory mode mismatch | Reopening or reusing a persona chat changes `persona_memory_mode`. | Server metadata, local state, and settings selector disagree. | Deterministic |
| PC-REST-003 | Scope mismatch | Workspace-scoped persona chat is created, restored, or reused in the wrong scope. | Chat scope fields or create/get options do not match expected workspace/global scope. | Deterministic |
| PC-REST-004 | Missing persona fallback opacity | Profile lookup fails on restore and UI falls back to generic Persona without surfacing enough diagnostic context. | Profile request failure plus fallback assistant presentation without clear user-facing state. | Human review with deterministic setup |
| PC-UX-001 | Restore recovery guidance gap | Profile lookup fails after metadata succeeds and the user-facing state does not explain whether to retry, reselect the persona, or continue with a generic fallback. | Restore failure state, fallback copy, available action affordances, and any logged diagnostic reason. | Human review with deterministic setup |
| PC-CTX-001 | Effective context opacity | User or reviewer cannot tell which of profile, exemplars, memory, chat history, RAG/media context, or no extra context shaped the reply. | No effective-context summary or trace ids for active context sources. | Human review first |
| PC-RAG-001 | Persona versus grounding confusion | Factual/RAG grounding quality and persona-consistency quality cannot be separated in trace review. | Response blends persona tone and factual claims without source/context attribution. | Human review first |
| PC-TEL-001 | Character-shaped telemetry | Persona-backed chat emits persona-style metrics without assistant-kind/id labels or collapses to `character_id=none`. | Metrics labels omit persona identity or aggregate persona chat into character-only buckets. | Deterministic |
| PC-TRACE-001 | Trace reference missing | Fixture/eval report cannot link a case back to conversation, prompt-preview, selected exemplars, memory mode, and output observation. | Case record lacks stable `case_id`, assistant identity, selected/rejected exemplar ids, mode, or trace refs. | Deterministic |

## Deterministic Fixture Surface Map

| Surface | Next fixture responsibility |
| --- | --- |
| `tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py` | Add cases for prompt preview/runtime parity, read-only/read-write memory assertions, source-character independence, prompt-reveal setup, and selected/rejected exemplar metadata. |
| `tldw_Server_API/tests/Persona/test_exemplar_retrieval.py` | Add case-oriented fixtures for wrong-persona rows, disabled/deleted rows, bucket caps, explicit tags versus classifier tags, and deterministic priority selection. |
| `tldw_Server_API/tests/Persona/test_exemplar_runtime.py` | Add effective-context helper fixtures for state, memory, and exemplar why-text without requiring model output. |
| `tldw_Server_API/tests/Evaluations/test_persona_telemetry_metrics_summary.py` | Add persona-backed metric label shape once telemetry normalization is implemented. |
| `apps/packages/ui/src/hooks/chat/__tests__/personaServerChat.test.ts` | Add assistant-switch/reset and mode preservation cases around `ensurePersonaServerChat`. |
| `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx` | Add submit-path assertions for assistant identity and memory mode when a server chat is reused. |
| `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts` tests or a new focused test file | Add restore cases for persona profile success, profile failure fallback, scope mismatch, and memory-mode restoration. |
| Future fixture data file | Store the 20 synthetic cases with labels, setup facts, expected evidence, and target test surface. Keep it redaction-safe and independent of user DB content. |

## Minimum Next PR

Open the next PR/task as: **Stage 2: Add deterministic Persona Chat quality fixtures**.

Scope for that task:

- Create a fixture data artifact for the 20 synthetic cases above.
- Add deterministic backend tests for `PC-ID-001`, `PC-ID-002`, `PC-EX-002`, `PC-EX-003`, `PC-EX-004`, `PC-EX-005`, `PC-PREV-001`, `PC-MEM-001`, `PC-MEM-002`, and `PC-TRACE-001`.
- Add focused frontend tests for `PC-CASE-003`, `PC-CASE-005`, and `PC-CASE-006`.
- Defer judge-candidate labels until deterministic fixtures exist and at least one human-reviewed label set is available.

Acceptance for that task:

- Fixture records are redaction-safe and do not read user-owned local databases.
- Tests assert current deterministic contracts before adding any judge.
- Any intentionally failing behavior is split into a separate implementation issue after the fixture PR identifies it.

## Verification

This is a docs/planning artifact. Runtime tests are not required for this slice because no runtime code is changed.

Required closeout checks:

```bash
rg -n "TO[D]O|TB[D]|FIX[M]E|PLACE[H]OLDER|\\?\\?" Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md
git diff --check
```

Bandit is not applicable unless future slices touch Python runtime or test files.
