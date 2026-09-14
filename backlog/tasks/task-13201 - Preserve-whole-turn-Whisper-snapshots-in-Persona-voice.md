---
id: TASK-13201
title: Preserve whole-turn Whisper snapshots in Persona voice
status: Done
assignee: []
created_date: 2026-09-06 01:06
updated_date: 2026-09-06 02:11
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A user said Reply with once, but the submitted transcript duplicated and corrupted that prefix. Real local-model reproduction shows five-second finalized fragments plus overlapping audio produce the duplicate; removing overlap instead cuts words.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona Whisper revises one transcript for the buffered turn instead of concatenating independently decoded overlapping fragments.
- [x] #2 Intentional repeated speech remains intact, reset and cleanup discard prior audio, and input beyond the existing 30-second buffer fails explicitly instead of trimming speech.
- [x] #3 Focused automated regressions and local real-model boundary probes cover speech after leading silence; human acceptance remains accurately documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: amend existing Docs/ADR/046-persona-live-conversation-and-voice-runtime.md. Reason: specialize the existing local Persona Whisper lifecycle without new provider or storage authority. Add a Persona-specific subclass using the existing Whisper model/filter and a bounded whole-turn buffer; expose replacement snapshots and no timed fragment finalization. Keep the existing 30-second memory bound and fail explicitly on overflow. Do not alter other streaming endpoints/backends. Add deterministic regressions for a sentence spanning five seconds, true repeats, reset, empty revisions and overflow; verify with the same local Kokoro/Whisper boundary corpus. Reject lexical deduplication because it erases intentional repeats, and reject zero overlap because real-model probes lost boundary words.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added PersonaWhisperTranscriber using the existing Whisper loader and speech filter with one bounded whole-turn snapshot. It revises empty results, preserves intentional repeats, and rejects more than 30 seconds before AudioBuffer can trim earlier speech. Existing owned STT failure handling shows a safe shorter-turn retry message. Factory changes apply only to Persona Whisper. Three new regressions failed before repair; final Persona/Whisper scope passed 134 tests, including bounded failure correlation and cleanup. Four real local-model boundary cases now retain the complete phrase; zero-overlap baseline still corrupted words. Bandit zero findings; Ruff one unchanged endpoint SIM114; Black checks for both new files pass. ADR046 and user guide/mirrors updated. Source-hashed synthetic receipts and human baseline recorded in Docs/Reviews/assets/migu-buddy-browser-voice-2026-09-05. Fresh human acceptance remains under TASK-13202/13198.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
