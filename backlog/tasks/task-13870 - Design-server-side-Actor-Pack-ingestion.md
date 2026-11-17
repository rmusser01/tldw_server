---
id: TASK-13870
title: Design server-side Actor Pack ingestion
status: To Do
labels:
- design
- character-chat
- persona
- portability
- security
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define an implementation-ready server contract for ingesting Chatbook .tldw-actor-pack archives using tldw.actor-pack/v1 without weakening the existing Character, Persona, and Persona Visual ownership boundaries. The design should make portable actors reviewable and importable on the server while preserving explicit activation, user scoping, hostile-archive defenses, and compatibility with Chatbook-produced packs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The design inventories the current Character import, Persona profile, Persona Visual portability, Jobs, storage, and authentication boundaries and identifies which existing services can be reused versus which Actor Pack-specific capabilities are missing.
- [ ] #2 The design defines a versioned authenticated API and job lifecycle for upload, bounded validation, review preview, cancellation and cleanup, explicit commit, and status retrieval; preview performs no live actor or visual mutation.
- [ ] #3 The archive trust boundary specifies canonical member handling, declared-file and digest integrity, size, count, decompression, and image limits, private staging, path, link, encryption, nesting, and collision rejection, external-reference rejection, cleanup authority, and path-free client diagnostics.
- [ ] #4 The actor mapping supports exactly one Character or one Persona, requires an actor record and portrait, and defines how optional Shared Visual Identity and Persona Visual sections map to server capabilities without merging their distinct schemas or runtimes.
- [ ] #5 The design defines a server-owned portable UUID registry and the exact no-match, same-kind match, and cross-kind conflict actions, including Create New, Create Copy, explicitly confirmed Update Existing, provenance, stale-review revalidation, and preservation of omitted optional visual bindings.
- [ ] #6 The design records ownership, transaction, idempotency, concurrency, rollback, crash-recovery, and post-commit invalidation semantics across Character, Persona, portrait, and visual stores, including behavior when an optional section is valid but unsupported by the server.
- [ ] #7 The compatibility contract covers tldw.actor-pack/v1, required and optional feature negotiation, unknown fields and features, deterministic fixtures shared with tldw_chatbook, backward-compatible evolution, and a round-trip interoperability test matrix.
- [ ] #8 A canonical ADR is created for the cross-module API, storage, identity, security, and activation decisions, and the design decomposes implementation into atomic dependency-ordered Backlog tasks that can each ship in one PR.
- [ ] #9 The verification strategy includes focused unit and integration tests, hostile archive and race and crash cases, real SQLite coverage where applicable, authorization isolation, no-mutation preview proof, Bandit on future Python scope, documentation updates, and live API interoperability evidence with a Chatbook-generated fixture.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Preserved from PR #2810, commit 724db16258c74226ba38d8a8c5a55b43488bcef1; original TASK-13111, created 2026-08-23 19:48, priority high, no assignee or dependencies, status To Do. Current dev uses TASK-13111 for unrelated completed work, which is unchanged. All nine original acceptance criteria are retained verbatim. This is a design request, not a completed implementation.
Source: https://github.com/rmusser01/tldw_server/pull/2810

Original references (retained as historical references; cross-repo link contents not revalidated in this preservation):
- https://github.com/rmusser01/tldw_chatbook/pull/2020
- https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md
- https://github.com/rmusser01/tldw_chatbook/blob/dev/backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md

Original documentation associations:
- Docs/Code_Documentation/Character_Chat.md
- Docs/Code_Documentation/Persona_Visual_Packs.md
- Docs/Design/Personas.md

ADR required: no for this record preservation (ADR-002 and ADR-059). The future Actor Pack design must assess and record durable cross-module API, identity, storage, security, and activation decisions; original criterion 8 explicitly requires a canonical ADR. No architecture decision is accepted by this filing.
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
