# ADR-047: Per-User Audio Preset Ownership

**Status:** Accepted
**Date:** 2026-09-25
**Backfilled from:** `Docs/Design/Audio_Presets_Ownership_2026_05.md`
**Decision owner:** TASK-12356 recorded the owner-selected per-user server-state direction
**Related task:** TASK-13357
**Related spec/plan:** `Docs/superpowers/specs/2026-05-18-tts-stt-webui-extension-workflows-prd-design.md`, `Docs/Design/Audio_Presets_Ownership_2026_05.md`

## Decision

Store reusable TTS/STT presets as authenticated per-user Audio API state in the user's Media DB v2 database, separate from speech history, transcripts, generated artifacts, and comparison results.

## Context

The TTS/STT WebUI and extension need reusable settings that survive reloads and can be shared by the same user's clients. TASK-12356 recorded the ownership decision before preset CRUD. TASK-12363 implemented it: `audio/audio.py` includes the `/api/v1/audio/presets` router, and Media DB v2 has an `audio_presets` table and runtime helpers for SQLite and PostgreSQL.

The endpoints derive `user_id` from the authenticated request principal and open that user's Media DB. List, create, update, delete, and validate operations query by that user ID. Active names are unique per user and kind, and only one active default exists per user and kind. Deletion soft-deletes the preset configuration without deleting speech outputs or history.

Browser TTS configurations may be saved, but the endpoint marks them `browser_local` and `requires_browser_revalidation` and returns a warning on validation. Gateway-backed TTS presets resolve a configured gateway identity and reject route-authority fields at save time. The request schema rejects credential-like keys in preset config.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep reusable presets only in each browser's local storage | The same user's WebUI and extension would have separate copies and no common server-owned state. |
| Store presets in TTS history, STT transcripts, or generated artifact rows | Those records have output and retention lifecycles different from reusable configuration. |
| Store one shared preset collection without authenticated user ownership | It would make settings visible or mutable across users and bypass the existing per-user Media DB boundary. |
| Treat Browser TTS settings as portable server speech settings | Browser speech capabilities and voices depend on the current client, so saved values require browser-side revalidation. |

## Consequences

- The Audio API owns preset CRUD under `/api/v1/audio/presets`; clients cannot choose the owner ID.
- Media DB v2 owns preset rows and their active-name, default, and soft-delete behavior on SQLite and PostgreSQL.
- Presets are configuration records; deleting one does not delete generated audio, transcripts, history, jobs, or comparison results.
- Browser TTS presets remain client-dependent even when stored on the server.
- The current validate endpoint returns the stored preset and a Browser TTS warning. This ADR does not claim that it recomputes live model/provider readiness.
- The request schema admits `tts`, `stt`, and `speech` kinds even though the original design reserved `speech` for a later combined workflow. This ADR accepts the per-user ownership boundary, not a claim that all designed kind or config restrictions are fully enforced.

## Follow-up

- Use this ADR as the covering record for INV-022's implemented ownership decision.
- Keep any future live-readiness validation or stricter config policy work separately tracked and verified against the current endpoint contract.
