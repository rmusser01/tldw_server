# ADR Inventory Reconciliation - 2026-09-25

**Related task:** TASK-13356
**Baseline:** `origin/dev` at `a2f5e1b816` (2026-09-25)
**Scope:** Recheck the unresolved June inventory rows and proposed ADRs. The 2026-06-03 inventory's search counts and task numbers are historical, not a current repo-wide census.

**Subsequent follow-up:** TASK-13373 reconciled [task identities](2026-09-25-adr-task-reference-reconciliation.md), and TASK-13374 reviewed Evaluations persistence against `dev` at `3f909e133b` before recording ADR-048.

## Current Dispositions

| Item | Evidence reviewed | Disposition and next action |
| --- | --- | --- |
| INV-009 / INV-012, Evaluations persistence | `tldw_Server_API/app/core/DB_Management/Evaluations_DB.py` has SQLite `TEXT` and PostgreSQL `JSONB` structured fields, DB-owned primary CRUD JSON conversion, raw unified reads, configured/injected backends, and explicit path/resolver fallback caveats. | TASK-13374 backfilled the owner-approved bounded contract as [ADR-048](../048-backend-aware-evaluations-persistence.md). Old SQLite-only completeness claims remain superseded; RLS, migration parity, other helpers, and execution ownership are not promoted. |
| INV-014, Evaluations run ownership | `tldw_Server_API/app/core/Evaluations/eval_runner.py` tracks local async run tasks, while `recipe_runs_jobs_worker.py` owns Jobs-backed recipe execution. | Keep the broad historical async statement inventory-only. Review the core-run and recipe-run boundaries separately before any replacement ADR. |
| INV-022, TTS/STT preset storage | `Docs/Design/Audio_Presets_Ownership_2026_05.md` records the accepted per-user Audio API/Media DB decision; TASK-12356 and TASK-12363 are Done. `audio/audio_presets.py`, the audio router, and Media DB `audio_presets` storage exist. | The June "needs owner review" disposition is stale. TASK-13357 backfilled the implemented ownership boundary as ADR-047. ADR-011's older follow-up remains historical text. |
| INV-029, `SecretManager` adoption | Application references to `Security.secret_manager` still appear only in that helper; the ACP `TriggerSecretManager` is a separate implementation. | Keep repository-wide `SecretManager` adoption inventory-only. Define an implementation slice with explicit migrations or exemptions before claiming centralized lookup. ADR-027 and ADR-028 already cover the separate crypto and restricted-pickle rules. |
| ADR-029, static PyPI WebUI bundle | ADR status is Proposed; TASK-12158 completed planning only. `tldw_Server_API/app/main.py` has no `/ui` static mount. | Leave Proposed. Recheck release implementation and package artifact gates before an acceptance decision. |
| ADR-040, moodboard/Studio Sync | The ADR file says Accepted; TASK-13007 records requester approval of the written design and remains In Progress for implementation. | Correct the ADR index to Accepted. Design acceptance does not imply implementation completion. |
| ADR-043, managed llama.cpp snapshots | The ADR file says Accepted and is present in `Docs/ADR/`, but absent from the index. | Add its index entry in numeric order. |

## Next Review

INV-022 is covered by ADR-047, and INV-009/INV-012 by ADR-048. INV-014's execution ownership and `SecretManager` adoption need separate decision or implementation work. ADR-029 remains Proposed pending release implementation evidence. Unresolved historical task identities and duplicate current IDs remain explicitly identified in the task-reference reconciliation, not guessed or silently reassigned.
