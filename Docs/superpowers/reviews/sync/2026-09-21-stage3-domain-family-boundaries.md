# Stage 3 — Domain adapters, materializers, bootstrappers: the sibling-family duplication

## Scope

The three parallel per-domain families Sync grows one member at a time:
`v2/domain_adapters/` (12 files, 3,444 LOC), `v2/materializers/` (12 files, 3,757 LOC), and the
five `notes_*_bootstrap.py` / `notes_*_coordinator.py` pairs. The question for this stage is not
"do these files look similar" — they are siblings, they should — but **which knowledge is
duplicated across them and has already drifted**.

## Code Paths Reviewed

### Existing shared modules (the destinations, both already adopted)

- `v2/domain_adapters/_lineage.py (1-146)` — `prior_envelopes (14-24)`, `current_head`,
  `delete_update_conflict`, `incoming_references_exact_head`. Imported by 8 of the 12 adapters.
- `v2/materializers/base.py (1-41)` — `MaterializationResult (16-24)`,
  `SyncMaterializer (27-38)`. Imported by every materializer.
- `v2/materializers/guarded_product_mutation.py (1-77)`

### The helper family that sits outside them

Domain adapters:
- `_is_delete` — `chat.py:74`, `media.py:345-350`, `notes.py:142-147`, `source_cache.py:90-95`,
  `workspaces.py:102-107` (the last four byte-identical)
- `_manual_delete_conflict` — `chat.py:64-71`, `media.py:353-360`, `notes.py:122-129`,
  `source_cache.py:80-87`, `workspaces.py:92-99` (pairwise 0.97-0.99 similar; differ only in the
  message string)
- `_has_base` — `notes_link.py:218-227` == `notes_organization.py:472-481`;
  `notes_task.py:241-251` == `notes_task_activity.py:412-422`
- `_conflict` — `notes_link.py:284-295` == `notes_organization.py:576-587` (byte-identical);
  `notes_task.py:373-385`, `notes_task_activity.py:450`
- `_rejected` — `notes_link.py:272-281` == `notes_organization.py:564-573` (byte-identical);
  `notes_task.py:360`, `notes_task_activity.py:440`
- `_get_head` — `notes_link.py:151-162` == `notes_organization.py:265-276` (byte-identical);
  `notes_task.py:216`, `notes_task_activity.py:220`
- `_is_deleted` — `notes_link.py:268-269` == `notes_organization.py:560-561`;
  `notes_task.py:265-268`, `notes_task_activity.py:425-428`
- `_base_conflict` — `notes_link.py:298-299` ≈ `notes_organization.py:590-591`
- `_metadata_value` — `chat.py:104-105` == `workspaces.py:135-136`

Materializers:
- `_next_object_revision` — `attachment_refs.py:520-528` == `media_metadata.py:136-144` ==
  `source_cache.py:136-144` (byte-identical); `chat.py:498-503` (0.98);
  plus two staticmethod copies, `notes.py:253-261` and `notes_organization.py:302-309`
- `_is_already_materialized` — `chat.py:506-518`, `notes.py:264-279`,
  `notes_organization.py:311-325`
- `_conflict_result` — `notes.py:290-316` ≈ `notes_organization.py:364-389` (0.98);
  `notes_task.py:328-336` ≈ `notes_task_activity.py:444-452` (0.90); also
  `attachment_refs.py:531`, `notes_link.py:247`, `chat.py:435`
- `_tombstone_conflict_result` — `attachment_refs.py:551-568`, `media_metadata.py:167-184`,
  `source_cache.py:167-184`
- `_hash_conflict_result` — `media_metadata.py:147-164` ≈ `source_cache.py:147-164` (0.91)
- `_record_applied` / `_mark_conflict` / `_mark_failed` — `notes_task.py:251-275, 278-298,
  301-325` ≈ `notes_task_activity.py:377-395, 398-418, 421-441` (0.92-0.97)
- `_projection_client_id` — `chat.py:494-495` == `notes.py:286-287`

Bootstrappers / readiness:
- `v2/store.py:transition_notes_task_readiness (618-647)` vs
  `transition_notes_task_activity_readiness (650-679)` — identical apart from the
  `readiness_key="notes_task_v1"` / `"notes_task_activity_v1"` literal and the docstring
- `_block` — `notes_task_bootstrap.py:590-613` ≈ `notes_task_activity_bootstrap.py:756-779` (0.97)
- `_readiness` — `notes_task_bootstrap.py:616-622` ≈ `notes_task_activity_bootstrap.py:782-788`
- `_optional_string` — `notes_task_bootstrap.py:625-628` ≈
  `notes_task_activity_bootstrap.py:813-816` (0.99)
- `_bootstrap_id` — `notes_task_bootstrap.py:631-637` ≈ `notes_task_activity_bootstrap.py:819-825`
- `_non_negative_int` — `notes_attachment_bootstrap.py:962-963` ==
  `notes_link_bootstrap.py:336-337` (byte-identical)
- `note_db` — `notes_task_bootstrap.py:193-196` ≈ `notes_task_activity_bootstrap.py:203-206`
- Exception twins — `notes_link_bootstrap.py:29,33` vs `notes_organization_bootstrap.py:55,59`
- `_parse_uuid` — `notes_moodboard_studio_readiness.py:238-245` ==
  `notes_task_readiness.py:270-277` (byte-identical); `as_metadata` —
  `notes_moodboard_studio_readiness.py:86-96` == `notes_task_readiness.py:78-88`
- Contract twins — `notes_moodboard_studio_contract.py` (1,592 LOC) vs `notes_task_contract.py`
  (1,202 LOC): `_sha256` (1557-1558 == 936-937), `_validate_ids` (392-393 == 243-244),
  `_validate_note_id` (606-607 == 414-415), `_reject_mutation` (218-219, 240-241 == 141-142,
  163-164), `__copy__`/`__deepcopy__` (221-225, 243-247 == 144-148, 166-170), `_freeze_json`
  (1439-1446 ≈ 868-875), `_canonical_uuid4` (1493-1502 ≈ 792-801)

### Whole-file clone

- `v2/materializers/media_metadata.py` (187 LOC) vs `v2/materializers/source_cache.py` (187 LOC)

### Error-redaction policy

- `v2/materializers/chat.py:_safe_error_message (525-527)`
- `v2/materializers/notes.py:_safe_error_message (319-321)`
- `v2/materializers/notes_link.py:_safe_error_message (282-292)`
- `v2/materializers/notes_task.py:_safe_error_message (339-351)`
- `v2/materializers/notes_task_activity.py:_safe_error_message (455-464)`
- `v2/materializers/notes_organization.py:_safe_error_message (392-405)`
- `v2/service.py:_safe_projection_error_message (380-382)`
- Exposure path: `apply_error_message` is set from these at `materializers/notes.py:94,185` and
  `materializers/chat.py:118,248,300`, lands on the API schemas at
  `api/v1/schemas/sync_v2_models.py:1930` and `:2122`, and is returned to clients at
  `api/v1/endpoints/notes.py:437`, `api/v1/endpoints/character_messages.py:213`,
  `api/v1/endpoints/character_chat_sessions.py:763`, `api/v1/endpoints/notes_sync_errors.py:108`.

### ADRs consulted

`Docs/ADR/031`, `037`, `038`, `039`, `040` define the per-domain sync contracts and the derived
projections. None of them mandates a separate helper per domain; they constrain payload shape,
authority, and projection semantics, all of which survive a shared helper unchanged.

## Tests Reviewed

- `tests/Sync/test_sync_v2_domain_adapters.py` — the adapter family's shared behaviour. Its own
  `_envelope` helper at `:71` is another copy of the fixture duplication in `sync-11`.
- `tests/Sync/test_sync_v2_notes_materializer.py`, `..._chat_materializer.py`,
  `..._notes_link_materializer.py`, `..._notes_organization_materializer.py`,
  `..._notes_task_materializer.py`, `..._notes_task_activity_materializer.py`,
  `..._attachment_materializer.py` — one file per materializer, mirroring the source duplication.
  These cover the applied/conflict/failed outcomes well; none of them asserts what
  `apply_error_message` contains on the failure path, which is why `sync-3` has survived.
- `tests/Sync/test_sync_v2_media_compat.py` covers `media_metadata`; there is no
  `test_sync_v2_source_cache_materializer.py` — `source_cache` is exercised indirectly via
  `test_sync_v2_domain_adapters.py`. The clone is therefore asymmetrically covered.
- `tests/Sync/test_sync_v2_notes_task_bootstrap.py`, `..._notes_task_activity_bootstrap.py`,
  `..._notes_link_bootstrap.py`, `..._notes_organization_bootstrap.py`,
  `..._notes_attachment_bootstrap.py` — the five bootstrappers, one suite each.

## Validation Commands

```
# AST + difflib scan for cross-file near-duplicate function bodies inside core/Sync/
# (similarity > 0.80, same name, different file). Selected output:
## _next_object_revision
   1.0  materializers/attachment_refs.py:520-528 <-> materializers/media_metadata.py:136-144
   1.0  materializers/attachment_refs.py:520-528 <-> materializers/source_cache.py:136-144
   1.0  materializers/media_metadata.py:136-144  <-> materializers/source_cache.py:136-144
   0.98 materializers/chat.py:498-503            <-> materializers/media_metadata.py:136-144
## _is_delete
   1.0  domain_adapters/media.py:345-350 <-> domain_adapters/notes.py:142-147
   1.0  domain_adapters/media.py:345-350 <-> domain_adapters/source_cache.py:90-95
   1.0  domain_adapters/media.py:345-350 <-> domain_adapters/workspaces.py:102-107
## _conflict
   1.0  domain_adapters/notes_link.py:284-295 <-> domain_adapters/notes_organization.py:576-587
## _get_head
   1.0  domain_adapters/notes_link.py:151-162 <-> domain_adapters/notes_organization.py:265-276
## _non_negative_int
   1.0  notes_attachment_bootstrap.py:962-963 <-> notes_link_bootstrap.py:336-337
## _safe_error_message
   1.0  materializers/chat.py:525-527 <-> materializers/notes.py:319-321
   0.92 materializers/notes_task.py:339-351 <-> materializers/notes_task_activity.py:455-464
   0.88 materializers/notes_link.py:282-292 <-> materializers/notes_task.py:339-351

$ diff .../materializers/media_metadata.py .../materializers/source_cache.py | grep -c "^[<>]"
     56          # i.e. 28 changed line-pairs across two 187-line files, identifiers only

$ grep -rn --include='*.py' "_lineage" tldw_Server_API/app/core/Sync/ | grep "^.*domain_adapters" | wc -l
      8          # _lineage.py is already imported by 8 of the 12 adapters

$ git log --diff-filter=A --format=%ad --date=short -1 -- .../materializers/notes.py
2026-05-23
$ git log --diff-filter=A --format=%ad --date=short -1 -- .../materializers/chat.py
2026-05-23
$ git log --diff-filter=A --format=%ad --date=short -1 -- .../materializers/notes_link.py
2026-08-10
$ git log --diff-filter=A --format=%ad --date=short -1 -- .../materializers/notes_task.py
2026-08-21
```

## Findings

### FINDING sync-3

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       Passthrough policy (echoes raw exception text):
               v2/materializers/chat.py:_safe_error_message (525-527)
               v2/materializers/notes.py:_safe_error_message (319-321)
               — both exactly `message = str(exc).strip(); return message[:200] if message
                 else type(exc).__name__`
             Classify-and-redact policy (never echoes exception text):
               v2/materializers/notes_link.py:_safe_error_message (282-292)
               v2/materializers/notes_task.py:_safe_error_message (339-351)
               v2/materializers/notes_task_activity.py:_safe_error_message (455-464)
               v2/materializers/notes_organization.py:_safe_error_message (392-405)
             Type-name-only policy (strictest):
               v2/service.py:_safe_projection_error_message (380-382)
             Consumers of the output: materializers/notes.py:94,99,185,190;
               materializers/chat.py:118,123,248,253,300,305
             Client exposure: api/v1/schemas/sync_v2_models.py:1930, :2122;
               api/v1/endpoints/notes.py:437; api/v1/endpoints/character_messages.py:213;
               api/v1/endpoints/character_chat_sessions.py:763;
               api/v1/endpoints/notes_sync_errors.py:108
canonical:   The classify-and-redact form is correct. Its docstring states the intent —
             notes_task.py:340, "Map internal projection failures to bounded public messages."
             The two passthrough copies are the OLDER ones (chat.py and notes.py both added
             2026-05-23; notes_link.py 2026-08-10; notes_task.py 2026-08-21), so the policy was
             tightened for later domains and never back-fixed on the first two.
destination: One `v2/materializers/error_messages.py` owning "what a projection failure is allowed
             to tell a client", exporting a `safe_error_message(exc, *, domain: SyncDomain)` that
             takes the domain label as a parameter — the per-domain prefix is the only thing the
             four correct copies actually vary.
knowledge:   The redaction policy for product-database failures crossing the Sync boundary. Six
             copies, three answers.
scenario:    Push a `notes.note` upsert whose projection raises a `CharactersRAGDBError` or a bare
             `sqlite3.DatabaseError` (a UNIQUE constraint on a note title, a disk-I/O error, a
             busy timeout). `materializers/notes.py:185` sets
             `apply_error_message=_safe_error_message(exc)` = `str(exc)[:200]`, which for sqlite
             carries the failing SQL fragment, the constraint name and often a row value; for
             `CharactersRAGDBError` it carries whatever the DB layer formatted, which can include
             a filesystem path to the per-user database. That string is persisted on the envelope
             and returned verbatim to the client at api/v1/endpoints/notes.py:437 and at
             api/v1/schemas/sync_v2_models.py:1930. The identical push against `notes.task`
             returns "notes.task product database operation failed" and nothing else.
impact:      Medium, not High: the recipient is the data's own owner, so this is
             internal-detail disclosure rather than cross-user leakage, and a `[:200]` cap bounds
             it. It is reported because the codebase has already decided this is not acceptable —
             four newer copies of the same function exist specifically to prevent it — and
             because internal DB error text on a client-facing field is a Bandit/security-review
             finding waiting to happen on the next touch of those files.
cost-driver: n/a
tests:       import-grep reachability, not measured coverage —
             tests/Sync/test_sync_v2_notes_materializer.py and
             tests/Sync/test_sync_v2_chat_materializer.py cover the failure outcomes but assert
             on `status`/`error_code`, not on `apply_error_message` content. A single assertion
             that `apply_error_message` never contains `str(exc)` would pin all six.
effort:      cheap — one shared function parameterised by the domain label, four call-site
             updates, two behaviour fixes. Small enough to skip the design doc; needs a Backlog
             task because it changes a client-visible string.
owner-only:  no for the core/ fix; the consuming endpoints under app/api/v1/** are owner-only but
             need no change.
confidence:  confirmed (all six bodies read directly; the exposure path traced to the schema and
             four endpoints; the age ordering from git)
```

### FINDING sync-6

```
axis:        duplication
class:       adoption-gap
severity:    Medium
sites:       Adapter family, destination `v2/domain_adapters/_lineage.py` (already imported by 8
             of 12 adapters):
               `_is_delete` — chat.py:74-79, media.py:345-350, notes.py:142-147,
                 source_cache.py:90-95, workspaces.py:102-107
               `_manual_delete_conflict` — chat.py:64-71, media.py:353-360, notes.py:122-129,
                 source_cache.py:80-87, workspaces.py:92-99
               `_has_base` — notes_link.py:218-227, notes_organization.py:472-481,
                 notes_task.py:241-251, notes_task_activity.py:412-422
               `_get_head` — notes_link.py:151-162, notes_organization.py:265-276,
                 notes_task.py:216-…, notes_task_activity.py:220-…
               `_conflict` — notes_link.py:284-295, notes_organization.py:576-587,
                 notes_task.py:373-385, notes_task_activity.py:450-…
               `_rejected` — notes_link.py:272-281, notes_organization.py:564-573,
                 notes_task.py:360-…, notes_task_activity.py:440-…
               `_is_deleted` — notes_link.py:268-269, notes_organization.py:560-561,
                 notes_task.py:265-268, notes_task_activity.py:425-428
               `_base_conflict` — notes_link.py:298-299, notes_organization.py:590-591
               `_metadata_value` — chat.py:104-105, workspaces.py:135-136
             Materializer family, destination `v2/materializers/base.py` (imported by all):
               `_next_object_revision` — attachment_refs.py:520-528, chat.py:498-503,
                 media_metadata.py:136-144, source_cache.py:136-144, plus staticmethods at
                 notes.py:253-261 and notes_organization.py:302-309
               `_is_already_materialized` — chat.py:506-518, notes.py:264-279,
                 notes_organization.py:311-325
               `_projection_client_id` — chat.py:494-495, notes.py:286-287
               `_conflict_result` — attachment_refs.py:531-548, chat.py:435-…,
                 notes.py:290-316, notes_link.py:247-…, notes_organization.py:364-389,
                 notes_task.py:328-336, notes_task_activity.py:444-452
               `_tombstone_conflict_result` — attachment_refs.py:551-568,
                 media_metadata.py:167-184, source_cache.py:167-184
               `_hash_conflict_result` — media_metadata.py:147-164, source_cache.py:147-164
               `_record_applied`/`_mark_conflict`/`_mark_failed` —
                 notes_task.py:251-275/278-298/301-325 vs
                 notes_task_activity.py:377-395/398-418/421-441
             Bootstrap/readiness family:
               v2/store.py:transition_notes_task_readiness (618-647) vs
                 transition_notes_task_activity_readiness (650-679) — identical but for the
                 `readiness_key` literal
               `_block` — notes_task_bootstrap.py:590-613, notes_task_activity_bootstrap.py:756-779
               `_readiness` — notes_task_bootstrap.py:616-622,
                 notes_task_activity_bootstrap.py:782-788
               `_optional_string` — notes_task_bootstrap.py:625-628,
                 notes_task_activity_bootstrap.py:813-816
               `_bootstrap_id` — notes_task_bootstrap.py:631-637,
                 notes_task_activity_bootstrap.py:819-825
               `_non_negative_int` — notes_attachment_bootstrap.py:962-963,
                 notes_link_bootstrap.py:336-337
               `note_db` — notes_task_bootstrap.py:193-196,
                 notes_task_activity_bootstrap.py:203-206
               `_parse_uuid` / `as_metadata` — notes_moodboard_studio_readiness.py:238-245, 86-96
                 vs notes_task_readiness.py:270-277, 78-88
               contract twins — notes_moodboard_studio_contract.py:1557,392,606,218,240,221,243,
                 1439,1493 vs notes_task_contract.py:936,243,414,141,163,144,166,868,792
canonical:   `v2/domain_adapters/_lineage.py (1-146)` and `v2/materializers/base.py (1-41)` both
             already exist and are already imported by every member of their family. This is an
             adoption gap, not a missing abstraction.
destination: Extend the two existing modules; add `v2/notes_bootstrap_common.py` owning the
             shared dormant-readiness state machine (`_readiness`, `_block`, `_bootstrap_id`,
             `_optional_string`, `_non_negative_int`) for the five bootstrappers, and make
             `SyncV2Store.transition_notes_task_readiness` take `readiness_key` as a parameter
             instead of existing twice.
knowledge:   Three pieces. (a) "What counts as a delete envelope, and what conflict a manual
             delete produces" — 5 copies. (b) "How an object revision advances and when an
             envelope is already materialized" — 6 and 3 copies. (c) "How a dormant notes domain
             transitions between bootstrapping/blocked/ready and what its bootstrap identity is"
             — 2-5 copies. Each is a rule the ADRs describe once (031, 037, 038, 039) and the
             code states many times.
scenario:    n/a (duplication)
impact:      Medium. The change-amplification is measurable and recurring: this module adds a
             domain roughly every two months (notes_link 2026-08-10, notes_task 2026-08-21,
             moodboard/studio later), and each addition copies the whole helper set again — the
             `_safe_error_message` divergence in `sync-3` is what that looks like after two
             rounds. Concretely: adding a third outcome to `AdapterConflict` (say a
             `retry_after`) is one edit in `_lineage.py` and nine edits today. Severity is not
             High because no current copy is wrong; the cost is future edits, not present
             behaviour.
cost-driver: n/a
tests:       import-grep reachability, not measured coverage —
             tests/Sync/test_sync_v2_domain_adapters.py covers the adapter helpers; seven
             per-materializer suites cover the materializer helpers; five per-bootstrapper suites
             cover the bootstrap helpers. Coverage is good, which makes this cheap to do.
effort:      cheap-to-moderate, and it should be staged by family rather than done as one PR:
             (1) materializer helpers into base.py, (2) adapter helpers into _lineage.py,
             (3) the `readiness_key` parameterisation in store.py, (4) the bootstrap common
             module. Stages 1-3 need no design doc; stage 4 touches a state machine four ADRs
             reference and should get one.
owner-only:  no
confidence:  confirmed (every pair listed was produced by an AST+difflib scan and the
             byte-identical ones spot-checked by diff)
```

### FINDING sync-7

```
axis:        duplication
class:       true-duplication
severity:    Medium
sites:       v2/materializers/media_metadata.py (1-187) — `MediaMetadataMaterializer (13-112)`,
               `_record_media_metadata_state (114-133)`, `_next_object_revision (136-144)`,
               `_hash_conflict_result (147-164)`, `_tombstone_conflict_result (167-184)`
             v2/materializers/source_cache.py (1-187) — `SourceCacheMaterializer (13-112)`,
               `_record_source_cache_state (114-133)`, `_next_object_revision (136-144)`,
               `_hash_conflict_result (147-164)`, `_tombstone_conflict_result (167-184)`
             The two files are the same 187 lines. `diff` reports 28 changed line-pairs, and
             every one of them is an identifier or a message string: the class name, the
             `domain` literal (`"media.item"` vs `"source_cache.entry"`), the
             `_record_*_state` function name, and the `media_metadata_*` vs `source_cache_*`
             error-code prefixes. There is no structural difference anywhere.
canonical:   NONE — neither is more correct; they are the same code
destination: `v2/materializers/metadata_only.py` owning "project a metadata-only Sync domain into
             restoreable object state", exposing one
             `MetadataOnlyMaterializer(domain: SyncDomain, error_prefix: str, label: str)`.
             `MediaMetadataMaterializer` and `SourceCacheMaterializer` become three-line
             constructions. This is a cohesive module with one responsibility, not a dumping
             ground.
knowledge:   The whole metadata-only projection policy, stated twice: tombstoned objects may not
             be resurrected by an upsert (:52-63 / :52-63), a reused stable object ID with a
             different payload hash is a conflict rather than an overwrite (:66-79 / :66-79),
             `payload_hash` is mandatory (:33-46 / :33-46), object revision advances by one from
             current state (:136-144), and the tombstone path preserves the prior object hash
             (:90-99). Five rules × two files.
scenario:    n/a (duplication)
impact:      Medium. This is the largest single copy-paste inside the module and it is the exact
             shape flagged repo-wide as C9 (the discord/slack whole-module clone). The live cost
             is asymmetric coverage: `tests/Sync/test_sync_v2_media_compat.py` exercises
             `media_metadata` directly, while `source_cache` has no dedicated materializer suite
             — so a rule fixed on one side and missed on the other would not be caught. A third
             metadata-only domain would make it three files.
cost-driver: n/a
tests:       import-grep reachability, not measured coverage —
             tests/Sync/test_sync_v2_media_compat.py (media_metadata),
             tests/Sync/test_sync_v2_domain_adapters.py (source_cache, indirectly). The
             parameterised replacement would let one table-driven suite cover both.
effort:      cheap — mechanical, and the diff proves there is no behavioural difference to
             preserve. The only judgement call is the error-code strings, which must stay exactly
             as they are because clients match on them.
owner-only:  no
confidence:  confirmed (full diff of both files read line by line)
```

## Suggested Refactor/Actions

1. `sync-3` first — it is the only finding in this stage with a client-visible consequence. Move
   the classify-and-redact form into `v2/materializers/error_messages.py`, parameterise the domain
   label, and delete the two passthrough copies. Add the "never contains `str(exc)`" assertion to
   `tests/Sync/test_sync_v2_notes_materializer.py` and `..._chat_materializer.py`.
2. `sync-7` next — collapse the two metadata-only materializers onto one parameterised class and
   give the result a single table-driven test. Preserve the error-code strings verbatim.
3. `sync-6` staged as described in the finding. Stages 1-3 are cheap and well covered; do not
   bundle stage 4 (the bootstrap state machine) into them — that one needs
   `Docs/Design/YYYY-MM-DD-sync-notes-bootstrap-common-design.md` and a Backlog task, because it
   touches behaviour four ADRs (031, 037, 038, 039) describe.
4. Do **not** try to unify `notes_moodboard_studio_contract.py` and `notes_task_contract.py` as
   whole modules. Their payload schemas are genuinely different domains (ADR-039 vs ADR-040);
   only the leaf validators (`_sha256`, `_freeze_json`, `_canonical_uuid4`, `_validate_note_id`,
   `_validate_ids`) are shareable, and they belong in one small
   `v2/contract_primitives.py`, not in a merged contract module.
