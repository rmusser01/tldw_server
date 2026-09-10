# Independent Buddy API

Governance: [ADR-005](../../backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md).

All routes below require the current server user. Client slots identify preferences, never users or permissions. Paths are relative to `/api/v1/buddies`.

| Method and path | Contract |
| --- | --- |
| `GET` base path | `{buddies: BuddyProfile[]}`; `limit` defaults to 50, maximum 100; `offset` defaults to 0. |
| `POST` base path | Create an immutable art snapshot with `{name, source, optional_persona_id?, display_mode?}`. `source` is `{kind:"starter",starter_id}` or `{kind:"persona_pack",persona_id,pack_id}`. Returns a profile, status 201. |
| `GET /{buddy_id}` | Owner's active profile. |
| `PATCH /{buddy_id}` | `{expected_version, name?, optional_persona_id?, display_mode?}`. Explicit `optional_persona_id:null` clears the optional association. |
| `DELETE /{buddy_id}?expected_version=N` | Soft-delete the profile, status 204. |
| `GET /{buddy_id}/assets/{asset_id}/content` | Authenticated image bytes with a private, no-store cache policy. |
| `GET /conversation-targets/{conversation_id}` | Current owned conversation `{id,title,scope_type,workspace_id}` for selection before attachment. |
| `GET /conversation-targets/{conversation_id}/reply-settings` | Current attached target only: `{provider:string|null,model:string|null}` with private, no-store caching. Uses the same effective conversation settings as reply acceptance; no work is accepted. |
| `GET /attachment` | Freshly resolved attachment envelope. |
| `PUT /attachment` | `{expected_version,buddy_id,scope_type:"conversation"|"workspace",scope_id}`; returns the envelope. |
| `DELETE /attachment?expected_version=N` | Detach and increment the slot revision; returns the envelope. |
| `GET /attachment/conversations` | `{conversations:[{id,title,scope_type,workspace_id}],limit,offset}`. Same pagination bounds as profile listing. |
| `GET /attachment/activity` | `{items:[{conversation_id,title,workspace_id,result:{id,created_at,content},acknowledged}],limit,offset}`. Content is bounded to 2,000 characters. Only conversations with a persisted assistant result appear. |
| `POST /attachment/acknowledgements` | `{conversation_id,result_message_id}` → `{acknowledged:true}` for an exact, currently authorized assistant result. |

Reply-settings, attachment, conversation-list, activity, and acknowledgement routes accept `client_slot`, defaulting to `default`. Slot keys are bounded to 128 characters. An empty slot starts at version 0; updates and detachments retain monotonically increasing revisions. Stale profile or slot revisions return 409. Missing, deleted, or foreign resources return 404. Invalid input returns 422.

The attachment envelope is `{client_slot,version,attachment,target,unavailable_reason}`. `attachment` is `{buddy_id,scope_type,scope_id}` or null; `target` is `{title,workspace_id}` or null. A deleted Buddy or inaccessible target produces a null attachment with `buddy_unavailable` or `target_unavailable`. Reads never open a shared workspace owner's private conversation database. Use authorized conversation summaries to supply the workspace scope required by the existing transcript APIs.

Conversation summaries also include `created_at`, `version`, `assistant_kind`, `assistant_id`, and `assistant_name`. A Persona name is resolved through the authenticated owner's profile. Other or unavailable names are null; character behavior remains governed by its existing frozen snapshot and setup API.

Profiles contain `id`, `name`, `optional_persona_id`, `optional_persona_available`, `display_mode`, `version`, a native Persona visual `manifest`, copied `attribution`, and `assets`. Asset summaries include `id`, `mime_type`, `byte_size`, `width`, `height`, `checksum_sha256`, and `content_url`. Static/Dynamic is a display preference; neither changes conversation identity. A deleted source Persona makes its optional association unavailable while the copied artwork and attribution remain readable. Artwork is limited to 256 validated assets, 10 MiB per asset, and 64 MiB total. Profiles cannot change their artwork in place.

Activity uses canonical persisted-message ordering and exact result IDs. An acknowledgement of an older result cannot acknowledge a newer one. This projection does not infer execution or approval state; the accepted-turn API under `/turns` owns Buddy work status. Detachment and navigation do not stop accepted work.

Starter previews are available before creating a Persona or Buddy: authenticated `GET /api/v1/persona/visual-starter-packs/{starter_pack_id}/assets/{asset_key}/content`. Starter detail asset summaries expose width and height for native sprite rendering.

New workspace conversation creation applies `assistant_defaults` only when the request omits all assistant identity fields. Explicit None or an explicit assistant choice wins. Fork requests do not implicitly apply workspace defaults. Deleted or inactive default Personas produce 409 and require an explicit choice. Existing conversations are unchanged.

Schema version 66 adds independent profile, asset, attachment, exact-acknowledgement, and accepted-turn metadata tables. SQLite migration failure rolls back both tables and schema version; PostgreSQL adds forced tenant row policies for all six tables. No provider credential or prompt text is stored in Buddy metadata.

The profile deletion flag uses a boolean schema and bound boolean update values on both backends. This matches the ChaCha PostgreSQL adapter, which converts literal deletion comparisons to `FALSE` and `TRUE`.

An explicit provider/model selection in ordinary neutral workspace Chat is merged into that conversation's settings for later replies. This applies only when Chat requests persistence, both values were explicit, and the authenticated principal owns the conversation. Frozen Persona/character behavior keeps its existing policy. Temporary Buddy reply overrides never replace these saved defaults.


## Imported artwork credits

Native `source_context.artwork` is a canonical JSON string containing the
version-1 `creator`, `license`, `source_url` and `notices` record. Import validates
and stores it in the pack's existing manifest JSON under `tldw/artwork`.
Independent copies retain it in `attribution.artwork` and their owned manifest,
including after deletion of the source Persona. These fields are untrusted data;
source URLs are never fetched and notices do not grant tool permissions.

Native export removes the internal manifest field from an exported copy and
restores `source_context.artwork`, preserving Chatbook's strict animation schema.
Credits participate in the export fingerprint. Invalid or conflicting carriers
are rejected; credit-free packs retain their existing behavior. Field and byte
limits are recorded in [ADR-006](../../backlog/decisions/006-buddy-artwork-credit-portability.md).

Earlier imports may already have lost their credits. Re-import the credited
original archive and create a new independent Buddy; existing immutable copies
are not silently rewritten. Keep original notice files for archives which did
not embed credits. The existing import-job repository supports SQLite only;
PostgreSQL artwork snapshot and native export paths preserve the same credits.
