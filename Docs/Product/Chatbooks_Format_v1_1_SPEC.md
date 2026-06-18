# Chatbooks Format v1.1 Specification

- **Status:** Draft
- **Owner:** tldw_server core team
- **Backlog:** TASK-2332
- **Last Updated:** 2026-06-18
- **Related:** `Docs/Product/Chatbooks_PRD.md`, `Docs/Schemas/chatbooks_manifest_v1.json`, `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`

## Implementation Status

The current rollout implements this spec as an opt-in v1.1 path while keeping
v1.0.0 as the default export format. Completed implementation stages include:

- v1.1 manifest version support and `Docs/Schemas/chatbooks_manifest_v1_1.json`.
- Shared `chatbook_format_v1_1.py` helpers for the feature registry,
  `file_inventory` hashing, content envelopes, preview report generation, and
  pre-import validation.
- API/service `format_version` handling so clients must request `"1.1.0"` to
  produce v1.1 output.
- v1.1 manifest metadata, producer/source/compatibility data, and archive file
  inventory generation.
- Explainer session envelopes as the first v1.1 producer, including structured
  restore JSON and rendered Markdown representation.
- v1.1 preview report fields for compatibility, features, integrity, lossiness,
  source references, warnings, and errors.
- v1.1 import validation before writes, including checksum mismatches, missing
  inventory entries for import payloads, and bundled conversation attachment
  inventory coverage.

The rollout intentionally does not make every content type a full v1.1 producer
yet. Future content-type work should add envelopes and validation coverage
incrementally while preserving v1-compatible fields.

## 1. Purpose

Chatbook v1.1 is a backward-compatible format update for making Chatbook
archives more reliable as interchange artifacts. It keeps the current ZIP plus
`manifest.json` architecture, but strengthens the contract around item payloads,
file integrity, feature negotiation, references, and import behavior.

The v1.0.0 format is good at listing exported objects. The v1.1 format must also
tell a reader how each object can be restored, what was bundled, what was left as
an external reference, what was redacted or truncated, and which unsupported
features would make an import lossy.

## 2. Goals

1. Preserve v1.0.0 archive compatibility where practical.
2. Make each content item self-describing enough for module-specific importers
   and external tools.
3. Require deterministic validation and preview behavior before import.
4. Record per-file integrity without putting a self-referential whole-archive
   hash inside `manifest.json`.
5. Separate relationships between bundled Chatbook items from references to
   external or unresolved sources.
6. Standardize the successful pattern already used by Explainer exports:
   structured content for restoration and rendered Markdown for human review.

## 3. Non-Goals

- Replacing the ZIP archive container.
- Requiring all content types to implement v1.1 envelopes at the same time.
- Making rendered Markdown mandatory for every content type.
- Implementing real-time sync, collaborative editing, or delta imports.
- Defining the final JSON Schema file in this document. A separate
  `chatbooks_manifest_v1_1.json` should be created when implementation begins.
- Storing the final archive checksum inside `manifest.json`.

## 4. Compatibility Model

### 4.1 Version Field

v1.1 uses the existing top-level `version` field:

```json
{
  "version": "1.1.0"
}
```

Do not add a parallel `manifest_version` field. The current `version` field
remains the single source of truth for manifest versioning.

### 4.2 Reader Compatibility

v1.1 readers MUST accept:

- `version: "1.1.0"`
- `version: "1.0.0"`
- legacy `version: "1.0"` where the current model already accepts it

v1.0 readers are not expected to understand all v1.1 fields. v1.1 producers
SHOULD keep v1-compatible fields populated when possible:

- `content_items[].file_path`
- `content_items[].checksum`
- `metadata.tags`
- `metadata.categories`
- `statistics`
- `relationships`

### 4.3 Compatibility Object

v1.1 manifests SHOULD add:

```json
{
  "compatibility": {
    "min_reader_version": "1.0.0",
    "recommended_reader_version": "1.1.0",
    "unsupported_feature_behavior": "warn_and_skip",
    "v1_compatibility": {
      "file_path_populated": true,
      "checksum_alias_populated": true
    }
  }
}
```

`unsupported_feature_behavior` values:

- `warn_and_skip`: unsupported items or features are skipped with warnings.
- `warn_lossy_import`: import can proceed but loses information.
- `reject_import`: import must fail before writing content.

## 5. Top-Level Manifest Additions

v1.1 keeps all required v1.0.0 fields and adds optional fields that v1.1 readers
understand.

```json
{
  "version": "1.1.0",
  "features_used": [
    "content_envelopes",
    "file_inventory",
    "integrity_metadata",
    "typed_source_refs",
    "representations",
    "lossiness_metadata"
  ],
  "producer": {
    "name": "tldw_server",
    "version": "0.1.0",
    "component": "chatbooks",
    "exported_at": "2026-06-18T12:00:00Z"
  },
  "source_instance": {
    "instance_id": "sha256:...",
    "deployment_kind": "self_hosted",
    "export_scope": "user"
  },
  "compatibility": {
    "min_reader_version": "1.0.0",
    "recommended_reader_version": "1.1.0",
    "unsupported_feature_behavior": "warn_and_skip"
  },
  "file_inventory": []
}
```

### 5.1 Feature Registry

`features_used` MUST use stable tokens. Initial tokens:

| Token | Meaning |
| --- | --- |
| `content_envelopes` | Items include a v1.1 envelope in `content_items[].metadata.envelope`. |
| `file_inventory` | Manifest includes top-level file records for bundled files. |
| `integrity_metadata` | Items and files use structured integrity objects. |
| `typed_source_refs` | Items declare source references separately from relationships. |
| `representations` | Items declare structured, rendered, or binary representations. |
| `lossiness_metadata` | Items declare whether export/import is lossless, reference-only, truncated, redacted, or rendered-only. |
| `schema_refs` | Items point to JSON Schema documents bundled in `schemas/` or published externally. |
| `redaction_profiles` | Items declare intentional redaction rules applied during export. |
| `external_rehydration` | Items include hints for recovering external media or source objects. |

Unknown feature tokens MUST be reported during preview. Whether import proceeds
depends on `compatibility.unsupported_feature_behavior`.

## 6. Archive Layout

v1.1 continues to use:

```text
manifest.json
README.md
content/<content_type>/...
```

v1.1 producers MAY add:

```text
schemas/
  tldw.explainer_session.v1.schema.json
rendered/
  <content_type>/...
```

A whole-archive checksum MUST NOT be stored inside the ZIP it hashes. Producers
that compute an archive checksum SHOULD expose it through export job metadata,
download response metadata, or a sidecar next to the archive, such as
`example.chatbook.zip.sha256`.

## 7. File Inventory

The top-level `file_inventory` records every bundled file except `manifest.json`
and whole-archive checksum sidecars. `manifest.json` cannot carry its own
content hash without becoming self-referential. Whole-archive checksum sidecars
must live outside the ZIP if they hash the final ZIP bytes.

```json
{
  "path": "content/explainer_sessions/session_exp_123.json",
  "media_type": "application/json",
  "size_bytes": 14520,
  "integrity": {
    "status": "verified",
    "algorithm": "sha256",
    "value": "sha256:..."
  },
  "role": "payload",
  "content_item_ids": ["exp_123"]
}
```

`role` values:

- `payload`
- `rendered`
- `attachment`
- `schema`
- `readme`
- `other`

Readers MUST verify file inventory entries before import when `features_used`
contains `file_inventory` or `integrity_metadata`. Readers verify
`manifest.json` through normal JSON parsing, schema validation, and optional
external archive checksum metadata rather than through `file_inventory`.

## 8. Content Item Envelope

v1.1 preserves the v1 `ContentItem` shape and adds an envelope under
`content_items[].metadata.envelope`. This avoids adding a second source of truth
for existing fields such as `file_path`.

```json
{
  "id": "exp_123",
  "type": "explainer_session",
  "title": "Learn attention",
  "file_path": "content/explainer_sessions/session_exp_123.json",
  "checksum": "sha256:...",
  "metadata": {
    "format": "tldw.explainer_session.v1",
    "envelope": {
      "format": "tldw.explainer_session.v1",
      "schema_version": 1,
      "schema_ref": "schemas/tldw.explainer_session.v1.schema.json",
      "media_type": "application/json",
      "representations": [],
      "integrity": {},
      "lossiness": {},
      "provenance": {},
      "source_refs": [],
      "attachments": [],
      "redaction_profile": null
    }
  }
}
```

### 8.1 Required Envelope Fields

For v1.1 content-envelope items, these fields are required:

| Field | Requirement |
| --- | --- |
| `format` | Stable payload format identifier, for example `tldw.explainer_session.v1`. |
| `schema_version` | Integer or semantic string used by the payload importer. |
| `media_type` | MIME type of the primary structured payload. |
| `representations` | Array of structured, rendered, binary, or reference representations. |
| `integrity` | Structured integrity status for the item. |
| `lossiness` | Structured statement of export fidelity. |
| `provenance` | Source identity and export details. |

### 8.2 `file_path` Compatibility

`content_items[].file_path` remains the v1-compatible pointer to the primary
payload. v1.1 MUST NOT introduce a second top-level `payload_path` field.

The v1.1 equivalent is expressed as:

```json
{
  "representations": [
    {
      "kind": "structured",
      "path": "content/explainer_sessions/session_exp_123.json",
      "media_type": "application/json",
      "primary": true
    }
  ]
}
```

When `file_path` and the primary structured representation disagree, v1.1
readers MUST report a validation error and reject that item before import.

## 9. Representations

Representations tell readers what files or references exist for one item.

```json
{
  "kind": "structured",
  "path": "content/explainer_sessions/session_exp_123.json",
  "media_type": "application/json",
  "primary": true,
  "role": "restore_payload"
}
```

Allowed `kind` values:

- `structured`: machine-readable payload intended for restoration.
- `markdown`: human-readable Markdown rendering.
- `html`: human-readable HTML rendering.
- `plain_text`: human-readable plain text rendering.
- `binary`: bundled binary asset.
- `thumbnail`: bundled preview asset.
- `external_reference`: non-bundled external dependency.

Rendered representations are optional. Machine-oriented or binary-heavy content
types do not need Markdown or HTML, but when they provide a human-readable view
they SHOULD use a representation rather than ad hoc metadata.

## 10. Integrity Metadata

v1.1 replaces the single nullable `checksum` concept with structured integrity
metadata while preserving `checksum` as a v1 compatibility alias.

```json
{
  "integrity": {
    "status": "verified",
    "algorithm": "sha256",
    "value": "sha256:...",
    "scope": "primary_payload"
  }
}
```

Allowed `status` values:

- `verified`: bundled bytes have a supported checksum.
- `not_bundled`: item is reference-only and has no bundled payload.
- `external_unverified`: item references external data that was not fetched.
- `missing`: exporter expected a file but could not include it.
- `redacted`: bytes or fields were intentionally removed.
- `unsupported`: exporter cannot compute integrity for this item type.

Rules:

1. Bundled files MUST have SHA-256 file inventory entries.
2. Bundled primary payloads SHOULD copy the SHA-256 into
   `content_items[].checksum` for v1 compatibility.
3. Reference-only items MUST NOT use a fake checksum. They MUST set an
   integrity status and reason.
4. Checksum mismatch during preview MUST block import unless the item is skipped
   before any writes occur.
5. Non-`verified` integrity statuses SHOULD include `reason` with a stable code,
   for example `large_media_not_bundled`, `redacted_by_policy`, or
   `unsupported_hash_source`.

## 11. Lossiness Metadata

Lossiness describes whether the exported item can be restored exactly.

```json
{
  "lossiness": {
    "mode": "reference_only",
    "reasons": [
      "large_media_not_bundled"
    ],
    "user_visible": true
  }
}
```

Allowed `mode` values:

- `lossless`
- `reference_only`
- `truncated`
- `rendered_only`
- `partially_redacted`
- `metadata_only`
- `unknown`

Readers MUST expose non-`lossless` items in preview results.

## 12. Provenance

Every v1.1 envelope SHOULD include provenance:

```json
{
  "provenance": {
    "source_instance_id": "sha256:...",
    "source_user_hash": "sha256:...",
    "original_id": "exp_123",
    "exported_id": "exp_123",
    "created_at": "2026-06-18T12:00:00Z",
    "exported_at": "2026-06-18T12:30:00Z",
    "producer": "tldw_server.chatbooks",
    "producer_version": "0.1.0"
  }
}
```

Importers SHOULD preserve `original_id` and `source_instance_id` in local
provenance fields when creating new records.

## 13. Source References

`relationships` describe links between Chatbook content items. `source_refs`
describe dependencies on source material that may or may not be bundled in the
archive.

```json
{
  "source_type": "media",
  "source_id": "media_42",
  "title": "Attention paper notes",
  "resolution_status": "external",
  "snapshot_hash": "sha256:...",
  "location": {
    "label": "chunk 3",
    "start_offset": 120,
    "end_offset": 178
  },
  "rehydration_hint": {
    "strategy": "media_db_lookup",
    "metadata": {
      "media_uuid": "uuid-42"
    }
  }
}
```

Allowed `resolution_status` values:

- `bundled`: referenced source is included in this archive.
- `external`: referenced source exists elsewhere and may be rehydrated.
- `unresolved`: source identity is preserved but no location is available.
- `redacted`: source was intentionally removed.
- `missing`: source was expected but unavailable during export.

Importers MUST NOT silently treat unresolved references as bundled content.

## 14. Relationships

The existing top-level `relationships` array remains available for links between
Chatbook items:

```json
{
  "source_id": "conversation_1",
  "target_id": "character_1",
  "relationship_type": "uses_character",
  "metadata": {
    "source_type": "conversation",
    "target_type": "character"
  }
}
```

v1.1 readers SHOULD validate that `source_id` and `target_id` refer to manifest
content items. Missing relationship targets MUST produce preview warnings. They
SHOULD block import only when the relationship is required for restoring the
source item.

## 15. Redaction Profiles

If sensitive fields are removed, the envelope SHOULD describe the policy without
revealing the removed values.

```json
{
  "redaction_profile": {
    "profile_id": "tldw.default.safe_export.v1",
    "redacted_fields": [
      "generationMetadata.api_key",
      "generationMetadata.system_prompt"
    ],
    "reason": "secret_scrubbing"
  }
}
```

This generalizes the current Explainer behavior that removes API keys, tokens,
passwords, raw prompts, and system prompts from exported metadata.

## 16. Reader Behavior

Preview and import MUST be deterministic. A v1.1 reader should produce a report
before writing content.

| Condition | Preview result | Import behavior |
| --- | --- | --- |
| v1.0.0 archive | Compatible | Import using v1 path. |
| legacy `1.0` archive | Compatible with warning | Import using v1 path. |
| v1.1 archive, supported features | Compatible | Import eligible items. |
| Unknown feature with `warn_and_skip` | Warning | Skip affected items. |
| Unknown feature with `warn_lossy_import` | Warning | Import only with lossy warning. |
| Unknown feature with `reject_import` | Error | Reject before writes. |
| Unknown content type | Warning | Skip item unless caller explicitly opts into raw preservation. |
| Checksum mismatch | Error | Reject affected item before writes. |
| Primary `file_path` missing | Error | Reject affected item before writes. |
| `file_path` disagrees with primary representation | Error | Reject affected item before writes. |
| Non-lossless item | Warning | Import if caller accepts lossy import. |
| Unresolved source reference | Warning | Import item with unresolved provenance. |

## 17. Preview Report Contract

v1.1 preview responses SHOULD include:

```json
{
  "compatibility": {
    "status": "compatible_with_warnings",
    "reader_version": "1.1.0",
    "manifest_version": "1.1.0"
  },
  "features": {
    "supported": ["content_envelopes", "file_inventory"],
    "unsupported": []
  },
  "integrity": {
    "verified_files": 12,
    "failed_files": []
  },
  "lossiness": {
    "lossless_items": 4,
    "lossy_items": [
      {
        "item_id": "media_42",
        "mode": "reference_only",
        "reasons": ["large_media_not_bundled"]
      }
    ]
  },
  "source_refs": {
    "bundled": 3,
    "external": 5,
    "unresolved": 1,
    "redacted": 0,
    "missing": 0
  },
  "warnings": [],
  "errors": []
}
```

The field name `manifest_version` in preview responses is acceptable because it
describes the report. It MUST NOT become a second version field inside
`manifest.json`.

## 18. Examples

### 18.1 Bundled Note Item

```json
{
  "id": "note_123",
  "type": "note",
  "title": "Research notes",
  "description": null,
  "created_at": "2026-06-18T12:00:00Z",
  "updated_at": "2026-06-18T12:10:00Z",
  "tags": ["research"],
  "file_path": "content/notes/note_123.md",
  "checksum": "sha256:abc123",
  "metadata": {
    "envelope": {
      "format": "tldw.note.markdown.v1",
      "schema_version": 1,
      "media_type": "text/markdown",
      "representations": [
        {
          "kind": "structured",
          "path": "content/notes/note_123.md",
          "media_type": "text/markdown",
          "primary": true
        }
      ],
      "integrity": {
        "status": "verified",
        "algorithm": "sha256",
        "value": "sha256:abc123",
        "scope": "primary_payload"
      },
      "lossiness": {
        "mode": "lossless",
        "reasons": []
      },
      "provenance": {
        "original_id": "note_123",
        "exported_id": "note_123",
        "producer": "tldw_server.chatbooks"
      },
      "source_refs": [],
      "attachments": []
    }
  }
}
```

### 18.2 Reference-Only Media Item

```json
{
  "id": "media_42",
  "type": "media",
  "title": "Long lecture",
  "description": "Metadata-only export for large media",
  "created_at": "2026-06-18T12:00:00Z",
  "updated_at": "2026-06-18T12:10:00Z",
  "tags": ["lecture"],
  "file_path": "content/media/media_42.json",
  "checksum": "sha256:def456",
  "metadata": {
    "envelope": {
      "format": "tldw.media_descriptor.v1",
      "schema_version": 1,
      "media_type": "application/json",
      "representations": [
        {
          "kind": "structured",
          "path": "content/media/media_42.json",
          "media_type": "application/json",
          "primary": true
        },
        {
          "kind": "external_reference",
          "uri": "media-db://media_42/original",
          "media_type": "video/mp4",
          "role": "source_media"
        }
      ],
      "integrity": {
        "status": "verified",
        "algorithm": "sha256",
        "value": "sha256:def456",
        "scope": "descriptor_only"
      },
      "lossiness": {
        "mode": "reference_only",
        "reasons": ["large_media_not_bundled"]
      },
      "source_refs": [
        {
          "source_type": "media",
          "source_id": "media_42",
          "resolution_status": "external",
          "rehydration_hint": {
            "strategy": "media_db_lookup"
          }
        }
      ],
      "attachments": []
    }
  }
}
```

### 18.3 Explainer Session Item

```json
{
  "id": "exp_123",
  "type": "explainer_session",
  "title": "Learn attention",
  "description": null,
  "created_at": "2026-06-18T12:00:00Z",
  "updated_at": "2026-06-18T12:10:00Z",
  "tags": [],
  "file_path": "content/explainer_sessions/session_exp_123.json",
  "checksum": "sha256:789abc",
  "metadata": {
    "format": "tldw.explainer_session.v1",
    "envelope": {
      "format": "tldw.explainer_session.v1",
      "schema_version": 1,
      "schema_ref": "schemas/tldw.explainer_session.v1.schema.json",
      "media_type": "application/json",
      "representations": [
        {
          "kind": "structured",
          "path": "content/explainer_sessions/session_exp_123.json",
          "media_type": "application/json",
          "primary": true,
          "role": "restore_payload"
        },
        {
          "kind": "markdown",
          "path": "rendered/explainer_sessions/session_exp_123.md",
          "media_type": "text/markdown",
          "primary": false,
          "role": "human_review"
        }
      ],
      "integrity": {
        "status": "verified",
        "algorithm": "sha256",
        "value": "sha256:789abc",
        "scope": "primary_payload"
      },
      "lossiness": {
        "mode": "lossless",
        "reasons": []
      },
      "source_refs": [
        {
          "source_type": "media",
          "source_id": "media_42",
          "title": "Attention paper notes",
          "resolution_status": "external",
          "snapshot_hash": "sha256:citationhash",
          "location": {
            "label": "chunk 3",
            "start_offset": 120,
            "end_offset": 178
          }
        }
      ],
      "redaction_profile": {
        "profile_id": "tldw.default.safe_export.v1",
        "redacted_fields": [
          "generationMetadata.api_key",
          "generationMetadata.system_prompt"
        ],
        "reason": "secret_scrubbing"
      }
    }
  }
}
```

## 19. Migration Path

### Stage 1: Producer-Only Envelopes

- Add envelopes to new exports for one content type, preferably
  `explainer_session`.
- Keep existing `file_path`, `checksum`, and `metadata.format` fields populated.
- Add file inventory entries for the new payloads.

### Stage 2: Preview Awareness

- Extend preview to report v1.1 features, file integrity, lossiness, and source
  references.
- Preserve current v1 preview fields for frontend compatibility.

### Stage 3: Import Enforcement

- Verify file inventory and item integrity before writing.
- Apply reader behavior rules consistently across sync and async imports.
- Surface unsupported features and lossy imports in job warnings.

### Stage 4: Schema Publication

- Add `Docs/Schemas/chatbooks_manifest_v1_1.json`.
- Add optional payload schemas under `Docs/Schemas/chatbooks/`.
- Add contract tests that validate real exports against the v1.1 schema.

## 20. Validation Requirements

The implementation should add tests for:

- v1.1 export still includes required v1 fields.
- v1.1 reader accepts v1.0.0 and legacy `1.0`.
- Unknown feature behavior follows the compatibility object.
- File inventory detects missing files, changed bytes, and unsupported hash
  algorithms.
- `file_path` and primary structured representation mismatch is rejected.
- Reference-only items report lossiness and do not use fake payload checksums.
- Explainer exports include envelope metadata without leaking scrubbed secrets.
- Preview reports unsupported content types, lossy items, unresolved refs, and
  checksum errors before import.

## 21. Open Questions

1. Should v1.1 support raw-preservation import for unknown content types, or only
   skip them with warnings?
2. Should `schemas/` be required for first-party tldw formats, or only for
   third-party/custom item formats?
3. Should whole-archive checksum be mandatory in job metadata once export workers
   support it?
4. Should lossy import require explicit API opt-in, or is a warning enough for
   non-destructive imports?

## 22. Summary

Chatbook v1.1 should remain a conservative evolution of the current format. The
core change is not a new container; it is a stronger contract around each item:
what format it uses, where its restorable payload lives, how its files are
verified, what was omitted or redacted, and how external source references should
be treated. This makes Chatbooks safer for backups, migrations, offline bundles,
and future third-party tooling without forcing every existing exporter to migrate
at once.
