# Chatbooks Format v1.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Chatbook v1.1 format contract as an opt-in, backward-compatible export/import path with content envelopes, file inventory, integrity validation, and deterministic preview reporting.

**Architecture:** Keep v1.0.0 as the default export format while adding v1.1 helpers, schema, and preview/import validation paths. Implement v1.1 first for Explainer sessions because that content type already has structured and rendered payloads, then make the shared helpers reusable for other content types.

**Tech Stack:** FastAPI, Pydantic, Python dataclasses, SQLite-backed ChatbookService, ZIP archives, JSON Schema, pytest, Loguru.

---

## Scope Check

This plan covers one reviewable vertical slice: Chatbook v1.1 format support from schema to opt-in export, preview, and import validation. It does not implement every content type as a full v1.1 producer. The first producer is `explainer_session`; other content types keep v1-compatible output until follow-up tasks add envelopes.

Default behavior must remain compatible with existing v1.0.0 tests. v1.1 should be opt-in through an explicit API/service format-version field until frontend and external clients are updated.

## File Structure

- Create `Docs/Schemas/chatbooks_manifest_v1_1.json`
  - Canonical JSON Schema for v1.1 manifests.
- Modify `Docs/Product/Chatbooks_Format_v1_1_SPEC.md`
  - Add implementation status notes only if implementation changes the agreed contract.
- Modify `Docs/API-related/Chatbook_API_Documentation.md`
  - Document optional export `format_version` and preview compatibility report.
- Modify `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
  - Add `ChatbookVersion.V1_1` and typed helpers only if they are small and shared.
- Create `tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py`
  - Shared v1.1 feature registry, envelope builders, file inventory, integrity helpers, and preview report helpers.
- Modify `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
  - Accept requested format version, build v1.1 manifests, write Explainer rendered Markdown files, compute file inventory, validate v1.1 on preview/import.
- Modify `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
  - Add optional `format_version`, v1.1 preview report response models, and response examples.
- Modify `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
  - Pass `format_version` to the service and surface preview report data.
- Modify `tldw_Server_API/app/core/Explainer/chatbook_adapter.py`
  - Reuse existing structured/rendered payload; expose helper accessors if needed, without duplicating export logic.
- Create `tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py`
  - Schema contract tests and real opt-in export validation.
- Create `tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_file_inventory.py`
  - File inventory, checksum, and self-reference tests.
- Create `tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_preview.py`
  - Preview compatibility, warning, and error report tests.
- Modify `tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py`
  - Add v1.1 Explainer envelope/export assertions.

## Task 1: Add v1.1 Schema Contract

**Files:**
- Create: `Docs/Schemas/chatbooks_manifest_v1_1.json`
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`

- [ ] **Step 1: Add the failing version enum test**

Add to `tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py`:

```python
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ChatbookVersion


def test_chatbook_version_accepts_v1_1():
    assert ChatbookVersion("1.1.0") is ChatbookVersion.V1_1
```

- [ ] **Step 2: Run the version enum test and verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py::test_chatbook_version_accepts_v1_1 -v
```

Expected: fail with `ValueError` or missing `V1_1`.

- [ ] **Step 3: Add `ChatbookVersion.V1_1`**

Modify `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`:

```python
class ChatbookVersion(str, Enum):
    V1 = "1.0.0"
    V1_LEGACY = "1.0"
    V1_1 = "1.1.0"
    V2 = "2.0.0"
```

- [ ] **Step 4: Run the version enum test and verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py::test_chatbook_version_accepts_v1_1 -v
```

Expected: pass.

- [ ] **Step 5: Add the v1.1 JSON Schema**

Create `Docs/Schemas/chatbooks_manifest_v1_1.json`. Start from `Docs/Schemas/chatbooks_manifest_v1.json`, then add:

```json
{
  "$id": "https://schemas.tldw.ai/chatbooks/manifest/v1.1.json",
  "properties": {
    "version": {
      "type": "string",
      "const": "1.1.0"
    },
    "features_used": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "content_envelopes",
          "file_inventory",
          "integrity_metadata",
          "typed_source_refs",
          "representations",
          "lossiness_metadata",
          "schema_refs",
          "redaction_profiles",
          "external_rehydration"
        ]
      }
    },
    "producer": {
      "type": "object",
      "additionalProperties": true
    },
    "source_instance": {
      "type": "object",
      "additionalProperties": true
    },
    "compatibility": {
      "type": "object",
      "additionalProperties": true
    },
    "file_inventory": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["path", "media_type", "size_bytes", "integrity", "role"],
        "additionalProperties": true
      }
    }
  }
}
```

Keep v1 fields required. Do not require v1.1 fields globally until the service can produce them for all selected content. Instead, require them in tests for v1.1 opt-in exports.

- [ ] **Step 6: Add schema validation tests**

Add a minimal manifest fixture test:

```python
import json
from pathlib import Path

import jsonschema


def test_minimal_v1_1_manifest_matches_schema():
    schema_path = Path("Docs/Schemas/chatbooks_manifest_v1_1.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    manifest = {
        "version": "1.1.0",
        "name": "v1.1 contract",
        "description": "contract",
        "author": None,
        "created_at": "2026-06-18T12:00:00+00:00",
        "updated_at": "2026-06-18T12:00:00+00:00",
        "export_id": "contract-export",
        "content_items": [],
        "relationships": [],
        "configuration": {
            "include_media": False,
            "include_embeddings": False,
            "include_generated_content": True,
            "media_quality": "compressed",
            "max_file_size_mb": 100,
        },
        "statistics": {
            "total_conversations": 0,
            "total_notes": 0,
            "total_characters": 0,
            "total_media_items": 0,
            "total_prompts": 0,
            "total_evaluations": 0,
            "total_embeddings": 0,
            "total_world_books": 0,
            "total_dictionaries": 0,
            "total_documents": 0,
            "total_explainer_sessions": 0,
            "total_size_bytes": 0,
        },
        "metadata": {
            "tags": [],
            "categories": [],
            "language": "en",
            "license": None,
        },
        "user_info": {"user_id": None},
        "features_used": [],
        "producer": {"name": "tldw_server"},
        "source_instance": {},
        "compatibility": {"min_reader_version": "1.0.0"},
        "file_inventory": [],
    }
    jsonschema.validate(manifest, schema)
```

- [ ] **Step 7: Run schema tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py -v
```

Expected: pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add Docs/Schemas/chatbooks_manifest_v1_1.json \
  tldw_Server_API/app/core/Chatbooks/chatbook_models.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py
git commit -m "feat: add chatbook manifest v1.1 schema"
```

## Task 2: Add Shared v1.1 Format Helpers

**Files:**
- Create: `tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py`
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_file_inventory.py`

- [ ] **Step 1: Write failing helper tests**

Create tests for:

```python
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_format_v1_1 import (
    build_file_inventory,
    ensure_known_features,
)


def test_build_file_inventory_excludes_manifest_and_hashes_payload(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    payload = tmp_path / "content" / "notes" / "note_1.md"
    payload.parent.mkdir(parents=True)
    payload.write_text("hello", encoding="utf-8")

    inventory = build_file_inventory(tmp_path)

    assert [item["path"] for item in inventory] == ["content/notes/note_1.md"]
    assert inventory[0]["integrity"]["algorithm"] == "sha256"
    assert inventory[0]["integrity"]["value"].startswith("sha256:")


def test_ensure_known_features_reports_unknown_tokens():
    report = ensure_known_features(["content_envelopes", "future_feature"])
    assert report["supported"] == ["content_envelopes"]
    assert report["unsupported"] == ["future_feature"]
```

- [ ] **Step 2: Run helper tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_file_inventory.py -v
```

Expected: import failure because helper module does not exist.

- [ ] **Step 3: Implement helper module**

Create `tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py`:

```python
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any

FEATURE_REGISTRY = {
    "content_envelopes",
    "file_inventory",
    "integrity_metadata",
    "typed_source_refs",
    "representations",
    "lossiness_metadata",
    "schema_refs",
    "redaction_profiles",
    "external_rehydration",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def ensure_known_features(features: list[str]) -> dict[str, list[str]]:
    supported = [feature for feature in features if feature in FEATURE_REGISTRY]
    unsupported = [feature for feature in features if feature not in FEATURE_REGISTRY]
    return {"supported": supported, "unsupported": unsupported}


def build_file_inventory(work_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(work_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(work_dir).as_posix()
        if rel == "manifest.json" or rel.endswith(".sha256"):
            continue
        media_type = mimetypes.guess_type(rel)[0] or "application/octet-stream"
        entries.append(
            {
                "path": rel,
                "media_type": media_type,
                "size_bytes": path.stat().st_size,
                "integrity": {
                    "status": "verified",
                    "algorithm": "sha256",
                    "value": sha256_file(path),
                },
                "role": _role_for_path(rel),
                "content_item_ids": [],
            }
        )
    return entries


def _role_for_path(rel: str) -> str:
    if rel.startswith("rendered/"):
        return "rendered"
    if rel.startswith("schemas/"):
        return "schema"
    if rel == "README.md":
        return "readme"
    return "payload"
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_file_inventory.py -v
```

Expected: pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_file_inventory.py
git commit -m "feat: add chatbook v1.1 format helpers"
```

## Task 3: Add Opt-In v1.1 Export Request Support

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py`

- [ ] **Step 1: Write failing API schema test**

Add to `test_chatbooks_manifest_v1_1_contract.py`:

```python
from tldw_Server_API.app.api.v1.schemas.chatbook_schemas import CreateChatbookRequest


def test_create_chatbook_request_accepts_format_version_v1_1():
    request = CreateChatbookRequest(
        name="v1.1",
        description="v1.1",
        content_selections={},
        format_version="1.1.0",
    )
    assert request.format_version.value == "1.1.0"
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py::test_create_chatbook_request_accepts_format_version_v1_1 -v
```

Expected: validation error or missing field.

- [ ] **Step 3: Add request field**

Modify `CreateChatbookRequest`:

```python
format_version: ChatbookVersion = Field(
    ChatbookVersion.V1,
    description="Chatbook manifest format version to produce"
)
```

Keep default as `ChatbookVersion.V1` to avoid breaking existing clients.

- [ ] **Step 4: Pass `format_version` through endpoint and service**

In `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`, pass
`request.format_version` into `service.create_chatbook(...)`.

In `ChatbookService.create_chatbook(...)` and `_create_chatbook_sync_wrapper(...)`,
accept `format_version: ChatbookVersion = ChatbookVersion.V1`.

- [ ] **Step 5: Run existing sync contract tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_sync_contracts.py -v
```

Expected: pass, proving default v1.0.0 behavior is unchanged.

- [ ] **Step 6: Commit Task 3**

```bash
git add tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py
git commit -m "feat: add opt-in chatbook format version"
```

## Task 4: Produce v1.1 Manifest Metadata and File Inventory

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py`

- [ ] **Step 1: Write failing real-export schema test**

Add a test that calls `create_chatbook(..., format_version=ChatbookVersion.V1_1)`,
opens `manifest.json`, and asserts:

```python
assert manifest["version"] == "1.1.0"
assert manifest["features_used"] == [
    "content_envelopes",
    "file_inventory",
    "integrity_metadata",
    "representations",
    "lossiness_metadata",
]
assert "file_inventory" in manifest
assert all(entry["path"] != "manifest.json" for entry in manifest["file_inventory"])
```

Also validate against `Docs/Schemas/chatbooks_manifest_v1_1.json`.

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py::test_real_v1_1_export_matches_schema -v
```

Expected: v1.1 fields missing.

- [ ] **Step 3: Extend manifest serialization safely**

Option A, preferred: add optional fields to `ChatbookManifest`:

```python
features_used: list[str] = field(default_factory=list)
producer: dict[str, Any] = field(default_factory=dict)
source_instance: dict[str, Any] = field(default_factory=dict)
compatibility: dict[str, Any] = field(default_factory=dict)
file_inventory: list[dict[str, Any]] = field(default_factory=list)
```

In `to_dict`, include these fields only when non-empty or when
`version == ChatbookVersion.V1_1`.

In `from_dict`, parse these fields with safe defaults.

- [ ] **Step 4: Add service v1.1 manifest initialization**

In `_create_chatbook_sync_wrapper`, when `format_version == ChatbookVersion.V1_1`:

```python
manifest.version = ChatbookVersion.V1_1
manifest.features_used = [
    "content_envelopes",
    "file_inventory",
    "integrity_metadata",
    "representations",
    "lossiness_metadata",
]
manifest.producer = {
    "name": "tldw_server",
    "component": "chatbooks",
}
manifest.compatibility = {
    "min_reader_version": "1.0.0",
    "recommended_reader_version": "1.1.0",
    "unsupported_feature_behavior": "warn_and_skip",
}
```

Do not add whole-archive checksum data to `manifest.json`.

- [ ] **Step 5: Build file inventory after content files are written**

After README and all content files are written, but before the final
`manifest.json` write for v1.1:

```python
if manifest.version == ChatbookVersion.V1_1:
    manifest.file_inventory = build_file_inventory(work_dir)
```

Then write `manifest.json`, zip, calculate `total_size_bytes`, rewrite manifest,
rebuild inventory if needed only when inventory-excluded files changed. Because
`manifest.json` is excluded from inventory, inventory does not self-reference.

- [ ] **Step 6: Run v1 and v1.1 contract tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py -v
```

Expected: pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add tldw_Server_API/app/core/Chatbooks/chatbook_models.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py
git commit -m "feat: emit chatbook v1.1 manifest metadata"
```

## Task 5: Add v1.1 Explainer Content Envelopes

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/app/core/Explainer/chatbook_adapter.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py`

- [ ] **Step 1: Write failing Explainer v1.1 export test**

Add a test that exports one Explainer session with `format_version=ChatbookVersion.V1_1`.
Assert:

```python
item = next(item for item in manifest["content_items"] if item["type"] == "explainer_session")
envelope = item["metadata"]["envelope"]
assert envelope["format"] == "tldw.explainer_session.v1"
assert envelope["representations"][0]["kind"] == "structured"
assert envelope["representations"][0]["path"] == item["file_path"]
assert any(rep["kind"] == "markdown" for rep in envelope["representations"])
assert envelope["integrity"]["status"] == "verified"
assert envelope["lossiness"]["mode"] == "lossless"
assert "rendered/explainer_sessions/" in zf.namelist()
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py::test_v1_1_explainer_export_writes_envelope_and_rendered_markdown -v
```

Expected: envelope missing.

- [ ] **Step 3: Add envelope builder helper**

In `chatbook_format_v1_1.py`, add:

```python
def build_content_envelope(
    *,
    format_id: str,
    schema_version: int | str,
    media_type: str,
    structured_path: str,
    integrity_value: str | None,
    lossiness_mode: str = "lossless",
    rendered: list[dict[str, Any]] | None = None,
    source_refs: list[dict[str, Any]] | None = None,
    redaction_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    representations = [
        {
            "kind": "structured",
            "path": structured_path,
            "media_type": media_type,
            "primary": True,
            "role": "restore_payload",
        }
    ]
    representations.extend(rendered or [])
    return {
        "format": format_id,
        "schema_version": schema_version,
        "media_type": media_type,
        "representations": representations,
        "integrity": {
            "status": "verified" if integrity_value else "unsupported",
            "algorithm": "sha256" if integrity_value else None,
            "value": integrity_value,
            "scope": "primary_payload",
        },
        "lossiness": {"mode": lossiness_mode, "reasons": []},
        "provenance": {},
        "source_refs": source_refs or [],
        "attachments": [],
        "redaction_profile": redaction_profile,
    }
```

- [ ] **Step 4: Write rendered Explainer Markdown for v1.1**

In `_collect_explainer_sessions`, when manifest version is v1.1:

1. Write existing JSON payload under `content/explainer_sessions/...`.
2. Write `payload["rendered"]["markdown"]` to
   `rendered/explainer_sessions/session_<id>.md`.
3. Compute the structured payload SHA-256 after writing the JSON file.
4. Add `metadata.envelope` to the `ContentItem`.
5. Keep `metadata.format` and `file_path` populated for v1 compatibility.

- [ ] **Step 5: Run Explainer export/import tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py \
  tldw_Server_API/tests/Explainer/test_explainer_chatbook_export.py -v
```

Expected: pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py \
  tldw_Server_API/app/core/Explainer/chatbook_adapter.py \
  tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py
git commit -m "feat: add v1.1 explainer chatbook envelopes"
```

## Task 6: Add v1.1 Preview Report

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_preview.py`

- [ ] **Step 1: Write failing preview report test**

Create a v1.1 archive fixture and assert preview returns:

```python
response = client.post("/api/v1/chatbooks/preview", files={"file": ("v11.chatbook", data, "application/zip")})
body = response.json()
assert body["compatibility"]["manifest_version"] == "1.1.0"
assert "file_inventory" in body["features"]["supported"]
assert body["integrity"]["verified_files"] >= 1
```

- [ ] **Step 2: Run preview test and verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_preview.py -v
```

Expected: response schema has no preview report fields.

- [ ] **Step 3: Add response models**

In `chatbook_schemas.py`, add Pydantic models:

```python
class ChatbookPreviewCompatibility(BaseModel):
    status: str
    reader_version: str = "1.1.0"
    manifest_version: str | None = None


class ChatbookPreviewFeatures(BaseModel):
    supported: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)


class ChatbookPreviewIntegrity(BaseModel):
    verified_files: int = 0
    failed_files: list[dict[str, Any]] = Field(default_factory=list)
```

Extend `PreviewChatbookResponse` with optional:

```python
compatibility: Optional[ChatbookPreviewCompatibility] = None
features: Optional[ChatbookPreviewFeatures] = None
integrity: Optional[ChatbookPreviewIntegrity] = None
lossiness: Optional[dict[str, Any]] = None
source_refs: Optional[dict[str, int]] = None
warnings: list[str] = Field(default_factory=list)
errors: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Implement preview report helper**

In `chatbook_format_v1_1.py`, add `build_preview_report(manifest, extract_dir)`.
It should:

1. Detect manifest version.
2. Call `ensure_known_features`.
3. Verify file inventory paths exist and hashes match.
4. Count lossiness modes from envelopes.
5. Count `source_refs` by `resolution_status`.
6. Return `warnings` and `errors`.

- [ ] **Step 5: Wire preview endpoint**

Update `ChatbookService.preview_chatbook` to optionally return a report:

Preferred low-risk approach:

```python
def preview_chatbook(self, file_path: str) -> tuple[ChatbookManifest | None, str | None, dict[str, Any] | None]:
```

Then update endpoint callers. If this is too invasive, add a new helper
`preview_chatbook_with_report` and keep the existing method for compatibility.

- [ ] **Step 6: Run preview tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_preview.py -v
```

Expected: pass.

- [ ] **Step 7: Commit Task 6**

```bash
git add tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_preview.py
git commit -m "feat: add chatbook v1.1 preview report"
```

## Task 7: Enforce v1.1 Integrity Before Import

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbook_service_preview_import_safety.py`

- [ ] **Step 1: Write failing checksum mismatch import test**

Create a v1.1 archive with file inventory hash for `content/notes/note_1.md`,
then mutate that file before import. Assert import fails before writing.

Expected test shape:

```python
success, message, details = service._import_chatbook_sync(
    file_path=str(archive_path),
    content_selections=None,
    conflict_resolution=ConflictResolution.SKIP,
    prefix_imported=False,
    import_media=False,
    import_embeddings=False,
)
assert success is False
assert "checksum" in message.lower() or any("checksum" in warning.lower() for warning in details["warnings"])
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py::test_v1_1_import_rejects_checksum_mismatch_before_writes -v
```

Expected: import currently ignores file inventory.

- [ ] **Step 3: Add validation entry point**

In `chatbook_format_v1_1.py`, add:

```python
def validate_v1_1_before_import(manifest: ChatbookManifest, extract_dir: Path) -> tuple[bool, list[str], list[str]]:
    report = build_preview_report(manifest, extract_dir)
    errors = list(report.get("errors") or [])
    warnings = list(report.get("warnings") or [])
    return not errors, warnings, errors
```

- [ ] **Step 4: Call validation after manifest parse and before imports**

In `_import_chatbook_sync`, after `manifest = ChatbookManifest.from_dict(manifest_data)`:

```python
if manifest.version == ChatbookVersion.V1_1:
    ok, warnings, errors = validate_v1_1_before_import(manifest, extract_dir)
    if not ok:
        return False, f"Chatbook v1.1 validation failed: {errors[0]}", {
            "imported_items": {},
            "warnings": warnings,
            "errors": errors,
        }
```

- [ ] **Step 5: Run import validation tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_preview_import_safety.py -v
```

Expected: pass.

- [ ] **Step 6: Commit Task 7**

```bash
git add tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_preview_import_safety.py
git commit -m "feat: validate chatbook v1.1 integrity before import"
```

## Task 8: Update Documentation and API Examples

**Files:**
- Modify: `Docs/API-related/Chatbook_API_Documentation.md`
- Modify: `Docs/Code_Documentation/Chatbook_Developer_Guide.md`
- Modify: `tldw_Server_API/app/core/Chatbooks/README.md`
- Modify: `Docs/Product/Chatbooks_Format_v1_1_SPEC.md`

- [ ] **Step 1: Update API docs**

Document:

- `format_version` in export request body.
- v1.1 default remains opt-in.
- preview response compatibility report fields.
- checksum mismatch behavior.

- [ ] **Step 2: Update developer guide**

Document:

- `chatbook_format_v1_1.py` helper responsibilities.
- v1.1 export flow.
- v1.1 preview/import validation flow.
- where to add envelopes for future content types.

- [ ] **Step 3: Update module README**

Add a short v1.1 section with:

- format helper module
- opt-in export behavior
- first producer: Explainer sessions
- tests to run

- [ ] **Step 4: Update spec implementation status**

In `Docs/Product/Chatbooks_Format_v1_1_SPEC.md`, add an implementation status
note with completed stages. Do not rewrite the contract unless implementation
found a genuine issue.

- [ ] **Step 5: Commit docs**

```bash
git add Docs/API-related/Chatbook_API_Documentation.md \
  Docs/Code_Documentation/Chatbook_Developer_Guide.md \
  tldw_Server_API/app/core/Chatbooks/README.md \
  Docs/Product/Chatbooks_Format_v1_1_SPEC.md
git commit -m "docs: document chatbook v1.1 rollout"
```

## Task 9: Final Verification

**Files:**
- Modify: `backlog/tasks/<task-id>.md` for final notes if using Backlog.md.

- [ ] **Step 1: Run focused Chatbooks tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_file_inventory.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_preview.py \
  tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run relevant endpoint preview tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py -v
```

Expected: pass.

- [ ] **Step 3: Run Bandit on touched Python scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/Chatbooks \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py \
  -f json -o /tmp/bandit_chatbooks_v1_1.json
```

Expected: no new findings in touched code. Investigate and fix new findings
before finalizing.

- [ ] **Step 4: Run schema JSON parse check**

Run:

```bash
python -m json.tool Docs/Schemas/chatbooks_manifest_v1_1.json >/tmp/chatbooks_manifest_v1_1.pretty.json
```

Expected: exit 0.

- [ ] **Step 5: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output, exit 0.

- [ ] **Step 6: Update Backlog task final summary**

Record:

- implemented stages
- tests run with exact commands
- Bandit result path
- known skips or follow-up tasks

- [ ] **Step 7: Final commit**

```bash
git add backlog/tasks/<task-id>.md
git commit -m "chore: record chatbook v1.1 verification"
```

## Follow-Up Tasks

Create separate Backlog tasks for these after the first v1.1 slice lands:

1. Add v1.1 envelopes for notes, dictionaries, generated documents, and media descriptors.
2. Add whole-archive checksum metadata to export job records and download responses.
3. Add frontend preview UI for v1.1 compatibility, lossiness, and integrity reports.
4. Add raw-preservation import mode for unknown content types if product chooses that behavior.
5. Add `schemas/` payload schema publication for each first-party content format.
