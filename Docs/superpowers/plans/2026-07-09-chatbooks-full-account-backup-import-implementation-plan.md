# Chatbooks Full Account Backup Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Chatbooks full user-account backup and archive import acceptance-ready from the WebUI and browser extension, including stored media artifacts, derived media data, embeddings, sensitive account data handling, Settings demotion, and OpenWebUI hydration mapping.

**Architecture:** Add an authoritative account-data inventory as the contract source for full export, scope preview, manifest/job summaries, import validation, and tests. Preserve selective export as explicit allowlist mode while treating omitted or empty `content_selections` as full account export. Restore archive content by inventory category rather than silently dropping unsupported content types.

**Tech Stack:** FastAPI, Pydantic, Python dataclasses, SQLite-backed ChatbookService, ChromaDB, ZIP archives, Next.js shared UI package, Vitest, Playwright, pytest, Bandit.

---

## Scope Check

This is one coordinated feature because the reviewed failures are contract mismatches across backend, WebUI, extension, docs, and tests. The implementation is split into reviewable tasks with commit boundaries:

1. Inventory and scope summary.
2. Export API contract.
3. Full export contents and manifest/job summaries.
4. Full archive import restore coverage.
5. WebUI and extension Backup all flow.
6. Settings demotion and naming cleanup.
7. OpenWebUI import mapping and hydration scope reuse.
8. Documentation, end-to-end coverage, and security verification.

Do not narrow "full account export" to selected Chatbook content types. It includes every tldw-owned account record and stored account artifact listed by the inventory. If tldw stores a file artifact for the account, export the bytes. If tldw stores only a URL, local path, or other provenance pointer, export that stored pointer and show a pointer-only warning. Do not invent source file bytes that tldw does not store.

## Finding Coverage Map

- P0 backup-all failure: Tasks 1, 2, 3, and 5 add the backend contract, full inventory expansion, scope summary, and WebUI/extension primary Backup all path.
- P0 default archive restore failure: Task 4 makes Chatbook archive import restore restorable archive media, stored/derived media data, embeddings, prompts, and evaluations by default.
- P0 data-loss risk: Tasks 1, 3, and 4 require every inventory row to be exported/restored or explicitly marked non-restorable with a visible warning.
- P0 sensitive-data risk: Tasks 1, 3, 4, and 8 require sensitive values to be included only when required for restore and redacted from previews, manifests summaries, job warnings, logs, and audit metadata.
- P1 Settings ambiguity: Task 6 demotes Settings to an entry point and labels any remaining shortcut as conversation-only.
- P1 OpenWebUI hydration recall burden: Task 7 persists source-to-destination mappings and uses a visible import scope instead of normal-path manual conversation IDs.
- P2 acceptance coverage gap: Tasks 1 through 8 add the backend, frontend, extension, E2E, documentation, and Bandit verification gates.

## File Structure

- Create `tldw_Server_API/app/core/Chatbooks/chatbook_account_inventory.py`
  - Owns the account-data inventory, sensitivity classification, restorable/pointer-only/non-restorable status, count keys, and redacted summary helpers.
- Modify `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
  - Adds manifest fields for account inventory summary, warning counts, pointer-only counts, sensitive-category counts, archive verification, and any new `ContentType.ACCOUNT_DATA` representation needed for account-owned non-chat content.
- Modify `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
  - Expands full-account selection, builds scope estimates, exports all inventory rows, restores all restorable rows, validates non-restorable rows, records post-write verification, and redacts sensitive values from summaries/log-facing structures.
- Modify `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
  - Makes export `content_selections` optional, adds scope summary response models, adds import defaults that restore archive media/embedding data by default, and adds OpenWebUI mapping/scope response models.
- Modify `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
  - Preserves omitted selections as full-account mode, exposes scope preview, maps new service summaries into API responses, and removes backend media/embedding import rejection for Chatbook archives.
- Modify `tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py`
  - Accepts persisted source-to-destination conversation mappings in addition to manually entered conversation IDs.
- Modify `tldw_Server_API/app/core/Chatbooks/openwebui_hydration_jobs.py`
  - Carries selected import scope IDs and visible scope summaries through hydration job payload/result.
- Modify `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Makes `content_selections` optional for `exportChatbook`, adds `getChatbookExportScope`, and adds OpenWebUI import scope methods.
- Modify `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
  - Adds primary Backup all mode, uses backend scope summary, sends omitted selections for full export, fixes archive import defaults, shows post-export verification, and uses OpenWebUI import scope mapping for hydration.
- Modify `apps/packages/ui/src/components/Option/Settings/chatbooks.tsx`
  - Turns Settings into a full Backup & Import entry point with an explicitly secondary conversation export shortcut.
- Modify locale and navigation files:
  - `apps/packages/ui/src/assets/locale/en/option.json`
  - `apps/packages/ui/src/assets/locale/en/settings.json`
  - `apps/packages/ui/src/public/_locales/en/option.json`
  - `apps/packages/ui/src/public/_locales/en/settings.json`
  - `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
  - `apps/packages/ui/src/routes/route-metadata.ts`
- Modify docs:
  - `Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md`
  - `Docs/API-related/Chatbook_API_Documentation.md`
  - `CHANGELOG.md`
  - `Docs/RELEASE_NOTES.md`
- Add and modify backend tests:
  - Create `tldw_Server_API/tests/Chatbooks/test_chatbooks_account_inventory.py`
  - Create `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py`
  - Create `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py`
  - Create `tldw_Server_API/tests/Chatbooks/test_chatbooks_sensitive_export_redaction.py`
  - Modify existing Chatbooks import/export tests where the expected contract changed.
- Add and modify frontend tests:
  - Create `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx`
  - Modify `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx`
  - Modify `apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts`
  - Add or modify a Settings Chatbooks component test if one exists; otherwise add `apps/packages/ui/src/components/Option/Settings/__tests__/chatbooks.test.tsx`
  - Modify `apps/extension/tests/e2e/chatbooks-export-download.spec.ts`
  - Modify `apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks.spec.ts`
  - Modify `apps/tldw-frontend/e2e/utils/page-objects/ChatbooksPage.ts`

## Task 1: Account Inventory And Scope Summary

**Files:**
- Create: `tldw_Server_API/app/core/Chatbooks/chatbook_account_inventory.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_account_inventory.py`

- [x] **Step 1: Write failing inventory coverage tests**

Add tests that require every inventory row to declare source, export representation, import handler key, dependency notes, sensitivity, restore status, manifest count key, and user-facing warning behavior.

```python
from tldw_Server_API.app.core.Chatbooks.chatbook_account_inventory import ACCOUNT_DATA_INVENTORY


def test_inventory_rows_have_required_restore_contract_fields():
    required = {
        "category",
        "source",
        "export_representation",
        "manifest_count_key",
        "import_handler_key",
        "dependencies",
        "sensitivity",
        "restore_status",
    }

    assert ACCOUNT_DATA_INVENTORY
    for row in ACCOUNT_DATA_INVENTORY:
        assert required <= row.to_summary().keys()
        assert row.manifest_count_key
        assert row.restore_status in {"restorable", "pointer_only", "non_restorable"}
```

Also add a test that the inventory contains at least these categories:

```python
EXPECTED_CATEGORIES = {
    "account_profile",
    "account_settings",
    "conversations",
    "notes",
    "characters",
    "world_books",
    "dictionaries",
    "prompts",
    "evaluations",
    "generated_documents",
    "explainer_sessions",
    "media_records",
    "media_transcripts",
    "media_chunks",
    "media_stored_artifacts",
    "media_pointers",
    "embeddings",
    "tags_categories_relationships",
    "sensitive_user_values",
}
```

- [x] **Step 2: Run the inventory tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_account_inventory.py -v
```

Expected: fail because `chatbook_account_inventory.py` does not exist.

- [x] **Step 3: Implement the inventory module**

Create `chatbook_account_inventory.py` with small dataclasses and no database calls.

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Sensitivity = Literal["public", "personal", "sensitive", "secret"]
RestoreStatus = Literal["restorable", "pointer_only", "non_restorable"]


@dataclass(frozen=True)
class AccountInventoryEntry:
    category: str
    label: str
    source: str
    export_representation: str
    manifest_count_key: str
    import_handler_key: str
    dependencies: tuple[str, ...]
    sensitivity: Sensitivity
    restore_status: RestoreStatus
    warning: str | None = None

    def to_summary(self) -> dict[str, object]:
        return asdict(self)
```

Populate `ACCOUNT_DATA_INVENTORY` from the PRD scope. Rows for deployment-local or unrecoverable secrets may be `non_restorable`, but each must have a visible warning. Rows for external media source paths/URLs are `pointer_only`, not "missing data".

- [x] **Step 4: Add scope summary service and API models**

Add models such as:

```python
class ChatbookAccountScopeCategory(BaseModel):
    category: str
    label: str
    count: int = 0
    restore_status: str
    sensitivity: str
    warning: str | None = None


class ChatbookAccountScopeResponse(BaseModel):
    mode: Literal["full_account"] = "full_account"
    categories: list[ChatbookAccountScopeCategory]
    total_items: int = 0
    pointer_only_count: int = 0
    sensitive_category_count: int = 0
    warning_count: int = 0
    estimated_size_bytes: int | None = None
```

Expose `GET /api/v1/chatbooks/export/scope` and implement `ChatbookService.get_full_account_export_scope()` using best-effort counts. Redact secret values; summaries may include labels and counts only.

- [x] **Step 5: Verify tests pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_account_inventory.py -v
```

Expected: pass.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Chatbooks/chatbook_account_inventory.py tldw_Server_API/app/core/Chatbooks/chatbook_models.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/tests/Chatbooks/test_chatbooks_account_inventory.py
git commit -m "feat: add chatbooks account export inventory"
```

Completed in commits `e8aab93f869258b67999765dd85297ae89ae651c` and `58f46d9fb9d6879328c441234c4cae78669637e3`. Focused verification: `python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_account_inventory.py -v` passed with 9 tests; Bandit on touched Chatbooks scope reported 0 findings. Independent spec and quality reviews found no blocking Task 1 issues after follow-up.

## Task 2: Export Contract For Full Account And Explicit Allowlists

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts`

- [x] **Step 1: Write failing backend contract tests**

Cover these cases:

- omitted `content_selections` exports full account mode
- `content_selections: {}` exports full account mode
- non-empty object is explicit allowlist mode
- empty arrays inside allowlist mode mean export none for that type
- a non-empty allowlist resolving to zero exportable items returns a 4xx

Test the schema directly and at the API boundary. For endpoint tests, assert omitted and `{}` do not raise Pydantic validation errors and do call the service with a full-account mode marker or `None`.

- [x] **Step 2: Run contract tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py -v
```

Expected: fail because `CreateChatbookRequest.content_selections` is required and the endpoint iterates it unconditionally.

- [x] **Step 3: Make `content_selections` optional and preserve full-account mode**

Change `CreateChatbookRequest` to:

```python
content_selections: dict[ContentType, list[str]] | None = Field(
    None,
    description=(
        "Omit or pass {} for full user-account export. "
        "Pass a non-empty object for explicit allowlist export."
    ),
)
```

In `chatbooks.py`, do not collapse omitted selections into `{}` too early. Use:

```python
content_selections = None
if request_data.content_selections:
    content_selections = {
        ContentType(content_type.value if hasattr(content_type, "value") else str(content_type)): ids
        for content_type, ids in request_data.content_selections.items()
    }
elif request_data.content_selections == {}:
    content_selections = None
```

In `ChatbookService.create_chatbook`, replace `content_selections = content_selections or {}` with an explicit selection mode:

```python
selection_mode = "full_account" if content_selections is None or content_selections == {} else "allowlist"
```

For async export payloads, include `selection_mode` and store `content_selections` as `None` for full account mode. Update the jobs adapter path if it currently assumes a mapping.

- [x] **Step 4: Reject zero-item allowlists**

In service validation, reject non-empty allowlists with no IDs before creating a job or archive:

```python
if selection_mode == "allowlist" and sum(len(ids) for ids in content_selections.values()) == 0:
    return False, "Export allowlist contains no exportable items.", None
```

The API should map this to a 4xx response. Keep selective export with some empty arrays valid as long as at least one selected type has at least one ID.

- [x] **Step 5: Update TypeScript client type**

Change:

```ts
content_selections: Record<string, string[]>
```

to:

```ts
content_selections?: Record<string, string[]>
```

Add a client test that `exportChatbook({ name, description, async_mode: true })` sends no `content_selections` field and does not fail type checking.

- [x] **Step 6: Verify tests pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py -v
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts
```

Expected: pass.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py apps/packages/ui/src/services/tldw/TldwApiClient.ts tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts
git commit -m "feat: support chatbooks full account export contract"
```

Completed in commits `377338c7aef165aa5c3d2fd5205e174bc096cbff` and `2db382f157`. Focused verification after review follow-up: `python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py -v` passed with 10 tests; `python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py -k core_jobs -v` passed with 9 selected tests; `bunx vitest run ../packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts` passed with 8 tests; Bandit on touched backend files reported 0 findings. Full `test_chatbooks_manifest_v1_1_contract.py` still has an unrelated schema failure for pre-existing `statistics.total_explainer_sessions`.

## Task 3: Full Export Contents, Manifest Summary, And Verification

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/jobs_adapter.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_sensitive_export_redaction.py`

- [x] **Step 1: Write failing full-export archive tests**

Build a fixture user with conversations, notes, characters, prompts, evaluations, media records with transcripts/derived metadata, and embedding stubs or vectors where local fixtures allow. Export with omitted selections and assert the ZIP contains manifest entries for every restorable inventory category with data.

Assertions:

- manifest `account_inventory` lists every inventory category
- manifest statistics include counts for every `manifest_count_key`
- media records are exported
- media transcripts and derived metadata are exported
- stored account file artifacts are present when the fixture creates them
- pointer-only media sources show pointer warnings but still export stored pointer metadata
- sensitive values are not in `manifest.json`, job result summaries, warnings, or logs captured by `caplog`
- archive size is present and post-write verification is true

- [x] **Step 2: Run full-export tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py tldw_Server_API/tests/Chatbooks/test_chatbooks_sensitive_export_redaction.py -v
```

Expected: fail because full-account mode does not expand to all inventory rows and manifest/job summaries do not carry the required inventory data.

- [x] **Step 3: Implement full-account expansion**

Add a helper that returns all export IDs and inventory summaries for the current user:

```python
def _expand_full_account_content_selections(self) -> dict[ContentType, list[str]]:
    return {
        ContentType.CONVERSATION: self._list_all_conversation_ids(),
        ContentType.NOTE: self._list_all_note_ids(),
        ContentType.CHARACTER: self._list_all_character_ids(),
        ContentType.WORLD_BOOK: self._list_all_world_book_ids(),
        ContentType.DICTIONARY: self._list_all_dictionary_ids(),
        ContentType.PROMPT: self._list_all_prompt_ids(),
        ContentType.EVALUATION: self._list_all_evaluation_ids(),
        ContentType.MEDIA: self._list_all_media_ids(),
        ContentType.EMBEDDING: self._list_all_embedding_collection_ids(),
        ContentType.GENERATED_DOCUMENT: self._list_all_generated_document_ids(),
        ContentType.EXPLAINER_SESSION: self._list_all_explainer_session_ids(),
    }
```

Use existing DB abstractions and helper methods. Do not query unrelated users. If a source database is absent, add an inventory warning for that category instead of omitting it silently.

- [x] **Step 4: Export stored file artifacts and pointer-only media correctly**

Replace the current media collector warning that only exports metadata when stored artifacts exist. The implementation should:

- include Media DB records, transcripts, chunks, captions, summaries, tags, links, prompts, processing results, and other stored/derived media account data exposed by Media DB helpers
- copy stored account file artifacts from configured account storage locations into `content/media/files/` when the file is under an allowed account-owned storage root
- export external URLs, external local paths, and provenance pointers as pointer metadata only
- add pointer-only warnings for unavailable source bytes without claiming the bytes were exported
- keep archive path traversal checks for copied files

- [x] **Step 5: Add manifest and job summaries**

Extend `ChatbookManifest.to_dict()` with non-secret summary fields:

```python
"account_inventory": [row.to_summary() for row in inventory_rows],
"account_inventory_summary": {
    "counts": counts,
    "pointer_only_count": pointer_only_count,
    "sensitive_category_count": sensitive_category_count,
    "warning_count": warning_count,
    "post_write_verification": post_write_verification,
}
```

For completed sync and async jobs, store a redacted summary in job metadata and return it from job status endpoints. Do not put secret values in metadata.

- [x] **Step 6: Verify archive after final manifest write**

After the final ZIP write, reopen the archive and verify:

- `manifest.json` exists and parses
- every manifest `file_path` exists in the ZIP
- archive size matches the final file size
- category counts match exported content files for inventory rows that can be counted deterministically

Set `post_write_verification` to true only after these checks pass. Fail the export if verification fails.

- [x] **Step 7: Verify tests pass**

Actual verification:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py -v
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py tldw_Server_API/tests/Chatbooks/test_chatbooks_sync_contracts.py -v
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbook_service.py tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py tldw_Server_API/tests/Chatbooks/test_chatbooks_account_inventory.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py -q
python -m json.tool Docs/Schemas/chatbooks_manifest_v1_1.json
python -m bandit -f json -o /tmp/bandit_chatbooks_full_account_task3.json tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/core/Chatbooks/chatbook_models.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py
```

Expected: pass.

Completed with focused and regression verification: `python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py -v` passed with 16 tests; `python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py tldw_Server_API/tests/Chatbooks/test_chatbooks_sync_contracts.py -v` passed with 29 tests; `python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbook_service.py tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py tldw_Server_API/tests/Chatbooks/test_chatbooks_account_inventory.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py -q` passed with 83 tests; `python -m json.tool Docs/Schemas/chatbooks_manifest_v1_1.json` passed; Bandit on touched backend files reported 0 findings.

- [x] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/app/core/Chatbooks/chatbook_models.py tldw_Server_API/app/core/Chatbooks/jobs_adapter.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py tldw_Server_API/tests/Chatbooks/test_chatbooks_sensitive_export_redaction.py
git commit -m "feat: export chatbooks full account inventory"
```

Completed in the Task 3 commit.

## Task 4: Full Archive Import Restore Coverage

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_account_inventory.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbook_service.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_sync_contracts.py`

- [x] **Step 1: Write failing full-import restore tests**

Create a `.chatbook` fixture containing media records, stored/derived media data, prompts, evaluations, embeddings, conversations, notes, characters, world books, dictionaries, generated documents, and explainer sessions. Import with default options and assert every restorable category is imported or restored. Assert non-restorable rows produce visible warnings and do not count as silent skips.

Also assert:

- `import_media` and `import_embeddings` default to restoring Chatbook archive data
- explicit media/embedding restore requests are not rejected
- import fails clearly if the archive contains an inventory category marked restorable but no handler is registered
- sensitive values are not logged or surfaced in warnings

- [x] **Step 2: Run import tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py -v
```

Expected: fail because media and embedding import paths are rejected or skipped, and prompts/evaluations are not restored.

- [x] **Step 3: Change archive import defaults and remove Chatbook reject paths**

For `source_format=chatbook`, default to restoring archive media records and embeddings when present. Keep OpenWebUI import behavior separate because OpenWebUI import is not a `.chatbook` archive restore.

Make the schema source-aware by using nullable request flags and computing effective defaults after `source_format` is known:

```python
import_media: bool | None = Field(
    None,
    description="Omit to restore media data for Chatbook archives and ignore media restore for OpenWebUI imports.",
)
import_embeddings: bool | None = Field(
    None,
    description="Omit to restore embeddings for Chatbook archives and ignore embedding restore for OpenWebUI imports.",
)
```

Then in the endpoint or service:

```python
effective_import_media = (
    import_request.import_media
    if import_request.import_media is not None
    else source_format_value == "chatbook"
)
effective_import_embeddings = (
    import_request.import_embeddings
    if import_request.import_embeddings is not None
    else source_format_value == "chatbook"
)
```

Remove the generic rejection:

```python
if import_media or import_embeddings:
    return False, ...
```

Replace it with source-specific validation:

```python
if source_format_value in {"openwebui_json", "openwebui_db"} and (import_media or import_embeddings):
    return False, "OpenWebUI imports do not use archive media or embedding restore options.", None
```

- [x] **Step 4: Register restore handlers for every restorable inventory row**

Extend `supported_types` and add import functions:

- `_import_media_items`
- `_import_embeddings`
- `_import_prompts`
- `_import_evaluations`
- `_import_account_data` if `ContentType.ACCOUNT_DATA` is added

Use the same path index and safe extracted path helpers used by existing importers. Restore stored file artifacts only into account-owned storage paths. Preserve original external pointers as metadata; do not resolve arbitrary local paths from the archive.

- [x] **Step 5: Stop silent skips**

Before importing, compare manifest categories and content item types to the inventory. If a category is restorable but no handler exists, fail with a clear error. If a category is pointer-only or non-restorable, add a warning and count it in the result.

Return sync import results with:

```python
{
    "imported_items": imported_items,
    "warnings": warnings,
    "inventory_summary": redacted_inventory_summary,
    "skipped_non_restorable": skipped_non_restorable,
}
```

- [x] **Step 6: Verify tests pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py tldw_Server_API/tests/Chatbooks/test_chatbooks_sensitive_export_redaction.py -v
```

Expected: pass.

Completed with expanded verification:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks -v
# 360 passed, 6 warnings

source .venv/bin/activate && python -m py_compile tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py tldw_Server_API/app/core/Chatbooks/chatbook_models.py tldw_Server_API/app/core/Chatbooks/chatbook_account_inventory.py
# passed

git diff --check
# passed

source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py tldw_Server_API/app/core/Chatbooks/chatbook_models.py tldw_Server_API/app/core/Chatbooks/chatbook_account_inventory.py -f json -o /tmp/bandit_chatbooks_full_account_task4.json
# results: []
```

Review follow-ups addressed before commit: v1.1 media artifact inventory validation now covers bundled media artifact bytes; Chroma collection embedding restore honors `conflict_resolution=skip`; `tags_categories_relationships` is no longer advertised as silently restorable; rename conflicts for prompt/evaluation/media/generated-document imports use a generic title suffix; async import job status persists imported item, inventory, skipped, warning, and metadata results; generated document restore persists metadata.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py tldw_Server_API/tests/Chatbooks/test_chatbooks_sensitive_export_redaction.py
git commit -m "feat: restore full chatbook archive contents"
```

## Task 5: WebUI And Extension Backup All Flow

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/settings.json`
- Modify: `apps/packages/ui/src/public/_locales/en/settings.json`
- Test: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx`
- Test: `apps/extension/tests/e2e/chatbooks-export-download.spec.ts`

- [x] **Step 1: Write failing UI tests**

Add tests that render the Chatbooks page and assert:

- visible heading says `Chatbooks Backup & Import`
- primary Backup all path is visible
- scope summary shows categories, counts, pointer-only count, sensitive-category count, warning count, estimated size when provided
- starting Backup all calls `exportChatbook` without `content_selections`
- selective export still blocks zero-item allowlists
- default archive import does not send unsupported raw-source-file options as normal enabled options
- completed job rows show archive size, warning count, and post-write verification status when present

- [x] **Step 2: Run UI tests and verify failure**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx
```

Expected: fail because the component requires selections and has no Backup all mode.

- [x] **Step 3: Add client method and UI state**

Add:

```ts
async getChatbookExportScope(): Promise<ChatbookAccountScopeResponse> {
  return await bgRequest<ChatbookAccountScopeResponse>({
    path: "/api/v1/chatbooks/export/scope",
    method: "GET"
  })
}
```

In `ChatbooksPlaygroundPage.tsx`, add an export mode state such as:

```ts
const [exportMode, setExportMode] = React.useState<"full_account" | "selective">("full_account")
```

When `exportMode === "full_account"`, load and show the scope summary and send a payload without `content_selections`.

- [x] **Step 4: Fix archive import defaults**

For `.chatbook` archive imports, use defaults that restore all archive data. Hide or disable any legacy raw-source-file option. Keep OpenWebUI import flags omitted or false because OpenWebUI import is a different source format.

- [x] **Step 5: Verify UI and extension tests pass**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx
npx playwright test ../extension/tests/e2e/chatbooks-export-download.spec.ts
```

Expected: pass against the available local test setup. If Playwright requires a running server, record the required server command and run it before the test.

Actual verification:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx
# 2 files passed, 12 tests passed

cd apps/extension
npx playwright test tests/e2e/chatbooks-export-download.spec.ts
# extension production build passed; 1 real-server guarded test skipped

git diff --check
# passed

node -e "JSON.parse(require('fs').readFileSync(process.argv[1], 'utf8'))" apps/packages/ui/src/assets/locale/en/settings.json
node -e "JSON.parse(require('fs').readFileSync(process.argv[1], 'utf8'))" apps/packages/ui/src/public/_locales/en/settings.json
# both passed
```

Completed in this task: primary WebUI Backup all mode, full-account scope summary, omitted `content_selections` for backup-all export, archive import default-restore payload cleanup, completed export job metadata display, and extension E2E update to use Backup all.

- [x] **Step 6: Commit**

```bash
git add apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx apps/packages/ui/src/assets/locale/en/settings.json apps/packages/ui/src/public/_locales/en/settings.json apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx apps/extension/tests/e2e/chatbooks-export-download.spec.ts
git commit -m "feat: add chatbooks backup all flow"
```

## Task 6: Settings Demotion And Naming Cleanup

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Settings/chatbooks.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/option.json`
- Modify: `apps/packages/ui/src/assets/locale/en/settings.json`
- Modify: `apps/packages/ui/src/public/_locales/en/option.json`
- Modify: `apps/packages/ui/src/public/_locales/en/settings.json`
- Modify: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- Modify: `apps/packages/ui/src/routes/route-metadata.ts`
- Modify: `apps/tldw-frontend/e2e/utils/page-objects/ChatbooksPage.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks.spec.ts`
- Test: `apps/packages/ui/src/components/Option/Settings/__tests__/chatbooks.test.tsx`

- [x] **Step 1: Write failing Settings and naming tests**

Assert Settings shows a primary link/button to `Chatbooks Backup & Import`. If the conversation shortcut remains, assert it is labeled `Conversation export shortcut` and requires conversation IDs. Assert page-object and E2E headings no longer look for `Chatbooks Playground`.

- [x] **Step 2: Run tests and verify failure**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Settings/__tests__/chatbooks.test.tsx
```

Expected: fail if the test file is new or because Settings still presents Chatbooks controls as a backup-like surface.

- [x] **Step 3: Demote Settings workflow**

Make Settings default to a full Backup & Import entry point. Keep a secondary conversation-only shortcut only if it remains valuable and clearly scoped. Set `chatbookImportMedia` default to the same valid archive restore behavior or remove the Settings archive import shortcut if it cannot preview/restore safely.

- [x] **Step 4: Rename visible copy**

Replace visible `Chatbooks Playground` labels with `Chatbooks Backup & Import`. Internal route names and component names can stay if renaming them creates unnecessary churn.

- [x] **Step 5: Verify tests pass**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Settings/__tests__/chatbooks.test.tsx ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx
npx playwright test e2e/workflows/tier-2-features/chatbooks.spec.ts
```

Expected: pass with updated headings and Settings behavior.

- [x] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/Settings/chatbooks.tsx apps/packages/ui/src/assets/locale/en/option.json apps/packages/ui/src/assets/locale/en/settings.json apps/packages/ui/src/public/_locales/en/option.json apps/packages/ui/src/public/_locales/en/settings.json apps/packages/ui/src/components/Layouts/header-shortcut-items.ts apps/packages/ui/src/routes/route-metadata.ts apps/tldw-frontend/e2e/utils/page-objects/ChatbooksPage.ts apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks.spec.ts apps/packages/ui/src/components/Option/Settings/__tests__/chatbooks.test.tsx
git commit -m "feat: clarify chatbooks backup import surfaces"
```

Completed implementation before commit:
- Settings Chatbooks now links to `/chatbooks` as the full Backup & Import workflow and keeps only a selected-conversation export shortcut.
- Removed unused Settings import/job locale keys and changed visible Settings export copy to conversation-only language.
- Header shortcut and route metadata now label `/chatbooks` as `Chatbooks Backup & Import`; `/chatbooks-playground` is metadata-only legacy alias to `/chatbooks`.
- Updated the Chatbooks E2E page object/spec to use `/chatbooks`, Backup & Import heading, backup-all export CTA, and current archive dropzone copy.

Verification:
- Initial red test: `bunx vitest run ../packages/ui/src/components/Option/Settings/__tests__/chatbooks.test.tsx` failed on missing Backup & Import link, stale Settings export/import UI, and `Chatbooks Playground` shortcut label.
- Passed: `bunx vitest run ../packages/ui/src/components/Option/Settings/__tests__/chatbooks.test.tsx ../packages/ui/src/components/Layouts/__tests__/header-shortcut-descriptions.test.ts ../packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts ../packages/ui/src/routes/__tests__/route-registry.visibility.test.ts`.
- Passed: `npx playwright test e2e/workflows/tier-2-features/chatbooks.spec.ts`.
- Passed: locale JSON parse check for the four changed English locale files.
- Passed: `git diff --check`.
- Noted baseline: optional `route-governance.metadata-coverage.test.ts` still fails on existing broad metadata gaps unrelated to Task 6.
- Bandit skipped: frontend, locale, route metadata, E2E, and documentation-only change set.

## Task 7: OpenWebUI Import Mapping And Hydration Scope Reuse

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/openwebui_hydration_jobs.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
- Test: `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py`
- Test: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx`

- [x] **Step 1: Write failing backend mapping tests**

After an OpenWebUI JSON or DB import, assert the result persists:

- source format
- source user when available
- source conversation ID
- imported tldw conversation ID
- chat title or display label
- attachment-reference summary when available

Then assert hydration preview can use the last or selected import scope without manual `conversation_ids`.

- [x] **Step 2: Run backend OpenWebUI tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py -v
```

Expected: fail because hydration currently depends on manually supplied imported tldw conversation IDs.

- [x] **Step 3: Persist import mappings**

Reuse existing OpenWebUI metadata when possible. If a dedicated table is needed, create it through the existing Chatbooks DB initialization path and store owner user ID with every row. Expose helpers:

```python
def _record_openwebui_import_mapping(...)
def list_openwebui_import_scopes(...)
def resolve_openwebui_hydration_scope(...)
```

Do not store raw local file paths in user-facing scope summaries.

- [x] **Step 4: Add API for scope listing and hydration selection**

Add models:

```python
class OpenWebUIImportScopeSummary(BaseModel):
    scope_id: str
    source_format: str
    source_user_id: str | None = None
    conversation_count: int
    attachment_reference_count: int = 0
    created_at: datetime | None = None
```

Add `GET /api/v1/chatbooks/openwebui/import-scopes` and extend hydration requests with `import_scope_id` while preserving manual IDs as an advanced override.

- [x] **Step 5: Update UI hydration flow**

After an OpenWebUI import, show the last import scope as the default hydration scope. Show source format, source user, conversation count, and attachment-reference summary. Keep manual conversation ID entry under an advanced control.

- [x] **Step 6: Verify backend and UI tests pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py -v
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx ../packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts
```

Expected: pass.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py tldw_Server_API/app/core/Chatbooks/openwebui_hydration_jobs.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/api/v1/endpoints/chatbooks.py apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts
git commit -m "feat: reuse openwebui import scope for hydration"
```

Completed implementation before commit:
- OpenWebUI JSON and database imports now persist an import scope id and import timestamp in existing per-conversation OpenWebUI metadata; scope summaries are rebuilt from user-owned ChaCha conversation/settings/message metadata without storing or exposing source filesystem paths.
- Added `list_openwebui_import_scopes()` and `resolve_openwebui_hydration_scope()` so preview/run hydration can use `import_scope_id` while preserving manual `conversation_ids` as an override.
- Added `GET /api/v1/chatbooks/openwebui/import-scopes`, `import_scope_id` request support, and response schemas for user-safe scope summaries.
- Updated the WebUI client and Chatbooks Backup & Import OpenWebUI hydration UI to default to the latest import scope after sync import or background import completion, with manual conversation IDs kept behind a manual-scope control.

Verification:
- Initial red backend run failed on missing scope listing/resolution/schema/endpoint support: `python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py -v`.
- Initial red frontend run failed on missing `listOpenWebUIImportScopes`; the UI test also failed until background import completion refreshed scopes.
- Passed: `python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py -v` => 51 passed, 5 warnings.
- Passed: `bunx vitest run ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx ../packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts` => 19 passed.
- Passed: `git diff --check`.
- Passed: `python -m bandit -r tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py -f json -o /tmp/bandit_chatbooks_task7.json` => 0 findings.

## Task 8: Documentation, E2E, Security, And Final Review

**Files:**
- Modify: `Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md`
- Modify: `Docs/API-related/Chatbook_API_Documentation.md`
- Modify: `CHANGELOG.md`
- Modify: `Docs/RELEASE_NOTES.md`
- Modify: `apps/extension/tests/e2e/chatbooks-export-download.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks.spec.ts`
- Modify: `backlog/tasks/task-12098.1 - P0-Chatbooks-backup-restore-correctness-remediation.md`
- Modify: `backlog/tasks/task-12098.2 - P1-Chatbooks-backup-import-UX-clarity-remediation.md`
- Modify: `backlog/tasks/task-12098.3 - P2-Chatbooks-backup-import-acceptance-coverage.md`

- [ ] **Step 1: Update docs and migration notes**

Docs must state:

- omitted or `{}` export selections mean full user-account export
- non-empty selections are explicit allowlists
- zero-item allowlists are invalid
- the inventory defines full-account scope
- full account export includes media records, derived media data, and stored account file artifacts
- pointer-only sources export stored pointers, not unavailable source bytes
- archive import restores all restorable account data present by default
- sensitive values are treated as archive contents only when required, and redacted from preview/log/summary surfaces
- Settings is not the full backup/restore workflow if a shortcut remains

Update `CHANGELOG.md` and `Docs/RELEASE_NOTES.md` with the compatibility note that omitted or `{}` export selections now mean full user-account export.

- [ ] **Step 2: Update E2E tests**

Update WebUI and extension E2E so a backup-all export that does not fire is a failure. Assert the browser extension route inherits the Backup all behavior.

- [ ] **Step 3: Run targeted backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_account_inventory.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py tldw_Server_API/tests/Chatbooks/test_chatbooks_sensitive_export_redaction.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py -v
```

Expected: pass.

- [ ] **Step 4: Run targeted frontend tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx ../packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts
```

Expected: pass.

- [ ] **Step 5: Run E2E where environment is available**

Start the backend and frontend according to repo instructions, then run:

```bash
cd apps/tldw-frontend
npx playwright test e2e/workflows/tier-2-features/chatbooks.spec.ts
npx playwright test ../extension/tests/e2e/chatbooks-export-download.spec.ts
```

Expected: pass. If the local server cannot be started in the environment, record the exact blocker and keep the unit/integration coverage passing.

- [ ] **Step 6: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/core/Chatbooks -f json -o /tmp/bandit_chatbooks_full_account_backup.json
```

Expected: no new findings in touched code. Fix new findings before finalizing.

- [ ] **Step 7: Update Backlog tasks**

Using Backlog.md MCP or CLI if available, update TASK-12098.1, TASK-12098.2, and TASK-12098.3 with implemented files, verification results, known skips, and final summary. If MCP/CLI is unavailable, edit only the existing task files for these tasks and mention the fallback in the commit message.

- [ ] **Step 8: Commit final docs and verification records**

```bash
git add Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md Docs/API-related/Chatbook_API_Documentation.md CHANGELOG.md Docs/RELEASE_NOTES.md apps/extension/tests/e2e/chatbooks-export-download.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks.spec.ts backlog/tasks/task-12098.1\\ -\\ P0-Chatbooks-backup-restore-correctness-remediation.md backlog/tasks/task-12098.2\\ -\\ P1-Chatbooks-backup-import-UX-clarity-remediation.md backlog/tasks/task-12098.3\\ -\\ P2-Chatbooks-backup-import-acceptance-coverage.md
git commit -m "docs: document chatbooks full account backup restore"
```

## Final Verification Checklist

- [ ] Backend account inventory tests pass.
- [ ] Backend full export contract tests pass.
- [ ] Backend full import restore tests pass.
- [ ] Backend sensitive-data redaction tests pass.
- [ ] OpenWebUI mapping and hydration tests pass.
- [ ] WebUI Backup all tests pass.
- [ ] Settings clarity tests pass.
- [ ] Extension `/chatbooks` Backup all E2E passes or environment blocker is documented.
- [ ] WebUI Chatbooks E2E passes or environment blocker is documented.
- [ ] Bandit touched-scope scan reports no new findings.
- [ ] API docs, user guide, and release notes match runtime behavior.
- [ ] Backlog task notes include implementation summary, verification, and known skips.

## Review Guardrails

- Full account export is not complete unless every inventory row is exported/restored or explicitly non-restorable with a visible warning.
- Do not log, preview, summarize, or include secret values in manifest summaries or job metadata.
- Do not silently skip restorable archive categories during import.
- Do not use client-side fetching of every ID as the Backup all implementation.
- Do not label pointer-only sources as exported file contents.
- Do not leave Settings looking like the full backup/restore workflow unless it delegates to the real Backup & Import surface.
