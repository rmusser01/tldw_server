# Wave 6 ChaCha Lifecycle-First Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the ChaChaNotes runtime, character, conversation, and message lifecycle seams into focused internal modules while preserving the `CharactersRAGDB` facade and current caller-visible behavior.

**Architecture:** Keep `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` as the public facade and orchestration owner while moving the lifecycle implementation into a new `tldw_Server_API/app/core/DB_Management/chacha/` package. Move init/cache/shutdown logic into a transport-neutral runtime manager, then peel character, conversation, and message logic behind thin facade wrappers, with `ChaCha_Notes_DB_Deps.py` remaining the only HTTP translation layer.

**Tech Stack:** Python, FastAPI, asyncio, SQLite/PostgreSQL backends, pytest, loguru, Bandit

---

## File Map

### New Core Runtime Package

- Create: `tldw_Server_API/app/core/DB_Management/chacha/__init__.py`
  - Export the runtime manager and extracted store classes without changing public callers.
- Create: `tldw_Server_API/app/core/DB_Management/chacha/runtime.py`
  - Encapsulate ChaCha runtime state, cache/init coordination, shutdown handling, health snapshots, and test reset hooks behind an explicit interface.
- Create: `tldw_Server_API/app/core/DB_Management/chacha/character_store.py`
  - Hold character-card CRUD, versioning, restore, and default-character support code extracted from `ChaChaNotes_DB.py`.
- Create: `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py`
  - Hold conversation CRUD, scope/search, settings, and restore paths extracted from `ChaChaNotes_DB.py`.
- Create: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`
  - Hold message CRUD, images, metadata, and citation-related paths extracted from `ChaChaNotes_DB.py`.
- Create only if duplication is real after Tasks 2-4: `tldw_Server_API/app/core/DB_Management/chacha/shared.py`
  - Pure helper functions shared by at least two extracted stores. Skip this file if duplication never becomes concrete.

### Existing Facade And HTTP Boundary

- Modify: `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`
  - Replace module-global runtime orchestration with the new runtime manager while keeping all HTTP mapping in the dependency layer.
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Preserve the facade, transaction ownership, and cross-store orchestration while delegating character/conversation/message method bodies to extracted stores.
- Modify only if import ordering or lifecycle hooks require it: `tldw_Server_API/app/core/DB_Management/README.md`
  - Document the new internal `chacha/` package boundary for maintainers.

### Focused Test Surface

- Create: `tldw_Server_API/tests/ChaChaNotesDB/conftest.py`
  - Shared ChaChaNotes DB fixtures for the new extracted-store test files so setup is not duplicated across files.
- Create: `tldw_Server_API/tests/Chat/test_chacha_runtime_contract.py`
  - New explicit tests for runtime-manager surface, reset semantics, and transport-neutral failures.
- Modify: `tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_sqlite_policy.py`
  - Keep dependency-layer behavior covered, especially HTTP `503` translation and shutdown/reset behavior.
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py`
  - Focused characterization tests for extracted character-store behavior.
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py`
  - Focused characterization tests for extracted conversation/settings behavior.
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py`
  - Focused characterization tests for extracted message behavior.
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py`
  - Keep the public `CharactersRAGDB` behavior green after extraction.
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_conversation_scope_db.py`
  - Guard conversation scope, cascade, and deleted-parent behavior during conversation/message extraction.
- Modify: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`
  - Keep workspace-scoped conversation behavior stable through conversation-store extraction.
- Modify: `tldw_Server_API/tests/Characters/test_character_chat_greetings_api.py`
  - Add a narrow orchestration regression proving seeded greeting creation still persists the first assistant message and `greetingsChecksum`.

## Constraints

- Do not change imports for external callers of `CharactersRAGDB`.
- Do not let runtime or store modules raise `HTTPException`.
- Do not move workflow orchestration out of `CharactersRAGDB` in this wave.
- Do not pull notes, persona, prompt preset, flashcard, moodboard, or note-studio logic into the extracted stores.
- Stop and split follow-on work if `shared.py` starts becoming a dumping ground or if verification grows beyond the focused lifecycle slice.

### Task 1: Extract A Transport-Neutral ChaCha Runtime Manager

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/__init__.py`
- Create: `tldw_Server_API/app/core/DB_Management/chacha/runtime.py`
- Create: `tldw_Server_API/tests/Chat/test_chacha_runtime_contract.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`
- Modify: `tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_sqlite_policy.py`

- [ ] **Step 1: Write the failing runtime-contract and HTTP-translation tests**

```python
import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.core.DB_Management.chacha.runtime import (
    ChaChaRuntimeManager,
    ChaChaRuntimeUnavailableError,
)


def test_runtime_manager_exposes_explicit_resettable_surface():
    runtime = ChaChaRuntimeManager()

    assert hasattr(runtime, "get_or_create")
    assert hasattr(runtime, "shutdown")
    assert hasattr(runtime, "snapshot")
    runtime.reset_for_tests()


@pytest.mark.asyncio
async def test_dependency_maps_runtime_unavailable_to_503(monkeypatch):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    class _Runtime:
        async def get_or_create(self, *_args, **_kwargs):
            raise ChaChaRuntimeUnavailableError("ChaChaNotes shutdown in progress")

    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", _Runtime())

    with pytest.raises(HTTPException) as exc:
        await deps.get_chacha_db_for_user_id(1, "1")

    assert exc.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "shutdown" in exc.value.detail.lower()
```

- [ ] **Step 2: Run the new runtime and dependency tests to confirm the missing boundary**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/test_chacha_runtime_contract.py tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_sqlite_policy.py -k "runtime_manager_exposes_explicit_resettable_surface or dependency_maps_runtime_unavailable_to_503" -v`

Expected: FAIL because `tldw_Server_API.app.core.DB_Management.chacha.runtime` and the `_CHACHA_RUNTIME` contract do not exist yet.

- [ ] **Step 3: Implement the explicit runtime manager and wire the dependency layer to it**

```python
class ChaChaRuntimeUnavailableError(RuntimeError):
    """Transport-neutral runtime failure for callers that translate to HTTP elsewhere."""


class ChaChaRuntimeManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._instances = LRUCache(maxsize=MAX_CACHED_CHACHA_DB_INSTANCES)
        self._init_events: dict[str, threading.Event] = {}
        self._init_errors: dict[str, Exception] = {}
        ...

    async def get_or_create(self, user_id: int, client_id: str | None) -> CharactersRAGDB:
        ...

    def snapshot(self) -> dict[str, Any]:
        ...

    async def shutdown(self, wait_timeout: float = 5.0) -> None:
        ...

    def reset_for_tests(self) -> None:
        ...


_CHACHA_RUNTIME = ChaChaRuntimeManager()


async def get_chacha_db_for_user_id(user_id: int, client_id: str | None = None) -> CharactersRAGDB:
    try:
        return await _CHACHA_RUNTIME.get_or_create(user_id, client_id)
    except ChaChaRuntimeUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
```

Implementation notes:
- Move the existing cache/init/shutdown/default-character orchestration from `ChaCha_Notes_DB_Deps.py` into `runtime.py` without changing caller-visible behavior.
- Keep `HTTPException` creation in the dependency layer only.
- Preserve the current recovery semantics around stale waiters, shutdown abort sentinels, and executor recreation.

- [ ] **Step 4: Re-run the focused runtime/dependency verification**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/test_chacha_runtime_contract.py tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_sqlite_policy.py -v`

Expected: PASS, proving the runtime has an explicit resettable interface and the dependency layer still owns HTTP `503` translation.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/chacha/__init__.py tldw_Server_API/app/core/DB_Management/chacha/runtime.py tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py tldw_Server_API/tests/Chat/test_chacha_runtime_contract.py tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_sqlite_policy.py
git commit -m "refactor: extract chacha runtime manager"
```

### Task 2: Extract Character Lifecycle Into A Focused Store

**Files:**
- Create: `tldw_Server_API/tests/ChaChaNotesDB/conftest.py`
- Create: `tldw_Server_API/app/core/DB_Management/chacha/character_store.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py`

- [ ] **Step 1: Write the shared ChaChaNotes fixtures and the failing character-store characterization test**

```python
# tldw_Server_API/tests/ChaChaNotesDB/conftest.py
@pytest.fixture
def client_id():
    return "test_client_001"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_db.sqlite"


@pytest.fixture
def db_instance(db_path, client_id):
    db = CharactersRAGDB(db_path, client_id)
    yield db
    db.close_connection()


@pytest.fixture
def character_id(db_instance):
    return db_instance.add_character_card({"name": "Seed Char", "description": "fixture"})


# tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py
from tldw_Server_API.app.core.DB_Management.chacha.character_store import CharacterStore


def test_character_store_add_update_restore_roundtrip(db_instance):
    store = CharacterStore(db_instance)

    char_id = store.add_character_card({
        "name": "Wave6 Char",
        "description": "store extraction guard",
        "personality": "steady",
    })
    assert char_id is not None

    updated = store.update_character_card(
        char_id,
        {"description": "updated"},
        expected_version=1,
    )
    assert updated is True

    assert store.soft_delete_character_card(char_id, expected_version=2) is True
    assert store.restore_character_card(char_id, expected_version=3) is True
```

- [ ] **Step 2: Run the new store test plus the existing character facade tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py -k "character_store_add_update_restore_roundtrip or TestCharacterCards or restore_character_card_conflict_then_success" -v`

Expected: FAIL because `CharacterStore` does not exist yet.

- [ ] **Step 3: Move character logic into `CharacterStore` and leave thin facade wrappers**

```python
class CharacterStore:
    def __init__(self, db: "CharactersRAGDB") -> None:
        self.db = db

    def add_character_card(self, card_data: dict[str, Any]) -> int | None:
        ...

    def update_character_card(self, character_id: int, card_data: dict[str, Any], expected_version: int) -> bool | None:
        ...

    def soft_delete_character_card(self, character_id: int, expected_version: int) -> bool | None:
        ...

    def restore_character_card(self, character_id: int, expected_version: int) -> bool | None:
        ...


class CharactersRAGDB:
    def __init__(...):
        ...
        self._character_store = CharacterStore(self)

    def add_character_card(self, card_data: dict[str, Any]) -> int | None:
        return self._character_store.add_character_card(card_data)
```

Implementation notes:
- Move method bodies, not behavior.
- Keep default-character support paths working for the runtime manager and existing chat helpers.
- Do not extract non-lifecycle character-adjacent domains such as exemplars in this task.

- [ ] **Step 4: Re-run the focused character verification**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py -k "CharacterCards or restore_character_card_conflict_then_success" -v`

Expected: PASS, with no behavior change at the facade.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/tests/ChaChaNotesDB/conftest.py tldw_Server_API/app/core/DB_Management/chacha/character_store.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py
git commit -m "refactor: extract chacha character store"
```

### Task 3: Extract Conversation Lifecycle And Settings While Preserving Facade Ownership

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_conversation_scope_db.py`
- Modify: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`

- [ ] **Step 1: Write the failing conversation-store and settings tests**

```python
from tldw_Server_API.app.core.DB_Management.chacha.conversation_store import ConversationStore


def test_conversation_store_roundtrip_preserves_scope_and_settings(db_instance, character_id):
    store = ConversationStore(db_instance)

    conv_id = store.add_conversation({
        "character_id": character_id,
        "title": "Scoped chat",
        "scope_type": "workspace",
        "workspace_id": "ws-1",
    })

    assert conv_id is not None
    assert store.upsert_conversation_settings(conv_id, {"greetingsChecksum": "abc123"}) is True

    conversation = store.get_conversation_by_id(conv_id)
    settings = store.get_conversation_settings(conv_id)

    assert conversation["scope_type"] == "workspace"
    assert conversation["workspace_id"] == "ws-1"
    assert settings["greetingsChecksum"] == "abc123"
```

- [ ] **Step 2: Run the conversation/scope tests before the extraction**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py tldw_Server_API/tests/ChaChaNotesDB/test_conversation_scope_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -k "conversation_store_roundtrip_preserves_scope_and_settings or ScopedChatSessions or TestConversationScope" -v`

Expected: FAIL because `ConversationStore` does not exist yet.

- [ ] **Step 3: Implement `ConversationStore` and delegate conversation/settings methods from the facade**

```python
class ConversationStore:
    def __init__(self, db: "CharactersRAGDB") -> None:
        self.db = db

    def add_conversation(self, conv_data: dict[str, Any]) -> str | None:
        ...

    def get_conversation_by_id(self, conversation_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
        ...

    def update_conversation(self, conversation_id: str, update_data: dict[str, Any], expected_version: int) -> bool | None:
        ...

    def upsert_conversation_settings(self, conversation_id: str, settings: dict[str, Any]) -> bool:
        ...

    def get_conversation_settings(self, conversation_id: str) -> dict[str, Any] | None:
        ...
```

Implementation notes:
- Keep scope helpers and workspace behavior intact.
- Keep `CharactersRAGDB` as the transaction/orchestration owner for any flow that spans conversation plus message behavior.
- Do not let `ConversationStore` call `MessageStore` directly.

- [ ] **Step 4: Re-run the focused conversation/workspace verification**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py tldw_Server_API/tests/ChaChaNotesDB/test_conversation_scope_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -v`

Expected: PASS, proving scope handling and settings behavior stayed stable.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py tldw_Server_API/tests/ChaChaNotesDB/test_conversation_scope_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py
git commit -m "refactor: extract chacha conversation store"
```

### Task 4: Extract Message Lifecycle And Keep Cross-Store Workflow Orchestration At The Facade

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py`
- Modify: `tldw_Server_API/tests/Characters/test_character_chat_greetings_api.py`

- [ ] **Step 1: Write the failing message-store and seeded-greeting orchestration tests**

```python
from tldw_Server_API.app.core.DB_Management.chacha.message_store import MessageStore


def test_message_store_add_and_fetch_roundtrip(db_instance, character_id):
    conversation_id = db_instance.add_conversation({"character_id": character_id, "title": "msg store"})
    store = MessageStore(db_instance)

    message_id = store.add_message({
        "conversation_id": conversation_id,
        "sender": "assistant",
        "content": "hello from store",
    })

    assert message_id is not None
    assert store.get_message_by_id(message_id)["content"] == "hello from store"


def test_seeded_chat_creation_persists_greeting_checksum(authenticated_client, mock_chacha_db, setup_dependencies):
    ...
    assert detail["settings"]["greetingsChecksum"]
```

- [ ] **Step 2: Run the new message and greetings regression tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py tldw_Server_API/tests/Characters/test_character_chat_greetings_api.py -k "message_store_add_and_fetch_roundtrip or greeting_checksum" -v`

Expected: FAIL because `MessageStore` and the new seeded-greeting checksum assertion path do not exist yet.

- [ ] **Step 3: Implement `MessageStore`, delegate message methods, and keep orchestration in `CharactersRAGDB`/endpoint flows**

```python
class MessageStore:
    def __init__(self, db: "CharactersRAGDB") -> None:
        self.db = db

    def add_message(self, msg_data: dict[str, Any]) -> str | None:
        ...

    def get_message_by_id(self, message_id: str) -> dict[str, Any] | None:
        ...

    def get_messages_for_conversation(...):
        ...

    def add_message_metadata(...):
        ...
```

Implementation notes:
- Extract message CRUD/images/metadata/citation reads.
- Keep cross-store flows in the facade or existing endpoint orchestration rather than letting `MessageStore` coordinate with `ConversationStore`.
- Reuse the existing transaction context from `CharactersRAGDB` for multi-step flows when needed.

- [ ] **Step 4: Re-run the focused message/orchestration verification**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py tldw_Server_API/tests/Characters/test_character_chat_greetings_api.py -k "TestConversationsAndMessages or MessageMetadata" -v`

Expected: PASS, proving seeded-message flows and public message behavior survived the extraction.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/chacha/message_store.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py tldw_Server_API/tests/Characters/test_character_chat_greetings_api.py
git commit -m "refactor: extract chacha message store"
```

### Task 5: Finalize Maintainer Clarity And Run Focused Verification

**Files:**
- Modify only if it improves clarity after extraction: `tldw_Server_API/app/core/DB_Management/README.md`
- Create only if Tasks 2-4 reveal true duplication: `tldw_Server_API/app/core/DB_Management/chacha/shared.py`

- [ ] **Step 1: Write the failing documentation or duplication guard**

```markdown
## ChaCha Internal Layout

- `ChaChaNotes_DB.py` remains the public facade and orchestration owner.
- `chacha/runtime.py` owns cache/init/shutdown behavior and exposes explicit resettable hooks for tests.
- `chacha/character_store.py`, `conversation_store.py`, and `message_store.py` hold extracted lifecycle implementations.
```

If two or more stores now share the same pure helper, write the failing import/use adjustment for `shared.py` in the touched store tests instead of updating the README first.

- [ ] **Step 2: Apply the smallest clarity-only follow-up**

```python
# shared.py only if duplication is concrete
def normalize_row(record: Any) -> dict[str, Any]:
    ...
```

or

```markdown
- `tldw_Server_API/app/core/DB_Management/chacha/`
  - Internal lifecycle package for ChaCha runtime + character/conversation/message stores.
```

- [ ] **Step 3: Run the focused Wave 6 verification set**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/test_chacha_runtime_contract.py tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_sqlite_policy.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py tldw_Server_API/tests/ChaChaNotesDB/test_conversation_scope_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py tldw_Server_API/tests/Characters/test_character_chat_greetings_api.py -v`

Expected: PASS.

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/chacha tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_wave6_chacha.json`

Expected: completes successfully with no new findings in touched code.

- [ ] **Step 4: Run one public-facade integration guard**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_role_normalization_and_search.py -k "chat_settings_roundtrip_persists_author_note_and_position or chat_settings_update_increments_conversation_version_once" -v`

Expected: PASS, confirming the preserved facade still behaves correctly through the character-chat endpoint surface.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/README.md tldw_Server_API/app/core/DB_Management/chacha/shared.py
git commit -m "docs: document chacha lifecycle decomposition"
```

Commit note:
- Inspect `/tmp/bandit_wave6_chacha.json` for new findings, but do not commit it.
- If `README.md` and `shared.py` are both unchanged because neither was needed, commit only the touched repo files from the final cleanup task. Do not create empty artifacts to satisfy the plan mechanically.
