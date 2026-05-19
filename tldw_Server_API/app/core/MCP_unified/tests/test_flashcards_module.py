import base64
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.implementations.flashcards_module import FlashcardsModule
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError


class FakeFlashcardsDB:
    def __init__(self) -> None:
        self._deck_id = 0
        self.decks: Dict[int, Dict[str, Any]] = {}
        self.cards: Dict[str, Dict[str, Any]] = {}
        self.list_flashcards_calls: List[Dict[str, Any]] = []
        self.count_flashcards_calls: List[Dict[str, Any]] = []
        self.export_flashcards_calls: List[Dict[str, Any]] = []

    def add_deck(self, name: str, description=None, workspace_id=None):
        self._deck_id += 1
        deck_id = self._deck_id
        self.decks[deck_id] = {
            "id": deck_id,
            "name": name,
            "description": description,
            "workspace_id": workspace_id,
            "deleted": 0,
        }
        return deck_id

    def list_decks(self, limit=100, offset=0, include_deleted=False, workspace_id=None, include_workspace_items=False):
        decks = list(self.decks.values())
        if workspace_id is not None:
            decks = [deck for deck in decks if deck.get("workspace_id") == workspace_id]
        elif not include_workspace_items:
            decks = [deck for deck in decks if deck.get("workspace_id") is None]
        return decks[offset: offset + limit]

    def get_deck(self, deck_id: int):
        return self.decks.get(deck_id)

    def _filtered_flashcards(
        self,
        deck_id=None,
        workspace_id=None,
        include_workspace_items=False,
        include_deleted=False,
    ):
        clause, params, _ = CharactersRAGDB._flashcard_visibility_filter(  # type: ignore[misc]
            self,
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
        )
        items = list(self.cards.values())
        if not include_deleted:
            items = [c for c in items if not c.get("deleted")]

        if clause == "":
            return items
        if clause == "f.deck_id = ?":
            return [c for c in items if c.get("deck_id") == params[0]]
        if clause in {"f.deck_id = ? AND d.workspace_id = ?", "d.workspace_id = ? AND f.deck_id = ?"}:
            return [
                c
                for c in items
                if c.get("deck_id") == deck_id and self.decks.get(c.get("deck_id"), {}).get("workspace_id") == workspace_id
            ]
        if clause == "d.workspace_id = ?":
            return [c for c in items if self.decks.get(c.get("deck_id"), {}).get("workspace_id") == params[0]]
        if clause == "d.workspace_id IS NULL":
            return [c for c in items if self.decks.get(c.get("deck_id"), {}).get("workspace_id") is None]
        return items

    def list_flashcards(
        self,
        deck_id=None,
        workspace_id=None,
        include_workspace_items=False,
        tag=None,
        due_status="all",
        q=None,
        include_deleted=False,
        limit=100,
        offset=0,
        order_by="due_at",
    ):
        self.list_flashcards_calls.append(
            {
                "deck_id": deck_id,
                "workspace_id": workspace_id,
                "include_workspace_items": include_workspace_items,
                "tag": tag,
                "due_status": due_status,
                "q": q,
                "include_deleted": include_deleted,
                "limit": limit,
                "offset": offset,
                "order_by": order_by,
            }
        )
        items = self._filtered_flashcards(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
            include_deleted=include_deleted,
        )
        return items[offset: offset + limit]

    def count_flashcards(
        self,
        deck_id=None,
        workspace_id=None,
        include_workspace_items=False,
        tag=None,
        due_status="all",
        q=None,
        include_deleted=False,
    ):
        self.count_flashcards_calls.append(
            {
                "deck_id": deck_id,
                "workspace_id": workspace_id,
                "include_workspace_items": include_workspace_items,
                "tag": tag,
                "due_status": due_status,
                "q": q,
                "include_deleted": include_deleted,
            }
        )
        return len(
            self._filtered_flashcards(
                deck_id=deck_id,
                workspace_id=workspace_id,
                include_workspace_items=include_workspace_items,
                include_deleted=include_deleted,
            )
        )

    def get_flashcard(self, card_uuid: str):
        return self.cards.get(card_uuid)

    def add_flashcard(self, card_data: Dict[str, Any]):
        uuid = f"card-{len(self.cards) + 1}"
        card = dict(card_data)
        card.update({"uuid": uuid, "version": 1, "deleted": 0})
        self.cards[uuid] = card
        return uuid

    def add_flashcards_bulk(self, cards: List[Dict[str, Any]]):
        uuids = []
        for card in cards:
            uuids.append(self.add_flashcard(card))
        return uuids

    def update_flashcard(self, card_uuid: str, updates: Dict[str, Any], expected_version=None, tags=None):
        card = self.cards.get(card_uuid)
        if not card:
            return False
        if expected_version is not None and card["version"] != expected_version:
            raise ConflictError("Version mismatch", entity="flashcards", identifier=card_uuid)
        card.update(updates)
        card["version"] += 1
        return True

    def soft_delete_flashcard(self, card_uuid: str, expected_version: int):
        card = self.cards.get(card_uuid)
        if not card:
            raise ConflictError("Not found", entity="flashcards", identifier=card_uuid)
        if card["version"] != expected_version:
            raise ConflictError("Version mismatch", entity="flashcards", identifier=card_uuid)
        card["deleted"] = 1
        card["version"] += 1
        return True

    def review_flashcard(self, card_uuid: str, rating: int, answer_time_ms=None):
        return {"card_uuid": card_uuid, "rating": rating, "answer_time_ms": answer_time_ms}

    def set_flashcard_tags(self, card_uuid: str, tags: List[str]):
        card = self.cards.get(card_uuid)
        if not card:
            return False
        card["tags"] = tags
        return True

    def get_keywords_for_flashcard(self, card_uuid: str):
        card = self.cards.get(card_uuid)
        if not card:
            return []
        return [{"keyword": t} for t in card.get("tags", [])]

    def export_flashcards_csv(
        self,
        deck_id=None,
        workspace_id=None,
        include_workspace_items=False,
        tag=None,
        q=None,
        delimiter=",",
        include_header=True,
        extended_header=False,
    ):
        self.export_flashcards_calls.append(
            {
                "deck_id": deck_id,
                "workspace_id": workspace_id,
                "include_workspace_items": include_workspace_items,
                "tag": tag,
                "q": q,
                "delimiter": delimiter,
                "include_header": include_header,
                "extended_header": extended_header,
            }
        )
        rows = self._filtered_flashcards(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
            include_deleted=False,
        )
        lines = []
        if include_header:
            lines.append("front,back")
        for row in rows:
            lines.append(f"{row.get('front','')},{row.get('back','')}")
        return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")

    def close_all_connections(self) -> None:
        return None


@pytest.mark.asyncio
async def test_flashcards_crud_and_export(monkeypatch, tmp_path):
    mod = FlashcardsModule(ModuleConfig(name="flashcards"))
    fake_db = FakeFlashcardsDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]

    ctx = SimpleNamespace(db_paths={"chacha": str(tmp_path / "chacha.db")})

    created_deck = await mod.execute_tool("flashcards.decks.create", {"name": "Deck"}, context=ctx)
    deck_id = created_deck["deck_id"]
    assert created_deck["success"] is True  # nosec B101

    card = await mod.execute_tool(
        "flashcards.create",
        {"deck_id": deck_id, "front": "Q", "back": "A", "model_type": "basic"},
        context=ctx,
    )
    card_uuid = card["card_uuid"]

    updated = await mod.execute_tool(
        "flashcards.update",
        {"card_uuid": card_uuid, "updates": {"front": "Q2"}},
        context=ctx,
    )
    assert updated["success"] is True  # nosec B101

    reviewed = await mod.execute_tool("flashcards.review", {"card_uuid": card_uuid, "rating": 3}, context=ctx)
    assert reviewed["review_result"]["rating"] == 3  # nosec B101

    tags = await mod.execute_tool("flashcards.tags.set", {"card_uuid": card_uuid, "tags": ["Tag1"]}, context=ctx)
    assert tags["tags"] == ["tag1"]  # nosec B101

    listed = await mod.execute_tool("flashcards.list", {}, context=ctx)
    assert listed["total"] == 1  # nosec B101
    assert fake_db.list_flashcards_calls[-1]["include_workspace_items"] is True  # nosec B101
    assert fake_db.count_flashcards_calls[-1]["include_workspace_items"] is True  # nosec B101

    csv_export = await mod.execute_tool("flashcards.export", {"format": "csv"}, context=ctx)
    assert csv_export["mime_type"].startswith("text/")  # nosec B101
    csv_bytes = base64.b64decode(csv_export["content_base64"])
    assert csv_bytes.startswith(b"front")  # nosec B101
    assert fake_db.export_flashcards_calls[-1]["include_workspace_items"] is True  # nosec B101

    # Stub APKG exporter
    from tldw_Server_API.app.core.Flashcards import apkg_exporter
    monkeypatch.setattr(apkg_exporter, "export_apkg_from_rows", lambda rows, include_reverse=False: b"apkg")

    apkg_export = await mod.execute_tool("flashcards.export", {"format": "apkg"}, context=ctx)
    apkg_bytes = base64.b64decode(apkg_export["content_base64"])
    assert apkg_bytes == b"apkg"  # nosec B101
    assert fake_db.list_flashcards_calls[-1]["include_workspace_items"] is True  # nosec B101


@pytest.mark.asyncio
async def test_flashcards_workspace_filters_list_count_and_export_with_foreign_deck(
    monkeypatch, tmp_path
):
    mod = FlashcardsModule(ModuleConfig(name="flashcards"))
    fake_db = FakeFlashcardsDB()
    foreign_deck_id = fake_db.add_deck("Foreign Deck", workspace_id="ws-2")
    fake_db.cards["card-foreign"] = {
        "uuid": "card-foreign",
        "deck_id": foreign_deck_id,
        "workspace_id": "ws-2",
        "front": "Foreign Front",
        "back": "Foreign Back",
        "version": 1,
        "deleted": 0,
    }
    monkeypatch.setattr(mod, "_open_db", lambda ctx: fake_db)

    ctx = SimpleNamespace(
        db_paths={"chacha": str(tmp_path / "chacha.db")},
        metadata={"workspace_id": "ws-1"},
        client_id="cli",
    )

    listed = await mod.execute_tool("flashcards.list", {"deck_id": foreign_deck_id}, context=ctx)
    csv_export = await mod.execute_tool(
        "flashcards.export",
        {"format": "csv", "deck_id": foreign_deck_id},
        context=ctx,
    )

    assert listed["total"] == 0  # nosec B101
    assert listed["cards"] == []  # nosec B101
    assert fake_db.list_flashcards_calls[-1]["workspace_id"] == "ws-1"  # nosec B101
    assert fake_db.list_flashcards_calls[-1]["deck_id"] == foreign_deck_id  # nosec B101
    assert fake_db.count_flashcards_calls[-1]["workspace_id"] == "ws-1"  # nosec B101
    assert fake_db.count_flashcards_calls[-1]["deck_id"] == foreign_deck_id  # nosec B101
    assert fake_db.export_flashcards_calls[-1]["workspace_id"] == "ws-1"  # nosec B101
    assert fake_db.export_flashcards_calls[-1]["deck_id"] == foreign_deck_id  # nosec B101
    csv_bytes = base64.b64decode(csv_export["content_base64"])
    assert b"Foreign Front" not in csv_bytes  # nosec B101
    assert b"Foreign Back" not in csv_bytes  # nosec B101


@pytest.mark.asyncio
async def test_flashcards_export_rejects_cross_workspace_card_in_apkg_path(monkeypatch, tmp_path):
    mod = FlashcardsModule(ModuleConfig(name="flashcards"))
    fake_db = FakeFlashcardsDB()
    foreign_deck_id = fake_db.add_deck("Foreign Deck", workspace_id="ws-2")
    fake_db.cards["card-foreign"] = {
        "uuid": "card-foreign",
        "deck_id": foreign_deck_id,
        "workspace_id": "ws-2",
        "front": "Foreign Front",
        "back": "Foreign Back",
        "version": 1,
        "deleted": 0,
    }
    monkeypatch.setattr(mod, "_open_db", lambda ctx: fake_db)

    captured: Dict[str, Any] = {}

    from tldw_Server_API.app.core.Flashcards import apkg_exporter

    def fake_export_apkg_from_rows(rows, include_reverse=False):
        captured["rows"] = list(rows)
        captured["include_reverse"] = include_reverse
        return b"apkg"

    monkeypatch.setattr(apkg_exporter, "export_apkg_from_rows", fake_export_apkg_from_rows)

    ctx = SimpleNamespace(
        db_paths={"chacha": str(tmp_path / "chacha.db")},
        metadata={"workspace_id": "ws-1"},
        client_id="cli",
    )

    apkg_export = await mod.execute_tool(
        "flashcards.export",
        {"format": "apkg", "deck_id": foreign_deck_id},
        context=ctx,
    )

    assert captured["rows"] == []  # nosec B101
    assert captured["include_reverse"] is False  # nosec B101
    assert fake_db.list_flashcards_calls[-1]["workspace_id"] == "ws-1"  # nosec B101
    assert fake_db.list_flashcards_calls[-1]["deck_id"] == foreign_deck_id  # nosec B101
    apkg_bytes = base64.b64decode(apkg_export["content_base64"])
    assert apkg_bytes == b"apkg"  # nosec B101


@pytest.mark.asyncio
async def test_flashcards_get_rejects_cross_workspace_card(monkeypatch, tmp_path):
    mod = FlashcardsModule(ModuleConfig(name="flashcards"))
    fake_db = FakeFlashcardsDB()
    fake_db.cards["card-cross"] = {
        "uuid": "card-cross",
        "deck_id": 2,
        "workspace_id": "ws-2",
        "front": "Q",
        "back": "A",
        "version": 1,
    }
    monkeypatch.setattr(mod, "_open_db", lambda ctx: fake_db)

    ctx = SimpleNamespace(
        db_paths={"chacha": str(tmp_path / "chacha.db")},
        metadata={"workspace_id": "ws-1"},
        client_id="cli",
    )

    with pytest.raises(PermissionError, match="Flashcard access denied for workspace"):
        await mod.execute_tool("flashcards.get", {"card_uuid": "card-cross"}, context=ctx)


@pytest.mark.asyncio
async def test_flashcards_update_rejects_cross_workspace_card(monkeypatch, tmp_path):
    mod = FlashcardsModule(ModuleConfig(name="flashcards"))
    fake_db = FakeFlashcardsDB()
    fake_db.cards["card-cross"] = {
        "uuid": "card-cross",
        "deck_id": 2,
        "workspace_id": "ws-2",
        "front": "Q",
        "back": "A",
        "version": 1,
    }
    monkeypatch.setattr(mod, "_open_db", lambda ctx: fake_db)

    ctx = SimpleNamespace(
        db_paths={"chacha": str(tmp_path / "chacha.db")},
        metadata={"workspace_id": "ws-1"},
        client_id="cli",
    )

    with pytest.raises(PermissionError, match="Flashcard access denied for workspace"):
        await mod.execute_tool(
            "flashcards.update",
            {"card_uuid": "card-cross", "updates": {"front": "Q2"}},
            context=ctx,
        )


@pytest.mark.asyncio
async def test_flashcards_update_rejects_move_to_foreign_deck(monkeypatch, tmp_path):
    mod = FlashcardsModule(ModuleConfig(name="flashcards"))
    fake_db = FakeFlashcardsDB()
    own_deck_id = fake_db.add_deck("Own Deck", workspace_id="ws-1")
    foreign_deck_id = fake_db.add_deck("Foreign Deck", workspace_id="ws-2")
    card_uuid = fake_db.add_flashcard(
        {
            "deck_id": own_deck_id,
            "workspace_id": "ws-1",
            "front": "Q",
            "back": "A",
        }
    )
    monkeypatch.setattr(mod, "_open_db", lambda ctx: fake_db)

    ctx = SimpleNamespace(
        db_paths={"chacha": str(tmp_path / "chacha.db")},
        metadata={"workspace_id": "ws-1"},
        client_id="cli",
    )

    with pytest.raises(PermissionError, match="Flashcard access denied for workspace"):
        await mod.execute_tool(
            "flashcards.update",
            {"card_uuid": card_uuid, "updates": {"deck_id": foreign_deck_id}},
            context=ctx,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("flashcards.delete", {"card_uuid": "card-cross", "expected_version": 1}),
        ("flashcards.review", {"card_uuid": "card-cross", "rating": 3}),
        ("flashcards.tags.set", {"card_uuid": "card-cross", "tags": ["Tag1"]}),
        ("flashcards.tags.get", {"card_uuid": "card-cross"}),
    ],
)
async def test_flashcards_direct_access_denied_across_workspace(monkeypatch, tmp_path, tool_name, arguments):
    mod = FlashcardsModule(ModuleConfig(name="flashcards"))
    fake_db = FakeFlashcardsDB()
    fake_db.cards["card-cross"] = {
        "uuid": "card-cross",
        "deck_id": 2,
        "workspace_id": "ws-2",
        "front": "Q",
        "back": "A",
        "version": 1,
        "tags": ["tag1"],
    }
    monkeypatch.setattr(mod, "_open_db", lambda ctx: fake_db)

    ctx = SimpleNamespace(
        db_paths={"chacha": str(tmp_path / "chacha.db")},
        metadata={"workspace_id": "ws-1"},
        client_id="cli",
    )

    with pytest.raises(PermissionError, match="Flashcard access denied for workspace"):
        await mod.execute_tool(tool_name, arguments, context=ctx)


@pytest.mark.asyncio
async def test_flashcards_decks_list_and_get_honor_workspace_scope(monkeypatch, tmp_path):
    mod = FlashcardsModule(ModuleConfig(name="flashcards"))
    fake_db = FakeFlashcardsDB()
    own_deck_id = fake_db.add_deck("Own Deck", workspace_id="ws-1")
    foreign_deck_id = fake_db.add_deck("Foreign Deck", workspace_id="ws-2")
    monkeypatch.setattr(mod, "_open_db", lambda ctx: fake_db)

    ctx = SimpleNamespace(
        db_paths={"chacha": str(tmp_path / "chacha.db")},
        metadata={"workspace_id": "ws-1"},
        client_id="cli",
    )

    listed = await mod.execute_tool("flashcards.decks.list", {}, context=ctx)

    assert [deck["id"] for deck in listed["decks"]] == [own_deck_id]  # nosec B101
    assert listed["total"] == 1  # nosec B101

    with pytest.raises(PermissionError, match="Flashcard access denied for workspace"):
        await mod.execute_tool("flashcards.decks.get", {"deck_id": foreign_deck_id}, context=ctx)


@pytest.mark.asyncio
async def test_flashcards_create_deck_uses_workspace_context(monkeypatch, tmp_path):
    mod = FlashcardsModule(ModuleConfig(name="flashcards"))
    fake_db = FakeFlashcardsDB()
    monkeypatch.setattr(mod, "_open_db", lambda ctx: fake_db)

    ctx = SimpleNamespace(
        db_paths={"chacha": str(tmp_path / "chacha.db")},
        metadata={"workspace_id": "ws-1"},
        client_id="cli",
    )

    created = await mod.execute_tool(
        "flashcards.decks.create",
        {"name": "Scoped Deck"},
        context=ctx,
    )

    assert created["deck"]["workspace_id"] == "ws-1"  # nosec B101


@pytest.mark.asyncio
async def test_flashcards_create_rejects_foreign_workspace_deck(monkeypatch, tmp_path):
    mod = FlashcardsModule(ModuleConfig(name="flashcards"))
    fake_db = FakeFlashcardsDB()
    foreign_deck_id = fake_db.add_deck("Foreign Deck", workspace_id="ws-2")
    monkeypatch.setattr(mod, "_open_db", lambda ctx: fake_db)

    ctx = SimpleNamespace(
        db_paths={"chacha": str(tmp_path / "chacha.db")},
        metadata={"workspace_id": "ws-1"},
        client_id="cli",
    )

    with pytest.raises(PermissionError, match="Flashcard access denied for workspace"):
        await mod.execute_tool(
            "flashcards.create",
            {"deck_id": foreign_deck_id, "front": "Q", "back": "A", "model_type": "basic"},
            context=ctx,
        )

    assert fake_db.cards == {}  # nosec B101


@pytest.mark.asyncio
async def test_flashcards_bulk_create_rejects_any_foreign_workspace_deck(monkeypatch, tmp_path):
    mod = FlashcardsModule(ModuleConfig(name="flashcards"))
    fake_db = FakeFlashcardsDB()
    own_deck_id = fake_db.add_deck("Own Deck", workspace_id="ws-1")
    foreign_deck_id = fake_db.add_deck("Foreign Deck", workspace_id="ws-2")
    monkeypatch.setattr(mod, "_open_db", lambda ctx: fake_db)

    ctx = SimpleNamespace(
        db_paths={"chacha": str(tmp_path / "chacha.db")},
        metadata={"workspace_id": "ws-1"},
        client_id="cli",
    )

    with pytest.raises(PermissionError, match="Flashcard access denied for workspace"):
        await mod.execute_tool(
            "flashcards.create_bulk",
            {
                "cards": [
                    {"deck_id": own_deck_id, "front": "Q1", "back": "A1"},
                    {"deck_id": foreign_deck_id, "front": "Q2", "back": "A2"},
                ]
            },
            context=ctx,
        )

    assert fake_db.cards == {}  # nosec B101
