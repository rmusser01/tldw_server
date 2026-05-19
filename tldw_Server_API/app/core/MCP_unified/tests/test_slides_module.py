import base64
import contextlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.implementations.slides_module import SlidesModule
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@dataclass
class FakeRow:
    id: str
    title: str
    description: str | None
    theme: str
    marp_theme: str | None
    template_id: str | None
    settings: str | None
    studio_data: str | None
    slides: str
    slides_text: str
    source_type: str | None
    source_ref: str | None
    source_query: str | None
    custom_css: str | None
    created_at: str
    last_modified: str
    deleted: int
    client_id: str
    version: int


@dataclass
class FakeVersionRow:
    presentation_id: str
    version: int
    payload_json: str
    created_at: str
    client_id: str


class FakeSlidesDB:
    def __init__(self) -> None:
        self.rows: Dict[str, FakeRow] = {}
        self.versions: Dict[str, Dict[int, FakeVersionRow]] = {}
        self._counter = 0

    def _snapshot_payload(self, row: FakeRow) -> str:
        return json.dumps(
            {
                "id": row.id,
                "title": row.title,
                "description": row.description,
                "theme": row.theme,
                "marp_theme": row.marp_theme,
                "template_id": row.template_id,
                "settings": row.settings,
                "studio_data": row.studio_data,
                "slides": row.slides,
                "custom_css": row.custom_css,
                "source_type": row.source_type,
                "source_ref": row.source_ref,
                "source_query": row.source_query,
                "created_at": row.created_at,
                "last_modified": row.last_modified,
                "deleted": row.deleted,
                "client_id": row.client_id,
                "version": row.version,
            }
        )

    def _store_version(self, row: FakeRow) -> None:
        self.versions.setdefault(row.id, {})[row.version] = FakeVersionRow(
            presentation_id=row.id,
            version=row.version,
            payload_json=self._snapshot_payload(row),
            created_at=row.last_modified,
            client_id=row.client_id,
        )

    def create_presentation(self, presentation_id, title, description, theme, marp_theme, template_id, settings, studio_data, slides, slides_text, source_type, source_ref, source_query, custom_css):
        self._counter += 1
        pid = presentation_id or f"pres-{self._counter}"
        row = FakeRow(
            id=pid,
            title=title,
            description=description,
            theme=theme,
            marp_theme=marp_theme,
            template_id=template_id,
            settings=settings,
            studio_data=studio_data,
            slides=slides,
            slides_text=slides_text,
            source_type=source_type,
            source_ref=source_ref,
            source_query=source_query,
            custom_css=custom_css,
            created_at="now",
            last_modified="now",
            deleted=0,
            client_id="test",
            version=1,
        )
        self.rows[pid] = row
        self._store_version(row)
        return row

    def list_presentations(self, limit, offset, include_deleted, sort_column, sort_direction) -> Tuple[List[FakeRow], int]:
        items = list(self.rows.values())
        return items[offset: offset + limit], len(items)

    def search_presentations(self, query, limit, offset, include_deleted):
        items = list(self.rows.values())
        return items[offset: offset + limit], len(items)

    def get_presentation_by_id(self, presentation_id, include_deleted=False):
        row = self.rows.get(presentation_id)
        if not row:
            raise KeyError("presentation_not_found")
        return row

    def update_presentation(self, presentation_id, update_fields, expected_version, operation=None):
        row = self.get_presentation_by_id(presentation_id, include_deleted=True)
        for k, v in update_fields.items():
            setattr(row, k, v)
        row.version += 1
        self._store_version(row)
        return row

    def list_presentation_versions(self, presentation_id, limit, offset):
        versions = sorted(self.versions.get(presentation_id, {}).values(), key=lambda row: row.version, reverse=True)
        return versions[offset: offset + limit], len(versions)

    def get_presentation_version(self, presentation_id, version):
        row = self.versions.get(presentation_id, {}).get(version)
        if not row:
            raise KeyError("presentation_version_not_found")
        return row

    def soft_delete_presentation(self, presentation_id, expected_version):
        row = self.get_presentation_by_id(presentation_id, include_deleted=True)
        row.deleted = 1
        row.version += 1
        return row

    def restore_presentation(self, presentation_id, expected_version):
        row = self.get_presentation_by_id(presentation_id, include_deleted=True)
        row.deleted = 0
        row.version += 1
        return row

    def close_connection(self):
        return None


class FakeGenerator:
    def generate_from_text(self, source_text, title_hint=None, provider=None, model=None, api_key=None, temperature=0.7, max_tokens=4000, max_source_tokens=None, max_source_chars=None, enable_chunking=True, chunk_size_tokens=1000, summary_tokens=200):
        return {"title": title_hint or "Generated", "slides": [{"order": 0, "title": "Slide", "content": "Content"}]}


def test_slides_module_get_media_content_uses_managed_media_database(monkeypatch, tmp_path):
    mod = SlidesModule(ModuleConfig(name="slides"))
    events = []

    class _FakeDb:
        def get_media_by_id(self, media_id):
            events.append(("get_media_by_id", media_id))
            return {"content": "slide source"}

    @contextlib.contextmanager
    def _fake_managed_media_database(client_id, **kwargs):
        events.append(("open", client_id, kwargs))
        yield _FakeDb()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.modules.implementations.slides_module.MediaDatabase",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("slides_module should not construct MediaDatabase directly")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.modules.implementations.slides_module.managed_media_database",
        _fake_managed_media_database,
        raising=False,
    )

    context = SimpleNamespace(db_paths={"media": str(tmp_path / "media.db")})

    result = mod._get_media_content(context, 23)

    _ensure(result == "slide source", f"Unexpected media content: {result!r}")
    _ensure(
        events == [
            ("open", "mcp_slides_gen", {"db_path": str(tmp_path / "media.db"), "initialize": False}),
            ("get_media_by_id", 23),
        ],
        f"Unexpected media database events: {events!r}",
    )


@pytest.mark.asyncio
async def test_slides_templates_export_and_rag(monkeypatch, tmp_path):
    mod = SlidesModule(ModuleConfig(name="slides"))
    fake_db = FakeSlidesDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]

    ctx = SimpleNamespace(
        db_paths={"media": str(tmp_path / "media.db"), "chacha": str(tmp_path / "chacha.db")},
        user_id="1",
    )

    created = await mod.execute_tool(
        "slides.presentations.create",
        {"title": "Deck", "slides": json.dumps({"slides": [{"order": 0, "title": "T", "content": "C"}]})},
        context=ctx,
    )
    pres_id = created["presentation_id"]

    templates = await mod.execute_tool("slides.templates.list", {}, context=ctx)
    template_ids = {t["id"] for t in templates["templates"]}
    _ensure("academic" in template_ids, f"Missing template ids: {template_ids!r}")

    template = await mod.execute_tool("slides.templates.get", {"template_id": "academic"}, context=ctx)
    _ensure(template["template"]["id"] == "academic", f"Unexpected template payload: {template!r}")

    from tldw_Server_API.app.core.Slides import slides_export
    monkeypatch.setattr(slides_export, "export_presentation_bundle", lambda **_kwargs: b"zip")
    monkeypatch.setattr(slides_export, "export_presentation_pdf", lambda **_kwargs: b"pdf")

    exported = await mod.execute_tool("slides.export", {"presentation_id": pres_id, "format": "reveal"}, context=ctx)
    _ensure(base64.b64decode(exported["content_base64"]) == b"zip", f"Unexpected reveal export payload: {exported!r}")

    exported_pdf = await mod.execute_tool("slides.export", {"presentation_id": pres_id, "format": "pdf"}, context=ctx)
    _ensure(base64.b64decode(exported_pdf["content_base64"]) == b"pdf", f"Unexpected pdf export payload: {exported_pdf!r}")

    async def _fake_rag_pipeline(**kwargs):
        doc = SimpleNamespace(metadata={"title": "Doc"}, content="RAG content")
        return SimpleNamespace(documents=[doc], generated_answer=None)

    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as rag_pipeline
    monkeypatch.setattr(rag_pipeline, "unified_rag_pipeline", _fake_rag_pipeline)
    mod._get_generator = lambda: FakeGenerator()  # type: ignore[attr-defined]

    rag_out = await mod.execute_tool(
        "slides.generate.from_rag",
        {"query": "test", "top_k": 1},
        context=ctx,
    )
    _ensure(rag_out["success"] is True, f"Unexpected RAG output: {rag_out!r}")


@pytest.mark.asyncio
async def test_slides_reorder_requires_exact_permutation():
    mod = SlidesModule(ModuleConfig(name="slides"))
    fake_db = FakeSlidesDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]
    ctx = SimpleNamespace(db_paths={"slides": "unused.db"})

    created = await mod.execute_tool(
        "slides.presentations.create",
        {"title": "Deck", "slides": json.dumps({"slides": [{"order": 0}, {"order": 1}]})},
        context=ctx,
    )

    with pytest.raises(ValueError, match=r"slide_order must be a permutation of 0\.\.1"):
        await mod.execute_tool(
            "slides.presentations.reorder",
            {"presentation_id": created["presentation_id"], "slide_order": [0, 0], "expected_version": 1},
            context=ctx,
        )


@pytest.mark.asyncio
async def test_slides_reorder_rejects_boolean_indices():
    mod = SlidesModule(ModuleConfig(name="slides"))
    fake_db = FakeSlidesDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]
    ctx = SimpleNamespace(db_paths={"slides": "unused.db"})

    created = await mod.execute_tool(
        "slides.presentations.create",
        {"title": "Deck", "slides": json.dumps({"slides": [{"order": 0}, {"order": 1}]})},
        context=ctx,
    )

    with pytest.raises(ValueError, match=r"slide_order must be a permutation of 0\.\.1"):
        await mod.execute_tool(
            "slides.presentations.reorder",
            {"presentation_id": created["presentation_id"], "slide_order": [False, True], "expected_version": 1},
            context=ctx,
        )


@pytest.mark.asyncio
async def test_slides_patch_normalizes_json_fields_and_slides_text():
    mod = SlidesModule(ModuleConfig(name="slides"))
    fake_db = FakeSlidesDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]
    ctx = SimpleNamespace(db_paths={"slides": "unused.db"})

    created = await mod.execute_tool(
        "slides.presentations.create",
        {"title": "Deck", "slides": json.dumps({"slides": [{"order": 0, "title": "A", "content": "Alpha"}]})},
        context=ctx,
    )
    pid = created["presentation_id"]

    await mod.execute_tool(
        "slides.presentations.patch",
        {
            "presentation_id": pid,
            "patch": {
                "slides": json.dumps({"slides": [{"order": 0, "title": "B", "content": "Beta"}]}),
                "slides_text": "STALE TEXT",
                "settings": {"theme": "serif"},
            },
            "expected_version": 1,
        },
        context=ctx,
    )

    row = fake_db.rows[pid]
    _ensure(row.settings == json.dumps({"theme": "serif"}), f"Unexpected normalized settings: {row.settings!r}")
    _ensure("Beta" in row.slides_text, f"Slides text was not refreshed: {row.slides_text!r}")
    _ensure("STALE TEXT" not in row.slides_text, f"Caller-supplied slides_text should be ignored: {row.slides_text!r}")


@pytest.mark.asyncio
async def test_slides_restore_version_normalizes_json_fields_and_slides_text():
    mod = SlidesModule(ModuleConfig(name="slides"))
    fake_db = FakeSlidesDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]
    ctx = SimpleNamespace(db_paths={"slides": "unused.db"})

    created = await mod.execute_tool(
        "slides.presentations.create",
        {
            "title": "Deck",
            "slides": json.dumps({"slides": [{"order": 0, "title": "A", "content": "Alpha"}]}),
            "settings": {"theme": "night"},
        },
        context=ctx,
    )
    pid = created["presentation_id"]

    await mod.execute_tool(
        "slides.presentations.patch",
        {
            "presentation_id": pid,
            "patch": {
                "slides": json.dumps({"slides": [{"order": 0, "title": "B", "content": "Beta"}]}),
                "settings": {"theme": "serif"},
            },
            "expected_version": 1,
        },
        context=ctx,
    )

    fake_db.versions[pid][1] = FakeVersionRow(
        presentation_id=pid,
        version=1,
        payload_json=json.dumps(
            {
                "title": "Deck",
                "theme": "black",
                "settings": {"theme": "night"},
                "slides": json.dumps({"slides": [{"order": 0, "title": "A", "content": "Alpha"}]}),
                "studio_data": {"layout": "restored"},
            }
        ),
        created_at="now",
        client_id="test",
    )

    await mod.execute_tool(
        "slides.versions.restore",
        {
            "presentation_id": pid,
            "version": 1,
            "expected_current_version": 2,
        },
        context=ctx,
    )

    restored_row = fake_db.rows[pid]
    _ensure(
        restored_row.settings == json.dumps({"theme": "night"}),
        f"Restore should normalize settings JSON: {restored_row.settings!r}",
    )
    _ensure(
        restored_row.studio_data == json.dumps({"layout": "restored"}),
        f"Restore should normalize studio_data JSON: {restored_row.studio_data!r}",
    )
    _ensure(
        "Alpha" in restored_row.slides_text and "Beta" not in restored_row.slides_text,
        f"Restore should refresh slides_text from restored slides: {restored_row.slides_text!r}",
    )
