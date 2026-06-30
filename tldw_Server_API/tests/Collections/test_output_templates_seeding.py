import json
import shutil
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase


pytestmark = pytest.mark.unit


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    template_dir = tmp_path / "watchlists_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "seeded.md").write_text("Seeded {{ title }}", encoding="utf-8")
    (template_dir / "seeded.meta.json").write_text(json.dumps({"description": "Seeded template"}), encoding="utf-8")
    monkeypatch.setenv("WATCHLIST_TEMPLATE_DIR", str(template_dir))

    base_dir = Path.cwd() / "Databases" / "test_user_dbs_seed"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    try:
        yield template_dir
    finally:
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_seed_watchlists_templates_into_outputs(seeded_db):
    cdb = CollectionsDatabase.for_user(user_id=901)
    items, total = cdb.list_output_templates(q=None, limit=50, offset=0)
    assert total >= 1
    seeded = next((tpl for tpl in items if tpl.name == "seeded"), None)
    assert seeded is not None
    assert seeded.format == "md"
    meta = json.loads(seeded.metadata_json or "{}")
    assert meta.get("seeded_from") == "watchlists_templates"

    # Seeding should be idempotent
    cdb_again = CollectionsDatabase.for_user(user_id=901)
    items_again, total_again = cdb_again.list_output_templates(q=None, limit=50, offset=0)
    names = [tpl.name for tpl in items_again]
    assert names.count("seeded") == 1


def test_output_templates_search_case_insensitive_sqlite(seeded_db):
    cdb = CollectionsDatabase.for_user(user_id=902)
    cdb.create_output_template(
        name="CaseTemplate",
        type_="summary",
        format_="md",
        body="Body",
        description="Mixed Case Template",
        is_default=False,
    )
    items, total = cdb.list_output_templates(q="casetemplate", limit=10, offset=0)
    assert total >= 1
    assert any(tpl.name == "CaseTemplate" for tpl in items)


def test_seed_watchlists_templates_refresh_updates_output_templates(seeded_db):
    template_dir = seeded_db
    cdb = CollectionsDatabase.for_user(user_id=903)
    seeded = cdb.get_output_template_by_name("seeded")
    assert seeded.body == "Seeded {{ title }}"
    assert seeded.description == "Seeded template"

    (template_dir / "seeded.md").write_text("Updated {{ title }}", encoding="utf-8")
    (template_dir / "seeded.meta.json").write_text(json.dumps({"description": "Updated template"}), encoding="utf-8")

    cdb_refresh = CollectionsDatabase.for_user(user_id=903)
    refreshed = cdb_refresh.get_output_template_by_name("seeded")
    assert refreshed.body == "Updated {{ title }}"
    assert refreshed.description == "Updated template"
    meta = json.loads(refreshed.metadata_json or "{}")
    assert meta.get("seeded_from") == "watchlists_templates"


def test_seed_watchlists_templates_refresh_format_change_md_to_html(seeded_db):
    template_dir = seeded_db
    cdb = CollectionsDatabase.for_user(user_id=904)
    seeded = cdb.get_output_template_by_name("seeded")
    assert seeded.format == "md"

    (template_dir / "seeded.md").unlink(missing_ok=True)
    (template_dir / "seeded.html").write_text("<h1>{{ title }}</h1>", encoding="utf-8")
    (template_dir / "seeded.meta.json").write_text(
        json.dumps({"description": "Seeded HTML template"}), encoding="utf-8"
    )

    cdb_refresh = CollectionsDatabase.for_user(user_id=904)
    refreshed = cdb_refresh.get_output_template_by_name("seeded")
    assert refreshed.format == "html"
    assert refreshed.type == "newsletter_html"
    assert refreshed.body == "<h1>{{ title }}</h1>"
    assert refreshed.description == "Seeded HTML template"


def test_seed_watchlists_templates_does_not_override_user_template(seeded_db, monkeypatch):
    monkeypatch.setenv("WATCHLISTS_SEED_OUTPUT_TEMPLATES", "false")

    cdb = CollectionsDatabase.for_user(user_id=905)
    cdb.create_output_template(
        name="seeded",
        type_="briefing_markdown",
        format_="md",
        body="User body",
        description="User description",
        is_default=False,
    )
    row = cdb.get_output_template_by_name("seeded")
    assert json.loads(row.metadata_json or "{}").get("seeded_from") is None

    monkeypatch.setenv("WATCHLISTS_SEED_OUTPUT_TEMPLATES", "true")
    cdb_refresh = CollectionsDatabase.for_user(user_id=905)
    refreshed = cdb_refresh.get_output_template_by_name("seeded")
    assert refreshed.body == "User body"
    assert refreshed.description == "User description"
    assert json.loads(refreshed.metadata_json or "{}").get("seeded_from") is None


def test_seed_watchlists_templates_rechecks_stale_existing_snapshot(seeded_db, monkeypatch):
    monkeypatch.setenv("WATCHLISTS_SEED_OUTPUT_TEMPLATES", "false")

    cdb = CollectionsDatabase.for_user(user_id=906)
    cdb.create_output_template(
        name="seeded",
        type_="briefing_markdown",
        format_="md",
        body="User body",
        description="User description",
        is_default=False,
    )

    monkeypatch.setenv("WATCHLISTS_SEED_OUTPUT_TEMPLATES", "true")
    monkeypatch.setattr(CollectionsDatabase, "_list_output_template_names", lambda self: set())

    original_create = CollectionsDatabase.create_output_template

    def fail_on_duplicate_insert(self, *args, **kwargs):
        name = kwargs.get("name") if kwargs else (args[0] if args else None)
        if name == "seeded":
            raise AssertionError("seeding attempted duplicate insert")
        return original_create(self, *args, **kwargs)

    monkeypatch.setattr(CollectionsDatabase, "create_output_template", fail_on_duplicate_insert)

    cdb_refresh = CollectionsDatabase.for_user(user_id=906)
    refreshed = cdb_refresh.get_output_template_by_name("seeded")
    assert refreshed.body == "User body"
    assert refreshed.description == "User description"
    assert json.loads(refreshed.metadata_json or "{}").get("seeded_from") is None
