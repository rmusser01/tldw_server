"""Behavioral coverage for the bundled character and its production bootstrap."""

from __future__ import annotations

import hashlib

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Visual_Identities.service import VisualIdentityService
from tldw_Server_API.app.core.Visual_Identities.storage import resolve_visual_identity_asset_path

pytestmark = pytest.mark.unit


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(
        DatabasePaths, "get_user_visual_identities_dir", staticmethod(lambda owner: tmp_path / str(owner))
    )
    database = CharactersRAGDB(tmp_path / "characters.db", client_id="42")
    yield database
    database.close_connection()


def seed(db, owner=42):
    from tldw_Server_API.app.core.Visual_Identities.builtin_pixel_migu import ensure_pixel_migu_character

    return ensure_pixel_migu_character(db, owner_user_id=owner)


def test_fresh_seed_resolves_all_expressions_and_preserves_bytes(db):
    character_id = seed(db)
    character = db.get_character_card_by_id(character_id)
    assert character["name"] == "pixel-migu"
    assert character["image"].startswith(b"\x89PNG")
    assert character["first_message"].startswith("Hi!")
    service = VisualIdentityService(db, owner_user_id=42)
    resolved = service.resolve_expression_asset("character", character_id, "happy")
    assets = service.repository.list_assets_for_version(resolved.pack_version_id, owner_user_id=42)
    assert len(assets) == 18
    assert len({asset["sha256"] for asset in assets}) == 18
    for asset in assets:
        result = service.resolve_expression_asset("character", character_id, asset["expression_key"])
        assert result.expression_key == asset["expression_key"]
        path = resolve_visual_identity_asset_path(owner_user_id=42, relpath=result.storage_relpath)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
    fallback = service.resolve_expression_asset("character", character_id, "custom:missing")
    assert fallback.expression_key == "neutral"
    assert service.repository.get_pack(resolved.pack_id, owner_user_id=43) is None


def test_replay_preserves_rename_customization_and_deleted_character(db):
    character_id = seed(db)
    db.update_character_card(character_id, {"name": "My Migu", "description": "My custom text"}, expected_version=1)
    assert seed(db) == character_id
    character = db.get_character_card_by_id(character_id)
    assert character["description"] == "My custom text"
    assert character["version"] == 2
    db.soft_delete_character_card(character_id, expected_version=2)
    assert seed(db) == character_id
    assert db.get_character_card_by_id(character_id) is None
    assert db.get_character_card_by_name("pixel-migu") is None


def test_existing_same_name_character_is_not_adopted(db):
    existing_id = db.add_character_card({"name": "pixel-migu", "description": "User content"})
    assert seed(db) is None
    db.update_character_card(existing_id, {"name": "Renamed"}, expected_version=1)
    assert seed(db) is None
    assert db.get_character_card_by_name("pixel-migu") is None


def test_failure_rolls_back_character_and_receipt_then_can_retry(db, monkeypatch):
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("activation failure")

        patch.setattr(VisualIdentityService, "activate_draft", fail)
        with pytest.raises(RuntimeError, match="activation failure"):
            seed(db)
    assert db.get_character_card_by_name("pixel-migu") is None
    assert seed(db) is not None


def test_production_db_factory_seeds_before_return(tmp_path, monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps

    monkeypatch.setattr(deps, "_get_chacha_db_path_for_user", lambda owner: tmp_path / str(owner) / "characters.db")
    monkeypatch.setattr(
        DatabasePaths, "get_user_visual_identities_dir", staticmethod(lambda owner: tmp_path / str(owner) / "visuals")
    )
    for owner in (42, 43):
        database = deps._create_and_prepare_db(owner, str(owner))
        try:
            character = database.get_character_card_by_name("pixel-migu")
            assert character is not None
            result = VisualIdentityService(database, owner).resolve_expression_asset(
                "character", character["id"], "happy"
            )
            path = resolve_visual_identity_asset_path(owner_user_id=owner, relpath=result.storage_relpath)
            assert str(owner) in path.parts
            assert path.is_file()
        finally:
            database.close_connection()


def test_reopen_keeps_user_unbinding(db):
    character_id = seed(db)
    service = VisualIdentityService(db, 42)
    repo = service.repository
    binding = repo.resolve_active_binding(owner_user_id=42, actor_kind="character", actor_id=character_id)
    repo.delete_binding(binding["id"], owner_user_id=42)
    db.close_connection()
    assert seed(db) == character_id
    assert repo.resolve_active_binding(owner_user_id=42, actor_kind="character", actor_id=character_id) is None


def test_concurrent_seed_attempts_publish_one_character(db):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    # Initialize schema before the race, matching database factory preparation.
    VisualIdentityService(db, 42).repository.initialize_schema()
    start = Barrier(2)

    def install():
        start.wait()
        try:
            return seed(db)
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as executor:
        ids = list(executor.map(lambda _: install(), range(2)))
    assert ids[0] == ids[1]
    assert len([card for card in db.list_character_cards() if card["name"] == "pixel-migu"]) == 1


def test_packaged_expression_manifest_matches_all_shipped_pngs():
    import json
    from importlib import resources
    from io import BytesIO

    from PIL import Image

    root = resources.files("tldw_Server_API.app.core.Visual_Identities").joinpath("assets", "pixel-migu")
    manifest = json.loads(root.joinpath("visual_identity_pack.json").read_text())
    assert len(manifest["assets"]) == 18
    for asset in manifest["assets"]:
        content = root.joinpath("expressions", asset["storage_relpath"].rsplit("/", 1)[-1]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == asset["sha256"]
        assert len(content) == asset["bytes"]
        with Image.open(BytesIO(content)) as image:
            assert image.size == (128, 128)
            assert image.mode == "RGBA"
            assert image.getpixel((0, 0))[3] == 0


def test_corrupt_bundled_expression_rolls_back_seed(db, tmp_path, monkeypatch):
    import shutil
    from importlib import resources

    from tldw_Server_API.app.core.Visual_Identities import builtin_pixel_migu

    package = "tldw_Server_API.app.core.Visual_Identities"
    original_files = resources.files
    copied = tmp_path / "package"
    shutil.copytree(original_files(package).joinpath("assets"), copied / "assets")
    expressions = copied / "assets/pixel-migu/expressions"
    (expressions / "angry.png").write_bytes((expressions / "happy.png").read_bytes())
    monkeypatch.setattr(
        builtin_pixel_migu.resources, "files", lambda name: copied if name == package else original_files(name)
    )
    with pytest.raises(ValueError, match="pixel_migu_asset_hash_mismatch"):
        seed(db)
    assert db.get_character_card_by_name("pixel-migu") is None


def test_replay_preserves_deleted_expression_pack(db):
    character_id = seed(db)
    service = VisualIdentityService(db, 42)
    result = service.resolve_expression_asset("character", character_id, "happy")
    service.repository.mark_pack_deleted(pack_id=result.pack_id, owner_user_id=42)
    assert seed(db) == character_id
    assert service.repository.get_pack(result.pack_id, owner_user_id=42) is None


def test_production_factory_preserves_preexisting_deleted_name(tmp_path, monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps

    db_path = tmp_path / "existing.db"
    database = CharactersRAGDB(db_path, client_id="42")
    existing_id = database.add_character_card({"name": "pixel-migu", "description": "Deleted user content"})
    database.soft_delete_character_card(existing_id, expected_version=1)
    database.close_connection()
    monkeypatch.setattr(deps, "_get_chacha_db_path_for_user", lambda owner: db_path)
    database = deps._create_and_prepare_db(42, "42")
    try:
        tombstone = database.get_character_card_by_id(existing_id, include_deleted=True)
        assert tombstone["deleted"] == 1
        assert tombstone["description"] == "Deleted user content"
        assert seed(database) is None
    finally:
        database.close_connection()
