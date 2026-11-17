import asyncio
import base64
import shutil
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.Storage.filesystem_storage import FileSystemStorage
from tldw_Server_API.app.core.Image_Generation import adapter_registry as image_registry
from tldw_Server_API.app.core.Image_Generation import config as image_config
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenResult
from tldw_Server_API.app.core.File_Artifacts.adapters import image_adapter as image_adapter_module


pytestmark = pytest.mark.integration

BASE_OPTIONS = {"persist": True}


class MockImageAdapter:
    name = "stable_diffusion_cpp"
    supported_formats = {"png", "jpg", "webp"}
    content: bytes = b"test-image"

    def generate(self, request):
        fmt = request.format
        content_type = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "webp": "image/webp",
        }[fmt]
        return ImageGenResult(content=self.content, content_type=content_type, bytes_len=len(self.content))


class MockSwarmUIImageAdapter:
    name = "swarmui"
    supported_formats = {"png", "jpg"}
    content: bytes = b"swarm-image"

    def generate(self, request):
        fmt = request.format
        content_type = {
            "png": "image/png",
            "jpg": "image/jpeg",
        }[fmt]
        return ImageGenResult(content=self.content, content_type=content_type, bytes_len=len(self.content))


class MockModelStudioImageAdapter:
    name = "modelstudio"
    supported_formats = {"png", "jpg", "webp"}

    def __init__(self) -> None:
        self.seen_requests = []

    def generate(self, request):
        self.seen_requests.append(request)
        fmt = request.format
        content_type = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "webp": "image/webp",
        }[fmt]
        return ImageGenResult(content=b"modelstudio-image", content_type=content_type, bytes_len=17)


@pytest.fixture()
def mock_image_registry(monkeypatch):
    registry = image_registry.ImageAdapterRegistry(
        config_override={
            "default_backend": "stable_diffusion_cpp",
            "enabled_backends": ["stable_diffusion_cpp"],
        }
    )
    registry.register_adapter("stable_diffusion_cpp", MockImageAdapter)
    monkeypatch.setattr(image_registry, "_registry", registry)
    yield registry
    image_registry.reset_registry()


@pytest.fixture()
def mock_swarmui_registry(monkeypatch):
    registry = image_registry.ImageAdapterRegistry(
        config_override={
            "default_backend": "swarmui",
            "enabled_backends": ["swarmui"],
        }
    )
    registry.register_adapter("swarmui", MockSwarmUIImageAdapter)
    monkeypatch.setattr(image_registry, "_registry", registry)
    yield registry
    image_registry.reset_registry()


@pytest.fixture()
def mock_modelstudio_registry(monkeypatch):
    registry = image_registry.ImageAdapterRegistry(
        config_override={
            "default_backend": "modelstudio",
            "enabled_backends": ["modelstudio"],
        }
    )
    registry.register_adapter("modelstudio", MockModelStudioImageAdapter)
    monkeypatch.setattr(image_registry, "_registry", registry)
    yield registry
    image_registry.reset_registry()


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (1, 1), (255, 0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _create_test_image_media(db: MediaDatabase, *, title: str, content_hash: str, content: str = "image content") -> int:
    cursor = db.execute_query(
        "INSERT INTO Media (title, type, content, author, content_hash, uuid, last_modified, client_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            title,
            "image",
            content,
            "Test Author",
            content_hash,
            f"uuid-{content_hash}",
            db._get_current_utc_timestamp_str(),
            db.client_id or "test_client",
        ),
        commit=True,
    )
    media_id = getattr(cursor, "lastrowid", None)
    if media_id:
        return media_id
    raise RuntimeError("Failed to create test image media")


async def _seed_reference_image_asset(
    *,
    user_id: int,
    title: str,
    filename: str,
    storage: FileSystemStorage,
) -> int:
    media_db = MediaDatabase(db_path=str(DatabasePaths.get_media_db_path(user_id)), client_id=str(user_id))
    media_db.initialize_db()
    try:
        media_id = _create_test_image_media(media_db, title=title, content_hash=f"{title}-hash")
        png_bytes = _png_bytes()
        storage_path = await storage.store(
            user_id=str(user_id),
            media_id=media_id,
            filename=filename,
            data=png_bytes,
            mime_type="image/png",
        )
        media_db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path=storage_path,
            original_filename=filename,
            file_size=len(png_bytes),
            mime_type="image/png",
        )
        row = media_db.get_media_file(media_id, "original")
        return row["id"]
    finally:
        media_db.close_connection()


@pytest.fixture()
def client_with_user(monkeypatch, fresh_services):
    async def override_user():
        return User(id=321, username="tester", email=None, is_active=True)

    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "321")

    base_dir = Path.cwd() / "Databases" / "test_user_dbs_files_image"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    users_db_path = base_dir / "users.db"
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{users_db_path}")

    app = None
    try:
        from importlib import import_module, reload
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_single_user_rbac_seed_if_needed
        from tldw_Server_API.app.core.DB_Management.Users_DB import reset_users_db
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        # Ensure settings/pools pick up the test database and seed the single-user row at id=321.
        reset_settings()
        asyncio.run(reset_users_db())
        asyncio.run(ensure_single_user_rbac_seed_if_needed())

        mod = import_module("tldw_Server_API.app.main")
        mod = reload(mod)
        app = mod.app
        app.dependency_overrides[get_request_user] = override_user
        with TestClient(app) as client:
            yield client
    finally:
        if app is not None:
            app.dependency_overrides.clear()
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_create_image_inline_export(client_with_user, mock_image_registry):
    MockImageAdapter.content = b"image-bytes"
    payload = {
        "file_type": "image",
        "title": "Mock",
        "payload": {
            "backend": "stable_diffusion_cpp",
            "prompt": "A test image",
        },
        "export": {"format": "png", "mode": "inline", "async_mode": "sync"},
        "options": BASE_OPTIONS,
    }
    response = client_with_user.post("/api/v1/files/create", json=payload)
    assert response.status_code == 200, response.text
    artifact = response.json()["artifact"]
    assert artifact["file_type"] == "image"
    export = artifact["export"]
    assert export["status"] == "none"
    assert export["format"] == "png"
    assert export["content_type"] == "image/png"
    assert export["content_b64"]
    assert base64.b64decode(export["content_b64"]) == MockImageAdapter.content
    assert "reference_file_id" not in artifact["structured"]
    assert "reference_image_provenance" not in artifact["structured"]


def test_create_image_inline_export_swarmui(client_with_user, mock_swarmui_registry):
    MockSwarmUIImageAdapter.content = b"swarm-bytes"
    payload = {
        "file_type": "image",
        "title": "Swarm",
        "payload": {
            "backend": "swarmui",
            "prompt": "A test image",
        },
        "export": {"format": "jpg", "mode": "inline", "async_mode": "sync"},
        "options": BASE_OPTIONS,
    }
    response = client_with_user.post("/api/v1/files/create", json=payload)
    assert response.status_code == 200, response.text
    artifact = response.json()["artifact"]
    assert artifact["file_type"] == "image"
    export = artifact["export"]
    assert export["status"] == "none"
    assert export["format"] == "jpg"
    assert export["content_type"] == "image/jpeg"
    assert export["content_b64"]
    assert base64.b64decode(export["content_b64"]) == MockSwarmUIImageAdapter.content


def test_image_export_mode_url_rejected(client_with_user, mock_image_registry):
    payload = {
        "file_type": "image",
        "payload": {
            "backend": "stable_diffusion_cpp",
            "prompt": "A test image",
        },
        "export": {"format": "png", "mode": "url", "async_mode": "sync"},
        "options": BASE_OPTIONS,
    }
    response = client_with_user.post("/api/v1/files/create", json=payload)
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "invalid_export_mode"


def test_image_inline_size_exceeded(client_with_user, mock_image_registry, monkeypatch):
    from tldw_Server_API.app.core.File_Artifacts import file_artifacts_service as fas

    original_get_config_value = fas.get_config_value

    def _fake_get_config_value(section, key, default=None, *, reload=False):
        if section == "Image-Generation" and key == "inline_max_bytes":
            return "4"
        return original_get_config_value(section, key, default, reload=reload)

    monkeypatch.setattr(fas, "get_config_value", _fake_get_config_value)

    MockImageAdapter.content = b"toolarge"
    payload = {
        "file_type": "image",
        "payload": {
            "backend": "stable_diffusion_cpp",
            "prompt": "A test image",
        },
        "export": {"format": "png", "mode": "inline", "async_mode": "sync"},
        "options": BASE_OPTIONS,
    }
    response = client_with_user.post("/api/v1/files/create", json=payload)
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "export_size_exceeded"


def test_image_extra_params_rejected_when_not_allowlisted(client_with_user, mock_image_registry, monkeypatch):
    from tldw_Server_API.app.core.File_Artifacts.adapters import image_adapter as image_adapter_module
    from tldw_Server_API.app.core.Image_Generation import config as image_config

    cfg = image_config.get_image_generation_config(reload=True)
    cfg = replace(cfg, sd_cpp_allowed_extra_params=[])
    monkeypatch.setattr(image_adapter_module, "get_image_generation_config", lambda: cfg)

    payload = {
        "file_type": "image",
        "payload": {
            "backend": "stable_diffusion_cpp",
            "prompt": "A test image",
            "extra_params": {"clip_skip": 2},
        },
        "export": {"format": "png", "mode": "inline", "async_mode": "sync"},
        "options": BASE_OPTIONS,
    }
    response = client_with_user.post("/api/v1/files/create", json=payload)
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    errors = detail.get("errors", [])
    assert any(error.get("code") == "image_params_invalid" for error in errors)
    assert any(error.get("path") == "extra_params.clip_skip" for error in errors)


def test_image_extra_params_allowed_when_allowlisted(client_with_user, mock_image_registry, monkeypatch):
    from tldw_Server_API.app.core.File_Artifacts.adapters import image_adapter as image_adapter_module
    from tldw_Server_API.app.core.Image_Generation import config as image_config

    cfg = image_config.get_image_generation_config(reload=True)
    cfg = replace(cfg, sd_cpp_allowed_extra_params=["clip_skip"])
    monkeypatch.setattr(image_adapter_module, "get_image_generation_config", lambda: cfg)

    payload = {
        "file_type": "image",
        "payload": {
            "backend": "stable_diffusion_cpp",
            "prompt": "A test image",
            "extra_params": {"clip_skip": 2},
        },
        "export": {"format": "png", "mode": "inline", "async_mode": "sync"},
        "options": BASE_OPTIONS,
    }
    response = client_with_user.post("/api/v1/files/create", json=payload)
    assert response.status_code == 200, response.text


def test_create_image_inline_export_with_reference_file_id_persists_provenance(
    client_with_user,
    mock_modelstudio_registry,
    monkeypatch,
):
    storage = FileSystemStorage(Path.cwd() / "Databases" / "test_user_dbs_files_image" / "reference_storage")
    reference_file_id = asyncio.run(
        _seed_reference_image_asset(
            user_id=321,
            title="Reference image",
            filename="reference.png",
            storage=storage,
        )
    )

    cfg = image_config.get_image_generation_config(reload=True)
    cfg = replace(cfg, reference_image_supported_models={"modelstudio": ["qwen-image-edit"]})
    monkeypatch.setattr(image_adapter_module, "get_image_generation_config", lambda: cfg)

    with patch(
        "tldw_Server_API.app.core.Image_Generation.reference_images.get_storage_backend",
        return_value=storage,
    ):
        payload = {
            "file_type": "image",
            "title": "Reference",
            "payload": {
                "backend": "modelstudio",
                "model": "qwen-image-edit-v1",
                "prompt": "A test image",
                "reference_file_id": reference_file_id,
            },
            "export": {"format": "png", "mode": "inline", "async_mode": "sync"},
            "options": BASE_OPTIONS,
        }
        response = client_with_user.post("/api/v1/files/create", json=payload)

    assert response.status_code == 200, response.text
    artifact = response.json()["artifact"]
    assert artifact["structured"]["reference_file_id"] == reference_file_id
    assert artifact["structured"]["reference_image_provenance"] == {
        "source": "managed_reference_image",
        "reference_file_id": reference_file_id,
        "snapshot": {
            "filename": "reference.png",
            "mime_type": "image/png",
            "width": 1,
            "height": 1,
        },
    }
    export = artifact["export"]
    assert export["status"] == "none"
    assert export["content_b64"]

    get_response = client_with_user.get(f"/api/v1/files/{artifact['file_id']}")
    assert get_response.status_code == 200, get_response.text
    persisted = get_response.json()["artifact"]
    assert persisted["structured"]["reference_image_provenance"]["snapshot"] == {
        "filename": "reference.png",
        "mime_type": "image/png",
        "width": 1,
        "height": 1,
    }

    registry = image_registry.get_registry()
    adapter = registry.get_adapter("modelstudio")
    assert adapter is not None
    assert adapter.seen_requests
    assert adapter.seen_requests[0].reference_image is not None
    assert adapter.seen_requests[0].reference_image.file_id == reference_file_id


def test_create_image_inline_export_rejects_unsupported_reference_image_backend(
    client_with_user,
    mock_swarmui_registry,
):
    payload = {
        "file_type": "image",
        "title": "Unsupported backend",
        "payload": {
            "backend": "swarmui",
            "prompt": "A test image",
            "reference_file_id": 123,
        },
        "export": {"format": "png", "mode": "inline", "async_mode": "sync"},
        "options": BASE_OPTIONS,
    }
    response = client_with_user.post("/api/v1/files/create", json=payload)
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "reference_image_unsupported_by_backend"


def test_create_image_inline_export_rejects_unsupported_reference_image_model(
    client_with_user,
    mock_modelstudio_registry,
    monkeypatch,
):
    cfg = image_config.get_image_generation_config(reload=True)
    cfg = replace(cfg, reference_image_supported_models={"modelstudio": ["qwen-image-edit"]})
    monkeypatch.setattr(image_adapter_module, "get_image_generation_config", lambda: cfg)

    payload = {
        "file_type": "image",
        "title": "Unsupported",
        "payload": {
            "backend": "modelstudio",
            "model": "other-model",
            "prompt": "A test image",
            "reference_file_id": 123,
        },
        "export": {"format": "png", "mode": "inline", "async_mode": "sync"},
        "options": BASE_OPTIONS,
    }
    response = client_with_user.post("/api/v1/files/create", json=payload)
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "reference_image_unsupported_by_model"


def test_create_image_inline_export_rejects_invalid_reference_file_id(
    client_with_user,
    mock_modelstudio_registry,
    monkeypatch,
):
    cfg = image_config.get_image_generation_config(reload=True)
    cfg = replace(cfg, reference_image_supported_models={"modelstudio": ["qwen-image-edit"]})
    monkeypatch.setattr(image_adapter_module, "get_image_generation_config", lambda: cfg)

    payload = {
        "file_type": "image",
        "title": "Invalid reference",
        "payload": {
            "backend": "modelstudio",
            "model": "qwen-image-edit-v1",
            "prompt": "A test image",
            "reference_file_id": "not-an-id",
        },
        "export": {"format": "png", "mode": "inline", "async_mode": "sync"},
        "options": BASE_OPTIONS,
    }
    response = client_with_user.post("/api/v1/files/create", json=payload)
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "reference_image_invalid"


def test_create_image_inline_export_rejects_missing_reference_file(
    client_with_user,
    mock_modelstudio_registry,
    monkeypatch,
):
    cfg = image_config.get_image_generation_config(reload=True)
    cfg = replace(cfg, reference_image_supported_models={"modelstudio": ["qwen-image-edit"]})
    monkeypatch.setattr(image_adapter_module, "get_image_generation_config", lambda: cfg)

    # The lookup must hit an initialized media DB, or it fails as invalid instead of not found;
    # no earlier test's cached handle may stand in for it.
    media_db = MediaDatabase(db_path=str(DatabasePaths.get_media_db_path(321)), client_id="321")
    media_db.initialize_db()
    media_db.close_connection()
    storage = FileSystemStorage(Path.cwd() / "Databases" / "test_user_dbs_files_image" / "reference_storage_missing")
    with patch(
        "tldw_Server_API.app.core.Image_Generation.reference_images.get_storage_backend",
        return_value=storage,
    ):
        payload = {
            "file_type": "image",
            "title": "Missing reference",
            "payload": {
                "backend": "modelstudio",
                "model": "qwen-image-edit-v1",
                "prompt": "A test image",
                "reference_file_id": 999999,
            },
            "export": {"format": "png", "mode": "inline", "async_mode": "sync"},
            "options": BASE_OPTIONS,
        }
        response = client_with_user.post("/api/v1/files/create", json=payload)

    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "reference_image_not_found"
