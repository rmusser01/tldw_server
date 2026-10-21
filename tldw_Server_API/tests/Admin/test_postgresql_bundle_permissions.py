"""Connected endpoint authority fences for whole-database archive operations."""

import io
import json
import zipfile
from datetime import datetime, timezone
from types import MappingProxyType
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile

from tldw_Server_API.app.api.v1.endpoints.admin import admin_bundle_ops as routes
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def org_admin():
    return AuthPrincipal(kind="user", user_id=41, roles=["org_admin"], is_admin=True)


@pytest.fixture
def database_metadata(monkeypatch):
    metadata = routes.svc.BundleMetadata(
        "database.zip", None, datetime.now(timezone.utc), 100, ("postgresql",), MappingProxyType({}), "test", 2, None
    )
    monkeypatch.setattr(routes.svc, "get_bundle_metadata", lambda _: metadata)
    monkeypatch.setattr(routes, "_emit_admin_audit_event", AsyncMock())
    return metadata


async def test_org_admin_cannot_create_database_archive(org_admin, monkeypatch):
    create = AsyncMock()
    monkeypatch.setattr(routes.svc, "create_bundle_async", create)
    with pytest.raises(HTTPException) as failure:
        await routes.create_bundle(
            routes.BundleCreateRequest(datasets=["postgresql"]), request=object(), principal=org_admin
        )
    assert failure.value.status_code == 403 and create.await_count == 0  # nosec B101 -- deny before dump/artifact admission.


@pytest.mark.parametrize("operation", ["metadata", "download", "delete"])
async def test_org_admin_cannot_access_database_archive(org_admin, database_metadata, monkeypatch, operation):
    writes = []
    monkeypatch.setattr(routes.svc, "get_bundle_path", lambda _: "/fixture/database.zip")
    monkeypatch.setattr(routes.svc, "delete_bundle", lambda _: writes.append(True))
    with pytest.raises(HTTPException) as failure:
        if operation == "metadata":
            await routes.get_bundle_metadata("database.zip", principal=org_admin)
        elif operation == "download":
            await routes.download_bundle("database.zip", request=object(), principal=org_admin)
        else:
            await routes.delete_bundle_endpoint("database.zip", request=object(), principal=org_admin)
    assert failure.value.status_code == 403 and writes == []  # nosec B101 -- knowing an archive id cannot cross platform authority.


@pytest.mark.parametrize("dry_run", [True, False])
async def test_org_admin_cannot_import_or_inspect_database_archive(org_admin, monkeypatch, dry_run):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as output:
        output.writestr(
            "manifest.json",
            json.dumps({"manifest_version": 2, "recovery_scope": "postgresql_database", "datasets": ["postgresql"]}),
        )
    content.seek(0)
    importing = AsyncMock(return_value={"status": "compatible"})
    monkeypatch.setattr(routes.svc, "import_bundle_async", importing)
    with pytest.raises(HTTPException) as failure:
        await routes.import_bundle(
            request=object(),
            file=UploadFile(file=content, filename="database.zip"),
            user_id=None,
            dry_run=dry_run,
            allow_downgrade=False,
            principal=org_admin,
        )
    assert failure.value.status_code == 403 and importing.await_count == 0  # nosec B101 -- archive scope is read before service admission for either operation.


async def test_org_admin_listing_filters_global_scope_before_pagination(org_admin, monkeypatch):
    seen = []

    def listing(**kwargs):
        seen.append(kwargs)
        return [], 0

    monkeypatch.setattr(routes.svc, "list_bundles", listing)
    await routes.list_bundles(user_id=None, limit=1, offset=2, principal=org_admin)
    assert seen == [{"user_id": None, "limit": 1, "offset": 2, "include_database": False}]  # nosec B101 -- hidden globals must not affect totals or pages.


async def test_platform_admin_can_reach_explicit_database_export(database_metadata, monkeypatch):
    creating = AsyncMock(return_value=database_metadata)
    monkeypatch.setattr(routes.svc, "create_bundle_async", creating)
    principal = AuthPrincipal(kind="user", user_id=1, roles=["admin"])
    result = await routes.create_bundle(
        routes.BundleCreateRequest(datasets=["postgresql"]), request=object(), principal=principal
    )
    assert result.item.datasets == ["postgresql"] and creating.await_count == 1  # nosec B101 -- the ordinary platform-admin path remains usable.


@pytest.mark.parametrize("claim", ["recovery_scope", "normalized_dataset"])
@pytest.mark.parametrize("operation", ["metadata", "download", "delete"])
async def test_actions_use_canonical_fail_closed_scope_claims(tmp_path, monkeypatch, org_admin, claim, operation):
    directory = tmp_path / "bundles"
    directory.mkdir()
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(tmp_path))
    manifest = {"manifest_version": 1, "user_id": None, "datasets": ["authnz"]}
    if claim == "recovery_scope":
        manifest["recovery_scope"] = "postgresql_database"
    else:
        manifest["datasets"] = [" POSTGRESQL "]
    with zipfile.ZipFile(directory / "claim.zip", "w") as output:
        output.writestr("manifest.json", json.dumps(manifest))
    monkeypatch.setattr(routes, "_emit_admin_audit_event", AsyncMock())
    with pytest.raises(HTTPException) as failure:
        if operation == "metadata":
            await routes.get_bundle_metadata("claim.zip", principal=org_admin)
        elif operation == "download":
            await routes.download_bundle("claim.zip", request=object(), principal=org_admin)
        else:
            await routes.delete_bundle_endpoint("claim.zip", request=object(), principal=org_admin)
    assert failure.value.status_code == 403 and (directory / "claim.zip").exists()  # nosec B101 -- every global claim is protected before exposure or deletion.


async def test_listing_does_not_trust_a_sidecar_for_archive_authority(tmp_path, monkeypatch):
    directory = tmp_path / "bundles"
    directory.mkdir()
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(tmp_path))
    with zipfile.ZipFile(directory / "database.zip", "w") as output:
        output.writestr(
            "manifest.json",
            json.dumps({"manifest_version": 2, "recovery_scope": "postgresql_database", "datasets": ["postgresql"]}),
        )
    (directory / "database.zip.manifest.json").write_text(json.dumps({"manifest_version": 1, "datasets": ["authnz"]}))
    assert routes.svc.list_bundles(include_database=False) == ([], 0)  # nosec B101 -- canonical ZIP scope governs visibility and totals.
