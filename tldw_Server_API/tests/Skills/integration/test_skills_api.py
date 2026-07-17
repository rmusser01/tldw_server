# tests/Skills/integration/test_skills_api.py
#
# Integration tests for Skills REST API endpoints
#

import gc
import os
import weakref
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

# Keep module-level app import lightweight for this suite.
os.environ.setdefault("MINIMAL_TEST_APP", "1")
os.environ.setdefault("TEST_MODE", "1")
_routes_disable = {
    part.strip() for part in str(os.environ.get("ROUTES_DISABLE", "")).split(",") if part and part.strip()
}
_routes_disable.update({"media", "audio", "audio-websocket"})
os.environ["ROUTES_DISABLE"] = ",".join(sorted(_routes_disable))

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.skills import (
    MAX_SKILL_IMPORT_PREVIEW_UPLOAD_BYTES,
    _metadata_to_summary,
    _skill_data_to_response,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Context_Integrity.resolver import clear_global_context_integrity_resolver
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Skills.exceptions import SkillsError
from tldw_Server_API.app.core.Skills.runtime_metadata import build_skill_runtime_metadata
from tldw_Server_API.app.core.Skills.skills_service import SkillsService

pytestmark = pytest.mark.integration

SKILLS_PREFIX = "/api/v1/skills"
TEST_USER_ID = 999


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Provide a TestClient with mocked auth and isolated user database."""
    from tldw_Server_API.app.main import app as fastapi_app

    user_base = tmp_path / "user_databases" / str(TEST_USER_ID)
    user_base.mkdir(parents=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))

    db_path = user_base / "ChaChaNotes.db"
    chacha_db = CharactersRAGDB(db_path=db_path, client_id="test_client")

    async def override_user() -> User:
        return User(id=TEST_USER_ID, username="skills-test-user", email=None, is_active=True)

    def override_chacha_db() -> CharactersRAGDB:
        return chacha_db

    # Monkeypatch DatabasePaths so SkillsService gets our temp dir
    monkeypatch.setattr(DatabasePaths, "get_user_base_directory", staticmethod(lambda uid: user_base))

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db

    try:
        clear_global_context_integrity_resolver()
        with TestClient(fastapi_app) as c:
            clear_global_context_integrity_resolver()
            yield c
    finally:
        clear_global_context_integrity_resolver()
        fastapi_app.dependency_overrides.clear()
        chacha_db.close_connection()


@pytest.fixture()
def principal_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Provide a TestClient that authenticates skills routes through AuthPrincipal only."""
    from tldw_Server_API.app.main import app as fastapi_app

    user_base = tmp_path / "user_databases" / str(TEST_USER_ID)
    user_base.mkdir(parents=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))

    db_path = user_base / "ChaChaNotes.db"
    chacha_db = CharactersRAGDB(db_path=db_path, client_id="principal_test_client")

    principal = AuthPrincipal(
        kind="user",
        user_id=TEST_USER_ID,
        username="skills-principal-user",
        roles=["user"],
        permissions=["skills.read", "skills.write"],
        subject=f"user:{TEST_USER_ID}",
        token_type="access",
    )

    async def override_principal(request: Request) -> AuthPrincipal:
        request.state.auth = AuthContext(principal=principal)
        request.state._auth_user = {
            "id": principal.user_id,
            "username": principal.username,
            "roles": list(principal.roles),
            "permissions": list(principal.permissions),
            "is_active": True,
            "is_verified": True,
        }
        request.state.user_id = principal.user_id
        request.state.api_key_id = principal.api_key_id
        request.state.org_ids = list(principal.org_ids)
        request.state.team_ids = list(principal.team_ids)
        return principal

    def override_chacha_db() -> CharactersRAGDB:
        return chacha_db

    monkeypatch.setattr(DatabasePaths, "get_user_base_directory", staticmethod(lambda uid: user_base))

    fastapi_app.dependency_overrides[auth_deps.get_auth_principal] = override_principal
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db

    try:
        clear_global_context_integrity_resolver()
        with TestClient(fastapi_app) as c:
            clear_global_context_integrity_resolver()
            yield c
    finally:
        clear_global_context_integrity_resolver()
        fastapi_app.dependency_overrides.clear()
        chacha_db.close_connection()


@pytest.fixture()
def auth_path_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Provide a TestClient that leaves get_auth_principal on the real auth path."""
    from tldw_Server_API.app.main import app as fastapi_app

    user_base = tmp_path / "user_databases" / str(TEST_USER_ID)
    user_base.mkdir(parents=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))

    db_path = user_base / "ChaChaNotes.db"
    chacha_db = CharactersRAGDB(db_path=db_path, client_id="auth_path_test_client")

    def override_chacha_db() -> CharactersRAGDB:
        return chacha_db

    monkeypatch.setattr(DatabasePaths, "get_user_base_directory", staticmethod(lambda uid: user_base))
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db

    try:
        clear_global_context_integrity_resolver()
        with TestClient(fastapi_app) as c:
            clear_global_context_integrity_resolver()
            yield c
    finally:
        clear_global_context_integrity_resolver()
        fastapi_app.dependency_overrides.clear()
        chacha_db.close_connection()


SAMPLE_SKILL = """---
name: test-skill
description: A test skill for API integration
argument-hint: "[text]"
context: inline
---

Process $ARGUMENTS with care.
"""


@contextmanager
def _capture_skills_endpoint_errors() -> Iterator[list[str]]:
    from tldw_Server_API.app.api.v1.endpoints import skills as skills_endpoint

    messages: list[str] = []
    sink_id = skills_endpoint.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        yield messages
    finally:
        skills_endpoint.logger.remove(sink_id)


class TestListSkills:
    def test_list_skills_empty(self, client):
        r = client.get(f"{SKILLS_PREFIX}/")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["skills"] == []
        assert data["total"] == 0

    def test_list_skills_reuses_service_for_cached_database(self, client, monkeypatch):
        """Repeated requests reuse debounce and maintenance state for one user DB."""
        constructor_calls = 0
        original_init = SkillsService.__init__

        def _counting_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            nonlocal constructor_calls
            constructor_calls += 1
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(SkillsService, "__init__", _counting_init)

        first = client.get(f"{SKILLS_PREFIX}/")
        second = client.get(f"{SKILLS_PREFIX}/")

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert constructor_calls == 1

    def test_cached_service_does_not_outlive_database_owner(self, tmp_path, monkeypatch):
        """Service reuse must not keep an evicted ChaCha database alive globally."""
        from tldw_Server_API.app.api.v1.endpoints import skills as skills_endpoint

        class StubDatabase:
            pass

        class StubService:
            def __init__(self, user_id, base_path, db):
                self.user_id = user_id
                self.base_path = base_path
                self.db = db

        clear_cache = getattr(skills_endpoint._get_cached_skills_service, "cache_clear", None)
        if clear_cache is not None:
            clear_cache()
        monkeypatch.setattr(skills_endpoint, "SkillsService", StubService)
        db = StubDatabase()
        service = skills_endpoint._get_cached_skills_service(42, tmp_path, db)
        assert skills_endpoint._get_cached_skills_service(42, tmp_path, db) is service
        db_ref = weakref.ref(db)
        service_ref = weakref.ref(service)

        del service
        del db
        gc.collect()
        try:
            assert db_ref() is None
            assert service_ref() is None
        finally:
            if clear_cache is not None:
                clear_cache()

    def test_list_skills_uses_current_principal_alias(self, principal_client):
        r = principal_client.get(f"{SKILLS_PREFIX}/")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["skills"] == []
        assert data["total"] == 0

    def test_list_skills_current_principal_accepts_single_user_api_key(
        self,
        auth_path_client,
        monkeypatch,
    ):
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        with monkeypatch.context() as env:
            env.setenv("AUTH_MODE", "single_user")
            env.setenv("SINGLE_USER_API_KEY", "phase34-skills-single-user-key")
            reset_settings()

            r = auth_path_client.get(
                f"{SKILLS_PREFIX}/",
                headers={"X-API-KEY": "phase34-skills-single-user-key"},
            )

        reset_settings()

        assert r.status_code == 200, r.text
        data = r.json()
        assert data["skills"] == []
        assert data["total"] == 0

    def test_list_skills_current_principal_accepts_jwt(
        self,
        auth_path_client,
        monkeypatch,
    ):
        from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as udh
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        async def fake_verify_jwt_and_fetch_user(request: Request, token: str) -> User:
            assert token == "jwt.header.signature"
            request.state.user_id = TEST_USER_ID
            request.state.org_ids = []
            request.state.team_ids = []
            return User(
                id=TEST_USER_ID,
                username="skills-jwt-user",
                roles=["user"],
                permissions=["skills.read"],
            )

        monkeypatch.setattr(udh, "verify_jwt_and_fetch_user", fake_verify_jwt_and_fetch_user)

        with monkeypatch.context() as env:
            env.setenv("AUTH_MODE", "multi_user")
            reset_settings()
            r = auth_path_client.get(
                f"{SKILLS_PREFIX}/",
                headers={"Authorization": "Bearer jwt.header.signature"},
            )

        reset_settings()

        assert r.status_code == 200, r.text
        data = r.json()
        assert data["skills"] == []
        assert data["total"] == 0

    def test_list_skills_current_principal_accepts_api_key(
        self,
        auth_path_client,
        monkeypatch,
    ):
        from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as udh
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        async def fake_authenticate_api_key_user(request: Request, api_key: str) -> User:
            assert api_key == "phase34-skills-api-key"
            request.state.user_id = TEST_USER_ID
            request.state.api_key_id = 321
            request.state.org_ids = []
            request.state.team_ids = []
            return User(
                id=TEST_USER_ID,
                username="skills-api-key-user",
                roles=["automation"],
                permissions=["skills.read"],
            )

        monkeypatch.setattr(udh, "authenticate_api_key_user", fake_authenticate_api_key_user)

        with monkeypatch.context() as env:
            env.setenv("AUTH_MODE", "multi_user")
            reset_settings()
            r = auth_path_client.get(
                f"{SKILLS_PREFIX}/",
                headers={"X-API-KEY": "phase34-skills-api-key"},
            )

        reset_settings()

        assert r.status_code == 200, r.text
        data = r.json()
        assert data["skills"] == []
        assert data["total"] == 0

    def test_list_skills_pagination(self, client):
        # Create 3 skills
        for i in range(3):
            r = client.post(
                f"{SKILLS_PREFIX}/",
                json={"name": f"skill-{i:02d}", "content": f"Content {i}"},
            )
            assert r.status_code == 201, r.text

        # Page 1
        r = client.get(f"{SKILLS_PREFIX}/?limit=2&offset=0")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert data["total"] == 3
        assert data["pagination"] == {
            "mode": "offset",
            "limit": 2,
            "offset": 0,
            "total": 3,
            "has_more": True,
            "next_offset": 2,
        }
        assert data["has_more"] is True
        assert data["next_offset"] == 2

        # Page 2
        r = client.get(f"{SKILLS_PREFIX}/?limit=2&offset=2")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["pagination"] == {
            "mode": "offset",
            "limit": 2,
            "offset": 2,
            "total": 3,
            "has_more": False,
            "next_offset": None,
        }
        assert data["has_more"] is False
        assert data["next_offset"] is None

    def test_list_skills_includes_version(self, client: TestClient) -> None:
        """List summaries expose the registry version for optimistic deletes."""
        create_resp = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "listed-version", "content": "content"},
        )
        assert create_resp.status_code == 201, create_resp.text

        r = client.get(f"{SKILLS_PREFIX}/?limit=50&offset=0")
        assert r.status_code == 200, r.text
        listed = {skill["name"]: skill for skill in r.json()["skills"]}
        assert listed["listed-version"]["version"] == create_resp.json()["version"]

    def test_metadata_summary_defaults_missing_version(self) -> None:
        """Legacy metadata without a version still produces a valid summary."""
        metadata = SimpleNamespace(
            name="legacy-version",
            description=None,
            argument_hint=None,
            user_invocable=True,
            disable_model_invocation=False,
            context="inline",
            version=None,
        )

        summary = _metadata_to_summary(metadata)

        assert summary.version == 1

    def test_metadata_summary_defaults_explicit_none_runtime_fields(self) -> None:
        """Legacy metadata with explicit null runtime fields still produces a valid summary."""
        metadata = SimpleNamespace(
            name="legacy-runtime",
            description=None,
            argument_hint=None,
            user_invocable=True,
            disable_model_invocation=None,
            allowed_tools=["Read"],
            model=None,
            context=None,
            version=1,
        )

        summary = _metadata_to_summary(metadata)

        assert summary.context == "inline"
        assert summary.disable_model_invocation is False
        assert summary.runtime.execution_mode == "inline"
        assert summary.runtime.declared_tool_count == 1

    def test_skill_response_defaults_explicit_none_runtime_fields(self) -> None:
        """Detail responses preserve schema defaults when legacy rows contain nulls."""
        now = datetime.now(timezone.utc)
        response = _skill_data_to_response(
            {
                "id": "legacy-runtime-id",
                "name": "legacy-runtime",
                "description": None,
                "argument_hint": None,
                "disable_model_invocation": None,
                "user_invocable": True,
                "allowed_tools": ["Read"],
                "model": None,
                "context": None,
                "content": "Body",
                "supporting_files": None,
                "directory_path": "/tmp/legacy-runtime",
                "created_at": now,
                "last_modified": now,
                "version": 1,
            }
        )

        assert response.context == "inline"
        assert response.disable_model_invocation is False
        assert response.runtime.execution_mode == "inline"
        assert response.runtime.declared_tool_count == 1

    def test_runtime_metadata_counts_single_tool_string_as_one(self) -> None:
        """Defensive runtime metadata treats one tool string as one declaration."""
        metadata = build_skill_runtime_metadata(
            context="fork",
            allowed_tools="Read",
            model=None,
            disable_model_invocation=False,
        )

        assert metadata["declares_tools"] is True
        assert metadata["declared_tool_count"] == 1

    def test_list_skills_search_filters_before_pagination(self, client):
        for i in range(12):
            r = client.post(
                f"{SKILLS_PREFIX}/",
                json={
                    "name": f"alpha-{i:02d}",
                    "content": "---\ndescription: General utility skill\n---\n\nCommon content",
                },
            )
            assert r.status_code == 201, r.text

        r = client.post(
            f"{SKILLS_PREFIX}/",
            json={
                "name": "omega-research",
                "content": (
                    "---\n"
                    "description: Needle workflow for longform research synthesis\n"
                    "---\n\n"
                    "Use this for longform synthesis."
                ),
            },
        )
        assert r.status_code == 201, r.text

        r = client.get(f"{SKILLS_PREFIX}/?q=needle&limit=5&offset=0")
        assert r.status_code == 200, r.text
        data = r.json()

        assert [skill["name"] for skill in data["skills"]] == ["omega-research"]
        assert data["count"] == 1
        assert data["total"] == 1
        assert data["pagination"] == {
            "mode": "offset",
            "limit": 5,
            "offset": 0,
            "total": 1,
            "has_more": False,
            "next_offset": None,
        }

    def test_list_skills_filters_and_sorts_before_pagination(self, client):
        for i in range(12):
            r = client.post(
                f"{SKILLS_PREFIX}/",
                json={
                    "name": f"alpha-{i:02d}",
                    "content": "---\ndescription: General utility skill\ncontext: inline\n---\n\nCommon content",
                },
            )
            assert r.status_code == 201, r.text

        for name, tool in (("beta-first", "Read"), ("beta-second", "Grep")):
            r = client.post(
                f"{SKILLS_PREFIX}/",
                json={
                    "name": name,
                    "content": (
                        "---\n"
                        "description: Forked tool skill\n"
                        "context: fork\n"
                        "allowed-tools:\n"
                        f"  - {tool}\n"
                        "model: gpt-4o\n"
                        "---\n\n"
                        "Use this with tools."
                    ),
                },
            )
            assert r.status_code == 201, r.text

        r = client.get(
            f"{SKILLS_PREFIX}/?context=fork&has_tools=true&model=gpt-4o"
            "&sort=name&order=desc&limit=1&offset=0"
        )
        assert r.status_code == 200, r.text
        data = r.json()

        assert [skill["name"] for skill in data["skills"]] == ["beta-second"]
        assert data["count"] == 1
        assert data["total"] == 2
        assert data["pagination"] == {
            "mode": "offset",
            "limit": 1,
            "offset": 0,
            "total": 2,
            "has_more": True,
            "next_offset": 1,
        }

    def test_list_and_detail_responses_include_runtime_metadata(self, client):
        r = client.post(
            f"{SKILLS_PREFIX}/",
            json={
                "name": "runtime-review",
                "content": (
                    "---\n"
                    "description: Runtime metadata review\n"
                    "context: fork\n"
                    "allowed-tools:\n"
                    "  - Read\n"
                    "  - \"Bash(git *)\"\n"
                    "model: gpt-4o\n"
                    "disable-model-invocation: true\n"
                    "---\n\n"
                    "Use this with runtime metadata."
                ),
            },
        )
        assert r.status_code == 201, r.text

        r = client.get(f"{SKILLS_PREFIX}/?q=runtime-review")
        assert r.status_code == 200, r.text
        list_body = r.json()
        assert list_body["count"] == 1
        summary = list_body["skills"][0]
        assert summary["allowed_tools"] == ["Read", "Bash(git *)"]
        assert summary["model"] == "gpt-4o"
        assert summary["runtime"] == {
            "execution_mode": "fork",
            "test_run_may_call_model": True,
            "declares_tools": True,
            "declared_tool_count": 2,
            "model_override": "gpt-4o",
            "auto_invocation_enabled": False,
        }

        r = client.get(f"{SKILLS_PREFIX}/runtime-review")
        assert r.status_code == 200, r.text
        detail = r.json()
        assert detail["allowed_tools"] == ["Read", "Bash(git *)"]
        assert detail["model"] == "gpt-4o"
        assert detail["runtime"] == summary["runtime"]

    def test_list_skills_explicit_hidden_filter(self, client):
        for name, user_invocable in (("visible", "true"), ("hidden", "false")):
            r = client.post(
                f"{SKILLS_PREFIX}/",
                json={
                    "name": name,
                    "content": f"---\nuser-invocable: {user_invocable}\n---\n\nContent",
                },
            )
            assert r.status_code == 201, r.text

        r = client.get(f"{SKILLS_PREFIX}/?user_invocable=false")
        assert r.status_code == 200, r.text
        data = r.json()

        assert [skill["name"] for skill in data["skills"]] == ["hidden"]
        assert data["total"] == 1

    def test_list_skills_rejects_unapproved_sort_field(self, client):
        r = client.get(f"{SKILLS_PREFIX}/?sort=directory_path")

        assert r.status_code == 422, r.text


class TestCreateAndGetSkill:
    def test_create_skill_and_get(self, client):
        source = (
            SAMPLE_SKILL.replace("name: test-skill", "name: new-skill")
            .replace(
                "context: inline",
                "custom-review-key: preserve-me\ncontext: inline",
            )
        )
        r = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "new-skill", "content": source},
        )
        assert r.status_code == 201, r.text
        created = r.json()
        assert created["name"] == "new-skill"
        assert created["description"] == "A test skill for API integration"
        assert created["version"] == 1
        assert created["raw_content"] == source

        # Get it back
        r = client.get(f"{SKILLS_PREFIX}/new-skill")
        assert r.status_code == 200
        got = r.json()
        assert got["name"] == "new-skill"
        assert "Process $ARGUMENTS" in got["content"]
        assert got["raw_content"] == source
        assert "custom-review-key: preserve-me" in got["raw_content"]

    def test_create_skill_invalid_name_400(self, client):
        r = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "Invalid_Name!", "content": "content"},
        )
        assert r.status_code == 422  # Pydantic validation error

    def test_create_skill_duplicate_409(self, client):
        client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "dup-skill", "content": "content"},
        )
        r = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "dup-skill", "content": "content again"},
        )
        assert r.status_code == 409

    def test_create_skill_sanitizes_skills_error(self, client, monkeypatch):
        async def _boom(self, *, name, content, supporting_files=None):  # noqa: ANN001, ANN202
            raise SkillsError("skills backend exploded at /private/create")

        monkeypatch.setattr(SkillsService, "create_skill", _boom)

        with _capture_skills_endpoint_errors() as messages:
            r = client.post(
                f"{SKILLS_PREFIX}/",
                json={"name": "new-skill", "content": SAMPLE_SKILL},
            )

        joined = "\n".join(messages)
        assert r.status_code == 500
        assert r.json()["detail"] == "Failed to create skill"
        assert "Error creating skill" in joined
        assert "skills backend exploded" not in joined
        assert "/private/" not in joined


class TestUpdateSkill:
    def test_update_skill_content(self, client):
        client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "upd-skill", "content": "original"},
        )

        r = client.put(
            f"{SKILLS_PREFIX}/upd-skill",
            json={"content": "---\ndescription: Updated\n---\nNew content"},
        )
        assert r.status_code == 200, r.text
        updated = r.json()
        assert updated["description"] == "Updated"
        assert updated["version"] == 2

    def test_update_skill_accepts_quoted_if_match(self, client: TestClient) -> None:
        """Standard quoted entity tags are accepted for optimistic updates."""
        created = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "quoted-update", "content": "v1"},
        )
        assert created.status_code == 201, created.text

        updated = client.put(
            f"{SKILLS_PREFIX}/quoted-update",
            json={"content": "v2"},
            headers={"If-Match": f'"{created.json()["version"]}"'},
        )

        assert updated.status_code == 200, updated.text
        assert updated.json()["version"] == created.json()["version"] + 1

    def test_update_skill_version_conflict_409(self, client):
        client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "ver-skill", "content": "v1"},
        )
        # Update to v2
        client.put(
            f"{SKILLS_PREFIX}/ver-skill",
            json={"content": "v2"},
        )
        # Try with stale version
        r = client.put(
            f"{SKILLS_PREFIX}/ver-skill",
            json={"content": "v3"},
            headers={"If-Match": "1"},
        )
        assert r.status_code == 409

    def test_update_skill_supporting_file_delete_with_null(self, client):
        create_resp = client.post(
            f"{SKILLS_PREFIX}/",
            json={
                "name": "upd-files",
                "content": "content",
                "supporting_files": {"remove.md": "to remove", "keep.md": "to keep"},
            },
        )
        assert create_resp.status_code == 201, create_resp.text

        r = client.put(
            f"{SKILLS_PREFIX}/upd-files",
            json={"supporting_files": {"remove.md": None}},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["supporting_files"] is not None
        assert "remove.md" not in data["supporting_files"]
        assert data["supporting_files"]["keep.md"] == "to keep"

    def test_update_skill_sanitizes_skills_error(self, client, monkeypatch):
        async def _boom(
            self, *, name, content=None, supporting_files=None, expected_version=None
        ):  # noqa: ANN001, ANN202
            raise SkillsError("skills backend exploded at /private/update")

        monkeypatch.setattr(SkillsService, "update_skill", _boom)

        with _capture_skills_endpoint_errors() as messages:
            r = client.put(
                f"{SKILLS_PREFIX}/upd-skill",
                json={"content": "updated"},
            )

        joined = "\n".join(messages)
        assert r.status_code == 500
        assert r.json()["detail"] == "Failed to update skill"
        assert "Error updating skill" in joined
        assert "skills backend exploded" not in joined
        assert "/private/" not in joined


class TestDeleteSkill:
    def test_delete_skill_204(self, client):
        client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "del-skill", "content": "content"},
        )
        r = client.delete(f"{SKILLS_PREFIX}/del-skill")
        assert r.status_code == 204

        r = client.get(f"{SKILLS_PREFIX}/del-skill")
        assert r.status_code == 404

    def test_delete_skill_accepts_matching_if_match(self, client: TestClient) -> None:
        """Deletes with a matching If-Match version remove the skill."""
        create_resp = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "del-versioned", "content": "content"},
        )
        assert create_resp.status_code == 201, create_resp.text
        version = create_resp.json()["version"]

        r = client.delete(
            f"{SKILLS_PREFIX}/del-versioned",
            headers={"If-Match": f'"{version}"'},
        )
        assert r.status_code == 204, r.text

        missing = client.get(f"{SKILLS_PREFIX}/del-versioned")
        assert missing.status_code == 404

    def test_delete_skill_stale_if_match_returns_409_and_keeps_skill(
        self,
        client: TestClient,
    ) -> None:
        """Stale If-Match deletes return 409 without deleting the current skill."""
        create_resp = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "del-stale", "content": "v1"},
        )
        assert create_resp.status_code == 201, create_resp.text

        update_resp = client.put(
            f"{SKILLS_PREFIX}/del-stale",
            json={"content": "v2"},
        )
        assert update_resp.status_code == 200, update_resp.text
        assert update_resp.json()["version"] == create_resp.json()["version"] + 1

        r = client.delete(
            f"{SKILLS_PREFIX}/del-stale",
            headers={"If-Match": str(create_resp.json()["version"])},
        )
        assert r.status_code == 409

        still_there = client.get(f"{SKILLS_PREFIX}/del-stale")
        assert still_there.status_code == 200, still_there.text
        assert still_there.json()["version"] == update_resp.json()["version"]

    def test_delete_skill_rejects_malformed_if_match_with_400(self, client: TestClient) -> None:
        """Malformed or weak entity tags fail as client input, not schema errors."""
        created = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "invalid-etag", "content": "content"},
        )
        assert created.status_code == 201, created.text

        response = client.delete(
            f"{SKILLS_PREFIX}/invalid-etag",
            headers={"If-Match": 'W/"1"'},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "If-Match must be a numeric skill version"

    def test_delete_skill_rejects_oversized_if_match_with_400(self, client: TestClient) -> None:
        """Unbounded numeric entity tags fail as input instead of integer conversion."""
        response = client.delete(
            f"{SKILLS_PREFIX}/missing-skill",
            headers={"If-Match": "9" * 5000},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "If-Match must be a numeric skill version"

    def test_delete_skill_not_found_404(self, client):
        r = client.delete(f"{SKILLS_PREFIX}/nonexistent")
        assert r.status_code == 404

    def test_delete_skill_sanitizes_skills_error(self, client, monkeypatch):
        async def _boom(self, name, *, expected_version=None):  # noqa: ANN001, ANN202
            raise SkillsError("skills backend exploded at /private/delete")

        monkeypatch.setattr(SkillsService, "delete_skill", _boom)

        with _capture_skills_endpoint_errors() as messages:
            r = client.delete(f"{SKILLS_PREFIX}/del-skill")

        joined = "\n".join(messages)
        assert r.status_code == 500
        assert r.json()["detail"] == "Failed to delete skill"
        assert "Error deleting skill" in joined
        assert "skills backend exploded" not in joined
        assert "/private/" not in joined


class TestBulkDeleteSkills:
    def test_bulk_delete_accepts_matching_versions(self, client: TestClient) -> None:
        """Bulk delete removes all selected skills when versions match."""
        first = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "bulk-delete-a", "content": "content a"},
        )
        second = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "bulk-delete-b", "content": "content b"},
        )
        assert first.status_code == 201, first.text
        assert second.status_code == 201, second.text

        r = client.post(
            f"{SKILLS_PREFIX}/bulk-delete",
            json={
                "skills": [
                    {"name": "bulk-delete-a", "version": first.json()["version"]},
                    {"name": "bulk-delete-b", "version": second.json()["version"]},
                ],
            },
        )

        assert r.status_code == 200, r.text
        assert r.json() == {
            "deleted": ["bulk-delete-a", "bulk-delete-b"],
            "count": 2,
        }
        assert client.get(f"{SKILLS_PREFIX}/bulk-delete-a").status_code == 404
        assert client.get(f"{SKILLS_PREFIX}/bulk-delete-b").status_code == 404

    def test_bulk_delete_keeps_unknown_version_compatible(
        self,
        client: TestClient,
    ) -> None:
        """Bulk delete remains compatible for legacy rows without known versions."""
        created = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "bulk-delete-legacy", "content": "content"},
        )
        assert created.status_code == 201, created.text

        r = client.post(
            f"{SKILLS_PREFIX}/bulk-delete",
            json={"skills": [{"name": "bulk-delete-legacy"}]},
        )

        assert r.status_code == 200, r.text
        assert r.json()["deleted"] == ["bulk-delete-legacy"]
        assert client.get(f"{SKILLS_PREFIX}/bulk-delete-legacy").status_code == 404

    def test_bulk_delete_stale_version_returns_409_without_partial_delete(
        self,
        client: TestClient,
    ) -> None:
        """A stale bulk delete conflicts before deleting any selected skill."""
        stale = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "bulk-stale", "content": "v1"},
        )
        fresh = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "bulk-fresh", "content": "content"},
        )
        assert stale.status_code == 201, stale.text
        assert fresh.status_code == 201, fresh.text

        updated = client.put(
            f"{SKILLS_PREFIX}/bulk-stale",
            json={"content": "v2"},
        )
        assert updated.status_code == 200, updated.text

        r = client.post(
            f"{SKILLS_PREFIX}/bulk-delete",
            json={
                "skills": [
                    {"name": "bulk-stale", "version": stale.json()["version"]},
                    {"name": "bulk-fresh", "version": fresh.json()["version"]},
                ],
            },
        )

        assert r.status_code == 409
        assert client.get(f"{SKILLS_PREFIX}/bulk-stale").status_code == 200
        assert client.get(f"{SKILLS_PREFIX}/bulk-fresh").status_code == 200

    def test_bulk_delete_sanitized_error_log_has_request_context(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bulk delete 500 logs should include request context without leaking details."""

        async def _boom(self, items):  # noqa: ANN001, ANN202
            raise SkillsError("bulk backend exploded at /private/bulk-delete")

        monkeypatch.setattr(SkillsService, "bulk_delete_skills", _boom)

        with _capture_skills_endpoint_errors() as messages:
            r = client.post(
                f"{SKILLS_PREFIX}/bulk-delete",
                json={"skills": [{"name": "bulk-log", "version": 1}]},
            )

        joined = "\n".join(messages)
        assert r.status_code == 500
        assert r.json()["detail"] == "Failed to bulk delete skills"
        assert "Error bulk deleting skills" in joined
        assert "selected_count=1" in joined
        assert f"user_id={TEST_USER_ID}" in joined
        assert "bulk backend exploded" not in joined
        assert "/private/" not in joined


class TestSkillTrash:
    def test_delete_list_restore_and_purge_roundtrip(self, client: TestClient) -> None:
        """Trash endpoints expose durable state and preserve optimistic versions."""
        created = client.post(
            f"{SKILLS_PREFIX}/",
            json={
                "name": "trash-api",
                "content": "---\ndescription: Trash API\n---\nBody",
                "supporting_files": {"notes.md": "keep"},
            },
        )
        assert created.status_code == 201, created.text

        deleted = client.delete(
            f"{SKILLS_PREFIX}/trash-api",
            headers={"If-Match": str(created.json()["version"])},
        )
        assert deleted.status_code == 204, deleted.text

        trash = client.get(f"{SKILLS_PREFIX}/trash")
        assert trash.status_code == 200, trash.text
        assert trash.json()["total"] == 1
        item = trash.json()["skills"][0]
        assert item["name"] == "trash-api"
        assert item["restorable"] is True
        assert item["version"] == created.json()["version"] + 1

        restored = client.post(
            f"{SKILLS_PREFIX}/trash-api/restore",
            headers={"If-Match": f'"{item["version"]}"'},
        )
        assert restored.status_code == 200, restored.text
        assert restored.json()["supporting_files"] == {"notes.md": "keep"}
        assert client.get(f"{SKILLS_PREFIX}/trash").json()["total"] == 0

        deleted_again = client.delete(
            f"{SKILLS_PREFIX}/trash-api",
            headers={"If-Match": str(restored.json()["version"])},
        )
        assert deleted_again.status_code == 204, deleted_again.text
        trashed_again = client.get(f"{SKILLS_PREFIX}/trash").json()["skills"][0]

        purged = client.delete(
            f"{SKILLS_PREFIX}/trash-api/purge",
            headers={"If-Match": f'"{trashed_again["version"]}"'},
        )
        assert purged.status_code == 204, purged.text
        assert client.get(f"{SKILLS_PREFIX}/trash").json()["total"] == 0

    def test_restore_and_purge_reject_stale_versions(self, client: TestClient) -> None:
        created = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "trash-stale", "content": "Body"},
        )
        client.delete(
            f"{SKILLS_PREFIX}/trash-stale",
            headers={"If-Match": str(created.json()["version"])},
        )

        restore = client.post(
            f"{SKILLS_PREFIX}/trash-stale/restore",
            headers={"If-Match": str(created.json()["version"])},
        )
        purge = client.delete(
            f"{SKILLS_PREFIX}/trash-stale/purge",
            headers={"If-Match": str(created.json()["version"])},
        )

        assert restore.status_code == 409
        assert purge.status_code == 409

    def test_trash_list_error_log_has_sanitized_request_context(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _boom(self, *, limit=100, offset=0):  # noqa: ANN001, ANN202
            raise SkillsError("trash backend exploded at /private/trash.db")

        monkeypatch.setattr(SkillsService, "list_trash", _boom)

        with _capture_skills_endpoint_errors() as messages:
            response = client.get(f"{SKILLS_PREFIX}/trash?limit=7&offset=3")

        joined = "\n".join(messages)
        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to list Skills Trash"
        assert "Error listing Skills Trash" in joined
        assert f"user_id={TEST_USER_ID}" in joined
        assert "limit=7" in joined
        assert "offset=3" in joined
        assert "error_type=SkillsError" in joined
        assert "trash backend exploded" not in joined
        assert "/private/" not in joined

    @pytest.mark.parametrize(
        ("method", "action", "service_method", "log_message", "detail"),
        [
            (
                "POST",
                "restore",
                "restore_skill",
                "Error restoring skill from Trash",
                "Failed to restore skill",
            ),
            (
                "DELETE",
                "purge",
                "purge_skill",
                "Error permanently deleting skill from Trash",
                "Failed to permanently delete skill",
            ),
        ],
    )
    def test_trash_mutation_error_logs_have_sanitized_request_context(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        action: str,
        service_method: str,
        log_message: str,
        detail: str,
    ) -> None:
        async def _boom(self, name, expected_version=None):  # noqa: ANN001, ANN202
            raise SkillsError("trash backend exploded at /private/trash.db")

        monkeypatch.setattr(SkillsService, service_method, _boom)

        with _capture_skills_endpoint_errors() as messages:
            response = client.request(
                method,
                f"{SKILLS_PREFIX}/trash-log/{action}",
                headers={"If-Match": '"3"'},
            )

        joined = "\n".join(messages)
        assert response.status_code == 500
        assert response.json()["detail"] == detail
        assert log_message in joined
        assert f"user_id={TEST_USER_ID}" in joined
        assert "skill_name=trash-log" in joined
        assert "expected_version=3" in joined
        assert "error_type=SkillsError" in joined
        assert "trash backend exploded" not in joined
        assert "/private/" not in joined

    @pytest.mark.parametrize(
        ("method", "action"),
        [("POST", "restore"), ("DELETE", "purge")],
    )
    def test_restore_and_purge_reject_invalid_skill_names(
        self,
        client: TestClient,
        method: str,
        action: str,
    ) -> None:
        response = client.request(method, f"{SKILLS_PREFIX}/INVALID!/{action}")

        assert response.status_code == 400
        assert "Invalid skill name" in response.json()["detail"]


class TestImportExport:
    def test_import_skill_preview_json_returns_metadata_and_conflict(self, client):
        client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "preview-conflict", "content": "Original"},
        )

        r = client.post(
            f"{SKILLS_PREFIX}/import/preview",
            json={
                "content": (
                    "---\n"
                    "name: preview-conflict\n"
                    "description: Reviewed import\n"
                    "argument-hint: \"[topic]\"\n"
                    "allowed-tools: Read, Grep\n"
                    "context: fork\n"
                    "---\n"
                    "Preview content"
                ),
                "supporting_files": {"ref.md": "Reference"},
            },
        )

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["valid"] is True
        assert body["errors"] == []
        assert body["name"] == "preview-conflict"
        assert body["description"] == "Reviewed import"
        assert body["argument_hint"] == "[topic]"
        assert body["allowed_tools"] == ["Read", "Grep"]
        assert body["context"] == "fork"
        assert body["supporting_file_count"] == 1
        assert body["conflict"] is True
        assert body["can_overwrite"] is True
        assert body["existing_version"] == 1
        persisted = client.get(f"{SKILLS_PREFIX}/preview-conflict")
        assert persisted.status_code == 200
        assert persisted.json()["content"] == "Original"

    def test_text_import_rejects_stale_preview_version(self, client):
        created = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "stale-text", "content": "Original"},
        )
        assert created.status_code == 201, created.text
        preview = client.post(
            f"{SKILLS_PREFIX}/import/preview",
            json={"name": "stale-text", "content": "Replacement"},
        )
        assert preview.status_code == 200, preview.text
        preview_version = preview.json()["existing_version"]
        updated = client.put(
            f"{SKILLS_PREFIX}/stale-text",
            json={"content": "Concurrent edit"},
            headers={"If-Match": str(preview_version)},
        )
        assert updated.status_code == 200, updated.text

        response = client.post(
            f"{SKILLS_PREFIX}/import",
            json={
                "name": "stale-text",
                "content": "Replacement",
                "overwrite": True,
                "expected_version": preview_version,
            },
        )

        assert response.status_code == 409
        persisted = client.get(f"{SKILLS_PREFIX}/stale-text")
        assert persisted.status_code == 200
        assert persisted.json()["content"] == "Concurrent edit"

    def test_file_import_rejects_stale_preview_version(self, client):
        created = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "stale-file", "content": "Original"},
        )
        assert created.status_code == 201, created.text
        preview = client.post(
            f"{SKILLS_PREFIX}/import/file/preview",
            files={"file": ("stale-file.md", b"Replacement", "text/markdown")},
        )
        assert preview.status_code == 200, preview.text
        preview_version = preview.json()["existing_version"]
        updated = client.put(
            f"{SKILLS_PREFIX}/stale-file",
            json={"content": "Concurrent edit"},
            headers={"If-Match": str(preview_version)},
        )
        assert updated.status_code == 200, updated.text

        response = client.post(
            f"{SKILLS_PREFIX}/import/file",
            params={"overwrite": "true", "expected_version": preview_version},
            files={"file": ("stale-file.md", b"Replacement", "text/markdown")},
        )

        assert response.status_code == 409
        persisted = client.get(f"{SKILLS_PREFIX}/stale-file")
        assert persisted.status_code == 200
        assert persisted.json()["content"] == "Concurrent edit"

    def test_import_skill_preview_invalid_content_returns_review_error(self, client):
        r = client.post(
            f"{SKILLS_PREFIX}/import/preview",
            json={
                "content": "---\nname: Invalid_Name!\n---\nInvalid content",
            },
        )

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["valid"] is False
        assert body["name"] is None
        assert body["conflict"] is False
        assert body["can_overwrite"] is False
        assert body["existing_version"] is None
        assert any("frontmatter skill name" in error for error in body["errors"])
        created = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "invalid-name", "content": "Created after invalid preview"},
        )
        assert created.status_code == 201, created.text
        assert created.json()["version"] == 1

    def test_import_skill_file_preview_md_returns_metadata_without_importing(self, client, tmp_path):
        skill_file = tmp_path / "review-file-skill.md"
        skill_file.write_text("---\ndescription: From preview file\n---\nFile preview content")

        with open(skill_file, "rb") as f:
            r = client.post(
                f"{SKILLS_PREFIX}/import/file/preview",
                files={"file": ("review-file-skill.md", f, "text/markdown")},
            )

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["valid"] is True
        assert body["errors"] == []
        assert body["name"] == "review-file-skill"
        assert body["description"] == "From preview file"
        assert body["conflict"] is False
        assert body["can_overwrite"] is False
        imported = client.post(
            f"{SKILLS_PREFIX}/import",
            json={
                "name": "review-file-skill",
                "content": "---\ndescription: From preview file\n---\nFile preview content",
            },
        )
        assert imported.status_code == 201, imported.text
        assert imported.json()["version"] == 1

    def test_import_skill_file_preview_zip_sniffs_content_without_zip_filename(self, client):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(
                "sniffed-skill/SKILL.md",
                "---\nname: sniffed-skill\ndescription: Sniffed zip\n---\nZip preview content",
            )

        r = client.post(
            f"{SKILLS_PREFIX}/import/file/preview",
            files={"file": ("skill.bundle", buffer.getvalue(), "application/octet-stream")},
        )

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["valid"] is True
        assert body["name"] == "sniffed-skill"
        assert body["description"] == "Sniffed zip"
        imported = client.post(
            f"{SKILLS_PREFIX}/import",
            json={
                "name": "sniffed-skill",
                "content": "---\ndescription: Sniffed zip\n---\nZip preview content",
            },
        )
        assert imported.status_code == 201, imported.text
        assert imported.json()["version"] == 1

    def test_import_skill_file_zip_sniffs_content_without_zip_filename(self, client):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(
                "sniffed-import/SKILL.md",
                "---\nname: sniffed-import\ndescription: Sniffed import\n---\nZip import content",
            )

        response = client.post(
            f"{SKILLS_PREFIX}/import/file",
            files={
                "file": (
                    "skill.bundle",
                    buffer.getvalue(),
                    "application/octet-stream",
                )
            },
        )

        assert response.status_code == 201, response.text
        assert response.json()["name"] == "sniffed-import"
        persisted = client.get(f"{SKILLS_PREFIX}/sniffed-import")
        assert persisted.status_code == 200
        assert persisted.json()["description"] == "Sniffed import"

    def test_import_skill_file_zip_non_utf8_skill_md_returns_400(self, client):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("bad-encoding/SKILL.md", b"\xff\xfe\xfd\xfc")
        zip_data = buffer.getvalue()

        for path in ("import/file/preview", "import/file"):
            r = client.post(
                f"{SKILLS_PREFIX}/{path}",
                files={"file": ("bad-encoding.zip", zip_data, "application/zip")},
            )

            assert r.status_code == 400, r.text
            assert "UTF-8" in r.json()["detail"]

    def test_import_skill_file_preview_rejects_oversized_upload(self, client):
        too_large = b"a" * (MAX_SKILL_IMPORT_PREVIEW_UPLOAD_BYTES + 1)

        r = client.post(
            f"{SKILLS_PREFIX}/import/file/preview",
            files={"file": ("too-large.md", too_large, "text/markdown")},
        )

        assert r.status_code == 413, r.text
        assert "exceeds" in r.json()["detail"]

    def test_import_skill_file_rejects_oversized_upload(self, client):
        """The mutating import path must enforce the same upload limit as preview."""
        too_large = b"a" * (MAX_SKILL_IMPORT_PREVIEW_UPLOAD_BYTES + 1)

        r = client.post(
            f"{SKILLS_PREFIX}/import/file",
            files={"file": ("too-large.md", too_large, "text/markdown")},
        )

        assert r.status_code == 413, r.text
        assert "exceeds" in r.json()["detail"]

    def test_import_skill_file_preview_rejects_non_utf8_non_zip_upload(self, client):
        r = client.post(
            f"{SKILLS_PREFIX}/import/file/preview",
            files={"file": ("payload.bin", b"\xff\xfe\xfd\xfc", "application/octet-stream")},
        )

        assert r.status_code == 400, r.text
        assert "UTF-8" in r.json()["detail"]

    def test_import_skill_json(self, client):
        r = client.post(
            f"{SKILLS_PREFIX}/import",
            json={
                "name": "imported",
                "content": "---\ndescription: Imported\n---\nImported content",
                "overwrite": False,
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["name"] == "imported"

    def test_import_skill_json_without_name_uses_frontmatter(self, client):
        r = client.post(
            f"{SKILLS_PREFIX}/import",
            json={
                "content": "---\nname: from-frontmatter\ndescription: Imported\n---\nImported content",
                "overwrite": False,
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["name"] == "from-frontmatter"

    def test_import_skill_invalid_frontmatter_name_400(self, client):
        r = client.post(
            f"{SKILLS_PREFIX}/import",
            json={
                "name": "safe-name",
                "content": "---\nname: Invalid_Name!\n---\nImported content",
                "overwrite": False,
            },
        )
        assert r.status_code == 400

    def test_import_skill_overwrite(self, client):
        client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "overwrite-me", "content": "original"},
        )
        r = client.post(
            f"{SKILLS_PREFIX}/import",
            json={
                "name": "overwrite-me",
                "content": "---\ndescription: Overwritten\n---\nNew",
                "overwrite": True,
            },
        )
        assert r.status_code == 201
        assert r.json()["description"] == "Overwritten"

    def test_import_skill_sanitizes_skills_error(self, client, monkeypatch):
        async def _boom(  # noqa: ANN001, ANN202
            self,
            *,
            content,
            name=None,
            supporting_files=None,
            overwrite=False,
            expected_version=None,
        ):
            raise SkillsError("skills backend exploded at /private/import")

        monkeypatch.setattr(SkillsService, "import_skill", _boom)

        with _capture_skills_endpoint_errors() as messages:
            r = client.post(
                f"{SKILLS_PREFIX}/import",
                json={
                    "name": "imported",
                    "content": "---\ndescription: Imported\n---\nImported content",
                    "overwrite": False,
                },
            )

        joined = "\n".join(messages)
        assert r.status_code == 500
        assert r.json()["detail"] == "Failed to import skill"
        assert "Error importing skill" in joined
        assert "skills backend exploded" not in joined
        assert "/private/" not in joined

    def test_import_skill_json_supporting_files_count_limit_422(self, client):
        files = {f"file{i:02d}.md": "content" for i in range(25)}
        r = client.post(
            f"{SKILLS_PREFIX}/import",
            json={
                "content": "---\nname: too-many-files\n---\nBody",
                "supporting_files": files,
                "overwrite": False,
            },
        )
        assert r.status_code == 422

    def test_import_skill_json_supporting_files_aggregate_limit_422(self, client):
        big_content = "x" * 400_000
        files = {f"file{i:02d}.md": big_content for i in range(15)}
        r = client.post(
            f"{SKILLS_PREFIX}/import",
            json={
                "content": "---\nname: too-big-files\n---\nBody",
                "supporting_files": files,
                "overwrite": False,
            },
        )
        assert r.status_code == 422

    def test_import_skill_file_md(self, client, tmp_path):
        skill_file = tmp_path / "my-file-skill.md"
        skill_file.write_text("---\ndescription: From file\n---\nFile content")

        with open(skill_file, "rb") as f:
            r = client.post(
                f"{SKILLS_PREFIX}/import/file",
                files={"file": ("my-file-skill.md", f, "text/markdown")},
            )
        assert r.status_code == 201, r.text

    def test_import_skill_file_sanitizes_skills_error(self, client, monkeypatch):
        async def _boom(  # noqa: ANN001, ANN202
            self,
            *,
            content,
            name=None,
            supporting_files=None,
            overwrite=False,
            expected_version=None,
        ):
            raise SkillsError("skills backend exploded at /private/import-file")

        monkeypatch.setattr(SkillsService, "import_skill", _boom)

        with _capture_skills_endpoint_errors() as messages:
            r = client.post(
                f"{SKILLS_PREFIX}/import/file",
                files={"file": ("skill.md", SAMPLE_SKILL, "text/markdown")},
            )

        joined = "\n".join(messages)
        assert r.status_code == 500
        assert r.json()["detail"] == "Failed to import skill from file"
        assert "Error importing skill from file" in joined
        assert "skills backend exploded" not in joined
        assert "/private/" not in joined

    def test_import_skill_file_invalid_frontmatter_name_400(self, client, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("---\nname: Invalid_Name!\n---\nFile content")

        with open(skill_file, "rb") as f:
            r = client.post(
                f"{SKILLS_PREFIX}/import/file",
                files={"file": ("SKILL.md", f, "text/markdown")},
            )
        assert r.status_code == 400

    def test_import_skill_file_zip(self, client, tmp_path):
        import zipfile
        from io import BytesIO

        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("zip-skill/SKILL.md", "---\ndescription: Zipped\n---\nZip content")
            zf.writestr("zip-skill/ref.md", "reference data")
        buf.seek(0)

        r = client.post(
            f"{SKILLS_PREFIX}/import/file",
            files={"file": ("skill.zip", buf, "application/zip")},
        )
        assert r.status_code == 201, r.text
        assert r.json()["name"] == "zip-skill"

    def test_import_skill_file_zip_path_traversal_400(self, client):
        import zipfile
        from io import BytesIO

        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("safe-skill/SKILL.md", "---\nname: safe-skill\n---\nZip content")
            zf.writestr("safe-skill/../escape.md", "escape")
        buf.seek(0)

        r = client.post(
            f"{SKILLS_PREFIX}/import/file",
            files={"file": ("skill.zip", buf, "application/zip")},
        )
        assert r.status_code == 400

    def test_import_skill_file_zip_supporting_files_count_limit_400(self, client):
        import zipfile
        from io import BytesIO

        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("zip-limit-count/SKILL.md", "---\nname: zip-limit-count\n---\nZip content")
            for i in range(21):
                zf.writestr(f"zip-limit-count/file{i:02d}.md", "tiny")
        buf.seek(0)

        r = client.post(
            f"{SKILLS_PREFIX}/import/file",
            files={"file": ("skill.zip", buf, "application/zip")},
        )
        assert r.status_code == 400

    def test_import_skill_file_zip_supporting_files_aggregate_limit_400(self, client):
        import zipfile
        from io import BytesIO

        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("zip-limit-size/SKILL.md", "---\nname: zip-limit-size\n---\nZip content")
            big_content = "x" * 400_000
            for i in range(15):
                zf.writestr(f"zip-limit-size/file{i:02d}.md", big_content)
        buf.seek(0)

        r = client.post(
            f"{SKILLS_PREFIX}/import/file",
            files={"file": ("skill.zip", buf, "application/zip")},
        )
        assert r.status_code == 400

    def test_export_skill_zip(self, client):
        create_response = client.post(
            f"{SKILLS_PREFIX}/",
            json={
                "name": "export-skill",
                "content": SAMPLE_SKILL.replace("name: test-skill", "name: export-skill"),
            },
        )
        assert create_response.status_code == 201, create_response.text
        r = client.get(f"{SKILLS_PREFIX}/export-skill/export")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/zip"
        # Zip magic bytes
        assert r.content[:2] == b"PK"

    def test_export_skill_not_found_404(self, client):
        r = client.get(f"{SKILLS_PREFIX}/no-such-skill/export")
        assert r.status_code == 404

    def test_export_skill_sanitizes_skills_error(self, client, monkeypatch):
        async def _boom(self, name):  # noqa: ANN001, ANN202
            raise SkillsError("skills backend exploded at /private/export")

        monkeypatch.setattr(SkillsService, "export_skill", _boom)

        with _capture_skills_endpoint_errors() as messages:
            r = client.get(f"{SKILLS_PREFIX}/export-skill/export")

        joined = "\n".join(messages)
        assert r.status_code == 500
        assert r.json()["detail"] == "Failed to export skill"
        assert "Error exporting skill" in joined
        assert "skills backend exploded" not in joined
        assert "/private/" not in joined

    def test_get_quarantined_skill_returns_423(self, client, monkeypatch):
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityBlocked,
        )

        async def _blocked(self, name, *args, **kwargs):  # noqa: ANN001, ANN202
            raise ContextIntegrityBlocked(
                asset_id=f"skill:user:1/{name}",
                state="changed_approved_executable",
            )

        monkeypatch.setattr(SkillsService, "get_skill", _blocked)

        r = client.get(f"{SKILLS_PREFIX}/blocked-skill")

        assert r.status_code == 423
        assert r.json()["detail"] == "Asset is quarantined pending admin review."

    def test_export_quarantined_skill_returns_423(self, client, monkeypatch):
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityBlocked,
        )

        async def _blocked(self, name, *args, **kwargs):  # noqa: ANN001, ANN202
            raise ContextIntegrityBlocked(
                asset_id=f"skill:user:1/{name}",
                state="changed_approved_executable",
            )

        monkeypatch.setattr(SkillsService, "export_skill", _blocked)

        r = client.get(f"{SKILLS_PREFIX}/blocked-skill/export")

        assert r.status_code == 423
        assert r.json()["detail"] == "Asset is quarantined pending admin review."


class TestExecuteSkill:
    def test_execute_skill_inline(self, client):
        client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "exec-skill", "content": "Do: $ARGUMENTS"},
        )
        r = client.post(
            f"{SKILLS_PREFIX}/exec-skill/execute",
            json={"args": "my test args"},
        )
        assert r.status_code == 200, r.text
        result = r.json()
        assert result["skill_name"] == "exec-skill"
        assert "my test args" in result["rendered_prompt"]
        assert result["execution_mode"] == "inline"
        assert result["dry_run"] is False

    def test_execute_skill_forwards_and_returns_dry_run(self, client, monkeypatch):
        """Verify the API forwards dry_run to the executor and returns it to clients."""
        client.post(
            f"{SKILLS_PREFIX}/",
            json={
                "name": "dry-run-exec-skill",
                "content": "---\ncontext: fork\n---\nDo: $ARGUMENTS",
            },
        )
        observed = {}

        async def fake_execute(
            self: object,
            *,
            skill_data: dict[str, Any],
            arguments: str,
            context: object | None,
            dry_run: bool,
        ) -> SimpleNamespace:
            """Capture endpoint-to-executor dry-run arguments without executing a skill."""
            observed["dry_run"] = dry_run
            observed["arguments"] = arguments
            return SimpleNamespace(
                skill_name=skill_data["name"],
                rendered_prompt=f"rendered {arguments}",
                allowed_tools=[],
                model_override=None,
                execution_mode="fork",
                fork_output=None,
                dry_run=dry_run,
            )

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.skills.SkillExecutor.execute",
            fake_execute,
        )

        r = client.post(
            f"{SKILLS_PREFIX}/dry-run-exec-skill/execute",
            json={"args": "safe args", "dry_run": True},
        )

        assert r.status_code == 200, r.text
        result = r.json()
        assert observed == {"dry_run": True, "arguments": "safe args"}
        assert result["dry_run"] is True
        assert result["fork_output"] is None

    def test_execute_skill_sanitizes_skills_error(self, client, monkeypatch):
        async def _boom(self, name):  # noqa: ANN001, ANN202
            raise SkillsError("skills backend exploded at /private/execute")

        monkeypatch.setattr(SkillsService, "get_skill", _boom)

        with _capture_skills_endpoint_errors() as messages:
            r = client.post(
                f"{SKILLS_PREFIX}/exec-skill/execute",
                json={"args": "my test args"},
            )

        joined = "\n".join(messages)
        assert r.status_code == 500
        assert r.json()["detail"] == "Failed to execute skill"
        assert "Error executing skill" in joined
        assert "skills backend exploded" not in joined
        assert "/private/" not in joined

    def test_execute_quarantined_skill_returns_423(self, client, monkeypatch):
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityBlocked,
        )

        async def _blocked(self, name, *args, **kwargs):  # noqa: ANN001, ANN202
            raise ContextIntegrityBlocked(
                asset_id=f"skill:user:1/{name}",
                state="changed_approved_executable",
            )

        monkeypatch.setattr(SkillsService, "get_skill", _blocked)

        r = client.post(
            f"{SKILLS_PREFIX}/blocked-skill/execute",
            json={"args": "my test args"},
        )

        assert r.status_code == 423
        assert r.json()["detail"] == "Asset is quarantined pending admin review."

    def test_execute_skill_request_context_uses_current_principal_alias(
        self,
        principal_client,
        monkeypatch,
    ):
        create_resp = principal_client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "principal-exec-skill", "content": "Do: $ARGUMENTS"},
        )
        assert create_resp.status_code == 201, create_resp.text
        observed = {}

        async def fake_execute(
            self: object,
            *,
            skill_data: dict[str, Any],
            arguments: str,
            context: Any | None,
            dry_run: bool,
        ) -> SimpleNamespace:
            """Capture principal context propagation while bypassing real execution."""
            observed["user_id"] = context.user_id if context else None
            observed["dry_run"] = dry_run
            return SimpleNamespace(
                skill_name=skill_data["name"],
                rendered_prompt=f"rendered {arguments}",
                allowed_tools=[],
                model_override=None,
                execution_mode="inline",
                fork_output=None,
                dry_run=False,
            )

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.skills.SkillExecutor.execute",
            fake_execute,
        )

        r = principal_client.post(
            f"{SKILLS_PREFIX}/principal-exec-skill/execute",
            json={"args": "principal args"},
        )

        assert r.status_code == 200, r.text
        assert observed["user_id"] == TEST_USER_ID
        assert observed["dry_run"] is False


class TestContextPayload:
    def test_get_context_payload(self, client):
        client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "ctx-skill", "content": "---\ndescription: Context test\n---\nBody"},
        )
        r = client.get(f"{SKILLS_PREFIX}/context")
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["available_skills"]) >= 1
        assert "ctx-skill" in data["context_text"]
        listed = {skill["name"]: skill for skill in data["available_skills"]}
        assert listed["ctx-skill"]["version"] == 1
        ctx_line = next(line for line in data["context_text"].splitlines() if "ctx-skill" in line)
        assert "version" not in ctx_line.lower()

    def test_get_context_payload_includes_runtime_metadata(self, client):
        r = client.post(
            f"{SKILLS_PREFIX}/",
            json={
                "name": "ctx-runtime",
                "content": (
                    "---\n"
                    "description: Context runtime test\n"
                    "argument-hint: \"[topic]\"\n"
                    "context: fork\n"
                    "allowed-tools:\n"
                    "  - Read\n"
                    "model: gpt-4o-mini\n"
                    "---\n\n"
                    "Context runtime body"
                ),
            },
        )
        assert r.status_code == 201, r.text

        r = client.get(f"{SKILLS_PREFIX}/context")
        assert r.status_code == 200, r.text
        data = r.json()
        listed = {skill["name"]: skill for skill in data["available_skills"]}
        assert listed["ctx-runtime"]["allowed_tools"] == ["Read"]
        assert listed["ctx-runtime"]["model"] == "gpt-4o-mini"
        assert listed["ctx-runtime"]["runtime"] == {
            "execution_mode": "fork",
            "test_run_may_call_model": True,
            "declares_tools": True,
            "declared_tool_count": 1,
            "model_override": "gpt-4o-mini",
            "auto_invocation_enabled": True,
        }
        ctx_line = next(line for line in data["context_text"].splitlines() if "ctx-runtime" in line)
        assert "gpt-4o-mini" not in ctx_line
        assert "Read" not in ctx_line

    def test_context_payload_defaults_explicit_none_context(self):
        service = SkillsService.__new__(SkillsService)
        service._is_skill_allowed = lambda _name, purpose: True

        payload = service._build_context_payload(
            [
                {
                    "name": "legacy-context",
                    "description": "Legacy context row",
                    "argument_hint": None,
                    "user_invocable": True,
                    "disable_model_invocation": False,
                    "allowed_tools": ["Read"],
                    "model": None,
                    "context": None,
                    "version": 1,
                }
            ]
        )

        skill = payload["available_skills"][0]
        assert skill["context"] == "inline"
        assert skill["runtime"]["execution_mode"] == "inline"
        assert skill["runtime"]["declared_tool_count"] == 1

    def test_get_context_payload_uses_async_service_method(self, client, monkeypatch):
        calls = {"async": 0}

        async def _fake_async_payload(self):
            calls["async"] += 1
            return {"available_skills": [], "context_text": ""}

        def _fake_sync_payload(self):
            raise AssertionError("sync context payload should not be called from async endpoint")

        monkeypatch.setattr(SkillsService, "get_context_payload_async", _fake_async_payload)
        monkeypatch.setattr(SkillsService, "get_context_payload", _fake_sync_payload)

        r = client.get(f"{SKILLS_PREFIX}/context")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["available_skills"] == []
        assert data["context_text"] == ""
        assert calls["async"] == 1


class TestReadErrorSanitization:
    def test_list_skills_sanitizes_skills_error(self, client, monkeypatch):
        async def _raise_skills_error(self, *args, **kwargs):
            _ = (self, args, kwargs)
            raise SkillsError("skills backend exploded at /private/skills.db")

        monkeypatch.setattr(SkillsService, "list_skills", _raise_skills_error)

        with _capture_skills_endpoint_errors() as messages:
            r = client.get(f"{SKILLS_PREFIX}/")

        joined = "\n".join(messages)
        assert r.status_code == 500
        assert r.json()["detail"] == "Failed to list skills"
        assert "Error listing skills" in joined
        assert "skills backend exploded" not in joined
        assert "/private/" not in joined

    def test_get_skills_context_sanitizes_skills_error(self, client, monkeypatch):
        async def _raise_skills_error(self):
            _ = self
            raise SkillsError("skills backend exploded at /private/context-cache")

        monkeypatch.setattr(SkillsService, "get_context_payload_async", _raise_skills_error)

        with _capture_skills_endpoint_errors() as messages:
            r = client.get(f"{SKILLS_PREFIX}/context")

        joined = "\n".join(messages)
        assert r.status_code == 500
        assert r.json()["detail"] == "Failed to get skills context"
        assert "Error getting skills context" in joined
        assert "skills backend exploded" not in joined
        assert "/private/" not in joined

    def test_get_skill_sanitizes_skills_error(self, client, monkeypatch):
        async def _raise_skills_error(self, name):
            _ = (self, name)
            raise SkillsError("skills backend exploded at /private/skill.md")

        monkeypatch.setattr(SkillsService, "get_skill", _raise_skills_error)

        with _capture_skills_endpoint_errors() as messages:
            r = client.get(f"{SKILLS_PREFIX}/broken-skill")

        joined = "\n".join(messages)
        assert r.status_code == 500
        assert r.json()["detail"] == "Failed to get skill"
        assert "Error getting skill" in joined
        assert "skills backend exploded" not in joined
        assert "/private/" not in joined


class TestSupportingFilesLimit:
    def test_supporting_files_aggregate_limit_rejected(self, client):
        """Regression Bug 7: aggregate supporting files over 5MB should be rejected."""
        big_content = "x" * 400_000
        files = {f"file{i:02d}.md": big_content for i in range(15)}
        r = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "big-files", "content": "content", "supporting_files": files},
        )
        assert r.status_code == 422  # Pydantic validation error


class TestSkillsEndToEndWorkflow:
    @staticmethod
    def _rename_skill_in_export_zip(zip_bytes: bytes, source_name: str, target_name: str) -> bytes:
        import zipfile
        from io import BytesIO

        src_prefix = f"{source_name}/"
        dst_prefix = f"{target_name}/"
        output = BytesIO()
        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as src_zip:
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as dst_zip:
                for entry in src_zip.infolist():
                    filename = entry.filename
                    if filename == source_name:
                        rewritten = target_name
                    elif filename.startswith(src_prefix):
                        rewritten = f"{dst_prefix}{filename[len(src_prefix):]}"
                    else:
                        rewritten = filename

                    if entry.is_dir():
                        dst_zip.writestr(rewritten, b"")
                    else:
                        dst_zip.writestr(rewritten, src_zip.read(entry.filename))
        return output.getvalue()

    def test_e2e_skills_lifecycle_workflow(self, client):
        skill_name = "e2e-workflow-skill"
        copied_skill_name = "e2e-workflow-copy"

        # 1) Create
        create_resp = client.post(
            f"{SKILLS_PREFIX}/",
            json={
                "name": skill_name,
                "content": (
                    "---\n" "description: E2E workflow skill\n" "context: inline\n" "---\n\n" "Analyze: $ARGUMENTS"
                ),
                "supporting_files": {
                    "guide.md": "workflow guide",
                    "notes.txt": "first revision",
                },
            },
        )
        assert create_resp.status_code == 201, create_resp.text
        created = create_resp.json()
        assert created["name"] == skill_name
        assert created["version"] == 1

        # 2) List + get
        list_resp = client.get(f"{SKILLS_PREFIX}/?limit=50&offset=0")
        assert list_resp.status_code == 200, list_resp.text
        listed_names = [skill["name"] for skill in list_resp.json()["skills"]]
        assert skill_name in listed_names

        get_resp = client.get(f"{SKILLS_PREFIX}/{skill_name}")
        assert get_resp.status_code == 200, get_resp.text
        fetched = get_resp.json()
        assert fetched["supporting_files"]["guide.md"] == "workflow guide"
        assert fetched["supporting_files"]["notes.txt"] == "first revision"

        # 3) Update (versioned) with supporting-file add/update/delete
        update_resp = client.put(
            f"{SKILLS_PREFIX}/{skill_name}",
            headers={"If-Match": str(created["version"])},
            json={
                "content": (
                    "---\n"
                    "description: E2E workflow skill updated\n"
                    "context: inline\n"
                    "---\n\n"
                    "Analyze updated: $ARGUMENTS"
                ),
                "supporting_files": {
                    "guide.md": "workflow guide v2",
                    "notes.txt": None,
                    "appendix.md": "appendix content",
                },
            },
        )
        assert update_resp.status_code == 200, update_resp.text
        updated = update_resp.json()
        assert updated["version"] == 2
        assert updated["description"] == "E2E workflow skill updated"
        assert updated["supporting_files"]["guide.md"] == "workflow guide v2"
        assert updated["supporting_files"]["appendix.md"] == "appendix content"
        assert "notes.txt" not in updated["supporting_files"]

        # 4) Execute preview
        execute_resp = client.post(
            f"{SKILLS_PREFIX}/{skill_name}/execute",
            json={"args": "e2e input"},
        )
        assert execute_resp.status_code == 200, execute_resp.text
        execute_data = execute_resp.json()
        assert execute_data["skill_name"] == skill_name
        assert "e2e input" in execute_data["rendered_prompt"]
        assert execute_data["execution_mode"] == "inline"

        # 5) Context payload reflects updated skill
        context_resp = client.get(f"{SKILLS_PREFIX}/context")
        assert context_resp.status_code == 200, context_resp.text
        context_data = context_resp.json()
        assert skill_name in context_data["context_text"]
        assert "E2E workflow skill updated" in context_data["context_text"]

        # 6) Export zip and import back as renamed copy
        export_resp = client.get(f"{SKILLS_PREFIX}/{skill_name}/export")
        assert export_resp.status_code == 200, export_resp.text
        assert export_resp.content[:2] == b"PK"
        renamed_zip = self._rename_skill_in_export_zip(
            export_resp.content,
            source_name=skill_name,
            target_name=copied_skill_name,
        )

        import_resp = client.post(
            f"{SKILLS_PREFIX}/import/file",
            files={"file": ("copied-skill.zip", renamed_zip, "application/zip")},
        )
        assert import_resp.status_code == 201, import_resp.text
        assert import_resp.json()["name"] == copied_skill_name

        copy_get_resp = client.get(f"{SKILLS_PREFIX}/{copied_skill_name}")
        assert copy_get_resp.status_code == 200, copy_get_resp.text
        copy_data = copy_get_resp.json()
        assert copy_data["description"] == "E2E workflow skill updated"
        assert copy_data["supporting_files"]["guide.md"] == "workflow guide v2"
        assert copy_data["supporting_files"]["appendix.md"] == "appendix content"

        # 7) Delete original
        delete_resp = client.delete(f"{SKILLS_PREFIX}/{skill_name}")
        assert delete_resp.status_code == 204, delete_resp.text
        missing_resp = client.get(f"{SKILLS_PREFIX}/{skill_name}")
        assert missing_resp.status_code == 404

        # 8) Seed builtin skills
        seed_resp = client.post(f"{SKILLS_PREFIX}/seed")
        assert seed_resp.status_code == 200, seed_resp.text
        seed_data = seed_resp.json()
        assert seed_data["count"] >= 3
        assert "summarize" in seed_data["seeded"]
        assert "code-review" in seed_data["seeded"]
        assert "feynman-technique" in seed_data["seeded"]

    def test_e2e_seed_endpoint_idempotent_and_overwrite(self, client):
        first_seed = client.post(f"{SKILLS_PREFIX}/seed")
        assert first_seed.status_code == 200, first_seed.text
        first_data = first_seed.json()
        assert first_data["count"] >= 3

        # idempotent without overwrite
        second_seed = client.post(f"{SKILLS_PREFIX}/seed")
        assert second_seed.status_code == 200, second_seed.text
        assert second_seed.json()["count"] == 0

        # mutate summarize, then verify overwrite restores builtin content
        mutate_resp = client.put(
            f"{SKILLS_PREFIX}/summarize",
            json={"content": "Custom summarize content"},
        )
        assert mutate_resp.status_code == 200, mutate_resp.text

        overwrite_seed = client.post(f"{SKILLS_PREFIX}/seed?overwrite=true")
        assert overwrite_seed.status_code == 200, overwrite_seed.text
        assert "summarize" in overwrite_seed.json()["seeded"]

        summarize_resp = client.get(f"{SKILLS_PREFIX}/summarize")
        assert summarize_resp.status_code == 200, summarize_resp.text
        assert "Custom summarize content" not in summarize_resp.json()["content"]
