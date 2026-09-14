from __future__ import annotations

import ast
import inspect
import io
import re
from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_Server_API.app.core.UserProfiles import overrides_repo


class _FailingOverridePool:
    pool = None

    async def execute(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("secret=/private/override.db token=override-secret")

    async def fetchall(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("secret=/private/override.db token=override-secret")

    async def fetchone(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("secret=/private/override.db token=override-secret")


def test_override_repository_sql_always_qualifies_candidate_relations() -> None:
    relation_names = (
        "user_config_overrides",
        "org_config_overrides",
        "team_config_overrides",
    )
    sql_keywords = ("SELECT", "INSERT", "UPDATE", "DELETE")
    tree = ast.parse(inspect.getsource(overrides_repo))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        sql = node.value
        if not any(keyword in sql.upper() for keyword in sql_keywords):
            continue
        for relation_name in relation_names:
            if relation_name not in sql:
                continue
            assert not re.search(
                rf"(?i)\b(?:FROM|INTO|UPDATE|DELETE\s+FROM)\s+{relation_name}\b",
                sql,
            ), sql


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "repo_type",
    [
        overrides_repo.UserProfileOverridesRepo,
        overrides_repo.OrgProfileOverridesRepo,
        overrides_repo.TeamProfileOverridesRepo,
    ],
)
@pytest.mark.parametrize("failure_mode", ["false", "exception"])
async def test_postgres_override_readiness_fails_closed_and_sanitizes(
    monkeypatch: pytest.MonkeyPatch,
    repo_type: type,
    failure_mode: str,
) -> None:
    sentinel = "private schema failure detail"

    async def _ensure(_pool: object) -> bool:
        if failure_mode == "exception":
            raise RuntimeError(sentinel)
        return False

    monkeypatch.setattr(overrides_repo, "ensure_authnz_core_tables_pg", _ensure)
    repo = repo_type(db_pool=SimpleNamespace(pool=object()))
    output = io.StringIO()
    sink = logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(RuntimeError) as raised:
            await repo.ensure_tables()
    finally:
        logger.remove(sink)

    assert str(raised.value) == (
        "PostgreSQL AuthNZ profile override schema readiness failed"
    )
    assert raised.value.__cause__ is None
    assert sentinel not in output.getvalue()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("repo_type", "method_name", "arguments"),
    [
        (overrides_repo.UserProfileOverridesRepo, "list_overrides_for_user", (7,)),
        (
            overrides_repo.UserProfileOverridesRepo,
            "upsert_override",
            {"user_id": 7, "key": "theme", "value": "dark", "updated_by": 7},
        ),
        (
            overrides_repo.UserProfileOverridesRepo,
            "delete_override",
            {"user_id": 7, "key": "theme"},
        ),
        (
            overrides_repo.UserProfileOverridesRepo,
            "get_latest_update_for_user",
            (7,),
        ),
        (overrides_repo.OrgProfileOverridesRepo, "list_overrides_for_orgs", ([3],)),
        (
            overrides_repo.OrgProfileOverridesRepo,
            "upsert_override",
            {"org_id": 3, "key": "theme", "value": "dark", "updated_by": 7},
        ),
        (
            overrides_repo.OrgProfileOverridesRepo,
            "delete_override",
            {"org_id": 3, "key": "theme"},
        ),
        (
            overrides_repo.OrgProfileOverridesRepo,
            "get_latest_update_for_orgs",
            ([3],),
        ),
        (overrides_repo.TeamProfileOverridesRepo, "list_overrides_for_teams", ([5],)),
        (
            overrides_repo.TeamProfileOverridesRepo,
            "upsert_override",
            {"team_id": 5, "key": "theme", "value": "dark", "updated_by": 7},
        ),
        (
            overrides_repo.TeamProfileOverridesRepo,
            "delete_override",
            {"team_id": 5, "key": "theme"},
        ),
        (
            overrides_repo.TeamProfileOverridesRepo,
            "get_latest_update_for_teams",
            ([5],),
        ),
    ],
)
async def test_override_repository_failure_logs_never_include_backend_details(
    repo_type: type,
    method_name: str,
    arguments: tuple[object, ...] | dict[str, object],
) -> None:
    repo = repo_type(db_pool=_FailingOverridePool())
    method = getattr(repo, method_name)
    output = io.StringIO()
    sink = logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(RuntimeError, match="override-secret"):
            if isinstance(arguments, dict):
                await method(**arguments)
            else:
                await method(*arguments)
    finally:
        logger.remove(sink)

    assert "override-secret" not in output.getvalue()
    assert "/private/override.db" not in output.getvalue()
