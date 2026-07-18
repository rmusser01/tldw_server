"""Durable Prompt Studio job credential-runtime and strict-failure regressions."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import gzip
import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FunctionType, MethodType
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ import orgs_teams
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    ProviderOverridePolicyError,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
)
from tldw_Server_API.app.core.Chat.bounded_daemon import (
    DaemonCapacityError,
    await_owned_worker,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatConfigurationError,
    ChatProviderError,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    DatabaseError,
    PromptStudioDatabase,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    job_processor as job_processor_module,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    optimization_engine as optimization_engine_module,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    prompt_executor as prompt_executor_module,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    test_runner as test_runner_module,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_processor import (
    JobProcessor,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services import (
    jobs_worker,
)

pytestmark = pytest.mark.unit

_SENTINEL = "TASK12963_DURABLE_RUNTIME_SECRET"
_RESOLVED_KEY = "TASK12963_RESOLVED_PROVIDER_KEY"
_RUNTIME_SENTINEL = "TASK12963_PROVIDER_RUNTIME_HANDLE"
_DB_PATHS: dict[int, Path] = {}
_MODEL_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "parameters": {"temperature": 0.17, "max_tokens": 64},
}

_DURABLE_SECRET_TOKENS = frozenset(
    {
        "apikey",
        "appconfig",
        "authorization",
        "auth",
        "password",
        "secret",
        "clientsecret",
        "accesstoken",
        "refreshtoken",
        "token",
        "cookie",
        "jwt",
    }
)


def _assert_payload_has_no_secret_aliases(value: Any, *sentinels: str) -> None:
    serialized = json.dumps(value, default=repr, sort_keys=True)
    for sentinel in sentinels:
        assert sentinel not in serialized

    def _walk(current: Any) -> None:
        if isinstance(current, dict):
            for key, nested in current.items():
                compact = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                assert compact not in _DURABLE_SECRET_TOKENS, (
                    f"secret alias survived durable payload: {key}"
                )
                _walk(nested)
        elif isinstance(current, (list, tuple)):
            for nested in current:
                _walk(nested)

    _walk(value)


def _inject_historical_archive_payload(
    db_path: Path,
    *,
    job_uuid: str,
    payload: dict[str, Any],
) -> None:
    """Bypass current archive hygiene to model a pre-migration row."""

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE jobs_archive SET payload = ?, payload_compressed = NULL "
            "WHERE uuid = ?",
            (json.dumps(payload), job_uuid),
        )
        conn.commit()
    finally:
        conn.close()


class _Handle:
    def __init__(
        self,
        *,
        provider: str = "openai",
        api_key: str | None,
        app_config: dict[str, Any],
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.app_config = app_config
        self.auth_source = "user" if api_key else None
        self.credentials_resolved = True
        self.runtime_handle_marker = _RUNTIME_SENTINEL

    def __repr__(self) -> str:
        return f"ProviderHandle({_RUNTIME_SENTINEL})"


class _Runtime:
    def __init__(
        self,
        *,
        scope: dict[str, Any],
        outcome: _Handle
        | BaseException
        | Callable[
            [dict[str, Any], int, str, str | None],
            _Handle | BaseException,
        ],
        index: int,
        events: list[Any],
    ) -> None:
        self.scope = scope
        self.outcome = outcome
        self.index = index
        self.resolved_outcome: _Handle | BaseException | None = None
        self.resolved_handles: list[_Handle] = []
        self.events = events
        self.resolve_calls: list[tuple[str, str | None]] = []
        self.mark_calls: list[object] = []
        self.close_count = 0
        events.append(("init", scope))

    async def resolve(self, provider: str, *, model: str | None = None) -> _Handle:
        self.resolve_calls.append((provider, model))
        self.events.append(("resolve", provider, model))
        outcome = (
            self.outcome(self.scope, self.index, provider, model)
            if callable(self.outcome)
            else self.outcome
        )
        self.resolved_outcome = outcome
        if isinstance(outcome, BaseException):
            raise outcome
        self.resolved_handles.append(outcome)
        return outcome

    async def mark_used(self, handle: object) -> bool:
        assert any(handle is candidate for candidate in self.resolved_handles)
        self.mark_calls.append(handle)
        self.events.append("mark")
        return True

    async def close(self) -> None:
        self.close_count += 1
        self.events.append("close")


class _RuntimeFactory:
    def __init__(
        self,
        outcome: _Handle
        | BaseException
        | Callable[
            [dict[str, Any], int, str, str | None],
            _Handle | BaseException,
        ],
        events: list[Any],
    ) -> None:
        self.outcome = outcome
        self.events = events
        self.instances: list[_Runtime] = []

    def __call__(self, **scope: Any) -> _Runtime:
        index = len(self.instances)
        runtime = _Runtime(
            scope=dict(scope),
            outcome=self.outcome,
            index=index,
            events=self.events,
        )
        self.instances.append(runtime)
        return runtime


def _new_db(tmp_path: Path, name: str) -> PromptStudioDatabase:
    path = tmp_path / f"{name}.db"
    db = PromptStudioDatabase(
        str(path),
        client_id=f"runtime-{name}",
    )
    _DB_PATHS[id(db)] = path
    return db


def _assert_all_durable_state_secret_free(
    db: PromptStudioDatabase,
    *sentinels: str,
) -> str:
    path = _DB_PATHS[id(db)]
    connection = sqlite3.connect(str(path))
    try:
        table_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        logical_dump = "\n".join(connection.iterdump())
    finally:
        connection.close()

    assert "prompt_studio_optimizations" in table_names
    assert "prompt_studio_test_runs" in table_names
    assert "prompt_studio_optimization_iterations" in table_names
    physical_bytes = b"".join(
        candidate.read_bytes()
        for candidate in sorted(path.parent.glob(f"{path.name}*"))
        if candidate.is_file()
    )
    for sentinel in (*sentinels, _RUNTIME_SENTINEL):
        assert sentinel not in logical_dump
        assert sentinel.encode() not in physical_bytes
    return logical_dump


def _seed_optimization(
    db: PromptStudioDatabase,
    *,
    name: str,
    max_iterations: int,
    with_test_case: bool = True,
    optimizer_type: str = "mipro",
    model_config: dict[str, Any] | None = None,
    strategy_params: dict[str, Any] | None = None,
) -> tuple[int, int, list[int]]:
    selected_model_config = copy.deepcopy(model_config or _MODEL_CONFIG)
    project = db.create_project(name=f"Project {name}", description="")
    prompt = db.create_prompt(
        project_id=int(project["id"]),
        name=f"Prompt {name}",
        system_prompt=f"Answer precisely for marker {name}.",
        user_prompt=f"Request marker {name}: {{question}}",
    )
    test_case_ids: list[int] = []
    if with_test_case:
        case = db.create_test_case(
            project_id=int(project["id"]),
            name=f"Case {name}",
            inputs={"question": f"input-{name}"},
            expected_outputs={"response": "expected only"},
        )
        test_case_ids.append(int(case["id"]))
    optimization = db.create_optimization(
        project_id=int(project["id"]),
        name=f"Optimization {name}",
        initial_prompt_id=int(prompt["id"]),
        optimizer_type=optimizer_type,
        optimization_config={
            "optimizer_type": optimizer_type,
            "target_metric": "accuracy",
            "model_config": selected_model_config,
            **(
                {"strategy_params": copy.deepcopy(strategy_params)}
                if strategy_params is not None
                else {}
            ),
        },
        max_iterations=max_iterations,
        status="pending",
    )
    db.update_optimization(
        int(optimization["id"]),
        {"test_case_ids": test_case_ids},
    )
    return int(optimization["id"]), int(prompt["id"]), test_case_ids


def _job(
    *,
    optimization_id: int,
    prompt_id: int,
    test_case_ids: list[int],
    owner_user_id: int,
    max_iterations: int,
    optimizer_type: str = "mipro",
    model_config: dict[str, Any] | None = None,
    strategy_params: dict[str, Any] | None = None,
    optimization_db: PromptStudioDatabase | None = None,
) -> dict[str, Any]:
    selected_model_config = copy.deepcopy(model_config or _MODEL_CONFIG)
    job = {
        "id": optimization_id,
        "uuid": f"job-{owner_user_id}-{optimization_id}",
        "job_type": "optimization",
        "owner_user_id": str(owner_user_id),
        "payload": {
            "optimization_id": optimization_id,
            "initial_prompt_id": prompt_id,
            "test_case_ids": list(test_case_ids),
            "optimizer_type": optimizer_type,
            "max_iterations": max_iterations,
            "optimization_config": {
                "optimizer_type": optimizer_type,
                "target_metric": "accuracy",
                "model_config": selected_model_config,
                **(
                    {"strategy_params": copy.deepcopy(strategy_params)}
                    if strategy_params is not None
                    else {}
                ),
            },
        },
    }
    if optimization_db is not None:
        optimization = optimization_db.get_optimization(
            optimization_id,
            include_deleted=True,
        ) or {}
        job["payload"]["optimization_uuid"] = str(optimization["uuid"])
    return job


def _install_runtime_and_memberships(
    monkeypatch: pytest.MonkeyPatch,
    factory: _RuntimeFactory,
) -> list[tuple[str, int]]:
    membership_calls: list[tuple[str, int]] = []

    async def _teams(user_id: int) -> list[dict[str, Any]]:
        owner = int(user_id)
        membership_calls.append(("team", owner))
        return [{"team_id": owner * 10 + 1, "org_id": owner * 10 + 2}]

    async def _orgs(user_id: int) -> list[dict[str, Any]]:
        owner = int(user_id)
        membership_calls.append(("org", owner))
        return [
            {"org_id": owner * 10 + 2, "status": "active"},
            {"org_id": owner * 10 + 3, "status": "revoked"},
        ]

    for module in (
        jobs_worker,
        job_processor_module,
        optimization_engine_module,
    ):
        monkeypatch.setattr(
            module,
            "ProviderCredentialRuntime",
            factory,
            raising=False,
        )
        monkeypatch.setattr(
            module,
            "list_active_team_memberships_for_user",
            _teams,
            raising=False,
        )
        monkeypatch.setattr(
            module,
            "list_org_memberships_for_user",
            _orgs,
            raising=False,
        )
        monkeypatch.setattr(
            module,
            "provider_requires_api_key",
            lambda provider: provider not in {"ollama", "llama.cpp"},
            raising=False,
        )
    monkeypatch.setattr(
        orgs_teams,
        "list_active_team_memberships_for_user",
        _teams,
        raising=True,
    )
    monkeypatch.setattr(
        orgs_teams,
        "list_org_memberships_for_user",
        _orgs,
        raising=True,
    )
    for module in (prompt_executor_module, test_runner_module):
        monkeypatch.setattr(
            module,
            "is_runtime_issued_provider_call_credentials",
            lambda value, *, provider=None: (
                isinstance(value, _Handle)
                and (provider is None or value.provider == provider)
            ),
            raising=False,
        )
    return membership_calls


def _install_safe_config_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (test_runner_module, prompt_executor_module):
        monkeypatch.setattr(
            module,
            "ensure_app_config",
            lambda config: config or {},
            raising=True,
        )
        monkeypatch.setattr(
            module,
            "resolve_provider_api_key_from_config",
            lambda *_args, **_kwargs: None,
            raising=True,
        )


def _assert_safe_job_error(
    exc: BaseException,
    *,
    code: str,
    retryable: bool,
) -> None:
    assert isinstance(exc, jobs_worker.PromptStudioJobError)
    assert exc.failure_code == code
    assert exc.retryable is retryable
    graph = "".join((repr(exc), repr(exc.__cause__), repr(exc.__context__)))
    assert _SENTINEL not in graph


def test_prompt_database_error_is_bounded_and_retryable() -> None:
    bounded = jobs_worker._bounded_optimization_error(
        DatabaseError(f"database unavailable: {_SENTINEL}")
    )

    _assert_safe_job_error(
        bounded,
        code="job_store_unavailable",
        retryable=True,
    )


def _retains_identity(owner: object, target: object, *, max_depth: int = 10) -> bool:
    seen: set[int] = set()
    pending: list[tuple[object, int]] = [(owner, 0)]
    while pending:
        value, depth = pending.pop()
        if value is target:
            return True
        value_id = id(value)
        if value_id in seen or depth >= max_depth:
            continue
        seen.add(value_id)
        nested: list[object] = []
        if isinstance(value, dict):
            nested.extend(value.keys())
            nested.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            nested.extend(value)
        if isinstance(value, FunctionType):
            nested.extend(value.__defaults__ or ())
            nested.extend((value.__kwdefaults__ or {}).values())
            for cell in value.__closure__ or ():
                try:
                    nested.append(cell.cell_contents)
                except ValueError:
                    continue
        elif isinstance(value, MethodType):
            nested.extend((value.__self__, value.__func__))
        if hasattr(value, "__dict__"):
            nested.extend(vars(value).values())
        for owner_type in type(value).__mro__:
            raw_slots = owner_type.__dict__.get("__slots__", ())
            slots = (raw_slots,) if isinstance(raw_slots, str) else raw_slots
            for slot in slots:
                if slot in {"__dict__", "__weakref__"}:
                    continue
                attribute = slot
                if slot.startswith("__") and not slot.endswith("__"):
                    attribute = f"_{owner_type.__name__.lstrip('_')}{slot}"
                try:
                    nested.append(getattr(value, attribute))
                except (AttributeError, TypeError):
                    continue
        pending.extend((item, depth + 1) for item in nested)
    return False


@pytest.mark.asyncio
async def test_owner_membership_scope_requires_explicit_normalized_active_org_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _teams(_user_id: int) -> list[dict[str, Any]]:
        return []

    async def _orgs(_user_id: int) -> list[dict[str, Any]]:
        return [
            {"org_id": 1},
            {"org_id": 2, "status": None},
            {"org_id": 3, "status": ""},
            {"org_id": 4, "status": " active "},
            {"org_id": 5, "status": "ACTIVE"},
            {"org_id": 6, "status": "inactive"},
        ]

    monkeypatch.setattr(
        jobs_worker,
        "list_active_team_memberships_for_user",
        _teams,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "list_org_memberships_for_user",
        _orgs,
        raising=True,
    )

    team_ids, org_ids = await jobs_worker._owner_membership_scope(7)

    assert team_ids == []
    assert org_ids == [4, 5]


@pytest.mark.asyncio
async def test_worker_rejects_before_persisting_completed_without_provider_success_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "missing-success-callback")
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )
    factory = _RuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)

    async def _optimize_without_callback(
        _self: Any,
        *,
        initial_prompt_id: int,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "initial_prompt_id": initial_prompt_id,
            "optimized_prompt_id": initial_prompt_id,
            "initial_score": 0.5,
            "final_score": 0.5,
            "improvement": 0.0,
            "iterations": 1,
            "iteration_history": [],
            "strategy": "MIPRO",
        }

    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="missing-success-callback",
        max_iterations=1,
    )
    monkeypatch.setattr(
        optimization_engine_module.MIPROOptimizer,
        "optimize",
        _optimize_without_callback,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(
                _job(
                    optimization_id=optimization_id,
                    prompt_id=prompt_id,
                    test_case_ids=case_ids,
                    owner_user_id=7,
                    max_iterations=1,
                )
            )
        _assert_safe_job_error(
            exc_info.value,
            code="provider_success_not_observed",
            retryable=False,
        )

        row = db.get_optimization(optimization_id, include_deleted=True) or {}
        assert row["status"] == "failed"
        assert len(factory.instances) == 1
        assert factory.instances[0].mark_calls == []
        assert factory.instances[0].close_count == 1
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_worker_retries_failed_usage_touch_across_concurrent_success_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "usage-touch-retry")
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )

    class _RetryRuntimeFactory(_RuntimeFactory):
        def __call__(self, **scope: Any) -> _Runtime:
            runtime = super().__call__(**scope)

            async def _mark_used(selected: _Runtime, candidate: object) -> bool:
                assert any(candidate is item for item in selected.resolved_handles)
                selected.mark_calls.append(candidate)
                selected.events.append("mark")
                return len(selected.mark_calls) > 1

            runtime.mark_used = MethodType(_mark_used, runtime)
            return runtime

    factory = _RetryRuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)

    class _ConcurrentCallbackProcessor:
        def __init__(self, database: PromptStudioDatabase) -> None:
            self.db = database

        async def process_optimization_job(
            self,
            _payload: dict[str, Any],
            optimization_id: int,
            **kwargs: Any,
        ) -> dict[str, Any]:
            callback = kwargs["on_provider_success"]
            await asyncio.gather(callback(), callback())
            self.db.complete_optimization(
                optimization_id,
                iterations_completed=1,
            )
            return {
                "optimization_id": optimization_id,
                "status": "completed",
                "iterations_completed": 1,
            }

    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="usage-touch-retry",
        max_iterations=1,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: _ConcurrentCallbackProcessor(db),
        raising=True,
    )

    try:
        result = await jobs_worker._handle_job(
            _job(
                optimization_id=optimization_id,
                prompt_id=prompt_id,
                test_case_ids=case_ids,
                owner_user_id=7,
                max_iterations=1,
            )
        )

        assert result["status"] == "completed"
        runtime = factory.instances[0]
        assert runtime.mark_calls == [handle, handle]
        assert runtime.close_count == 1
        assert events[-3:] == ["mark", "mark", "close"]
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_worker_allows_completed_distinct_scorer_when_cache_avoids_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "cached-scorer-without-dispatch")
    events: list[Any] = []
    base_model = "gpt-4o-mini"
    scorer_model = "gpt-scorer"
    base_handle = _Handle(
        api_key="base-model-key",
        app_config={"openai_api": {"model": base_model}},
    )
    scorer_handle = _Handle(
        api_key="scorer-model-key",
        app_config={"openai_api": {"model": scorer_model}},
    )

    def _resolve_for_model(
        _scope: dict[str, Any],
        _index: int,
        _provider: str,
        model: str | None,
    ) -> _Handle:
        return scorer_handle if model == scorer_model else base_handle

    factory = _RuntimeFactory(_resolve_for_model, events)
    _install_runtime_and_memberships(monkeypatch, factory)

    class _WarmScorerCacheProcessor:
        def __init__(self, database: PromptStudioDatabase) -> None:
            self.db = database

        async def process_optimization_job(
            self,
            _payload: dict[str, Any],
            optimization_id: int,
            **kwargs: Any,
        ) -> dict[str, Any]:
            assert kwargs["runtime_scorer_model_config"]["model"] == scorer_model
            assert kwargs["on_scorer_provider_success"] is not None
            await kwargs["on_provider_success"]()
            self.db.complete_optimization(
                optimization_id,
                iterations_completed=1,
            )
            return {
                "optimization_id": optimization_id,
                "status": "completed",
                "iterations_completed": 1,
                "provider_dispatches": {"primary": 1, "scorer": 0},
            }

    strategy_params = {"scorer_model": scorer_model}
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="cached-scorer-without-dispatch",
        max_iterations=1,
        optimizer_type="mcts",
        strategy_params=strategy_params,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: _WarmScorerCacheProcessor(db),
        raising=True,
    )

    try:
        result = await jobs_worker._handle_job(
            _job(
                optimization_id=optimization_id,
                prompt_id=prompt_id,
                test_case_ids=case_ids,
                owner_user_id=7,
                max_iterations=1,
                optimizer_type="mcts",
                strategy_params=strategy_params,
            )
        )

        assert result["status"] == "completed"
        assert "provider_dispatches" not in result
        assert "_provider_dispatches" not in result
        assert "scorer_provider_dispatched" not in result
        assert "_scorer_provider_dispatched" not in result
        runtime = factory.instances[0]
        assert runtime.resolve_calls == [
            ("openai", base_model),
            ("openai", scorer_model),
        ]
        assert runtime.mark_calls == [base_handle]
        assert runtime.close_count == 1
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_prompt_worker_rejects_wrong_jobs_domain_before_provider_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "wrong-jobs-domain")
    events: list[Any] = []
    factory = _RuntimeFactory(
        _Handle(
            api_key=_RESOLVED_KEY,
            app_config={"openai_api": {"model": "gpt-4o-mini"}},
        ),
        events,
    )
    _install_runtime_and_memberships(monkeypatch, factory)

    class _NoDispatchProcessor:
        def __init__(self, database: PromptStudioDatabase) -> None:
            self.db = database

        async def process_optimization_job(self, *_args: Any, **_kwargs: Any):
            raise AssertionError("provider dispatch must not be reached")

    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="wrong-jobs-domain",
        max_iterations=1,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: _NoDispatchProcessor(db),
        raising=True,
    )
    job = _job(
        optimization_id=optimization_id,
        prompt_id=prompt_id,
        test_case_ids=case_ids,
        owner_user_id=7,
        max_iterations=1,
    )
    job["domain"] = "other"

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(job)

        _assert_safe_job_error(
            exc_info.value,
            code="job_identity_invalid",
            retryable=False,
        )
        assert factory.instances == []
        row = db.get_optimization(optimization_id, include_deleted=True) or {}
        assert row["status"] == "pending"
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_prompt_worker_fails_closed_when_live_job_identity_guard_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "live-job-guard-rejects")
    events: list[Any] = []
    factory = _RuntimeFactory(
        _Handle(
            api_key=_RESOLVED_KEY,
            app_config={"openai_api": {"model": "gpt-4o-mini"}},
        ),
        events,
    )
    _install_runtime_and_memberships(monkeypatch, factory)
    calls: list[dict[str, Any]] = []

    class _RejectingJobManager:
        def replace_job_payload(self, job_id: int, **kwargs: Any) -> bool:
            calls.append({"job_id": job_id, **copy.deepcopy(kwargs)})
            return False

    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="live-job-guard-rejects",
        max_iterations=1,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_jobs_manager",
        lambda: _RejectingJobManager(),
        raising=True,
    )
    job = _job(
        optimization_id=optimization_id,
        prompt_id=prompt_id,
        test_case_ids=case_ids,
        owner_user_id=7,
        max_iterations=1,
    )
    job["domain"] = "prompt_studio"

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(job)

        _assert_safe_job_error(
            exc_info.value,
            code="job_identity_invalid",
            retryable=False,
        )
        assert len(calls) == 1
        assert calls[0]["expected_uuid"] == job["uuid"]
        assert calls[0]["expected_domain"] == "prompt_studio"
        assert factory.instances == []
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_stale_live_cancelled_job_scrubs_its_payload_without_replacing_newer_prompt_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "stale-live-cancelled-job")
    jobs_path = tmp_path / "stale-live-cancelled-job.sqlite"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_path))
    monkeypatch.setenv("JOBS_SECRET_REDACT", "0")
    monkeypatch.setenv("JOBS_SECRET_REJECT", "0")
    manager = JobManager(db_path=jobs_path)
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="stale-live-cancelled-job",
        max_iterations=1,
    )
    stale_payload = _job(
        optimization_id=optimization_id,
        prompt_id=prompt_id,
        test_case_ids=case_ids,
        owner_user_id=7,
        max_iterations=1,
        optimization_db=db,
    )["payload"]
    stale_payload["optimization_config"]["model_config"]["api_key"] = _SENTINEL
    job = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=stale_payload,
        owner_user_id="7",
    )
    newer_config = {
        "optimizer_type": "mipro",
        "target_metric": "f1",
        "model_config": {
            "provider": "openai",
            "model": "newer-model",
            "parameters": {"temperature": 0.9},
        },
        "strategy_params": {"newer_revision": 2},
    }
    db.update_optimization(
        optimization_id,
        {
            "status": "running",
            "optimization_config": newer_config,
        },
    )
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )

    try:
        assert manager.cancel_job(int(job["id"]), reason="stale cancellation")
        assert await jobs_worker._reconcile_cancelled_optimization_jobs(manager) == 1

        live = manager.get_job(int(job["id"])) or {}
        _assert_payload_has_no_secret_aliases(live.get("payload") or {}, _SENTINEL)
        optimization = db.get_optimization(
            optimization_id,
            include_deleted=True,
        ) or {}
        assert optimization["optimization_config"] == newer_config
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_preexisting_archived_cancelled_job_scrubs_without_mutating_missing_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "legacy-core-jobs")
    jobs_path = tmp_path / "legacy-core-jobs.sqlite"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_path))
    monkeypatch.setenv("JOBS_SECRET_REDACT", "0")
    monkeypatch.setenv("JOBS_SECRET_REJECT", "0")
    monkeypatch.setenv(
        "JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS",
        "prompt_studio",
    )
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "1")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", "1")
    manager = JobManager(db_path=jobs_path)
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="legacy-core-jobs",
        max_iterations=1,
    )
    legacy_payload = {
        "optimization_id": optimization_id,
        "initial_prompt_id": prompt_id,
        "test_case_ids": case_ids,
        "optimizer_type": "mipro",
        "authorization": _SENTINEL,
        "metadata": {
            "client-secret": _SENTINEL,
            "accessToken": _SENTINEL,
            "cookie": _SENTINEL,
        },
        "optimization_config": {
            "optimizer_type": "mipro",
            "target_metric": "accuracy",
            "model_config": {
                **copy.deepcopy(_MODEL_CONFIG),
                "api_key": _SENTINEL,
                "app_config": {
                    "openai_api": {
                        "api_key": _SENTINEL,
                        "model": "gpt-4o-mini",
                    }
                },
            },
            "strategy_params": {
                "nested": [
                    {
                        "refresh_token": _SENTINEL,
                        "JWT": _SENTINEL,
                    }
                ]
            },
        },
    }
    job = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=legacy_payload,
        owner_user_id="7",
    )
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )

    try:
        assert manager.cancel_job(
            int(job["id"]),
            reason="cancelled before migration",
        )
        assert manager.prune_jobs(
            statuses=["cancelled"],
            older_than_days=0,
            domain="prompt_studio",
        ) == 1
        _inject_historical_archive_payload(
            jobs_path,
            job_uuid=str(job["uuid"]),
            payload=legacy_payload,
        )
        archived_before = manager.get_job_or_archived(
            int(job["id"]),
            domain="prompt_studio",
        )
        assert archived_before is not None
        assert archived_before["archived"] is True
        assert _SENTINEL in json.dumps(archived_before["payload"])

        newer_config = {
            "optimizer_type": "mipro",
            "target_metric": "f1",
            "model_config": {
                "provider": "openai",
                "model": "newer-model",
                "parameters": {"temperature": 0.9},
            },
            "strategy_params": {"newer_revision": 2},
        }
        db.update_optimization(
            optimization_id,
            {
                "status": "running",
                "optimization_config": newer_config,
            },
        )

        assert await jobs_worker._reconcile_cancelled_optimization_jobs(
            manager,
            include_archived=True,
        ) == 1

        archived = manager.get_job_or_archived(
            int(job["id"]),
            domain="prompt_studio",
        )
        assert archived is not None
        assert archived["archived"] is True

        _assert_payload_has_no_secret_aliases(archived["payload"], _SENTINEL)
        optimization = db.get_optimization(
            optimization_id,
            include_deleted=True,
        ) or {}
        assert optimization["status"] == "running"
        assert optimization.get("completed_at") is None
        assert optimization["optimization_config"] == newer_config
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_malformed_archive_without_prompt_identity_is_still_scrubbed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs_path = tmp_path / "malformed-archive.sqlite"
    monkeypatch.setenv("JOBS_SECRET_REDACT", "0")
    monkeypatch.setenv("JOBS_SECRET_REJECT", "0")
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    manager = JobManager(db_path=jobs_path)
    job = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload={
            "authorization": _SENTINEL,
            "optimization_config": {
                "model_config": {
                    **copy.deepcopy(_MODEL_CONFIG),
                    "api_key": _SENTINEL,
                }
            },
        },
        owner_user_id=None,
    )
    assert manager.cancel_job(int(job["id"]), reason="malformed archive")
    assert manager.prune_jobs(
        statuses=["cancelled"],
        older_than_days=0,
        domain="prompt_studio",
    ) == 1
    _inject_historical_archive_payload(
        jobs_path,
        job_uuid=str(job["uuid"]),
        payload={
            "authorization": _SENTINEL,
            "optimization_config": {
                "model_config": {
                    **copy.deepcopy(_MODEL_CONFIG),
                    "api_key": _SENTINEL,
                }
            },
        },
    )
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: (_ for _ in ()).throw(
            AssertionError("malformed archive must not open a tenant Prompt DB")
        ),
        raising=True,
    )

    assert await jobs_worker._reconcile_cancelled_optimization_jobs(
        manager,
        include_archived=True,
    ) == 1
    archived = manager.get_job_or_archived(
        int(job["id"]),
        domain="prompt_studio",
    ) or {}
    _assert_payload_has_no_secret_aliases(
        archived.get("payload") or {},
        _SENTINEL,
    )


@pytest.mark.asyncio
async def test_live_optimization_payload_is_scrubbed_before_entity_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager(db_path=tmp_path / "malformed-live-job.sqlite")
    job = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload={
            "authorization": _SENTINEL,
            "optimization_config": {
                "model_config": {
                    **copy.deepcopy(_MODEL_CONFIG),
                    "api_key": _SENTINEL,
                }
            },
        },
        owner_user_id="7",
    )
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: (_ for _ in ()).throw(
            AssertionError("malformed job must not open a tenant Prompt DB")
        ),
        raising=True,
    )

    with pytest.raises(jobs_worker.PromptStudioJobError, match="Missing entity id"):
        await jobs_worker._handle_job(job, job_manager=manager)

    stored = manager.get_job(int(job["id"])) or {}
    _assert_payload_has_no_secret_aliases(stored.get("payload") or {}, _SENTINEL)


@pytest.mark.asyncio
async def test_archive_migration_scrubs_every_optimization_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs_path = tmp_path / "all-terminal-archives.sqlite"
    monkeypatch.setenv("PROMPT_STUDIO_CANCEL_RECONCILE_PAGE_SIZE", "1")
    manager = JobManager(db_path=jobs_path)
    statuses = (
        "queued",
        "processing",
        "completed",
        "failed",
        "cancelled",
        "quarantined",
    )
    conn = sqlite3.connect(jobs_path)
    try:
        for offset, status in enumerate(statuses, start=1):
            conn.execute(
                "INSERT INTO jobs_archive "
                "(id, uuid, domain, queue, job_type, payload, status, "
                "created_at, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    offset,
                    f"historical-{status}",
                    "prompt_studio",
                    "default",
                    "optimization",
                    json.dumps(
                        {
                            "optimization_id": offset,
                            "authorization": _SENTINEL,
                            "optimization_config": {
                                "model_config": {
                                    **copy.deepcopy(_MODEL_CONFIG),
                                    "api_key": _SENTINEL,
                                }
                            },
                        }
                    ),
                    status,
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                ),
            )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: (_ for _ in ()).throw(
            AssertionError("archive migration must not open a tenant Prompt DB")
        ),
        raising=True,
    )

    assert await jobs_worker._reconcile_cancelled_optimization_jobs(
        manager,
        include_archived=True,
    ) == len(statuses)

    for status in statuses:
        archived = manager.list_archived_jobs(
            domain="prompt_studio",
            status=status,
            job_type="optimization",
            limit=10,
        )
        assert len(archived) == 1
        _assert_payload_has_no_secret_aliases(archived[0]["payload"], _SENTINEL)


@pytest.mark.asyncio
async def test_archive_migration_clears_secret_bearing_compressed_copy_hidden_by_safe_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs_path = tmp_path / "hidden-compressed-secret.sqlite"
    manager = JobManager(db_path=jobs_path)
    safe_payload = {
        "optimization_id": 1,
        "version": "safe-primary",
        "optimization_config": {"model_config": copy.deepcopy(_MODEL_CONFIG)},
    }
    stale_payload = {
        **copy.deepcopy(safe_payload),
        "authorization": _SENTINEL,
    }
    compressed = "gzip64:" + base64.b64encode(
        gzip.compress(json.dumps(stale_payload).encode("utf-8"))
    ).decode("ascii")
    conn = sqlite3.connect(jobs_path)
    try:
        conn.execute(
            "INSERT INTO jobs_archive "
            "(id, uuid, domain, queue, job_type, payload, payload_compressed, "
            "status, created_at, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                "hidden-compressed-secret",
                "prompt_studio",
                "default",
                "optimization",
                json.dumps(safe_payload),
                compressed,
                "cancelled",
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: (_ for _ in ()).throw(
            AssertionError("archive scrub must not open a tenant Prompt DB")
        ),
        raising=True,
    )

    assert await jobs_worker._reconcile_cancelled_optimization_jobs(
        manager,
        include_archived=True,
    ) == 1

    conn = sqlite3.connect(jobs_path)
    try:
        raw_payload, raw_compressed = conn.execute(
            "SELECT payload, payload_compressed FROM jobs_archive"
        ).fetchone()
    finally:
        conn.close()
    assert json.loads(raw_payload) == safe_payload
    assert raw_compressed is None
    assert _SENTINEL not in raw_payload


@pytest.mark.asyncio
async def test_archive_migration_fail_closed_rewrites_malformed_payload_representations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs_path = tmp_path / "malformed-archive-representations.sqlite"
    manager = JobManager(db_path=jobs_path)
    rows = (
        (1, "json-list", json.dumps(["legacy", _SENTINEL]), None),
        (2, "invalid-json", f"not-json-{_SENTINEL}", None),
        (
            3,
            "corrupt-compressed",
            json.dumps({"version": "safe-primary"}),
            f"gzip64:not-valid-{_SENTINEL}",
        ),
    )
    conn = sqlite3.connect(jobs_path)
    try:
        for job_id, job_uuid, payload, payload_compressed in rows:
            conn.execute(
                "INSERT INTO jobs_archive "
                "(id, uuid, domain, queue, job_type, payload, payload_compressed, "
                "status, created_at, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    job_uuid,
                    "prompt_studio",
                    "default",
                    "optimization",
                    payload,
                    payload_compressed,
                    "cancelled",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                ),
            )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: (_ for _ in ()).throw(
            AssertionError("archive scrub must not open a tenant Prompt DB")
        ),
        raising=True,
    )

    assert await jobs_worker._reconcile_cancelled_optimization_jobs(
        manager,
        include_archived=True,
    ) == len(rows)

    conn = sqlite3.connect(jobs_path)
    try:
        stored = conn.execute(
            "SELECT uuid, payload, payload_compressed FROM jobs_archive ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert [(row[0], json.loads(row[1]), row[2]) for row in stored] == [
        ("json-list", {}, None),
        ("invalid-json", {}, None),
        ("corrupt-compressed", {"version": "safe-primary"}, None),
    ]
    assert _SENTINEL not in repr(stored)


@pytest.mark.asyncio
async def test_archive_migration_paginates_null_created_at_and_uuid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs_path = tmp_path / "null-archive-cursor.sqlite"
    monkeypatch.setenv("PROMPT_STUDIO_CANCEL_RECONCILE_PAGE_SIZE", "1")
    manager = JobManager(db_path=jobs_path)
    conn = sqlite3.connect(jobs_path)
    try:
        for version in ("first", "second"):
            conn.execute(
                "INSERT INTO jobs_archive "
                "(id, uuid, domain, queue, job_type, payload, status, "
                "created_at, archived_at) VALUES (?, NULL, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    9,
                    "prompt_studio",
                    "default",
                    "optimization",
                    json.dumps(
                        {
                            "version": version,
                            "authorization": _SENTINEL,
                            "optimization_config": {
                                "model_config": {"api_key": _SENTINEL}
                            },
                        }
                    ),
                    "cancelled",
                    "2026-01-01 00:00:00",
                ),
            )
        conn.commit()
    finally:
        conn.close()

    assert await jobs_worker._reconcile_cancelled_optimization_jobs(
        manager,
        include_archived=True,
    ) == 2
    archived = manager.list_archived_jobs(
        domain="prompt_studio",
        status="cancelled",
        job_type="optimization",
        limit=10,
    )
    assert len(archived) == 2
    for job in archived:
        _assert_payload_has_no_secret_aliases(job["payload"], _SENTINEL)


@pytest.mark.asyncio
async def test_archive_migration_decrypt_failure_preserves_original_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Security.crypto import encrypt_json_blob

    jobs_path = tmp_path / "encrypted-archive.sqlite"
    manager = JobManager(db_path=jobs_path)
    monkeypatch.setenv(
        "WORKFLOWS_ARTIFACT_ENC_KEY",
        base64.b64encode(b"A" * 32).decode("ascii"),
    )
    envelope = encrypt_json_blob(
        {
            "optimization_id": 1,
            "authorization": _SENTINEL,
        }
    )
    if envelope is None:
        pytest.skip("Crypto backend unavailable; skipping encryption test")

    conn = sqlite3.connect(jobs_path)
    try:
        conn.execute(
            "INSERT INTO jobs_archive "
            "(id, uuid, domain, queue, job_type, payload, payload_compressed, "
            "status, created_at, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                "encrypted-history",
                "prompt_studio",
                "default",
                "optimization",
                json.dumps({"_encrypted": envelope}, sort_keys=True),
                "legacy-compressed-copy",
                "cancelled",
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
            ),
        )
        conn.commit()
        before = conn.execute(
            "SELECT payload, payload_compressed FROM jobs_archive"
        ).fetchone()
    finally:
        conn.close()

    monkeypatch.setenv(
        "WORKFLOWS_ARTIFACT_ENC_KEY",
        base64.b64encode(b"B" * 32).decode("ascii"),
    )
    with pytest.raises(RuntimeError, match="decrypt") as exc_info:
        await jobs_worker._reconcile_cancelled_optimization_jobs(
            manager,
            include_archived=True,
        )
    assert type(exc_info.value).__name__ == "JobPayloadDecryptionError"

    conn = sqlite3.connect(jobs_path)
    try:
        after = conn.execute(
            "SELECT payload, payload_compressed FROM jobs_archive"
        ).fetchone()
    finally:
        conn.close()
    assert after == before


@pytest.mark.asyncio
async def test_archive_migration_database_scan_does_not_block_event_loop() -> None:
    heartbeat_ran = False

    class _SlowArchiveManager:
        archive_scanned = False

        def list_archived_jobs(self, **_kwargs: Any) -> list[dict[str, Any]]:
            if not self.archive_scanned:
                self.archive_scanned = True
                time.sleep(0.05)
            return []

        def list_jobs(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    async def _heartbeat() -> None:
        nonlocal heartbeat_ran
        await asyncio.sleep(0.01)
        heartbeat_ran = True

    heartbeat = asyncio.create_task(_heartbeat())
    await jobs_worker._reconcile_cancelled_optimization_jobs(
        _SlowArchiveManager(),  # type: ignore[arg-type]
        include_archived=True,
    )
    assert heartbeat_ran is True
    await heartbeat


@pytest.mark.asyncio
async def test_archive_reconciliation_paginates_reused_ids_with_same_created_at(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs_path = tmp_path / "archive-pagination.sqlite"
    monkeypatch.setenv("JOBS_SECRET_REDACT", "0")
    monkeypatch.setenv("JOBS_SECRET_REJECT", "0")
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("PROMPT_STUDIO_CANCEL_RECONCILE_PAGE_SIZE", "1")
    manager = JobManager(db_path=jobs_path)
    archived_jobs = []
    for version in ("first", "second"):
        job = manager.create_job(
            domain="prompt_studio",
            queue="default",
            job_type="optimization",
            payload={
                "version": version,
                "authorization": _SENTINEL,
                "optimization_config": {
                    "model_config": {"api_key": _SENTINEL}
                },
            },
            owner_user_id=None,
        )
        assert manager.cancel_job(int(job["id"]), reason="pagination regression")
        assert manager.prune_jobs(
            statuses=["cancelled"],
            older_than_days=0,
            domain="prompt_studio",
        ) == 1
        archived_jobs.append(job)

    for job, version in zip(
        archived_jobs,
        ("first", "second"),
        strict=True,
    ):
        _inject_historical_archive_payload(
            jobs_path,
            job_uuid=str(job["uuid"]),
            payload={
                "version": version,
                "authorization": _SENTINEL,
                "optimization_config": {
                    "model_config": {"api_key": _SENTINEL}
                },
            },
        )

    assert archived_jobs[0]["id"] == archived_jobs[1]["id"]
    conn = sqlite3.connect(jobs_path)
    try:
        conn.execute(
            "UPDATE jobs_archive SET created_at = ?",
            ("2026-01-01 00:00:00",),
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: (_ for _ in ()).throw(
            AssertionError("archive scrub must not open a tenant Prompt DB")
        ),
        raising=True,
    )

    assert await jobs_worker._reconcile_cancelled_optimization_jobs(
        manager,
        include_archived=True,
    ) == 2
    archived = manager.list_archived_jobs(
        domain="prompt_studio",
        status="cancelled",
        job_type="optimization",
        limit=10,
    )
    assert len(archived) == 2
    for job in archived:
        _assert_payload_has_no_secret_aliases(job["payload"], _SENTINEL)


@pytest.mark.asyncio
async def test_repeated_archived_cancellation_reconciliation_performs_no_durable_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "idempotent-archive-reconciliation")
    jobs_path = tmp_path / "idempotent-archive-reconciliation.sqlite"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_path))
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    manager = JobManager(db_path=jobs_path)
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="idempotent-archive-reconciliation",
        max_iterations=1,
    )
    job = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=_job(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=case_ids,
            owner_user_id=7,
            max_iterations=1,
            optimization_db=db,
        )["payload"],
        owner_user_id="7",
    )
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )

    try:
        assert manager.cancel_job(int(job["id"]), reason="already cancelled")
        assert await jobs_worker._reconcile_cancelled_optimization_jobs(manager) == 1
        assert manager.prune_jobs(
            statuses=["cancelled"],
            older_than_days=0,
            domain="prompt_studio",
        ) == 1

        writes = {"archive_payload": 0, "prompt_config": 0}
        replace_archived = manager.replace_archived_job_payload
        update_optimization = db.update_optimization

        def _count_archive_write(*args: Any, **kwargs: Any) -> bool:
            writes["archive_payload"] += 1
            return replace_archived(*args, **kwargs)

        def _count_prompt_write(*args: Any, **kwargs: Any) -> dict[str, Any]:
            updates = kwargs.get("updates")
            if updates is None and len(args) >= 2:
                updates = args[1]
            if isinstance(updates, dict) and "optimization_config" in updates:
                writes["prompt_config"] += 1
            return update_optimization(*args, **kwargs)

        monkeypatch.setattr(
            manager,
            "replace_archived_job_payload",
            _count_archive_write,
            raising=True,
        )
        monkeypatch.setattr(
            db,
            "update_optimization",
            _count_prompt_write,
            raising=True,
        )

        assert await jobs_worker._reconcile_cancelled_optimization_jobs(
            manager,
            include_archived=True,
        ) == 0
        assert await jobs_worker._reconcile_cancelled_optimization_jobs(
            manager,
            include_archived=True,
        ) == 0
        assert writes == {"archive_payload": 0, "prompt_config": 0}
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_archive_reconciliation_propagates_retryable_jobs_store_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "archive-store-failure")
    jobs_path = tmp_path / "archive-store-failure.sqlite"
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    manager = JobManager(db_path=jobs_path)
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="archive-store-failure",
        max_iterations=1,
    )
    payload = _job(
        optimization_id=optimization_id,
        prompt_id=prompt_id,
        test_case_ids=case_ids,
        owner_user_id=7,
        max_iterations=1,
    )["payload"]
    payload["authorization"] = _SENTINEL
    job = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=payload,
        owner_user_id="7",
    )
    assert manager.cancel_job(int(job["id"]), reason="archive store failure")
    assert manager.prune_jobs(
        statuses=["cancelled"],
        older_than_days=0,
        domain="prompt_studio",
    ) == 1
    _inject_historical_archive_payload(
        jobs_path,
        job_uuid=str(job["uuid"]),
        payload=payload,
    )

    def _unavailable(*_args: Any, **_kwargs: Any) -> bool:
        raise jobs_worker.PromptStudioJobError(
            "archive store unavailable",
            retryable=True,
            failure_code="job_store_unavailable",
        )

    monkeypatch.setattr(
        manager,
        "replace_archived_job_payload",
        _unavailable,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._reconcile_cancelled_optimization_jobs(
                manager,
                include_archived=True,
            )
        assert exc_info.value.failure_code == "job_store_unavailable"
        assert exc_info.value.retryable is True
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_retryable_archive_row_failure_advances_cycle_without_starving_older_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "archive-row-progress")
    jobs_path = tmp_path / "archive-row-progress.sqlite"
    manager = JobManager(db_path=jobs_path)
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("PROMPT_STUDIO_CANCEL_RECONCILE_PAGE_SIZE", "1")

    optimization_by_job_uuid: dict[str, int] = {}
    payload_by_job_uuid: dict[str, dict[str, Any]] = {}
    for suffix in ("older", "newer"):
        optimization_id, prompt_id, case_ids = _seed_optimization(
            db,
            name=f"archive-row-progress-{suffix}",
            max_iterations=1,
        )
        optimization = db.get_optimization(optimization_id) or {}
        payload = _job(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=case_ids,
            owner_user_id=7,
            max_iterations=1,
        )["payload"]
        payload["optimization_uuid"] = optimization["uuid"]
        created = manager.create_job(
            domain="prompt_studio",
            queue="default",
            job_type="optimization",
            payload=payload,
            owner_user_id="7",
        )
        assert manager.cancel_job(int(created["id"]), reason=suffix)
        job_uuid = str(created["uuid"])
        optimization_by_job_uuid[job_uuid] = optimization_id
        payload_by_job_uuid[job_uuid] = payload

    assert manager.prune_jobs(
        statuses=["cancelled"],
        older_than_days=0,
        domain="prompt_studio",
    ) == 2
    archived = manager.list_archived_jobs(
        domain="prompt_studio",
        status="cancelled",
        job_type="optimization",
        fail_on_decryption_error=True,
        limit=10,
    )
    assert len(archived) == 2
    failed_row, older_row = archived
    failed_uuid = str(failed_row["uuid"])
    older_uuid = str(older_row["uuid"])
    historical_payload = dict(payload_by_job_uuid[failed_uuid])
    historical_payload["authorization"] = _SENTINEL
    _inject_historical_archive_payload(
        jobs_path,
        job_uuid=failed_uuid,
        payload=historical_payload,
    )

    original_replace = manager.replace_archived_job_payload
    outage_active = True
    failed_row_replace_calls = 0

    def _replace_or_fail(*args: Any, **kwargs: Any) -> bool:
        nonlocal failed_row_replace_calls
        if kwargs.get("expected_uuid") == failed_uuid:
            failed_row_replace_calls += 1
            if outage_active:
                raise jobs_worker.PromptStudioJobError(
                    "archive row store unavailable",
                    retryable=True,
                    failure_code="job_store_unavailable",
                )
        return original_replace(*args, **kwargs)

    monkeypatch.setattr(
        manager,
        "replace_archived_job_payload",
        _replace_or_fail,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )

    state = jobs_worker._CancellationReconciliationState()
    failed_cursor = (
        jobs_worker._created_at_cursor(
            failed_row["_archive_cursor_created_at"]
        ),
        int(failed_row["id"]),
        str(failed_row.get("_archive_cursor_uuid") or ""),
        failed_row["_archive_locator"],
    )
    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._reconcile_cancelled_optimization_jobs(
                manager,
                include_archived=True,
                state=state,
            )
        assert exc_info.value.failure_code == "job_store_unavailable"
        assert state.archive_cursor == failed_cursor
        assert (
            db.get_optimization(optimization_by_job_uuid[failed_uuid]) or {}
        )["status"] == "pending"

        await jobs_worker._reconcile_cancelled_optimization_jobs(
            manager,
            include_archived=True,
            state=state,
        )
        assert (
            db.get_optimization(optimization_by_job_uuid[older_uuid]) or {}
        )["status"] == "cancelled"

        await jobs_worker._reconcile_cancelled_optimization_jobs(
            manager,
            include_archived=True,
            state=state,
        )
        assert state.archive_cursor is None

        outage_active = False
        await jobs_worker._reconcile_cancelled_optimization_jobs(
            manager,
            include_archived=True,
            state=state,
        )
        assert (
            db.get_optimization(optimization_by_job_uuid[failed_uuid]) or {}
        )["status"] == "cancelled"
        assert failed_row_replace_calls == 2
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_cyclic_archive_sweep_catches_job_pruned_ahead_of_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prune race missed by one page must converge after the cursor wraps."""

    db = _new_db(tmp_path, "archive-cyclic-prune-race")
    manager = JobManager(db_path=tmp_path / "archive-cyclic-prune-race.sqlite")
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("PROMPT_STUDIO_CANCEL_RECONCILE_PAGE_SIZE", "1")

    optimization_ids: list[int] = []
    for suffix in ("first", "racing"):
        optimization_id, prompt_id, case_ids = _seed_optimization(
            db,
            name=f"archive-cyclic-{suffix}",
            max_iterations=1,
        )
        optimization = db.get_optimization(optimization_id) or {}
        payload = _job(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=case_ids,
            owner_user_id=7,
            max_iterations=1,
        )["payload"]
        payload["optimization_uuid"] = optimization["uuid"]
        created = manager.create_job(
            domain="prompt_studio",
            queue="default",
            job_type="optimization",
            payload=payload,
            owner_user_id="7",
        )
        assert manager.cancel_job(int(created["id"]), reason=suffix)
        optimization_ids.append(optimization_id)
        if suffix == "first":
            assert manager.prune_jobs(
                statuses=["cancelled"],
                older_than_days=0,
                domain="prompt_studio",
            ) == 1

    archive_list_calls = 0
    original_list_archived = manager.list_archived_jobs
    original_list_jobs = manager.list_jobs
    skipped_live_cancelled_page = False

    def _list_live_before_prune(**kwargs: Any) -> list[dict[str, Any]]:
        nonlocal skipped_live_cancelled_page
        if kwargs.get("status") == "cancelled" and not skipped_live_cancelled_page:
            skipped_live_cancelled_page = True
            return []
        return original_list_jobs(**kwargs)

    def _list_then_prune(**kwargs: Any) -> list[dict[str, Any]]:
        nonlocal archive_list_calls
        rows = original_list_archived(**kwargs)
        archive_list_calls += 1
        if archive_list_calls == 1:
            assert manager.prune_jobs(
                statuses=["cancelled"],
                older_than_days=0,
                domain="prompt_studio",
            ) == 1
        return rows

    monkeypatch.setattr(
        manager,
        "list_archived_jobs",
        _list_then_prune,
        raising=True,
    )
    monkeypatch.setattr(
        manager,
        "list_jobs",
        _list_live_before_prune,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )

    state = jobs_worker._CancellationReconciliationState()
    try:
        for _ in range(3):
            await jobs_worker._reconcile_cancelled_optimization_jobs(
                manager,
                include_archived=True,
                state=state,
            )

        racing = db.get_optimization(optimization_ids[1]) or {}
        assert racing["status"] == "cancelled"
        assert archive_list_calls == 3
    finally:
        db.close_connection()


def test_archived_terminal_reconciliation_rejects_stale_optimization_uuid(
    tmp_path: Path,
) -> None:
    """An old archive cannot mutate a reused tenant-local numeric ID."""

    db = _new_db(tmp_path, "archive-stale-optimization-uuid")
    manager = JobManager(db_path=tmp_path / "archive-stale-uuid.sqlite")
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="archive-stale-optimization-uuid",
        max_iterations=1,
    )
    job = _job(
        optimization_id=optimization_id,
        prompt_id=prompt_id,
        test_case_ids=case_ids,
        owner_user_id=7,
        max_iterations=1,
    )
    job["status"] = "cancelled"
    job["payload"]["optimization_uuid"] = "stale-optimization-uuid"

    try:
        transitioned = jobs_worker._converge_terminal_prompt_state(
            processor=JobProcessor(db),
            optimization_id=optimization_id,
            job=job,
            job_manager=manager,
            archived=True,
            require_identity=True,
        )
        optimization = db.get_optimization(optimization_id) or {}
        assert transitioned is False
        assert optimization["status"] == "pending"
    finally:
        db.close_connection()


def test_archived_terminal_reconciliation_fails_closed_without_identity(
    tmp_path: Path,
) -> None:
    """Recurring archive repair only mutates jobs with durable row identity."""

    db = _new_db(tmp_path, "archive-missing-optimization-uuid")
    manager = JobManager(db_path=tmp_path / "archive-missing-uuid.sqlite")
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="archive-missing-optimization-uuid",
        max_iterations=1,
    )
    job = _job(
        optimization_id=optimization_id,
        prompt_id=prompt_id,
        test_case_ids=case_ids,
        owner_user_id=7,
        max_iterations=1,
    )
    job["status"] = "cancelled"

    try:
        transitioned = jobs_worker._converge_terminal_prompt_state(
            processor=JobProcessor(db),
            optimization_id=optimization_id,
            job=job,
            job_manager=manager,
            archived=True,
            require_identity=True,
        )
        optimization = db.get_optimization(optimization_id) or {}
        assert transitioned is False
        assert optimization["status"] == "pending"
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    ("helper", "replacement_status"),
    [
        (jobs_worker._mark_failed_safely, "pending"),
        (jobs_worker._mark_retry_pending_safely, "running"),
    ],
)
def test_failure_repair_cas_does_not_mutate_concurrently_replaced_optimization(
    tmp_path: Path,
    helper: Callable[..., None],
    replacement_status: str,
) -> None:
    """Late failure repair must target the immutable Prompt row identity."""

    db = _new_db(tmp_path, f"failure-repair-{replacement_status}")
    optimization_id, _prompt_id, _case_ids = _seed_optimization(
        db,
        name=f"failure-repair-{replacement_status}",
        max_iterations=1,
    )
    original = db.get_optimization(optimization_id) or {}
    original_uuid = str(original["uuid"])
    replacement_uuid = f"replacement-{replacement_status}"
    started = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []
    error = jobs_worker.PromptStudioJobError(
        "bounded failure",
        retryable=replacement_status == "running",
        failure_code="provider_unavailable",
    )

    def _late_repair() -> None:
        started.set()
        release.wait(timeout=2)
        try:
            helper(
                JobProcessor(db),
                optimization_id,
                error,
                expected_uuid=original_uuid,
            )
        except BaseException as exc:  # noqa: BLE001 - cross-thread assertion
            failures.append(exc)

    repair_thread = threading.Thread(target=_late_repair)
    repair_thread.start()
    assert started.wait(timeout=2)
    try:
        db.update_optimization(
            optimization_id,
            {"uuid": replacement_uuid, "status": replacement_status},
            expected_uuid=original_uuid,
        )
    finally:
        release.set()
        repair_thread.join(timeout=2)

    try:
        assert repair_thread.is_alive() is False
        assert failures == []
        replacement = db.get_optimization(optimization_id) or {}
        assert replacement["uuid"] == replacement_uuid
        assert replacement["status"] == replacement_status
        assert replacement.get("error_message") is None
    finally:
        db.close_connection()


def test_config_scrub_cas_does_not_overwrite_concurrently_replaced_optimization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delayed secret scrub must not overwrite a replacement Prompt row."""

    db = _new_db(tmp_path, "config-scrub-replacement")
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="config-scrub-replacement",
        max_iterations=1,
    )
    original = db.get_optimization(optimization_id) or {}
    original_uuid = str(original["uuid"])
    sensitive_config = {
        "optimizer_type": "mipro",
        "target_metric": "accuracy",
        "model_config": {**copy.deepcopy(_MODEL_CONFIG), "api_key": _SENTINEL},
    }
    replacement_config = {
        "optimizer_type": "mipro",
        "target_metric": "f1",
        "model_config": {
            "provider": "openai",
            "model": "replacement-model",
        },
    }
    job = _job(
        optimization_id=optimization_id,
        prompt_id=prompt_id,
        test_case_ids=case_ids,
        owner_user_id=7,
        max_iterations=1,
    )
    job["payload"]["optimization_uuid"] = original_uuid
    original_update = db.update_optimization
    original_get = db.get_optimization
    get_calls = 0
    raced = False

    def _historical_first_read(*args: Any, **kwargs: Any) -> Any:
        nonlocal get_calls
        get_calls += 1
        row = original_get(*args, **kwargs)
        if get_calls == 1 and row is not None:
            row = dict(row)
            row["optimization_config"] = sensitive_config
        return row

    def _racing_update(*args: Any, **kwargs: Any) -> Any:
        nonlocal raced
        updates = args[1] if len(args) > 1 else kwargs.get("updates", {})
        if not raced and "optimization_config" in updates:
            raced = True
            original_update(
                optimization_id,
                {
                    "uuid": "replacement-config-row",
                    "optimization_config": replacement_config,
                },
                expected_uuid=original_uuid,
            )
        return original_update(*args, **kwargs)

    monkeypatch.setattr(db, "get_optimization", _historical_first_read, raising=True)
    monkeypatch.setattr(db, "update_optimization", _racing_update, raising=True)

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            jobs_worker._secure_optimization_durable_state(
                processor=JobProcessor(db),
                optimization_id=optimization_id,
                job=job,
                payload=job["payload"],
                job_manager=None,
                require_valid_config=False,
            )
        assert exc_info.value.failure_code == "job_identity_invalid"
        replacement = db.get_optimization(optimization_id) or {}
        assert replacement["uuid"] == "replacement-config-row"
        assert replacement["optimization_config"] == replacement_config
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_live_worker_rejects_stale_optimization_uuid_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A queued job from an old tenant DB cannot execute against a reused ID."""

    db = _new_db(tmp_path, "live-stale-optimization-uuid")
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="live-stale-optimization-uuid",
        max_iterations=1,
    )
    job = _job(
        optimization_id=optimization_id,
        prompt_id=prompt_id,
        test_case_ids=case_ids,
        owner_user_id=7,
        max_iterations=1,
    )
    job["payload"]["optimization_uuid"] = "stale-optimization-uuid"
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )
    monkeypatch.setattr(jobs_worker, "_auth_mode", lambda: "single_user", raising=True)

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(job)
        optimization = db.get_optimization(optimization_id) or {}
        assert exc_info.value.failure_code == "job_identity_invalid"
        assert optimization["status"] == "pending"
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_cancellation_reconciliation_loop_scans_archive_on_every_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    include_archived_calls: list[bool] = []
    states: list[object | None] = []
    sleep_calls = 0

    async def _reconcile(
        _manager: object,
        *,
        include_archived: bool = False,
        state: object | None = None,
    ) -> int:
        include_archived_calls.append(include_archived)
        states.append(state)
        return 0

    async def _sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(
        jobs_worker,
        "_reconcile_cancelled_optimization_jobs",
        _reconcile,
        raising=True,
    )
    monkeypatch.setattr(asyncio, "sleep", _sleep, raising=True)

    with pytest.raises(asyncio.CancelledError):
        await jobs_worker._cancelled_job_reconciliation_loop(object())

    assert include_archived_calls == [True, True]
    assert states[0] is not None
    assert states[1] is states[0]


@pytest.mark.asyncio
async def test_cancellation_reconciliation_loop_retries_archive_after_store_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    include_archived_calls: list[bool] = []
    sleep_calls = 0

    async def _reconcile(
        _manager: object,
        *,
        include_archived: bool = False,
        state: object | None = None,
    ) -> int:
        del state
        include_archived_calls.append(include_archived)
        if len(include_archived_calls) == 1:
            raise RuntimeError("archive store unavailable")
        return 0

    async def _sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(
        jobs_worker,
        "_reconcile_cancelled_optimization_jobs",
        _reconcile,
        raising=True,
    )
    monkeypatch.setattr(asyncio, "sleep", _sleep, raising=True)

    with pytest.raises(asyncio.CancelledError):
        await jobs_worker._cancelled_job_reconciliation_loop(object())

    assert include_archived_calls == [True, True]


@pytest.mark.asyncio
async def test_acquired_cancelled_legacy_job_is_scrubbed_before_handler_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "acquired-cancelled-legacy-core-job")
    monkeypatch.setenv("JOBS_SECRET_REDACT", "0")
    monkeypatch.setenv("JOBS_SECRET_REJECT", "0")
    manager = JobManager(db_path=tmp_path / "acquired-cancelled-jobs.sqlite")
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="acquired-cancelled-legacy-core-job",
        max_iterations=1,
    )
    legacy_job = _job(
        optimization_id=optimization_id,
        prompt_id=prompt_id,
        test_case_ids=case_ids,
        owner_user_id=7,
        max_iterations=1,
        optimization_db=db,
    )
    legacy_job["payload"]["authorization"] = _SENTINEL
    legacy_job["payload"]["optimization_config"]["model_config"][
        "api_key"
    ] = _SENTINEL
    created = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=legacy_job["payload"],
        owner_user_id="7",
    )
    acquired = manager.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=60,
        worker_id="legacy-cancel-test",
    )
    assert acquired is not None
    assert acquired["id"] == created["id"]
    assert manager.cancel_job(int(acquired["id"]), reason="admin cancellation")
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(acquired, job_manager=manager)

        _assert_safe_job_error(
            exc_info.value,
            code="job_cancelled",
            retryable=False,
        )
        live = manager.get_job(int(acquired["id"])) or {}
        _assert_payload_has_no_secret_aliases(live.get("payload") or {}, _SENTINEL)
        optimization = db.get_optimization(
            optimization_id,
            include_deleted=True,
        ) or {}
        assert optimization["status"] == "cancelled"
        _assert_payload_has_no_secret_aliases(
            optimization.get("optimization_config") or {},
            _SENTINEL,
        )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_generic_jobs_cancellation_after_non_mcts_provider_success_stops_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "non-mcts-generic-cancel")
    manager = JobManager(db_path=tmp_path / "non-mcts-generic-cancel-jobs.sqlite")
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )
    factory = _RuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)
    dispatches: list[str] = []
    acquired: dict[str, Any] | None = None

    class _TwoDispatchProcessor:
        def __init__(self, database: PromptStudioDatabase) -> None:
            self.db = database

        async def process_optimization_job(
            self,
            _payload: dict[str, Any],
            optimization_id: int,
            **kwargs: Any,
        ) -> dict[str, Any]:
            assert acquired is not None
            dispatches.append("first")
            assert manager.cancel_job(
                int(acquired["id"]),
                reason="cancel after first provider call",
            )
            await kwargs["on_provider_success"]()
            dispatches.append("second")
            await kwargs["on_provider_success"]()
            return {
                "optimization_id": optimization_id,
                "status": "completed",
                "iterations_completed": 2,
            }

    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="non-mcts-generic-cancel",
        max_iterations=2,
        optimizer_type="iterative",
    )
    queued = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=_job(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=case_ids,
            owner_user_id=7,
            max_iterations=2,
            optimizer_type="iterative",
            optimization_db=db,
        )["payload"],
        owner_user_id="7",
    )
    acquired = manager.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=60,
        worker_id="non-mcts-cancel-test",
    )
    assert acquired is not None
    assert acquired["id"] == queued["id"]
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: _TwoDispatchProcessor(db),
        raising=True,
    )
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager, raising=True)

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(acquired, job_manager=manager)

        _assert_safe_job_error(
            exc_info.value,
            code="job_cancelled",
            retryable=False,
        )
        assert dispatches == ["first"]
        assert (manager.get_job(int(acquired["id"])) or {})["status"] == "cancelled"
        assert (
            db.get_optimization(optimization_id, include_deleted=True) or {}
        )["status"] == "cancelled"
        assert factory.instances[0].mark_calls == [handle]
        assert factory.instances[0].close_count == 1
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_generic_jobs_cancellation_wins_when_provider_raises_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "provider-error-cancel-race")
    manager = JobManager(db_path=tmp_path / "provider-error-cancel-race-jobs.sqlite")
    events: list[Any] = []
    factory = _RuntimeFactory(
        _Handle(
            api_key=_RESOLVED_KEY,
            app_config={"openai_api": {"model": "gpt-4o-mini"}},
        ),
        events,
    )
    _install_runtime_and_memberships(monkeypatch, factory)
    acquired: dict[str, Any] | None = None

    class _CancelledFailureProcessor:
        def __init__(self, database: PromptStudioDatabase) -> None:
            self.db = database

        async def process_optimization_job(
            self,
            _payload: dict[str, Any],
            _optimization_id: int,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            assert acquired is not None
            assert manager.cancel_job(
                int(acquired["id"]),
                reason="cancel while provider fails",
            )
            raise TimeoutError(_SENTINEL)

    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="provider-error-cancel-race",
        max_iterations=1,
    )
    queued = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=_job(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=case_ids,
            owner_user_id=7,
            max_iterations=1,
            optimization_db=db,
        )["payload"],
        owner_user_id="7",
        max_retries=0,
    )
    acquired = manager.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=60,
        worker_id="provider-error-cancel-test",
    )
    assert acquired is not None
    assert acquired["id"] == queued["id"]
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: _CancelledFailureProcessor(db),
        raising=True,
    )
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager, raising=True)

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(acquired, job_manager=manager)

        _assert_safe_job_error(
            exc_info.value,
            code="job_cancelled",
            retryable=False,
        )
        row = db.get_optimization(optimization_id, include_deleted=True) or {}
        assert row["status"] == "cancelled"
        assert _SENTINEL not in str(row.get("error_message") or "")
        assert factory.instances[0].close_count == 1
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_jobs_state_outage_during_provider_failure_keeps_prompt_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "provider-error-jobs-state-outage")
    manager = JobManager(
        db_path=tmp_path / "provider-error-jobs-state-outage.sqlite"
    )
    events: list[Any] = []
    factory = _RuntimeFactory(
        _Handle(
            api_key=_RESOLVED_KEY,
            app_config={"openai_api": {"model": "gpt-4o-mini"}},
        ),
        events,
    )
    _install_runtime_and_memberships(monkeypatch, factory)

    class _FailingProcessor:
        def __init__(self, database: PromptStudioDatabase) -> None:
            self.db = database

        async def process_optimization_job(
            self,
            _payload: dict[str, Any],
            _optimization_id: int,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            raise TimeoutError(_SENTINEL)

    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="provider-error-jobs-state-outage",
        max_iterations=1,
    )
    manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=_job(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=case_ids,
            owner_user_id=7,
            max_iterations=1,
            optimization_db=db,
        )["payload"],
        owner_user_id="7",
        max_retries=0,
    )
    acquired = manager.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=60,
        worker_id="provider-error-state-outage-test",
    )
    assert acquired is not None
    original_get_job = manager.get_job
    state_reads = 0

    def _flaky_get_job(job_id: int) -> dict[str, Any] | None:
        nonlocal state_reads
        state_reads += 1
        if state_reads > 1:
            raise RuntimeError(_SENTINEL)
        return original_get_job(job_id)

    monkeypatch.setattr(manager, "get_job", _flaky_get_job, raising=True)
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: _FailingProcessor(db),
        raising=True,
    )
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager, raising=True)

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(acquired, job_manager=manager)

        _assert_safe_job_error(
            exc_info.value,
            code="job_store_unavailable",
            retryable=True,
        )
        row = db.get_optimization(optimization_id, include_deleted=True) or {}
        assert row["status"] == "pending"
        assert row.get("completed_at") is None
        assert _SENTINEL not in str(row.get("error_message") or "")
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_core_cancel_after_final_gate_overrides_prompt_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "cancel-after-final-gate")
    manager = JobManager(db_path=tmp_path / "cancel-after-final-gate-jobs.sqlite")
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )
    factory = _RuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)
    acquired: dict[str, Any] | None = None

    class _FinalGateRaceProcessor:
        def __init__(self, database: PromptStudioDatabase) -> None:
            self.db = database

        async def process_optimization_job(
            self,
            _payload: dict[str, Any],
            optimization_id: int,
            **kwargs: Any,
        ) -> dict[str, Any]:
            assert acquired is not None
            await kwargs["on_provider_success"]()
            assert await kwargs["before_finalize"]() is False
            assert manager.cancel_job(
                int(acquired["id"]),
                reason="cancel in finalization window",
            )
            db.complete_optimization(
                optimization_id,
                iterations_completed=1,
                final_metrics={"score": 0.9},
            )
            return {
                "optimization_id": optimization_id,
                "status": "completed",
                "iterations_completed": 1,
            }

    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="cancel-after-final-gate",
        max_iterations=1,
    )
    queued = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=_job(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=case_ids,
            owner_user_id=7,
            max_iterations=1,
            optimization_db=db,
        )["payload"],
        owner_user_id="7",
    )
    acquired = manager.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=60,
        worker_id="final-gate-race-test",
    )
    assert acquired is not None
    assert acquired["id"] == queued["id"]
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: _FinalGateRaceProcessor(db),
        raising=True,
    )
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager, raising=True)

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(acquired, job_manager=manager)

        _assert_safe_job_error(
            exc_info.value,
            code="job_cancelled",
            retryable=False,
        )
        row = db.get_optimization(optimization_id, include_deleted=True) or {}
        assert row["status"] == "cancelled"
        assert (manager.get_job(int(acquired["id"])) or {})["status"] == "cancelled"
        assert factory.instances[0].mark_calls == [handle]
        assert factory.instances[0].close_count == 1
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_cancelled_prompt_outcome_cancels_processing_core_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "prompt-cancelled-core-processing")
    manager = JobManager(db_path=tmp_path / "prompt-cancelled-core-processing-jobs.sqlite")
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="prompt-cancelled-core-processing",
        max_iterations=1,
    )
    queued = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=_job(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=case_ids,
            owner_user_id=7,
            max_iterations=1,
            optimization_db=db,
        )["payload"],
        owner_user_id="7",
    )
    acquired = manager.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=60,
        worker_id="prompt-cancelled-test",
    )
    assert acquired is not None
    assert acquired["id"] == queued["id"]
    db.set_optimization_status(
        optimization_id,
        "cancelled",
        error_message="domain cancellation won first",
        mark_completed=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager, raising=True)

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(acquired, job_manager=manager)

        _assert_safe_job_error(
            exc_info.value,
            code="job_cancelled",
            retryable=False,
        )
        assert (manager.get_job(int(acquired["id"])) or {})["status"] == "cancelled"
        assert (
            db.get_optimization(optimization_id, include_deleted=True) or {}
        )["status"] == "cancelled"
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_completed_prompt_row_is_idempotent_recovery_for_processing_core_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "completed-prompt-recovery")
    manager = JobManager(db_path=tmp_path / "completed-prompt-recovery-jobs.sqlite")
    events: list[Any] = []
    factory = _RuntimeFactory(
        _Handle(
            api_key=_RESOLVED_KEY,
            app_config={"openai_api": {"model": "gpt-4o-mini"}},
        ),
        events,
    )
    _install_runtime_and_memberships(monkeypatch, factory)
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="completed-prompt-recovery",
        max_iterations=2,
    )
    db.complete_optimization(
        optimization_id,
        optimized_prompt_id=prompt_id,
        final_metrics={"score": 0.75},
        iterations_completed=2,
    )
    queued = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=_job(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=case_ids,
            owner_user_id=7,
            max_iterations=2,
            optimization_db=db,
        )["payload"],
        owner_user_id="7",
    )
    acquired = manager.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=60,
        worker_id="completed-recovery-test",
    )
    assert acquired is not None
    assert acquired["id"] == queued["id"]
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager, raising=True)

    try:
        result = await jobs_worker._handle_job(acquired, job_manager=manager)

        assert result["status"] == "completed"
        assert result["iterations_completed"] == 2
        assert result["best_prompt_id"] == prompt_id
        assert result["best_metric"] == 0.75
        assert factory.instances == []
        assert manager.complete_job(
            int(acquired["id"]),
            result=result,
            worker_id="completed-recovery-test",
            lease_id=str(acquired["lease_id"]),
            enforce=True,
        )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_durable_job_passes_one_resolved_snapshot_through_runner_and_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "snapshot")
    events: list[Any] = []
    handle = _Handle(
        api_key=_SENTINEL,
        app_config={
            "openai_api": {
                "model": "gpt-4o-mini",
                "api_key": _SENTINEL,
                "base_url": "https://configured.example/v1",
            }
        },
    )
    factory = _RuntimeFactory(handle, events)
    membership_calls = _install_runtime_and_memberships(monkeypatch, factory)
    _install_safe_config_fallbacks(monkeypatch)
    requests: list[tuple[str, dict[str, Any]]] = []

    class _Adapter:
        def __init__(self, boundary: str) -> None:
            self.boundary = boundary

        def chat(
            self,
            request: dict[str, Any],
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            requests.append((self.boundary, copy.deepcopy(request)))
            content = (
                "candidate instruction"
                if self.boundary == "executor"
                else "totally different"
            )
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {"total_tokens": 2},
            }

    class _Registry:
        @staticmethod
        def is_local_provider_name(_provider: str) -> bool:
            return False

        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter("executor")

    monkeypatch.setattr(
        test_runner_module,
        "get_adapter_or_raise",
        lambda _provider: _Adapter("runner"),
        raising=True,
    )
    monkeypatch.setattr(
        prompt_executor_module,
        "get_registry",
        lambda: _Registry(),
        raising=True,
    )
    processor = JobProcessor(db)
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: processor,
        raising=True,
    )

    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="snapshot",
        max_iterations=1,
    )
    job = _job(
        optimization_id=optimization_id,
        prompt_id=prompt_id,
        test_case_ids=case_ids,
        owner_user_id=7,
        max_iterations=1,
    )
    original_payload = copy.deepcopy(job["payload"])

    try:
        result = await jobs_worker._handle_job(job)

        assert result["final_score"] == pytest.approx(0.0)
        assert result["status"] == "completed"
        assert {boundary for boundary, _request in requests} == {
            "runner",
            "executor",
        }
        assert requests
        for _boundary, request in requests:
            assert request["api_key"] == _SENTINEL
            assert request["app_config"] == handle.app_config
            assert request["credentials_resolved"] is True

        assert len(factory.instances) == 1
        runtime = factory.instances[0]
        assert runtime.scope["user_id"] == 7
        assert runtime.scope["team_ids"] == [71]
        assert runtime.scope["org_ids"] == [72]
        assert runtime.scope["trusted_base_url_override"] is False
        assert runtime.resolve_calls == [("openai", "gpt-4o-mini")]
        assert runtime.mark_calls == [handle]
        assert runtime.close_count == 1
        assert events[-2:] == ["mark", "close"]
        assert sorted(membership_calls) == [("org", 7), ("team", 7)]

        assert job["payload"] == original_payload
        assert _SENTINEL not in json.dumps(job["payload"], sort_keys=True)
        row = db.get_optimization(optimization_id, include_deleted=True) or {}
        assert _SENTINEL not in json.dumps(
            row.get("optimization_config") or {},
            sort_keys=True,
        )
        assert _retains_identity(processor, runtime) is False
        assert _retains_identity(processor, handle) is False
        durable_dump = _assert_all_durable_state_secret_free(
            db,
            _SENTINEL,
        )
        assert 'INSERT INTO "prompt_studio_test_runs"' in durable_dump
        assert 'INSERT INTO "prompt_studio_optimizations"' in durable_dump
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_mcts_scorer_model_policy_denial_precedes_scorer_adapter_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "mcts-scorer-denied")
    events: list[Any] = []
    base_model = "gpt-4o-mini"
    scorer_model = "gpt-scorer-denied"
    base_handle = _Handle(
        api_key="base-model-key",
        app_config={"openai_api": {"model": base_model}},
    )

    def _resolve_for_model(
        _scope: dict[str, Any],
        _index: int,
        provider: str,
        model: str | None,
    ) -> _Handle | BaseException:
        if model == scorer_model:
            return ProviderOverridePolicyError("model_not_allowed", provider)
        return base_handle

    factory = _RuntimeFactory(_resolve_for_model, events)
    _install_runtime_and_memberships(monkeypatch, factory)
    _install_safe_config_fallbacks(monkeypatch)
    adapter_requests: list[dict[str, Any]] = []

    class _Adapter:
        def chat(
            self,
            request: dict[str, Any],
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            adapter_requests.append(copy.deepcopy(request))
            content = "8" if request.get("model") == scorer_model else "candidate"
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {"total_tokens": 1},
            }

    class _Registry:
        @staticmethod
        def is_local_provider_name(_provider: str) -> bool:
            return False

        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    monkeypatch.setattr(
        test_runner_module,
        "get_adapter_or_raise",
        lambda _provider: _Adapter(),
        raising=True,
    )
    monkeypatch.setattr(
        prompt_executor_module,
        "get_registry",
        lambda: _Registry(),
        raising=True,
    )
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="mcts-scorer-denied",
        max_iterations=1,
        optimizer_type="mcts",
        strategy_params={
            "mcts_simulations": 1,
            "mcts_max_depth": 1,
            "prompt_candidates_per_node": 1,
            "feedback_enabled": False,
            "scorer_model": scorer_model,
        },
    )
    processor = JobProcessor(db)
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: processor,
        raising=True,
    )

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(
                _job(
                    optimization_id=optimization_id,
                    prompt_id=prompt_id,
                    test_case_ids=case_ids,
                    owner_user_id=7,
                    max_iterations=1,
                    optimizer_type="mcts",
                    strategy_params={
                        "mcts_simulations": 1,
                        "mcts_max_depth": 1,
                        "prompt_candidates_per_node": 1,
                        "feedback_enabled": False,
                        "scorer_model": scorer_model,
                    },
                )
            )

        _assert_safe_job_error(
            exc_info.value,
            code="model_not_allowed",
            retryable=False,
        )
        runtime = factory.instances[0]
        assert runtime.resolve_calls == [
            ("openai", base_model),
            ("openai", scorer_model),
        ]
        assert [
            request
            for request in adapter_requests
            if request.get("model") == scorer_model
        ] == []
        assert runtime.mark_calls == []
        assert runtime.close_count == 1
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_mcts_scorer_dispatch_uses_separately_resolved_runtime_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "mcts-scorer-snapshot")
    events: list[Any] = []
    base_model = "gpt-4o-mini"
    scorer_model = "gpt-scorer-allowed"
    base_key = "TASK12963_BASE_MODEL_KEY"
    scorer_key = "TASK12963_SCORER_MODEL_KEY"
    base_handle = _Handle(
        api_key=base_key,
        app_config={"openai_api": {"model": base_model, "scope": "base"}},
    )
    scorer_handle = _Handle(
        api_key=scorer_key,
        app_config={"openai_api": {"model": scorer_model, "scope": "scorer"}},
    )

    def _resolve_for_model(
        _scope: dict[str, Any],
        _index: int,
        _provider: str,
        model: str | None,
    ) -> _Handle:
        return scorer_handle if model == scorer_model else base_handle

    factory = _RuntimeFactory(_resolve_for_model, events)
    _install_runtime_and_memberships(monkeypatch, factory)
    _install_safe_config_fallbacks(monkeypatch)
    adapter_requests: list[dict[str, Any]] = []

    class _Adapter:
        def chat(
            self,
            request: dict[str, Any],
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            adapter_requests.append(copy.deepcopy(request))
            content = "8" if request.get("model") == scorer_model else "candidate"
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {"total_tokens": 1},
            }

    class _Registry:
        @staticmethod
        def is_local_provider_name(_provider: str) -> bool:
            return False

        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    monkeypatch.setattr(
        test_runner_module,
        "get_adapter_or_raise",
        lambda _provider: _Adapter(),
        raising=True,
    )
    monkeypatch.setattr(
        prompt_executor_module,
        "get_registry",
        lambda: _Registry(),
        raising=True,
    )
    strategy_params = {
        "mcts_simulations": 1,
        "mcts_max_depth": 1,
        "prompt_candidates_per_node": 1,
        "feedback_enabled": False,
        "scorer_model": scorer_model,
    }
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="mcts-scorer-snapshot",
        max_iterations=1,
        optimizer_type="mcts",
        strategy_params=strategy_params,
    )
    processor = JobProcessor(db)
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: processor,
        raising=True,
    )

    try:
        result = await jobs_worker._handle_job(
            _job(
                optimization_id=optimization_id,
                prompt_id=prompt_id,
                test_case_ids=case_ids,
                owner_user_id=7,
                max_iterations=1,
                optimizer_type="mcts",
                strategy_params=strategy_params,
            )
        )

        assert result["status"] == "completed"
        runtime = factory.instances[0]
        assert runtime.resolve_calls == [
            ("openai", base_model),
            ("openai", scorer_model),
        ]
        scorer_requests = [
            request
            for request in adapter_requests
            if request.get("model") == scorer_model
        ]
        assert scorer_requests
        assert all(request["api_key"] == scorer_key for request in scorer_requests)
        assert all(
            request["app_config"] == scorer_handle.app_config
            for request in scorer_requests
        )
        assert scorer_handle in runtime.mark_calls
        assert runtime.close_count == 1
        _assert_all_durable_state_secret_free(db, base_key, scorer_key)
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_durable_job_requires_one_validated_baseline_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "zero-baseline")
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )
    factory = _RuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)
    _install_safe_config_fallbacks(monkeypatch)

    class _ForbiddenAdapter:
        def chat(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("no provider call is valid without a baseline case")

    monkeypatch.setattr(
        test_runner_module,
        "get_adapter_or_raise",
        lambda _provider: _ForbiddenAdapter(),
        raising=True,
    )
    processor = JobProcessor(db)
    monkeypatch.setattr(jobs_worker, "_get_processor", lambda _uid: processor, raising=True)
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="zero-baseline",
        max_iterations=0,
        with_test_case=False,
    )

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(
                _job(
                    optimization_id=optimization_id,
                    prompt_id=prompt_id,
                    test_case_ids=case_ids,
                    owner_user_id=7,
                    max_iterations=0,
                )
            )

        _assert_safe_job_error(
            exc_info.value,
            code="provider_configuration_invalid",
            retryable=False,
        )
        row = db.get_optimization(optimization_id, include_deleted=True) or {}
        assert row["status"] == "failed"
        assert len(factory.instances) == 1
        assert factory.instances[0].mark_calls == []
        assert factory.instances[0].close_count == 1
        _assert_all_durable_state_secret_free(db, _RESOLVED_KEY)
    finally:
        db.close_connection()


_RUNTIME_FAILURES = [
    pytest.param(
        ByokResolutionError("credential_scope_revoked", "openai"),
        "credential_scope_revoked",
        False,
        id="revoked-scope",
    ),
    pytest.param(
        ByokResolutionError("credential_store_unavailable", "openai"),
        "credential_store_unavailable",
        True,
        id="credential-store-unavailable",
    ),
    pytest.param(
        _Handle(
            api_key=None,
            app_config={"openai_api": {"model": "gpt-4o-mini"}},
        ),
        "missing_provider_credentials",
        False,
        id="missing-credentials",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("runtime_outcome", "expected_code", "retryable"), _RUNTIME_FAILURES)
async def test_durable_job_fails_closed_on_runtime_credential_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_outcome: _Handle | BaseException,
    expected_code: str,
    retryable: bool,
) -> None:
    db = _new_db(tmp_path, f"runtime-failure-{expected_code}")
    events: list[Any] = []
    factory = _RuntimeFactory(runtime_outcome, events)
    _install_runtime_and_memberships(monkeypatch, factory)
    _install_safe_config_fallbacks(monkeypatch)

    class _ForbiddenAdapter:
        def chat(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("credential failure reached the adapter")

    monkeypatch.setattr(
        test_runner_module,
        "get_adapter_or_raise",
        lambda _provider: _ForbiddenAdapter(),
        raising=True,
    )
    processor = JobProcessor(db)
    monkeypatch.setattr(jobs_worker, "_get_processor", lambda _uid: processor, raising=True)
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name=expected_code,
        max_iterations=0,
    )

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(
                _job(
                    optimization_id=optimization_id,
                    prompt_id=prompt_id,
                    test_case_ids=case_ids,
                    owner_user_id=7,
                    max_iterations=0,
                )
            )

        _assert_safe_job_error(
            exc_info.value,
            code=expected_code,
            retryable=retryable,
        )
        assert len(factory.instances) == 1
        assert factory.instances[0].mark_calls == []
        assert factory.instances[0].close_count == 1
        row = db.get_optimization(optimization_id, include_deleted=True) or {}
        assert row["status"] == "failed"
        assert _SENTINEL not in str(row.get("error_message") or "")
        _assert_all_durable_state_secret_free(db, _SENTINEL)
    finally:
        db.close_connection()


_JOB_PROVIDER_FAILURES = [
    pytest.param(
        {
            "error": {
                "code": "invalid_provider_credentials",
                "message": _SENTINEL,
            }
        },
        "invalid_provider_credentials",
        False,
        id="in-band-credential-error",
    ),
    pytest.param(
        ChatConfigurationError(provider="openai", message=_SENTINEL),
        "provider_configuration_invalid",
        False,
        id="configuration-error",
    ),
    pytest.param(
        DaemonCapacityError(_SENTINEL),
        "provider_unavailable",
        True,
        id="capacity-exhaustion",
    ),
    pytest.param(
        TimeoutError(_SENTINEL),
        "provider_unavailable",
        True,
        id="provider-timeout",
    ),
    pytest.param(
        ChatProviderError(
            provider="openai",
            message=_SENTINEL,
            details={"private": _SENTINEL},
        ),
        "provider_unavailable",
        True,
        id="provider-error",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("adapter_outcome", "expected_code", "retryable"), _JOB_PROVIDER_FAILURES)
async def test_strict_optimization_failure_reaches_jobs_with_safe_retry_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_outcome: Any,
    expected_code: str,
    retryable: bool,
) -> None:
    db = _new_db(tmp_path, f"job-failure-{expected_code}")
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )
    factory = _RuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)
    _install_safe_config_fallbacks(monkeypatch)
    adapter_calls = 0

    class _Adapter:
        def chat(self, *_args: Any, **_kwargs: Any) -> Any:
            nonlocal adapter_calls
            adapter_calls += 1
            if isinstance(adapter_outcome, BaseException):
                raise adapter_outcome
            return adapter_outcome

    monkeypatch.setattr(
        test_runner_module,
        "get_adapter_or_raise",
        lambda _provider: _Adapter(),
        raising=True,
    )
    processor = JobProcessor(db)
    monkeypatch.setattr(jobs_worker, "_get_processor", lambda _uid: processor, raising=True)
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name=f"strict-{expected_code}",
        max_iterations=0,
    )

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(
                _job(
                    optimization_id=optimization_id,
                    prompt_id=prompt_id,
                    test_case_ids=case_ids,
                    owner_user_id=7,
                    max_iterations=0,
                )
            )

        _assert_safe_job_error(
            exc_info.value,
            code=expected_code,
            retryable=retryable,
        )
        assert adapter_calls == 1
        assert len(factory.instances) == 1
        assert factory.instances[0].mark_calls == []
        assert factory.instances[0].close_count == 1
        row = db.get_optimization(optimization_id, include_deleted=True) or {}
        assert row["status"] == "failed"
        assert _SENTINEL not in str(row.get("error_message") or "")
        _assert_all_durable_state_secret_free(
            db,
            _SENTINEL,
            _RESOLVED_KEY,
        )
    finally:
        db.close_connection()


_POST_BASELINE_FAILURES = [
    pytest.param(
        "mipro",
        "generation",
        None,
        id="mipro-candidate-generation",
    ),
    pytest.param(
        "bootstrap",
        "bootstrap-evaluation",
        None,
        id="bootstrap-post-baseline-evaluation",
    ),
    pytest.param(
        "mcts",
        "scorer",
        {
            "mcts_simulations": 1,
            "mcts_max_depth": 1,
            "prompt_candidates_per_node": 1,
            "scorer_model": "gpt-4o-mini",
            "feedback_enabled": False,
        },
        id="mcts-quality-scorer",
    ),
    pytest.param(
        "mcts",
        "refiner",
        {
            "mcts_simulations": 1,
            "mcts_max_depth": 1,
            "prompt_candidates_per_node": 1,
            "feedback_enabled": True,
            "feedback_threshold": 10.0,
            "feedback_max_retries": 1,
        },
        id="mcts-feedback-refiner",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("optimizer_type", "failure_stage", "strategy_params"),
    _POST_BASELINE_FAILURES,
)
async def test_valid_baseline_then_optimizer_failure_fails_job_without_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    optimizer_type: str,
    failure_stage: str,
    strategy_params: dict[str, Any] | None,
) -> None:
    db = _new_db(tmp_path, f"post-baseline-{failure_stage}")
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )
    factory = _RuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)
    _install_safe_config_fallbacks(monkeypatch)
    runner_calls = 0
    complete_calls = 0

    class _RunnerAdapter:
        def chat(self, _request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
            nonlocal runner_calls
            runner_calls += 1
            if failure_stage == "bootstrap-evaluation" and runner_calls > 1:
                events.append("optimizer_failure")
                return {
                    "error": {
                        "code": "provider_unavailable",
                        "message": _SENTINEL,
                    }
                }
            events.append("validated_provider_result")
            return {
                "choices": [{"message": {"content": "valid mismatch"}}],
                "usage": {"total_tokens": 2},
            }

    class _ExecutorAdapter:
        def chat(self, request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
            assert request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handle
            request_text = json.dumps(
                {
                    key: value
                    for key, value in request.items()
                    if key != PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY
                },
                sort_keys=True,
            )
            should_fail = failure_stage == "generation"
            should_fail = should_fail or (
                failure_stage == "scorer"
                and "Rate the clarity and effectiveness" in request_text
            )
            should_fail = should_fail or (
                failure_stage == "refiner"
                and "Analyze these errors" in request_text
            )
            if should_fail:
                events.append("optimizer_failure")
                raise ChatProviderError(
                    provider="openai",
                    message=_SENTINEL,
                    details={"private": _SENTINEL},
                )
            events.append("validated_provider_result")
            return {
                "choices": [{"message": {"content": "candidate instruction"}}],
                "usage": {"total_tokens": 2},
            }

    class _Registry:
        @staticmethod
        def is_local_provider_name(_provider: str) -> bool:
            return False

        @staticmethod
        def get_adapter(_provider: str) -> _ExecutorAdapter:
            return _ExecutorAdapter()

    monkeypatch.setattr(
        test_runner_module,
        "get_adapter_or_raise",
        lambda _provider: _RunnerAdapter(),
        raising=True,
    )
    monkeypatch.setattr(
        prompt_executor_module,
        "get_registry",
        lambda: _Registry(),
        raising=True,
    )
    original_complete = db.complete_optimization

    def _record_complete(*args: Any, **kwargs: Any) -> Any:
        nonlocal complete_calls
        complete_calls += 1
        return original_complete(*args, **kwargs)

    monkeypatch.setattr(db, "complete_optimization", _record_complete, raising=True)
    processor = JobProcessor(db)
    monkeypatch.setattr(jobs_worker, "_get_processor", lambda _uid: processor, raising=True)
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name=f"post-baseline-{failure_stage}",
        max_iterations=1,
        optimizer_type=optimizer_type,
        strategy_params=strategy_params,
    )

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(
                _job(
                    optimization_id=optimization_id,
                    prompt_id=prompt_id,
                    test_case_ids=case_ids,
                    owner_user_id=7,
                    max_iterations=1,
                    optimizer_type=optimizer_type,
                    strategy_params=strategy_params,
                )
            )

        _assert_safe_job_error(
            exc_info.value,
            code="provider_unavailable",
            retryable=True,
        )
        assert runner_calls >= 1
        assert "validated_provider_result" in events
        assert "optimizer_failure" in events
        assert len(factory.instances) == 1
        runtime = factory.instances[0]
        assert runtime.mark_calls == [handle]
        assert runtime.close_count == 1
        assert events.index("optimizer_failure") < len(events) - 1
        assert events[-1] == "close"
        assert complete_calls == 0
        row = db.get_optimization(optimization_id, include_deleted=True) or {}
        assert row["status"] == "failed"
        assert row.get("optimized_prompt_id") is None
        assert _SENTINEL not in str(row.get("error_message") or "")
        _assert_all_durable_state_secret_free(
            db,
            _SENTINEL,
            _RESOLVED_KEY,
        )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_concurrent_same_and_different_owner_jobs_use_distinct_runtimes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_db = _new_db(tmp_path, "concurrent-owner-7")
    second_db = _new_db(tmp_path, "concurrent-owner-8")
    events: list[Any] = []

    def _handle_for_scope(
        scope: dict[str, Any],
        _index: int,
        provider: str,
        model: str | None,
    ) -> _Handle:
        owner = int(scope["user_id"])
        section = "anthropic_api" if provider == "anthropic" else "openai_api"
        marker = f"owner-{owner}:{provider}:{model}"
        return _Handle(
            provider=provider,
            api_key=f"key:{marker}",
            app_config={section: {"model": model, "runtime_marker": marker}},
        )

    factory = _RuntimeFactory(_handle_for_scope, events)
    membership_calls = _install_runtime_and_memberships(monkeypatch, factory)
    _install_safe_config_fallbacks(monkeypatch)
    barrier = threading.Barrier(3)
    adapter_requests: list[tuple[str, str, dict[str, Any]]] = []
    barrier_models: set[str] = set()
    request_lock = threading.Lock()

    class _Adapter:
        def __init__(self, boundary: str, provider: str) -> None:
            self.boundary = boundary
            self.provider = provider

        def chat(self, request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
            model = str(request["model"])
            provider_credentials = request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY]
            captured_request = copy.deepcopy(
                {
                    key: value
                    for key, value in request.items()
                    if key != PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY
                }
            )
            captured_request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] = (
                provider_credentials
            )
            with request_lock:
                adapter_requests.append(
                    (self.boundary, self.provider, captured_request)
                )
                wait_at_barrier = model not in barrier_models
                barrier_models.add(model)
            if wait_at_barrier:
                barrier.wait(timeout=2)
            content = (
                "candidate instruction"
                if self.boundary == "executor"
                else "valid mismatch"
            )
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {"total_tokens": 2},
            }

    class _Registry:
        @staticmethod
        def is_local_provider_name(_provider: str) -> bool:
            return False

        @staticmethod
        def get_adapter(provider: str) -> _Adapter:
            return _Adapter("executor", provider)

    monkeypatch.setattr(
        test_runner_module,
        "get_adapter_or_raise",
        lambda provider: _Adapter("runner", provider),
        raising=True,
    )
    monkeypatch.setattr(
        prompt_executor_module,
        "get_registry",
        lambda: _Registry(),
        raising=True,
    )
    processors = {
        "7": JobProcessor(first_db),
        "8": JobProcessor(second_db),
    }

    with jobs_worker._CACHE_LOCK:
        saved_db_cache = copy.copy(jobs_worker._DB_CACHE)
        saved_processor_cache = copy.copy(jobs_worker._PROCESSOR_CACHE)
        saved_active_counts = dict(jobs_worker._ACTIVE_USER_COUNTS)
        saved_pending_close = copy.copy(jobs_worker._PENDING_CLOSE)
        jobs_worker._DB_CACHE.clear()
        jobs_worker._PROCESSOR_CACHE.clear()
        jobs_worker._ACTIVE_USER_COUNTS.clear()
        jobs_worker._PENDING_CLOSE.clear()
        jobs_worker._DB_CACHE.update({"7": first_db, "8": second_db})
        jobs_worker._PROCESSOR_CACHE.update(processors)

    first_config = {
        "provider": "openai",
        "model": "gpt-owner7-a",
        "parameters": {"temperature": 0.11, "max_tokens": 31},
    }
    second_config = {
        "provider": "anthropic",
        "model": "claude-owner7-b",
        "parameters": {"temperature": 0.22, "max_tokens": 32},
    }
    third_config = {
        "provider": "openai",
        "model": "gpt-owner8-c",
        "parameters": {"temperature": 0.33, "max_tokens": 33},
    }

    first_id, first_prompt, first_cases = _seed_optimization(
        first_db,
        name="owner7-openai-a",
        max_iterations=1,
        optimizer_type="mipro",
        model_config=first_config,
    )
    second_id, second_prompt, second_cases = _seed_optimization(
        first_db,
        name="owner7-anthropic-b",
        max_iterations=1,
        optimizer_type="mipro",
        model_config=second_config,
    )
    third_id, third_prompt, third_cases = _seed_optimization(
        second_db,
        name="owner8-openai-c",
        max_iterations=1,
        optimizer_type="mipro",
        model_config=third_config,
    )
    jobs = [
        _job(
            optimization_id=first_id,
            prompt_id=first_prompt,
            test_case_ids=first_cases,
            owner_user_id=7,
            max_iterations=1,
            optimization_db=first_db,
            optimizer_type="mipro",
            model_config=first_config,
        ),
        _job(
            optimization_id=second_id,
            prompt_id=second_prompt,
            test_case_ids=second_cases,
            owner_user_id=7,
            max_iterations=1,
            optimization_db=first_db,
            optimizer_type="mipro",
            model_config=second_config,
        ),
        _job(
            optimization_id=third_id,
            prompt_id=third_prompt,
            test_case_ids=third_cases,
            owner_user_id=8,
            max_iterations=1,
            optimization_db=second_db,
            optimizer_type="mipro",
            model_config=third_config,
        ),
    ]

    try:
        results = await asyncio.gather(
            *(jobs_worker._handle_job(job) for job in jobs)
        )

        assert len(results) == 3
        assert len(factory.instances) == 3
        assert len({id(runtime) for runtime in factory.instances}) == 3
        assert {
            first_db.get_optimization(first_id)["optimizer_type"],
            first_db.get_optimization(second_id)["optimizer_type"],
            second_db.get_optimization(third_id)["optimizer_type"],
        } == {"mipro"}
        owner_seven = [
            runtime
            for runtime in factory.instances
            if runtime.scope["user_id"] == 7
        ]
        assert len(owner_seven) == 2
        assert owner_seven[0] is not owner_seven[1]
        assert all(runtime.close_count == 1 for runtime in factory.instances)
        assert all(len(runtime.mark_calls) == 1 for runtime in factory.instances)

        expected_requests = {
            "owner7-openai-a": (7, first_config),
            "owner7-anthropic-b": (7, second_config),
            "owner8-openai-c": (8, third_config),
        }
        requests_by_marker: dict[
            str,
            list[tuple[str, str, dict[str, Any]]],
        ] = {marker: [] for marker in expected_requests}
        for boundary, request_provider, request in adapter_requests:
            provider_credentials = request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY]
            serialized_request = json.dumps(
                {
                    key: value
                    for key, value in request.items()
                    if key != PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY
                },
                sort_keys=True,
            )
            matching_markers = [
                marker
                for marker in expected_requests
                if marker in serialized_request
            ]
            assert len(matching_markers) == 1
            marker = matching_markers[0]
            requests_by_marker[marker].append(
                (boundary, request_provider, request)
            )
            owner, config = expected_requests[marker]
            expected_runtime_marker = (
                f"owner-{owner}:{config['provider']}:{config['model']}"
            )
            expected_section = (
                "anthropic_api"
                if config["provider"] == "anthropic"
                else "openai_api"
            )
            assert request_provider == config["provider"]
            assert request["model"] == config["model"]
            assert request["api_key"] == f"key:{expected_runtime_marker}"
            assert request["app_config"] == {
                expected_section: {
                    "model": config["model"],
                    "runtime_marker": expected_runtime_marker,
                }
            }
            assert request["credentials_resolved"] is True
            matching_handles = [
                runtime.resolved_outcome
                for runtime in factory.instances
                if isinstance(runtime.resolved_outcome, _Handle)
                and runtime.resolved_outcome.app_config
                == request["app_config"]
                and runtime.resolved_outcome.api_key == request["api_key"]
            ]
            assert len(matching_handles) == 1
            assert provider_credentials is matching_handles[0]

        for marker, marker_requests in requests_by_marker.items():
            assert marker_requests
            assert {boundary for boundary, _provider, _request in marker_requests} == {
                "runner",
                "executor",
            }, marker

        for runtime in factory.instances:
            owner = int(runtime.scope["user_id"])
            assert isinstance(runtime.resolved_outcome, _Handle)
            assert runtime.mark_calls == [runtime.resolved_outcome]
            assert runtime.scope["team_ids"] == [owner * 10 + 1]
            assert runtime.scope["org_ids"] == [owner * 10 + 2]
            assert runtime.scope["trusted_base_url_override"] is False

        assert sorted(membership_calls) == sorted(
            [
                ("team", 7),
                ("org", 7),
                ("team", 7),
                ("org", 7),
                ("team", 8),
                ("org", 8),
            ]
        )
        for processor in jobs_worker._PROCESSOR_CACHE.values():
            assert all(
                _retains_identity(processor, runtime) is False
                for runtime in factory.instances
            )
            assert all(
                _retains_identity(processor, runtime.resolved_outcome) is False
                for runtime in factory.instances
            )
        for runtime in factory.instances:
            assert _retains_identity(jobs_worker._PROCESSOR_CACHE, runtime) is False
            assert (
                _retains_identity(
                    jobs_worker._PROCESSOR_CACHE,
                    runtime.resolved_outcome,
                )
                is False
            )

        class _SlottedRetentionProbe:
            __slots__ = ("callback",)

            def __init__(self, callback: Callable[[], object]) -> None:
                self.callback = callback

        def _capture_in_closure(value: object) -> Callable[[], object]:
            def _captured() -> object:
                return value

            return _captured

        closure_probe = _SlottedRetentionProbe(
            _capture_in_closure(factory.instances[0])
        )
        defaults_probe = _SlottedRetentionProbe(
            lambda retained=factory.instances[0].resolved_outcome: retained
        )
        assert _retains_identity(closure_probe, factory.instances[0]) is True
        assert (
            _retains_identity(
                defaults_probe,
                factory.instances[0].resolved_outcome,
            )
            is True
        )

        dynamic_sentinels: list[str] = []
        for runtime in factory.instances:
            assert isinstance(runtime.resolved_outcome, _Handle)
            section_config = next(
                iter(runtime.resolved_outcome.app_config.values())
            )
            dynamic_sentinels.extend(
                (
                    str(runtime.resolved_outcome.api_key),
                    str(section_config["runtime_marker"]),
                )
            )
        first_dump = _assert_all_durable_state_secret_free(
            first_db,
            *dynamic_sentinels,
        )
        second_dump = _assert_all_durable_state_secret_free(
            second_db,
            *dynamic_sentinels,
        )
        assert 'INSERT INTO "prompt_studio_test_runs"' in first_dump
        assert 'INSERT INTO "prompt_studio_test_runs"' in second_dump
    finally:
        with jobs_worker._CACHE_LOCK:
            jobs_worker._DB_CACHE.clear()
            jobs_worker._DB_CACHE.update(saved_db_cache)
            jobs_worker._PROCESSOR_CACHE.clear()
            jobs_worker._PROCESSOR_CACHE.update(saved_processor_cache)
            jobs_worker._ACTIVE_USER_COUNTS.clear()
            jobs_worker._ACTIVE_USER_COUNTS.update(saved_active_counts)
            jobs_worker._PENDING_CLOSE.clear()
            jobs_worker._PENDING_CLOSE.update(saved_pending_close)
        first_db.close_connection()
        second_db.close_connection()


def _install_owned_worker_cancellation_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> asyncio.Event:
    from tldw_Server_API.app.core.Chat import bounded_daemon

    entered = asyncio.Event()
    original = bounded_daemon._drain_owned_task

    async def _ack(task: asyncio.Future[Any]) -> tuple[bool, Any]:
        entered.set()
        return await original(task)

    monkeypatch.setattr(bounded_daemon, "_drain_owned_task", _ack, raising=True)
    return entered


async def _wait_for_thread_event(
    event: threading.Event,
    *,
    timeout: float = 2.0,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            raise AssertionError("Timed out waiting for adapter event")
        await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_cancelled_durable_job_drains_late_success_before_runtime_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "cancelled")
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )
    factory = _RuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)
    _install_safe_config_fallbacks(monkeypatch)
    started = threading.Event()
    release = threading.Event()
    adapter_done = threading.Event()
    drain_entered = _install_owned_worker_cancellation_ack(monkeypatch)

    class _Adapter:
        def chat(self, _request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
            started.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test release was not signalled")
            adapter_done.set()
            events.append("adapter_done")
            return {"choices": [{"message": {"content": "valid mismatch"}}]}

    monkeypatch.setattr(
        test_runner_module,
        "get_adapter_or_raise",
        lambda _provider: _Adapter(),
        raising=True,
    )
    processor = JobProcessor(db)
    monkeypatch.setattr(jobs_worker, "_get_processor", lambda _uid: processor, raising=True)
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="cancelled",
        max_iterations=0,
    )
    task = asyncio.create_task(
        jobs_worker._handle_job(
            _job(
                optimization_id=optimization_id,
                prompt_id=prompt_id,
                test_case_ids=case_ids,
                owner_user_id=7,
                max_iterations=0,
            )
        )
    )

    try:
        await _wait_for_thread_event(started)
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=2)
        assert len(factory.instances) == 1
        runtime = factory.instances[0]
        assert adapter_done.is_set() is False
        assert runtime.close_count == 0

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

        assert adapter_done.is_set() is True
        assert runtime.mark_calls == [handle]
        assert runtime.close_count == 1
        assert events[-3:] == ["adapter_done", "mark", "close"]
        _assert_all_durable_state_secret_free(db, _RESOLVED_KEY)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task
        db.close_connection()


@pytest.mark.asyncio
async def test_worker_cancellation_stops_after_inflight_call_before_next_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "cancel-before-next-dispatch")
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )
    factory = _RuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)
    started = asyncio.Event()
    release = asyncio.Event()
    dispatches: list[str] = []

    class _TwoDispatchProcessor:
        def __init__(self, database: PromptStudioDatabase) -> None:
            self.db = database

        async def process_optimization_job(
            self,
            _payload: dict[str, Any],
            optimization_id: int,
            **kwargs: Any,
        ) -> dict[str, Any]:
            async def _first_provider_call() -> str:
                started.set()
                await release.wait()
                dispatches.append("first")
                return "ok"

            await await_owned_worker(
                _first_provider_call(),
                on_cancel_success=kwargs["on_provider_success"],
            )
            dispatches.append("second")
            await kwargs["on_provider_success"]()
            return {
                "optimization_id": optimization_id,
                "status": "completed",
                "iterations_completed": 1,
            }

    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="cancel-before-next-dispatch",
        max_iterations=1,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: _TwoDispatchProcessor(db),
        raising=True,
    )
    task = asyncio.create_task(
        jobs_worker._handle_job(
            _job(
                optimization_id=optimization_id,
                prompt_id=prompt_id,
                test_case_ids=case_ids,
                owner_user_id=7,
                max_iterations=1,
            )
        )
    )

    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

        assert dispatches == ["first"]
        assert factory.instances[0].mark_calls == [handle]
        assert factory.instances[0].close_count == 1
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task
        db.close_connection()


@pytest.mark.asyncio
async def test_retryable_optimization_attempt_stays_nonterminal_until_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "retry-state-success")
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )
    factory = _RuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)
    status_updates: list[tuple[str, bool]] = []
    original_set_status = db.set_optimization_status

    def _record_status(
        optimization_id: int,
        status: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        status_updates.append((status, bool(kwargs.get("mark_completed"))))
        return original_set_status(optimization_id, status, **kwargs)

    monkeypatch.setattr(db, "set_optimization_status", _record_status, raising=True)

    attempts = 0

    async def _flaky_optimize(
        _self: Any,
        optimization_id: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError(_SENTINEL)
        await kwargs["on_provider_success"]()
        db.complete_optimization(
            optimization_id,
            iterations_completed=1,
        )
        return {
            "optimization_id": optimization_id,
            "status": "completed",
            "iterations_completed": 1,
        }

    monkeypatch.setattr(
        optimization_engine_module.OptimizationEngine,
        "optimize",
        _flaky_optimize,
        raising=True,
    )

    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="retry-state-success",
        max_iterations=1,
    )
    processor = JobProcessor(db)
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: processor,
        raising=True,
    )
    job = _job(
        optimization_id=optimization_id,
        prompt_id=prompt_id,
        test_case_ids=case_ids,
        owner_user_id=7,
        max_iterations=1,
    )
    job.update({"retry_count": 0, "max_retries": 1})

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(job)

        _assert_safe_job_error(
            exc_info.value,
            code="provider_unavailable",
            retryable=True,
        )
        first_attempt = dict(
            db.get_optimization(optimization_id, include_deleted=True) or {}
        )
        first_attempt_updates = list(status_updates)

        retry_job = copy.deepcopy(job)
        retry_job["retry_count"] = 1
        result = await jobs_worker._handle_job(retry_job)
        final = db.get_optimization(optimization_id, include_deleted=True) or {}

        assert first_attempt["status"] not in {"completed", "failed"}
        assert first_attempt.get("completed_at") is None
        assert all(
            status not in {"completed", "failed"} and not mark_completed
            for status, mark_completed in first_attempt_updates
        )
        assert result["status"] == "completed"
        assert final["status"] == "completed"
        assert final.get("error_message") is None
        assert len(factory.instances) == 2
        assert factory.instances[0].mark_calls == []
        assert factory.instances[1].mark_calls == [handle]
        assert all(runtime.close_count == 1 for runtime in factory.instances)
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("authoritative_status", "expected_code", "retryable"),
    [
        ("failed", "prompt_studio_job_failed", False),
        ("pending", "job_state_unavailable", True),
    ],
)
async def test_worker_rejects_noncompleted_authoritative_prompt_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authoritative_status: str,
    expected_code: str,
    retryable: bool,
) -> None:
    """A stale engine success must never complete Jobs over Prompt state."""

    db = _new_db(tmp_path, f"authoritative-{authoritative_status}")
    manager = JobManager(
        db_path=tmp_path / f"authoritative-{authoritative_status}-jobs.sqlite"
    )
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )
    factory = _RuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)

    async def _losing_completion(
        _self: Any,
        optimization_id: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        await kwargs["on_provider_success"]()
        if authoritative_status == "failed":
            db.set_optimization_status(
                optimization_id,
                "failed",
                error_message="authoritative failure",
                mark_completed=True,
            )
        return {
            "optimization_id": optimization_id,
            "status": "completed",
            "iterations_completed": 1,
        }

    monkeypatch.setattr(
        optimization_engine_module.OptimizationEngine,
        "optimize",
        _losing_completion,
        raising=True,
    )
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name=f"authoritative-{authoritative_status}",
        max_iterations=1,
    )
    queued = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=_job(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=case_ids,
            owner_user_id=7,
            max_iterations=1,
            optimization_db=db,
        )["payload"],
        owner_user_id="7",
        max_retries=2,
    )
    acquired = manager.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=60,
        worker_id="authoritative-outcome-test",
    )
    assert acquired is not None
    assert acquired["id"] == queued["id"]
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(acquired, job_manager=manager)

        _assert_safe_job_error(
            exc_info.value,
            code=expected_code,
            retryable=retryable,
        )
        assert (manager.get_job(int(acquired["id"])) or {})["status"] == "processing"
        prompt_status = str(
            (db.get_optimization(optimization_id, include_deleted=True) or {}).get(
                "status"
            )
        )
        assert prompt_status == authoritative_status
        assert factory.instances[0].mark_calls == [handle]
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_usage_persistence_failure_aborts_before_prompt_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed credential touch is a provider-boundary failure, not success."""

    db = _new_db(tmp_path, "usage-persistence-failure")
    manager = JobManager(
        db_path=tmp_path / "usage-persistence-failure-jobs.sqlite"
    )
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )
    factory = _RuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)
    touch_calls = 0

    async def _touch_fails(_runtime: Any, _credentials: Any) -> bool:
        nonlocal touch_calls
        touch_calls += 1
        return False

    async def _optimize(
        _self: Any,
        optimization_id: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        await kwargs["on_provider_success"]()
        db.complete_optimization(optimization_id, iterations_completed=1)
        return {
            "optimization_id": optimization_id,
            "status": "completed",
            "iterations_completed": 1,
        }

    monkeypatch.setattr(
        jobs_worker,
        "mark_provider_credential_used",
        _touch_fails,
        raising=True,
    )
    monkeypatch.setattr(
        optimization_engine_module.OptimizationEngine,
        "optimize",
        _optimize,
        raising=True,
    )
    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="usage-persistence-failure",
        max_iterations=1,
    )
    queued = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=_job(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=case_ids,
            owner_user_id=7,
            max_iterations=1,
            optimization_db=db,
        )["payload"],
        owner_user_id="7",
    )
    acquired = manager.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=60,
        worker_id="usage-persistence-test",
    )
    assert acquired is not None
    assert acquired["id"] == queued["id"]
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(acquired, job_manager=manager)

        _assert_safe_job_error(
            exc_info.value,
            code="provider_success_not_observed",
            retryable=False,
        )
        assert touch_calls == 1
        assert (manager.get_job(int(acquired["id"])) or {})["status"] == "processing"
        assert (
            db.get_optimization(optimization_id, include_deleted=True) or {}
        )["status"] == "failed"
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_exhausted_retryable_optimization_attempt_ends_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "retry-state-exhausted")
    events: list[Any] = []
    handle = _Handle(
        api_key=_RESOLVED_KEY,
        app_config={"openai_api": {"model": "gpt-4o-mini"}},
    )
    factory = _RuntimeFactory(handle, events)
    _install_runtime_and_memberships(monkeypatch, factory)

    async def _failing_optimize(
        _self: Any,
        _optimization_id: int,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        raise TimeoutError(_SENTINEL)

    monkeypatch.setattr(
        optimization_engine_module.OptimizationEngine,
        "optimize",
        _failing_optimize,
        raising=True,
    )

    optimization_id, prompt_id, case_ids = _seed_optimization(
        db,
        name="retry-state-exhausted",
        max_iterations=1,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )
    job = _job(
        optimization_id=optimization_id,
        prompt_id=prompt_id,
        test_case_ids=case_ids,
        owner_user_id=7,
        max_iterations=1,
    )
    job.update({"retry_count": 1, "max_retries": 1})

    try:
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await jobs_worker._handle_job(job)

        _assert_safe_job_error(
            exc_info.value,
            code="provider_unavailable",
            retryable=True,
        )
        row = db.get_optimization(optimization_id, include_deleted=True) or {}
        assert row["status"] == "failed"
        assert row.get("completed_at") is not None
        assert _SENTINEL not in str(row.get("error_message") or "")
        assert len(factory.instances) == 1
        assert factory.instances[0].mark_calls == []
        assert factory.instances[0].close_count == 1
    finally:
        db.close_connection()
