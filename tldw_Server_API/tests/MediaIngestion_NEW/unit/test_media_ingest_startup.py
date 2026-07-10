import pytest

from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerLifecycleContext
from tldw_Server_API.app.services.startup_content_jobs_pollers import (
    provide_content_jobs_worker_specs,
)


pytestmark = pytest.mark.unit


def _worker_context(
    *, sidecar_mode: bool = False, route_allowed: bool = True
) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=object(),
        settings={},
        test_mode=False,
        route_enabled=lambda *_args, **_kwargs: route_allowed,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
        sidecar_mode=sidecar_mode,
    )


def _content_spec(name: str):
    specs = {spec.name: spec for spec in provide_content_jobs_worker_specs()}
    return specs[name]


def test_should_start_inprocess_worker_uses_route_policy_when_flag_unset(monkeypatch):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    monkeypatch.delenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", raising=False)
    monkeypatch.setenv("ROUTES_ENABLE", "media-ingest-heavy-jobs")

    assert should_start_inprocess_worker(
        "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
        "media-ingest-heavy-jobs",
        sidecar_mode=False,
        default_stable=False,
        test_mode=False,
    )


def test_should_start_inprocess_worker_uses_injected_route_policy_when_flag_unset(
    monkeypatch,
):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    monkeypatch.delenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", raising=False)

    assert should_start_inprocess_worker(
        "MEDIA_INGEST_JOBS_WORKER_ENABLED",
        "media",
        sidecar_mode=False,
        default_stable=True,
        test_mode=False,
        route_enabled=lambda route_key, **_kwargs: route_key == "media",
    )


def test_should_start_inprocess_worker_honors_injected_route_disabled_when_flag_unset(
    monkeypatch,
):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    monkeypatch.delenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", raising=False)

    assert not should_start_inprocess_worker(
        "MEDIA_INGEST_JOBS_WORKER_ENABLED",
        "media",
        sidecar_mode=False,
        default_stable=True,
        test_mode=False,
        route_enabled=lambda *_args, **_kwargs: False,
    )


def test_should_start_inprocess_worker_supports_single_arg_injected_route_policy(
    monkeypatch,
):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    monkeypatch.delenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", raising=False)

    assert not should_start_inprocess_worker(
        "MEDIA_INGEST_JOBS_WORKER_ENABLED",
        "media",
        sidecar_mode=False,
        default_stable=True,
        test_mode=False,
        route_enabled=lambda _route_key: False,
    )


def test_should_start_inprocess_worker_supports_non_introspectable_single_arg_route_policy(
    monkeypatch,
):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    class NonIntrospectableRoutePolicy:
        @property
        def __signature__(self):
            raise ValueError("signature unavailable")

        def __call__(self, route_key):
            return route_key == "media"

    monkeypatch.delenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", raising=False)

    assert should_start_inprocess_worker(
        "MEDIA_INGEST_JOBS_WORKER_ENABLED",
        "media",
        sidecar_mode=False,
        default_stable=True,
        test_mode=False,
        route_enabled=NonIntrospectableRoutePolicy(),
    )


def test_should_start_inprocess_worker_does_not_mask_route_policy_type_errors(
    monkeypatch,
):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    def broken_route_enabled(_route_key, **_kwargs):
        raise TypeError("broken route policy")

    monkeypatch.delenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", raising=False)

    assert not should_start_inprocess_worker(
        "MEDIA_INGEST_JOBS_WORKER_ENABLED",
        "media",
        sidecar_mode=False,
        default_stable=True,
        test_mode=False,
        route_enabled=broken_route_enabled,
    )


def test_should_start_inprocess_worker_respects_explicit_enable_in_test_mode(monkeypatch):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    monkeypatch.setenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", "true")
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)

    assert should_start_inprocess_worker(
        "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
        "media-ingest-heavy-jobs",
        sidecar_mode=False,
        default_stable=False,
        test_mode=True,
    )


def test_should_start_inprocess_worker_disables_startup_in_sidecar_mode(monkeypatch):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    monkeypatch.setenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", "true")

    assert not should_start_inprocess_worker(
        "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
        "media-ingest-heavy-jobs",
        sidecar_mode=True,
        default_stable=False,
        test_mode=False,
    )


def test_media_ingest_lifecycle_spec_uses_route_policy_when_flag_unset(monkeypatch):
    monkeypatch.delenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", raising=False)

    spec = _content_spec("media_ingest_jobs_task")

    assert spec.enabled(_worker_context(route_allowed=True)) is True


def test_media_ingest_lifecycle_spec_disables_when_route_policy_disabled(monkeypatch):
    monkeypatch.delenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", raising=False)

    spec = _content_spec("media_ingest_jobs_task")

    assert spec.enabled(_worker_context(route_allowed=False)) is False


def test_media_ingest_lifecycle_spec_respects_explicit_false(monkeypatch):
    monkeypatch.setenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", "false")

    spec = _content_spec("media_ingest_jobs_task")

    assert spec.enabled(_worker_context(route_allowed=True)) is False


def test_media_ingest_lifecycle_spec_skips_in_sidecar_mode(monkeypatch):
    monkeypatch.setenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", "true")

    spec = _content_spec("media_ingest_jobs_task")

    assert spec.enabled(_worker_context(sidecar_mode=True, route_allowed=True)) is False


def test_media_ingest_heavy_lifecycle_spec_remains_disabled_by_default(monkeypatch):
    route_calls = []

    def _route_enabled(*args, **kwargs):
        route_calls.append((args, kwargs))
        return kwargs.get("default_stable", True)

    monkeypatch.delenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", raising=False)

    spec = _content_spec("media_ingest_heavy_jobs_task")

    assert spec.enabled(
        WorkerLifecycleContext(
            app=object(),
            settings={},
            test_mode=False,
            route_enabled=_route_enabled,
            logger=None,
            startup_guard_exceptions=(),
            import_exceptions=(),
            sidecar_mode=False,
        )
    ) is False
    assert route_calls == [(("media-ingest-heavy-jobs",), {"default_stable": False})]
