"""Checkpoint authorization, bounded hostile input and native writer exclusion."""

import asyncio
import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Web_Scraping.crawl import checkpoints
from tldw_Server_API.app.core.Web_Scraping.crawl.checkpoints import CheckpointError, CheckpointStore
from tldw_Server_API.app.core.Web_Scraping.crawl.engine import run_crawl
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CheckpointLimits, CheckpointScope, FrontierSnapshot
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import COUNTERS, WorkHandle
from tldw_Server_API.tests.Web_Scraping.test_phase5_checkpoints import state
from tldw_Server_API.tests.Web_Scraping.test_phase5_checkpoints import store as store
from tldw_Server_API.tests.Web_Scraping.test_phase5_frontier import BASE, entry, plan

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field_path",
    [("budget", counter) for counter in COUNTERS]
    + [
        ("budget", "accounting_complete"),
        ("budget", "remaining_seconds"),
        ("accounting_complete",),
        ("pages_scraped",),
    ],
)
async def test_missing_version1_wire_fields_fail_before_acquisition(store, tmp_path, field_path):
    expected = state(store)
    store.save(expected)
    path = tmp_path / "progress.json"
    data = json.loads(path.read_text())
    container = data if len(field_path) == 1 else data[field_path[0]]
    del container[field_path[-1]]
    path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError, match="^checkpoint_failed$"):
        store.load(plan=expected.plan, scope=store.scope)

    acquired = []

    async def acquire(item):
        acquired.append(item.url)
        raise AssertionError("malformed checkpoint reached page admission")

    report = await run_crawl(
        expected.plan,
        context=CrawlContext(expected.plan.limits, clock=lambda: 0),
        acquire=acquire,
        checkpoint=store,
    )
    assert report.outcome.stop_reason == "checkpoint_failed"
    assert acquired == []


@pytest.mark.parametrize("container", ["root", "scope", "binding", "budget", "frontier", "entry"])
def test_version1_wire_objects_reject_unknown_fields(store, tmp_path, container):
    expected = state(store)
    store.save(expected)
    path = tmp_path / "progress.json"
    data = json.loads(path.read_text())
    target = data if container == "root" else data["frontier"]["queued"][0] if container == "entry" else data[container]
    target["unexpected"] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError, match="^checkpoint_failed$"):
        store.load(plan=expected.plan, scope=store.scope)


@pytest.mark.asyncio
@pytest.mark.parametrize("pages", [None, True, False, 1.0, "1", -1, 1001])
async def test_legacy_page_count_must_be_bounded_integer_before_acquisition(store, tmp_path, pages):
    resolved = plan("legacy_recursive", delay=0)
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "visited": [],
                "to_visit": [[BASE, 0]],
                "scraped_articles": [],
                "pages_scraped": pages,
            }
        )
    )
    with pytest.raises(CheckpointError, match="^checkpoint_failed$"):
        store.load(plan=resolved, scope=store.scope)

    acquired = []

    async def acquire(item):
        acquired.append(item.url)
        raise AssertionError("malformed legacy counter reached page admission")

    report = await run_crawl(
        resolved,
        context=CrawlContext(resolved.limits, clock=lambda: 0),
        acquire=acquire,
        checkpoint=store,
    )
    assert report.outcome.stop_reason == "checkpoint_failed"
    assert acquired == []


def test_legacy_unknown_usage_stays_explicit_after_versioned_roundtrip(store, tmp_path):
    resolved = plan("legacy_recursive", delay=0)
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "visited": [],
                "to_visit": [[BASE, 0]],
                "scraped_articles": [],
                "pages_scraped": 0,
            }
        )
    )
    migrated = store.load(plan=resolved, scope=store.scope)
    assert not migrated.accounting_complete and not migrated.budget.accounting_complete
    assert all(getattr(migrated.budget, counter) == 0 for counter in COUNTERS)
    store.save(migrated)
    loaded = store.load(plan=resolved, scope=store.scope)
    assert loaded == migrated
    assert loaded.pages_scraped == 0


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"scope": {"owner": "other", "namespace": "crawl"}},
        {"budget": {"page_attempts": -1}},
        {"budget": {"page_attempts": True}},
        {"accounting_complete": 1},
        {"stop_reason": "private/path"},
    ],
)
def test_invalid_version_binding_counters_rejected_before_execution(store, tmp_path, mutation):
    expected = state(store)
    store.save(expected)
    path = tmp_path / "progress.json"
    value = json.loads(path.read_text())
    value.update(mutation)
    path.write_text(json.dumps(value))
    with pytest.raises(CheckpointError, match="^checkpoint_failed$"):
        store.load(plan=expected.plan, scope=store.scope)


@pytest.mark.parametrize("content", [b"{", b'{"x": NaN}', b'{"schema_version": 1,"schema_version": 1}', b"[]"])
def test_corrupt_json_has_no_path_or_content_in_error(store, tmp_path, content):
    (tmp_path / "progress.json").write_bytes(content)
    with pytest.raises(CheckpointError) as error:
        store.load(plan=plan(), scope=store.scope)
    assert str(error.value) == "checkpoint_failed"
    assert error.value.__suppress_context__


def test_read_is_bounded_before_json_parser(store, tmp_path, monkeypatch):
    store.limits = replace(store.limits, max_file_bytes=32)
    (tmp_path / "progress.json").write_bytes(b" " * 33)

    def unexpected(*args, **kwargs):
        pytest.fail("oversized input reached JSON decoder")

    monkeypatch.setattr(checkpoints, "json", SimpleNamespace(loads=unexpected))
    with pytest.raises(CheckpointError):
        store.load(plan=plan(), scope=store.scope)


@pytest.mark.parametrize("location", ["checkpoint", "lock", "descendant", "escape", "missing_root"])
def test_service_root_rejects_symlinks_and_escape(tmp_path, location):
    root = tmp_path / "authorized"
    root.mkdir()
    outside = tmp_path / "private"
    outside.mkdir()
    path = root / "progress.json"
    if location in {"checkpoint", "lock"}:
        target = outside / "file"
        target.write_text("untouched")
        (path if location == "checkpoint" else root / "progress.json.lock").symlink_to(target)
    elif location == "descendant":
        (root / "link").symlink_to(outside, target_is_directory=True)
        path = root / "link" / "progress.json"
    elif location == "escape":
        path = root / ".." / "private" / "progress.json"
    scope = CheckpointScope("owner", "crawl", root=None if location == "missing_root" else root)
    with pytest.raises(CheckpointError, match="^checkpoint_failed$"):
        CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())


def test_trusted_root_alias_is_canonicalized_but_descendants_are_not(tmp_path):
    alias = tmp_path / "alias"
    root = tmp_path / "actual"
    root.mkdir()
    alias.symlink_to(root, target_is_directory=True)
    scope = CheckpointScope("owner", "crawl", root=alias)
    value = CheckpointStore.open(alias / "progress.json", scope=scope, limits=CheckpointLimits())
    value.close()


@pytest.mark.parametrize("absolute", [False, True])
def test_explicit_trusted_local_paths_remain_supported(tmp_path, monkeypatch, absolute):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "progress.json" if absolute else Path("progress.json")
    scope = CheckpointScope("local", "crawl", trusted_local_path=True)
    value = CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
    value.save(state(value))
    assert value.load(plan=plan(), scope=scope)
    value.close()


def test_stable_lock_inode_and_same_process_exclusion(store, tmp_path):
    lock = tmp_path / "progress.json.lock"
    before = lock.stat().st_ino
    for _ in range(2):
        store.save(state(store))
        with pytest.raises(CheckpointError, match="checkpoint_in_use"):
            CheckpointStore.open(tmp_path / "progress.json", scope=store.scope, limits=store.limits)
    store.close()
    other = CheckpointStore.open(tmp_path / "progress.json", scope=store.scope, limits=store.limits)
    other.close()
    assert lock.stat().st_ino == before


def _contender(path, root, connection):
    try:
        value = CheckpointStore.open(
            path, scope=CheckpointScope("owner", "crawl", root=root), limits=CheckpointLimits()
        )
    except CheckpointError as exc:
        connection.send(exc.code)
    else:
        value.close()
        connection.send("acquired")
    finally:
        connection.close()


def test_cross_process_exclusion_is_native_not_only_registry(store, tmp_path):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_contender, args=(tmp_path / "progress.json", tmp_path, child))
    process.start()
    child.close()
    try:
        assert parent.poll(10), "child did not report"
        assert parent.recv() == "checkpoint_in_use"
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()


def test_credential_free_json_preserves_intentional_prose_and_extra_fields(store, tmp_path):
    prose = "Article discusses https://example.com/?token=public-example and /research/data"
    article = {
        "content": prose,
        "raw_content": prose,
        "title": "Title",
        "custom_field": {"kept": 7},
        "metadata": {
            "Authorization": "secret",
            "source": "https://user:pass@example.com/?token=secret",
            "file_path": "/private/secret",
        },
        "error": "secret /private/secret",
        "api_key": "secret",
    }
    candidate = replace(state(store), frontier=FrontierSnapshot(completed=(entry(BASE),), results=(article,)))
    store.save(candidate)
    raw = (tmp_path / "progress.json").read_text()
    assert "secret" not in raw and "user:pass" not in raw
    actual = store.load(plan=candidate.plan, scope=store.scope).frontier.results[0]
    assert actual["content"] == actual["raw_content"] == prose
    assert actual["custom_field"] == {"kept": 7}


@pytest.mark.parametrize("url", ["https://user:pass@example.com/", "https://example.com/?access_token=secret"])
def test_execution_credential_urls_fail_closed_without_serializing_or_rewriting_identity(store, url, tmp_path):
    candidate = replace(state(store), frontier=FrontierSnapshot(queued=(entry(url),)))
    with pytest.raises(CheckpointError, match="^checkpoint_failed$"):
        store.save(candidate)
    assert not (tmp_path / "progress.json").exists()


@pytest.mark.parametrize("counter", COUNTERS)
@pytest.mark.parametrize("value", [-1, True, 1.5, 2**64])
def test_all_budget_counters_are_strict_and_bounded(store, counter, value):
    expected = state(store)
    with pytest.raises(CheckpointError):
        store.save(replace(expected, budget=replace(expected.budget, **{counter: value})))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, 1801])
def test_remaining_duration_is_finite_bounded_and_not_boolean(store, value):
    expected = state(store)
    with pytest.raises(CheckpointError):
        store.save(replace(expected, budget=replace(expected.budget, remaining_seconds=value)))


def test_url_bytes_are_independent_of_file_limit_at_exact_and_one_over(store):
    expected = state(store)
    target = BASE + "/a"
    resolved = replace(expected.plan, limits=replace(expected.plan.limits, max_url_bytes=len(target)))
    store.save(replace(expected, plan=resolved, frontier=FrontierSnapshot(queued=(entry(target),))))
    with pytest.raises(CheckpointError):
        store.save(replace(expected, plan=resolved, frontier=FrontierSnapshot(queued=(entry(target + "x"),))))


def test_aggregate_url_bytes_are_independent_of_file_limit(store):
    expected = state(store)
    snapshot = FrontierSnapshot(queued=(entry(BASE + "/a"), entry(BASE + "/b", 1)))
    size = sum(len(item.url) for item in snapshot.queued)
    resolved = replace(expected.plan, limits=replace(expected.plan.limits, discovered_url_bytes=size))
    expected = replace(expected, budget=replace(expected.budget, discovered_url_bytes=0))
    store.save(replace(expected, plan=resolved, frontier=snapshot))
    resolved = replace(resolved, limits=replace(resolved.limits, discovered_url_bytes=size - 1))
    with pytest.raises(CheckpointError):
        store.save(replace(expected, plan=resolved, frontier=snapshot))


@pytest.mark.parametrize("collection", ["queued", "inflight", "completed", "results", "visited"])
def test_collection_limits_reject_one_over_without_discarding_results(store, collection):
    expected = state(store)
    store.limits = replace(store.limits, max_frontier=1, max_results=1, max_candidates=1)
    entries = (entry(BASE), entry(BASE + "/b", 1))
    if collection == "results":
        snapshot = FrontierSnapshot(completed=entries, results=({"content": "one"}, {"content": "two"}))
    elif collection == "visited":
        snapshot = FrontierSnapshot(visited=tuple(item.url for item in entries))
    elif collection == "completed":
        snapshot = FrontierSnapshot(completed=entries, results=({}, {}))
    else:
        snapshot = FrontierSnapshot(**{collection: entries})
    with pytest.raises(CheckpointError):
        store.save(replace(expected, frontier=snapshot))


def test_open_parent_substitution_is_detected_before_load_write_or_clear(tmp_path):
    root = tmp_path / "root"
    directory = root / "crawl"
    directory.mkdir(parents=True)
    scope = CheckpointScope("owner", "crawl", root=root)
    value = CheckpointStore.open(directory / "progress.json", scope=scope, limits=CheckpointLimits())
    value.save(state(value))
    moved = tmp_path / "moved"
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    try:
        previous = (moved / "progress.json").read_bytes()
        for operation in (lambda: value.load(plan=plan(), scope=scope), lambda: value.save(state(value)), value.clear):
            with pytest.raises(CheckpointError):
                operation()
        assert (moved / "progress.json").read_bytes() == previous
    finally:
        value.close()


@pytest.mark.parametrize("target", ["progress.json", "progress.json.lock"])
def test_late_symlink_is_refused_without_touching_victim(store, tmp_path, target):
    store.save(state(store))
    victim = tmp_path / "victim"
    victim.write_text("untouched")
    (tmp_path / target).unlink()
    (tmp_path / target).symlink_to(victim)
    for operation in (lambda: store.save(state(store)), store.clear):
        with pytest.raises(CheckpointError):
            operation()
    assert victim.read_text() == "untouched"


def _fork_contender(inherited, path, root, connection):
    inherited.close()
    try:
        inherited.save(state(inherited))
    except CheckpointError:
        _contender(path, root, connection)
    else:
        connection.send("inherited-write")
        connection.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX descriptor inheritance")
def test_fork_child_cannot_write_or_unlock_parent_store():
    from tldw_Server_API.tests.Web_Scraping.fork_assertions import run_single_thread_fork_check

    run_single_thread_fork_check(__name__, "_assert_fork_checkpoint_in_fresh_directory")


def _assert_fork_checkpoint_in_fresh_directory():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="phase5-fork-checkpoint-") as directory:
        root = Path(directory)
        scope = CheckpointScope("owner", "crawl", root=root)
        store = CheckpointStore.open(root / "progress.json", scope=scope, limits=CheckpointLimits())
        try:
            _assert_fork_child_cannot_write_or_unlock_parent_store(store, root)
        finally:
            store.close()


def _assert_fork_child_cannot_write_or_unlock_parent_store(store, tmp_path):
    ctx = multiprocessing.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_fork_contender, args=(store, tmp_path / "progress.json", tmp_path, child))
    process.start()
    child.close()
    try:
        assert parent.poll(5), "fork child did not report"
        assert parent.recv() == "checkpoint_in_use"
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
    store.save(state(store))


def test_binding_does_not_accept_boolean_as_numeric_option(store, tmp_path):
    resolved = plan(max_pages=1)
    store.save(state(store, resolved))
    path = tmp_path / "progress.json"
    data = json.loads(path.read_text())
    data["binding"]["options"]["max_pages"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError):
        store.load(plan=resolved, scope=store.scope)


def test_arbitrary_metadata_local_paths_do_not_escape_projection(store, tmp_path):
    candidate = replace(
        state(store),
        frontier=FrontierSnapshot(
            completed=(entry(BASE),),
            results=(
                {
                    "content": "/private/public-prose",
                    "metadata": {"note": "/private/secret"},
                },
            ),
        ),
    )
    store.save(candidate)
    raw = (tmp_path / "progress.json").read_text()
    assert "/private/secret" not in raw
    assert "/private/public-prose" in raw


def test_failed_temp_cleanup_cannot_leak_paths_or_replace_primary_error(store, monkeypatch):
    store.save(state(store))

    def failed(*args, **kwargs):
        raise OSError("/private/secret filesystem error")

    monkeypatch.setattr(os, "replace", failed)
    monkeypatch.setattr(os, "unlink", failed)
    with pytest.raises(CheckpointError, match="^checkpoint_failed$"):
        store.save(state(store))


def test_native_unlock_failure_still_closes_descriptors_and_registry_without_path_leak(store, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.crawl import checkpoints

    original = checkpoints._release_platform_file_lock

    def failed(*args, **kwargs):
        raise OSError("/private/secret unlock error")

    monkeypatch.setattr(checkpoints, "_release_platform_file_lock", failed)
    with pytest.raises(CheckpointError, match="^checkpoint_failed$"):
        store.close()
    monkeypatch.setattr(checkpoints, "_release_platform_file_lock", original)
    value = CheckpointStore.open(tmp_path / "progress.json", scope=store.scope, limits=store.limits)
    value.close()


@pytest.mark.parametrize(
    "url", ["file:///private/secret", "/private/secret", "https://example.com/?X-Amz-Signature=secret"]
)
def test_non_network_or_signed_execution_urls_fail_closed(store, url):
    candidate = replace(state(store), frontier=FrontierSnapshot(queued=(entry(url),)))
    with pytest.raises(CheckpointError, match="^checkpoint_failed$"):
        store.save(candidate)


@pytest.mark.asyncio
@pytest.mark.parametrize("store", [False, True], indirect=True)
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("target", ["https://foreign.example/blog/story", "https://example.com/unrelated/blog/story"])
async def test_resume_current_profile_rules_precede_private_credential_dispatch(
    store, tmp_path, monkeypatch, legacy, target
):
    from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl
    from tldw_Server_API.tests.Web_Scraping.test_phase5_frontier import page

    profile = "legacy_recursive" if legacy else "enhanced_recursive"
    options = {"delay": 0, "max_pages": 3, "score_threshold": 0, "respect_robots": True}
    resolved = plan(profile, **options)
    if legacy:
        (tmp_path / "progress.json").write_text(
            json.dumps(
                {
                    "visited": [],
                    "to_visit": [[target, 1]],
                    "scraped_articles": [],
                    "pages_scraped": 0,
                }
            )
        )
    else:
        # An untrusted file cannot turn an arbitrary URL into an authorized seed
        # by changing parent/depth/sequence, even with a matching owner binding.
        store.save(replace(state(store, resolved), frontier=FrontierSnapshot(queued=(entry(target),))))
    dispatched, policy_calls = [], []

    async def policy(url, **kwargs):
        policy_calls.append((url, kwargs["respect_robots"], kwargs["stage"]))
        return SimpleNamespace(allowed=True)

    def start(url, cookies, **kwargs):
        dispatched.append((url, cookies, kwargs["prepare_plan"]))
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(page(url))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    monkeypatch.setattr(crawl, "decide_web_outbound_policy", policy)
    monkeypatch.setattr(crawl.article, "start_page", start)
    report = await crawl.execute_crawl(
        resolved.request,
        profile=profile,
        options=options,
        supplied=frozenset(options),
        config={},
        credentials={
            "custom_headers": {"Authorization": "private"},
            "custom_cookies": [{"name": "sid", "value": "private"}],
        },
        checkpoint=store,
    )
    denied = legacy or "foreign.example" in target
    assert bool(dispatched) is not denied
    if denied:
        assert report.outcome.stop_reason == "policy_denied"
    else:
        assert (target, True, "discovery") in policy_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed,robots", [(False, True), (True, False), (True, True)])
async def test_authorized_external_resume_still_requires_current_egress_and_robots(store, monkeypatch, allowed, robots):
    from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl
    from tldw_Server_API.tests.Web_Scraping.test_phase5_frontier import page

    target = "https://foreign.example/blog/story"
    options = {"include_external": True, "score_threshold": 0, "respect_robots": True, "max_pages": 3}
    resolved = plan(**options)
    store.save(replace(state(store, resolved), frontier=FrontierSnapshot(queued=(entry(target),))))
    dispatched, policy_calls = [], []

    async def policy(url, **kwargs):
        policy_calls.append((url, kwargs["respect_robots"]))
        return SimpleNamespace(allowed=allowed and (robots or not kwargs["respect_robots"]))

    def start(url, cookies, **kwargs):
        dispatched.append(url)
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(page(url))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    monkeypatch.setattr(crawl, "decide_web_outbound_policy", policy)
    monkeypatch.setattr(crawl.article, "start_page", start)
    report = await crawl.execute_crawl(
        resolved.request,
        profile=resolved.profile,
        options=options,
        supplied=frozenset(options),
        config={},
        credentials={
            "custom_headers": {"Authorization": "private"},
            "custom_cookies": [{"name": "sid", "value": "private"}],
        },
        checkpoint=store,
    )
    assert (target, True) in policy_calls
    assert dispatched == ([target] if allowed and robots else [])
    assert report.outcome.state == ("completed" if allowed and robots else "failed")
