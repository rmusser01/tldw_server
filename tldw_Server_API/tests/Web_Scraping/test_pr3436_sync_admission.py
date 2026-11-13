"""Synchronous crawl admission belongs to the existing service container."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from tldw_Server_API.app.core.Utils import app_services as services_module
from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy


def test_sync_admission_uses_each_service_containers_own_gate(monkeypatch):
    first = services_module.AppServices()
    second = services_module.AppServices()
    monkeypatch.setattr(services_module, "_SERVICES", first)
    first_gate = legacy._get_sync_crawl_admission()
    release = first_gate.acquire()
    try:
        monkeypatch.setattr(services_module, "_SERVICES", second)
        second_gate = legacy._get_sync_crawl_admission()
        assert second_gate is not first_gate
        assert second_gate.active_count == 0
        assert first_gate.active_count == 1
        assert legacy._get_sync_crawl_admission() is second_gate
    finally:
        release()
    assert first_gate.active_count == 0


def test_concurrent_first_calls_share_one_admission_gate(monkeypatch):
    services = services_module.AppServices()
    monkeypatch.setattr(services_module, "_SERVICES", services)
    create_gate = legacy._SyncCrawlAdmission
    both_creating = Barrier(2, timeout=5)

    def create_together():
        both_creating.wait()
        return create_gate()

    monkeypatch.setattr(legacy, "_SyncCrawlAdmission", create_together)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(legacy._get_sync_crawl_admission)
        second = executor.submit(legacy._get_sync_crawl_admission)
        assert first.result() is second.result() is legacy._get_sync_crawl_admission()


def test_first_call_can_race_with_gate_publication_before_capacity_lookup(monkeypatch):
    winner = legacy._SyncCrawlAdmission()

    class AdmissionRegistry(dict):
        def values(self):
            registry = self

            class LiveView:
                def __reversed__(self):
                    values = reversed(dict.values(registry))
                    registry[winner.pid] = winner
                    return values

            return LiveView()

        def copy(self):
            snapshot = super().copy()
            self[winner.pid] = winner
            return snapshot

    services = services_module.AppServices(sync_crawl_admissions=AdmissionRegistry())
    monkeypatch.setattr(services_module, "_SERVICES", services)
    assert legacy._get_sync_crawl_admission() is winner


def test_sync_admission_replaces_child_gate_before_touching_inherited_lock(monkeypatch):
    services = services_module.AppServices()
    monkeypatch.setattr(services_module, "_SERVICES", services)
    parent = legacy._get_sync_crawl_admission()
    parent.capacity = 2
    release_parent = parent.acquire()
    pid = legacy.os.getpid()
    parent._lock.acquire()
    try:
        with monkeypatch.context() as child:
            child.setattr(legacy.os, "getpid", lambda: pid + 1)
            gate = legacy._get_sync_crawl_admission()
            assert gate is not parent
            assert gate.capacity == 2
            release_child = gate.acquire()
            release_parent()
            assert gate.active_count == 1
            release_child()
            assert gate.active_count == 0
    finally:
        parent._lock.release()
        release_parent()
    assert parent.active_count == 0
