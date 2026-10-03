"""Guarded synthetic local API flag/parity/rollback checks, used by archive probes."""
from __future__ import annotations

if not __debug__:
    raise RuntimeError("Synthetic validation requires Python without optimization")

import os
from typing import Any


async def validate_local_release(
    client: Any, *, read_headers: dict[str, str], sample_media_id: int, expected_media_ids: set[int],
) -> dict[str, Any]:
    """Exercise real HTTP handlers while reloading effective local flag settings.

    This validates reversible application configuration on a disposable local
    deployment; it does not represent production rollout or owner approval.
    """
    from tldw_Server_API.app.core.config import load_settings, settings
    from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
    from tldw_Server_API.app.core.testing import env_flag_enabled

    flags = ('EMAIL_NATIVE_PERSIST_ENABLED', 'EMAIL_OPERATOR_SEARCH_ENABLED',
             'EMAIL_MEDIA_SEARCH_DELEGATION_MODE', 'EMAIL_GMAIL_CONNECTOR_ENABLED', 'CONNECTORS_WORKER_ENABLED')
    def effective_flags() -> dict[str, Any]:
        return {key: env_flag_enabled(key) if key == 'CONNECTORS_WORKER_ENABLED' else settings.get(key) for key in flags}

    baseline = effective_flags()
    previous_env = {key: os.environ.get(key) for key in flags}
    expected_media_ids = set(expected_media_ids)
    assert expected_media_ids and sample_media_id in expected_media_ids  # nosec B101 - validation assertion; optimized execution rejected
    assert not baseline['EMAIL_GMAIL_CONNECTOR_ENABLED'] and not baseline['CONNECTORS_WORKER_ENABLED']  # nosec B101 - validation assertion; optimized execution rejected

    def reload_flags(**values: str) -> None:
        os.environ.update(values)
        loaded = load_settings()
        settings.update({key: loaded[key] for key in flags if key in loaded})

    async def media_ids(query: str, mode: str | None = None) -> set[int]:
        found: set[int] = set()
        page = 1
        while True:
            body = {'query': query, 'media_types': ['email']}
            if mode:
                body['email_query_mode'] = mode
            response = await client.post('/api/v1/media/search', json=body, headers=read_headers,
                                         params={'page': page, 'results_per_page': 100})
            assert response.status_code == 200, (response.status_code, response.text[:500])  # nosec B101 - validation assertion; optimized execution rejected
            payload = response.json()
            found.update(row['id'] for row in payload['items'])
            if page >= payload['pagination']['total_pages']:
                break
            page += 1
        return found

    evidence: dict[str, Any] = {'environment': 'guarded_local_reference', 'baseline_flags': baseline, 'parity': []}
    try:
        queries = ['ArchiveThroughput', '"ArchiveThroughput 0-000"', '"Unique synthetic archive body batch 0 message 0"']
        baseline_legacy_ids: set[int] = set()
        for query in queries:
            legacy = await media_ids(query, 'legacy')
            native = await media_ids(query, 'operators')
            assert legacy and legacy == native, (query, len(legacy), len(native))  # nosec B101 - validation assertion; optimized execution rejected
            if query == 'ArchiveThroughput':
                assert legacy == expected_media_ids  # nosec B101 - validation assertion; optimized execution rejected
                baseline_legacy_ids = legacy
            print(f'RELEASE_PARITY query_index={queries.index(query)} matching_ids={len(legacy)}', flush=True)
            evidence['parity'].append({'query': query, 'legacy_count': len(legacy), 'native_count': len(native), 'equal_ids': True})
        reload_flags(EMAIL_MEDIA_SEARCH_DELEGATION_MODE='auto_email')
        query = 'subject:ArchiveThroughput'
        automatic = await media_ids(query)
        explicit = await media_ids(query, 'operators')
        assert automatic == explicit == expected_media_ids  # nosec B101 - validation assertion; optimized execution rejected
        print(f'RELEASE_AUTO_EMAIL matching_ids={len(automatic)}', flush=True)
        evidence['auto_email'] = {'configured': settings['EMAIL_MEDIA_SEARCH_DELEGATION_MODE'], 'same_ids_as_explicit': True, 'count': len(automatic)}
        reload_flags(EMAIL_OPERATOR_SEARCH_ENABLED='false', EMAIL_NATIVE_PERSIST_ENABLED='false', EMAIL_MEDIA_SEARCH_DELEGATION_MODE='opt_in')
        search = await client.get('/api/v1/email/search', headers=read_headers)
        detail = await client.get(f'/api/v1/email/messages/{sample_media_id}', headers=read_headers)
        operators = await client.post('/api/v1/media/search', json={'query': query, 'media_types': ['email'], 'email_query_mode': 'operators'}, headers=read_headers)
        legacy = await media_ids('ArchiveThroughput', 'legacy')
        assert search.status_code == 404 and detail.status_code == 404 and operators.status_code == 422  # nosec B101 - validation assertion; optimized execution rejected
        assert legacy == baseline_legacy_ids == expected_media_ids  # nosec B101 - validation assertion; optimized execution rejected
        print(f'RELEASE_ROLLBACK retained_ids={len(legacy)}', flush=True)
        evidence['rollback'] = {'flags': effective_flags(), 'email_search_status': search.status_code,
                                'email_detail_status': detail.status_code, 'operator_bridge_status': operators.status_code,
                                'legacy_search_count': len(legacy), 'legacy_content_preserved': True}
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        loaded = load_settings()
        settings.update({key: loaded[key] for key in flags if key in loaded})
    restored = await client.get(f'/api/v1/email/messages/{sample_media_id}', headers=read_headers)
    assert restored.status_code == 200  # nosec B101 - validation assertion; optimized execution rejected
    evidence['restored_detail_status'] = restored.status_code
    exported = get_metrics_registry().export_prometheus_format()
    prefixes = ('email_ingestion_parse_total', 'email_ingestion_parse_seconds', 'email_ingestion_persist_total',
                'email_ingestion_persist_seconds', 'email_ingestion_dedupe_total', 'email_native_persist_total', 'email_native_persist_seconds', 'email_native_search_')
    samples = [line for line in exported.splitlines() if not line.startswith('#') and line.startswith(prefixes)]
    assert any(line.startswith('email_ingestion_parse_total') for line in samples)  # nosec B101 - validation assertion; optimized execution rejected
    assert any(line.startswith('email_ingestion_dedupe_total') for line in samples)  # nosec B101 - validation assertion; optimized execution rejected
    evidence['metric_export_samples'] = samples
    evidence['restored_flags'] = effective_flags()
    return evidence
