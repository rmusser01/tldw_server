import os
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
import tldw_Server_API.app.api.v1.endpoints.setup as setup_endpoint


def _make_client():


    return TestClient(app)


def test_update_config_blocked_for_remote_via_forwarded_header(mocker, monkeypatch):


     # Ensure remote access is not allowed
    monkeypatch.delenv('TLDW_SETUP_ALLOW_REMOTE', raising=False)
    monkeypatch.setenv('TLDW_SETUP_TRUST_PROXY', '1')

    # Pretend setup is enabled and needed
    mocker.patch.object(
        setup_endpoint.setup_manager,
        'get_status_snapshot',
        return_value={'enabled': True, 'needs_setup': True},
    )

    payload = {
        'updates': {'Setup': {'enable_first_time_setup': True}}
    }

    with _make_client() as client:
        # Simulate a non-local client via X-Forwarded-For
        response = client.post(
            '/api/v1/setup/config', json=payload, headers={'X-Forwarded-For': '8.8.8.8'}
        )

    assert response.status_code == 403
    body = response.json()
    detail = (body.get('detail') or '').lower()
    assert 'local' in detail
    assert 'remote setup access' in detail


def test_complete_blocked_for_remote_via_forwarded_header(mocker, monkeypatch):


     # Ensure remote access is not allowed
    monkeypatch.delenv('TLDW_SETUP_ALLOW_REMOTE', raising=False)
    monkeypatch.setenv('TLDW_SETUP_TRUST_PROXY', '1')

    mocker.patch.object(
        setup_endpoint.setup_manager,
        'get_status_snapshot',
        return_value={'enabled': True, 'needs_setup': True},
    )

    payload = {
        'disable_first_time_setup': False,
        'install_plan': {
            'stt': [], 'tts': [], 'embeddings': {'huggingface': [], 'custom': [], 'onnx': []}
        },
    }

    with _make_client() as client:
        response = client.post(
            '/api/v1/setup/complete', json=payload, headers={'X-Forwarded-For': '1.2.3.4'}
        )

    assert response.status_code == 403
    body = response.json()
    detail = (body.get('detail') or '').lower()
    assert 'local' in detail
    assert 'remote setup access' in detail
def test_get_config_blocked_for_remote_when_trust_proxy_enabled(mocker, monkeypatch):
     # Setup enabled
    mocker.patch.object(
        setup_endpoint.setup_manager,
        'get_status_snapshot',
        return_value={'enabled': True, 'needs_setup': True},
    )
    # Trust proxy and treat forwarded as remote
    monkeypatch.delenv('TLDW_SETUP_ALLOW_REMOTE', raising=False)
    monkeypatch.setenv('TLDW_SETUP_TRUST_PROXY', '1')

    with _make_client() as client:
        response = client.get('/api/v1/setup/config', headers={'X-Forwarded-For': '203.0.113.10'})

    assert response.status_code == 403
