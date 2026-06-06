from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
import tldw_Server_API.app.api.v1.endpoints.setup as setup_endpoint


def _make_client():
    return TestClient(app)


def test_update_config_rejects_unknown_section(mocker):
    mocker.patch.object(
        setup_endpoint.setup_manager,
        'get_status_snapshot',
        return_value={'enabled': True, 'needs_setup': True},
    )
    payload = {'updates': {'NopeSection': {'foo': 'bar'}}}
    with _make_client() as client:
        resp = client.post('/api/v1/setup/config', json=payload)
    assert resp.status_code == 400
    assert 'Unknown section' in resp.text


def test_update_config_rejects_unknown_key(mocker):
    mocker.patch.object(
        setup_endpoint.setup_manager,
        'get_status_snapshot',
        return_value={'enabled': True, 'needs_setup': True},
    )
    # Known section, fake key
    payload = {'updates': {'Setup': {'does_not_exist': '1'}}}
    with _make_client() as client:
        resp = client.post('/api/v1/setup/config', json=payload)
    assert resp.status_code == 400
    assert 'Unknown key' in resp.text


def test_update_config_type_validation_boolean(mocker):
    mocker.patch.object(
        setup_endpoint.setup_manager,
        'get_status_snapshot',
        return_value={'enabled': True, 'needs_setup': True},
    )
    # allow_remote_setup_access should be boolean-like; provide invalid string.
    # Lifecycle setup flags are rejected by the endpoint before config type validation.
    payload = {'updates': {'Setup': {'allow_remote_setup_access': 'not_boolean'}}}
    with _make_client() as client:
        resp = client.post('/api/v1/setup/config', json=payload)
    assert resp.status_code == 400
    assert 'Invalid boolean' in resp.text


def test_update_config_masks_internal_errors(mocker):
    mocker.patch.object(
        setup_endpoint.setup_manager,
        'get_status_snapshot',
        return_value={'enabled': True, 'needs_setup': True},
    )
    mocker.patch.object(
        setup_endpoint.setup_manager,
        'update_config',
        side_effect=RuntimeError('db blew up at /tmp/secret-config'),
    )

    with _make_client() as client:
        resp = client.post('/api/v1/setup/config', json={'updates': {'Setup': {'allow_remote_setup_access': 'true'}}})

    assert resp.status_code == 500
    assert resp.json() == {'detail': 'Failed to persist setup configuration.'}
