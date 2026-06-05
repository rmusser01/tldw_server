import os
from pathlib import Path

from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
import tldw_Server_API.app.api.v1.endpoints.setup as setup_endpoint


os.environ.setdefault('TLDW_SETUP_SKIP_DOWNLOADS', '1')


def _make_client():
    return TestClient(app)


def test_complete_setup_with_install_plan_triggers_background_task(mocker):
    install_calls = []

    class _FakeFirstRunStore:
        def validate_completion_ready(self):
            return None

        def mark_completed_with_legacy_flag(self, mark_legacy_complete):
            mark_legacy_complete()

    async def _fake_first_run_store_call(callback):
        return callback(_FakeFirstRunStore())

    mocker.patch.object(
        setup_endpoint.setup_manager,
        'get_status_snapshot',
        return_value={'enabled': True, 'needs_setup': True},
    )
    mock_mark = mocker.patch.object(setup_endpoint.setup_manager, 'mark_setup_completed')
    mocker.patch.object(setup_endpoint.setup_manager, 'update_config')
    mocker.patch.object(
        setup_endpoint,
        'execute_install_plan',
        side_effect=lambda payload: install_calls.append(payload),
    )
    mocker.patch.object(
        setup_endpoint,
        '_run_first_run_store_call',
        side_effect=_fake_first_run_store_call,
    )

    payload = {
        'disable_first_time_setup': False,
        'install_plan': {
            'stt': [{'engine': 'faster_whisper', 'models': ['medium']}],
            'tts': [],
            'embeddings': {
                'huggingface': ['sentence-transformers/all-MiniLM-L6-v2'],
                'custom': [],
                'onnx': [],
            },
        },
    }

    with _make_client() as client:
        response = client.post('/api/v1/setup/complete', json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['install_plan_submitted'] is True
    assert install_calls == [payload['install_plan']]
    mock_mark.assert_called_once_with(True)


def test_install_status_endpoint_returns_idle_when_missing_snapshot(mocker):
    mocker.patch.object(
        setup_endpoint.setup_manager,
        'get_status_snapshot',
        return_value={'enabled': True, 'needs_setup': True},
    )
    mocker.patch.object(
        setup_endpoint.install_manager,
        'get_install_status_snapshot',
        return_value=None,
    )

    with _make_client() as client:
        response = client.get('/api/v1/setup/install-status')

    assert response.status_code == 200
    assert response.json() == {'status': 'idle'}


def test_install_status_endpoint_exposes_snapshot(mocker):
    mocker.patch.object(
        setup_endpoint.setup_manager,
        'get_status_snapshot',
        return_value={'enabled': True, 'needs_setup': True},
    )
    snapshot = {
        'status': 'in_progress',
        'steps': [{'name': 'stt:faster_whisper', 'status': 'in_progress', 'detail': None, 'timestamp': 'now'}],
        'errors': [],
    }
    mocker.patch.object(
        setup_endpoint.install_manager,
        'get_install_status_snapshot',
        return_value=snapshot,
    )

    with _make_client() as client:
        response = client.get('/api/v1/setup/install-status')

    assert response.status_code == 200
    assert response.json() == snapshot


def test_setup_ui_script_exposes_bundle_first_audio_actions():
    with _make_client() as client:
        response = client.get('/static/setup/js/setup.js')

    assert response.status_code == 200
    source = response.text
    assert '/audio/recommendations' in source
    assert '/audio/provision' in source
    assert '/audio/verify' in source
    assert 'Recommended audio bundle' in source
    assert 'Recommended profile' in source
    assert 'Light' in source
    assert 'Balanced' in source
    assert 'Performance' in source
    assert 'Provision recommended bundle' in source
    assert 'Run verification' in source
    assert 'Safe rerun' in source


def test_setup_ui_script_mentions_kitten_tts():
    with _make_client() as client:
        response = client.get('/static/setup/js/setup.js')

    assert response.status_code == 200
    source = response.text
    assert 'kitten_tts' in source
    assert 'KittenTTS' in source
