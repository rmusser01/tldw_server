"""Ordinary-user transcription through normal PostgreSQL startup and auth."""

import json
import os
import subprocess  # nosec B404 - Fixed supported runtime child.
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

SCRIPT = r"""
import asyncio
import io
import json
import os
import re
import signal
import sys
import traceback
import wave

import asyncpg
from fastapi.testclient import TestClient
from tldw_Server_API.app.core.AuthNZ.database import _GuardedAsyncpgConnection, get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database
from tldw_Server_API.app.core.AuthNZ.create_admin import create_admin_user_non_interactive
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.testing import is_test_mode, is_explicit_pytest_runtime

def safe_failure(kind, value, tb):
    frames = [(frame.filename.rsplit('/tldw_Server_API/', 1)[-1], frame.lineno, frame.name)
        for frame in traceback.extract_tb(tb)]
    print('B05_RUNTIME_FAILURE=' + json.dumps({'type': kind.__name__, 'frames': frames,
        'result': globals().get('result', {}), 'databaseErrors': globals().get('errors', [])[:1]}))
sys.excepthook = safe_failure
def graceful_stop(_signal, _frame):
    raise SystemExit(143)
signal.signal(signal.SIGTERM, graceful_stop)
assert not is_test_mode() and not is_explicit_pytest_runtime()
settings = get_settings()
async def prepare():
    assert await setup_database()
    from tldw_Server_API.app.core.AuthNZ.repos.org_stt_settings_repo import AuthnzOrgSttSettingsRepo
    # Bootstrap must provide the policy schema before full app startup.
    assert await AuthnzOrgSttSettingsRepo(await get_db_pool()).get_settings(1) is None
    assert await create_admin_user_non_interactive(username=os.environ['ADMIN_USERNAME'],
        password=os.environ['ADMIN_PASSWORD'], email=os.environ['ADMIN_EMAIL'])
    await reset_db_pool()
asyncio.run(prepare())

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import ParakeetAdapter
calls = []
def transcribe(self, path, **kwargs):
    calls.append(kwargs['model'])
    return {'text': 'Runtime transcription.', 'language': 'en', 'segments': [], 'metadata': {}}
ParakeetAdapter.transcribe_batch = transcribe

errors = []
def observe(name, delegate):
    async def call(self, query, *args, **kwargs):
        try:
            return await delegate(self, query, *args, **kwargs)
        except asyncpg.UndefinedTableError as exc:
            # Schema identifiers and source locations only; never SQL values or credentials.
            match = re.search(r'relation "([a-zA-Z0-9_.]+)" does not exist', str(exc))
            errors.append({'operation': name, 'sqlstate': exc.sqlstate,
                'relation': match.group(1) if match else None,
                'frames': [(PathName.filename.rsplit('/tldw_Server_API/', 1)[-1], PathName.lineno, PathName.name)
                    for PathName in traceback.extract_stack() if '/tldw_Server_API/' in PathName.filename][-8:]})
            raise
    return call
for name in ('execute', 'fetch', 'fetchrow', 'fetchval'):
    setattr(_GuardedAsyncpgConnection, name, observe(name, getattr(_GuardedAsyncpgConnection, name)))

from tldw_Server_API.app.main import app
result = {}
with TestClient(app, raise_server_exceptions=False) as client:
    login = client.post('/api/v1/auth/login', data={'username': os.environ['ADMIN_USERNAME'], 'password': os.environ['ADMIN_PASSWORD']})
    result['adminLogin'] = login.status_code
    if login.status_code == 200:
        admin = {'Authorization': 'Bearer ' + login.json()['access_token']}
        for index in range(2, 13):
            username = 'runtime_audio_alice' if index == 12 else 'runtime_audio_setup' + str(index)
            created = client.post('/api/v1/admin/users', headers=admin,
                json={'username': username, 'email': username + '@example.com', 'password': os.environ['ADMIN_PASSWORD'], 'role': 'user'})
            result['lastCreateStatus'] = created.status_code
            assert created.status_code == 200, 'Normal user creation failed'
            assert created.json()['id'] == index and created.json()['role'] == 'user'
        result.pop('lastCreateStatus', None)
        login = client.post('/api/v1/auth/login', data={'username': 'runtime_audio_alice', 'password': os.environ['ADMIN_PASSWORD']})
        result['userLogin'] = login.status_code
        if login.status_code == 200:
            headers = {'Authorization': 'Bearer ' + login.json()['access_token']}
            me = client.get('/api/v1/auth/me', headers=headers)
            result['userId'] = me.json()['id']
            stream = io.BytesIO()
            with wave.open(stream, 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(bytes(3200))
            errors.clear()
            response = client.post('/api/v1/audio/transcriptions', headers=headers,
                files={'file': ('runtime.wav', stream.getvalue(), 'audio/wav')},
                data={'model': 'parakeet-tdt-0.6b-v3-onnx', 'language': 'en', 'response_format': 'json'})
            result.update(status=response.status_code, providerCalls=len(calls), firstDatabaseError=errors[:1])
print('B05_RUNTIME_RESULT=' + json.dumps(result))
"""


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_ordinary_postgres_user_can_transcribe_after_normal_startup(pg_restricted_backend, tmp_path):
    """Keep startup, auth, request transactions, policy and quotas real."""
    cfg = pg_restricted_backend.config
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env").write_text("")
    # Preserve HOME and use only this fixture's profile and database.
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER", "LANG", "TMPDIR") if key in os.environ}
    env.update(
        PYTHONPATH=str(ROOT),
        AUTH_MODE="multi_user",
        TLDW_USER_DB_BACKEND="postgresql",
        TLDW_CONTENT_DB_BACKEND="postgresql",
        DATABASE_URL=f"postgresql://{cfg.pg_user}:{cfg.pg_password}@{cfg.pg_host}:{cfg.pg_port}/{cfg.pg_database}",
        ADMIN_USERNAME="runtime_audio_admin",
        ADMIN_EMAIL="runtime_audio_admin@example.com",
        ADMIN_PASSWORD="RuntimeAudio@Test2026!",  # nosec B106 - Isolated ephemeral fixture credential.
        JWT_SECRET_KEY="runtime-audio-jwt-secret-at-least-32-bytes",
        API_KEY_HMAC_SECRET="runtime-audio-hmac-secret-at-least-32-bytes",
        USER_DB_BASE_DIR=str(tmp_path / "data/users"),
        TLDW_CONFIG_DIR=str(config),
        TLDW_ENV_FILE=str(config / ".env"),
        TLDW_ENV_FILE_EXCLUSIVE="1",
        SYSTEM_LOG_FILE_PATH=str(tmp_path / "data/system_logs.jsonl"),
        CIRCUIT_BREAKER_REGISTRY_DB_PATH=str(tmp_path / "data/circuit_breakers.db"),
        AUTO_DOWNLOAD_MODELS="false",
        JOBS_DB_URL=f"postgresql://{cfg.pg_user}:{cfg.pg_password}@{cfg.pg_host}:{cfg.pg_port}/{cfg.pg_database}",
        TLDW_CONTENT_PG_DSN=f"postgresql://{cfg.pg_user}:{cfg.pg_password}@{cfg.pg_host}:{cfg.pg_port}/{cfg.pg_database}",
        TLDW_PG_HOST=cfg.pg_host,
        TLDW_PG_PORT=str(cfg.pg_port),
        TLDW_PG_DATABASE=cfg.pg_database,
        TLDW_PG_USER=cfg.pg_user,
        TLDW_PG_PASSWORD=cfg.pg_password,
    )
    child = subprocess.Popen(  # nosec B603 - Fixed normal application child; private DSN stays in environment.
        [sys.executable, "-c", SCRIPT],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timed_out = False
    try:
        output, _private_errors = child.communicate(timeout=150)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                # Retain the official fixture until the owned child exits; never force it.
                child.communicate()
    if timed_out:
        raise AssertionError("Owned runtime exceeded150s; graceful cleanup completed") from None
    child_returncode = child.returncode
    failures = [
        json.loads(line.removeprefix("B05_RUNTIME_FAILURE="))
        for line in output.splitlines()
        if line.startswith("B05_RUNTIME_FAILURE=")
    ]
    assert child_returncode == 0, failures or "Runtime child failed; private diagnostics withheld"  # nosec B101 - Pytest assertion with sanitized diagnostics only.
    records = [
        json.loads(line.removeprefix("B05_RUNTIME_RESULT="))
        for line in output.splitlines()
        if line.startswith("B05_RUNTIME_RESULT=")
    ]
    assert len(records) == 1  # nosec B101 - Require exactly one safe runtime result.
    assert records[0] == {  # nosec B101 - Real startup/auth/SQL/endpoint contract; pytest assertions are intentional.
        "adminLogin": 200,
        "userLogin": 200,
        "userId": 12,
        "status": 200,
        "providerCalls": 1,
        "firstDatabaseError": [],
    }


def test_runtime_timeout_retains_fixture_until_owned_child_exits(tmp_path, monkeypatch):
    """A timed-out test must not force the child or release a live database."""
    from types import SimpleNamespace

    calls = []

    class Child:
        returncode = None

        def communicate(self, *, timeout=None):
            calls.append(("communicate", timeout))
            if timeout is None:
                self.returncode = 143
                return "", ""
            raise subprocess.TimeoutExpired("owned runtime", timeout)

        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate")

    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: Child())
    fixture = SimpleNamespace(
        config=SimpleNamespace(  # nosec B106 - Synthetic timeout-test credential; no process or database is contacted.
            pg_user="fixture", pg_password="private", pg_host="127.0.0.1", pg_port=15492, pg_database="fixture"
        )
    )
    with pytest.raises(AssertionError, match="Owned runtime exceeded150s; graceful cleanup completed") as failure:
        test_ordinary_postgres_user_can_transcribe_after_normal_startup(fixture, tmp_path)
    assert failure.value.__cause__ is None and failure.value.__suppress_context__ is True  # nosec B101 - The real timeout driver suppresses private child-error context.
    assert calls == [("communicate", 150), "terminate", ("communicate", 15), ("communicate", None)]  # nosec B101 - Require graceful ownership cleanup, never force.
