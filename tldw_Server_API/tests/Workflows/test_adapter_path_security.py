from pathlib import Path

import pytest

import tldw_Server_API.app.core.Workflows.adapters as wf_adapters
import tldw_Server_API.app.core.Workflows.adapters._common as workflow_common


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_prompt_adapter_sanitizes_artifact_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def _capture_artifact(**kwargs):

        captured["uri"] = kwargs.get("uri")

    context = {"step_run_id": "../escape", "add_artifact": _capture_artifact}
    result = await wf_adapters.run_prompt_adapter({"template": "hello", "save_artifact": True}, context)
    assert result.get("text") == "hello"

    uri = captured.get("uri")
    assert isinstance(uri, str) and uri.startswith("file://")
    path = Path(uri[len("file://") :])
    base_dir = (tmp_path / "Databases" / "artifacts").resolve()
    assert path.resolve().is_relative_to(base_dir)
    assert path.exists()


@pytest.mark.asyncio
async def test_tts_adapter_sanitizes_output_filename(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    from tldw_Server_API.app.core.Workflows.adapters.audio import tts as tts_adapter

    class _FakeTTSService:
        async def generate_speech(self, request, provider=None, fallback=True, voice_to_voice_start=None, voice_to_voice_route="audio.speech"):
            yield b"fake-audio"

    async def _fake_get_tts_service_v2(config=None):
        return _FakeTTSService()

    monkeypatch.setattr(tts_adapter, "get_tts_service_v2", _fake_get_tts_service_v2, raising=True)

    config = {
        "input": "hello",
        "response_format": "mp3",
        "output_filename_template": "../evil",
    }
    context = {"step_run_id": "../escape"}
    result = await wf_adapters.run_tts_adapter(config, context)

    uri = result.get("audio_uri")
    assert isinstance(uri, str) and uri.startswith("file://")
    path = Path(uri[len("file://") :])
    base_dir = (tmp_path / "Databases" / "artifacts").resolve()
    assert path.resolve().is_relative_to(base_dir)
    assert path.name.endswith(".mp3")
    assert path.exists()


@pytest.mark.asyncio
async def test_stt_transcribe_rejects_outside_base(monkeypatch, tmp_path):
    user_base = tmp_path / "user_dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_base))

    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF\x00\x00\x00WAVEfmt ")

    result = await wf_adapters.run_stt_transcribe_adapter(
        {"file_uri": f"file://{outside}"},
        {"user_id": 123},
    )
    assert result.get("error") == "file_access_denied"


@pytest.mark.asyncio
async def test_stt_transcribe_accepts_inside_base(monkeypatch, tmp_path):
    user_base = tmp_path / "user_dbs"
    user_id = 123
    user_dir = user_base / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_base))

    inside = user_dir / "valid.wav"
    inside.write_bytes(b"RIFF\x00\x00\x00WAVEfmt ")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Lib as stt_mod

    def _fake_speech_to_text(*_args, **_kwargs):

        return ([{"Text": "hello"}], "en")

    monkeypatch.setattr(stt_mod, "speech_to_text", _fake_speech_to_text, raising=True)

    result = await wf_adapters.run_stt_transcribe_adapter(
        {"file_uri": f"file://{inside}"},
        {"user_id": user_id},
    )
    assert result.get("text") == "hello"
    assert result.get("segments") == [{"Text": "hello"}]
    assert result.get("language") == "en"


@pytest.mark.asyncio
async def test_stt_transcribe_unsafe_access_requires_allowlist(monkeypatch, tmp_path):
    user_base = tmp_path / "user_dbs"
    user_id = 123
    user_dir = user_base / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_base))
    monkeypatch.setenv("WORKFLOWS_ALLOW_UNSAFE_FILE_ACCESS", "true")
    monkeypatch.setenv("WORKFLOWS_FILE_ALLOWLIST", str(user_dir))

    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF\x00\x00\x00WAVEfmt ")

    result = await wf_adapters.run_stt_transcribe_adapter(
        {"file_uri": f"file://{outside}"},
        {"user_id": user_id},
    )
    assert result.get("error") == "file_access_denied"


@pytest.mark.asyncio
async def test_stt_transcribe_unsafe_access_allows_allowlist(monkeypatch, tmp_path):
    user_base = tmp_path / "user_dbs"
    user_id = 123
    user_dir = user_base / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_base))
    monkeypatch.setenv("WORKFLOWS_ALLOW_UNSAFE_FILE_ACCESS", "true")
    monkeypatch.setenv("WORKFLOWS_FILE_ALLOWLIST", str(tmp_path))

    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF\x00\x00\x00WAVEfmt ")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Lib as stt_mod

    def _fake_speech_to_text(*_args, **_kwargs):
        return ([{"Text": "hello"}], "en")

    monkeypatch.setattr(stt_mod, "speech_to_text", _fake_speech_to_text, raising=True)

    result = await wf_adapters.run_stt_transcribe_adapter(
        {"file_uri": f"file://{outside}"},
        {"user_id": user_id},
    )
    assert result.get("text") == "hello"
    assert result.get("segments") == [{"Text": "hello"}]
    assert result.get("language") == "en"


@pytest.mark.asyncio
async def test_stt_transcribe_unsafe_access_tenant_allowlist_override(monkeypatch, tmp_path):
    user_base = tmp_path / "user_dbs"
    user_id = 123
    user_dir = user_base / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_base))
    monkeypatch.setenv("WORKFLOWS_ALLOW_UNSAFE_FILE_ACCESS", "true")
    monkeypatch.setenv("WORKFLOWS_FILE_ALLOWLIST", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_FILE_ALLOWLIST_ACME", str(user_dir))

    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF\x00\x00\x00WAVEfmt ")

    result = await wf_adapters.run_stt_transcribe_adapter(
        {"file_uri": f"file://{outside}"},
        {"user_id": user_id, "tenant_id": "acme"},
    )
    assert result.get("error") == "file_access_denied"


def test_resolve_workflow_file_path_allows_relative_under_base(monkeypatch, tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(base_dir))

    resolved = wf_adapters._resolve_workflow_file_path("subdir/file.txt", {})
    assert resolved == (base_dir / "subdir" / "file.txt").resolve(strict=False)
    assert resolved.resolve(strict=False).is_relative_to(base_dir.resolve(strict=False))


def test_resolve_workflow_file_path_allows_absolute_under_base(monkeypatch, tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(base_dir))

    inside = base_dir / "inside.txt"
    resolved = wf_adapters._resolve_workflow_file_path(str(inside), {})
    assert resolved == inside.resolve(strict=False)
    assert resolved.resolve(strict=False).is_relative_to(base_dir.resolve(strict=False))


def test_resolve_workflow_file_path_rejects_traversal(monkeypatch, tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(base_dir))

    with pytest.raises(wf_adapters.AdapterError):
        wf_adapters._resolve_workflow_file_path("../escape.txt", {})


def test_resolve_workflow_file_path_rejects_absolute_outside_base(monkeypatch, tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(base_dir))

    outside = tmp_path / "outside.txt"
    with pytest.raises(wf_adapters.AdapterError):
        wf_adapters._resolve_workflow_file_path(str(outside), {})


def test_resolve_workflow_file_path_unsafe_allows_allowlist(monkeypatch, tmp_path):
    base_dir = tmp_path / "base"
    allow_dir = tmp_path / "allow"
    base_dir.mkdir()
    allow_dir.mkdir()
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(base_dir))
    monkeypatch.setenv("WORKFLOWS_ALLOW_UNSAFE_FILE_ACCESS", "true")
    monkeypatch.setenv("WORKFLOWS_FILE_ALLOWLIST", str(allow_dir))

    target = allow_dir / "allowed.txt"
    resolved = wf_adapters._resolve_workflow_file_path(str(target), {})
    assert resolved == target.resolve(strict=False)
    assert resolved.resolve(strict=False).is_relative_to(allow_dir.resolve(strict=False))


def test_unsafe_file_access_flag_accepts_y(monkeypatch):
    monkeypatch.setenv("WORKFLOWS_ALLOW_UNSAFE_FILE_ACCESS", "y")
    assert wf_adapters._unsafe_file_access_allowed(None) is True


def test_artifacts_base_dir_accepts_test_mode_y(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("TEST_MODE", "y")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    base_dir = wf_adapters._artifacts_base_dir()
    assert base_dir == (tmp_path / "Databases" / "artifacts").resolve()


def test_resolve_workflow_file_path_unsafe_denies_without_allowlist(monkeypatch, tmp_path):
    base_dir = tmp_path / "base"
    allow_dir = tmp_path / "allow"
    base_dir.mkdir()
    allow_dir.mkdir()
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(base_dir))
    monkeypatch.setenv("WORKFLOWS_ALLOW_UNSAFE_FILE_ACCESS", "true")
    monkeypatch.setenv("WORKFLOWS_FILE_ALLOWLIST", str(base_dir))

    target = allow_dir / "blocked.txt"
    with pytest.raises(wf_adapters.AdapterError):
        wf_adapters._resolve_workflow_file_path(str(target), {})


def test_is_subpath_sanitizes_resolve_failure_logs():
    class _BrokenPath:
        def __init__(self, label: str) -> None:
            self.label = label

        def __fspath__(self) -> str:
            return f"/private/{self.label}"

        def __str__(self) -> str:
            return f"/private/{self.label}"

        def resolve(self, strict: bool = False):  # noqa: ARG002, ANN201
            raise OSError(f"secret resolve failure for {self.label}")

        def relative_to(self, _other):  # noqa: ANN001, ANN201
            raise ValueError

    messages: list[str] = []
    sink_id = workflow_common.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        assert workflow_common.is_subpath(_BrokenPath("parent-token"), _BrokenPath("child-token")) is False
    finally:
        workflow_common.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Failed to resolve workflow parent path" in joined
    assert "Failed to resolve workflow child path" in joined
    assert "parent-token" not in joined
    assert "child-token" not in joined
    assert "secret resolve failure" not in joined


def test_resolve_workflows_file_allowlist_sanitizes_invalid_path_logs(monkeypatch, tmp_path):
    original_resolve = workflow_common.Path.resolve
    allowlist_path = tmp_path / "allowlist-token"

    def broken_resolve(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if "allowlist-token" in str(self):
            raise OSError("secret allowlist backend at /private/allowlist.db")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(workflow_common.Path, "resolve", broken_resolve)

    messages: list[str] = []
    sink_id = workflow_common.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        assert workflow_common.resolve_workflows_file_allowlist_paths([str(allowlist_path)]) == []
    finally:
        workflow_common.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Workflow file allowlist: invalid path skipped" in joined
    assert "allowlist-token" not in joined
    assert "private/allowlist.db" not in joined


def test_resolve_workflows_file_allowlist_sanitizes_project_root_logs(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Utils import Utils as utils_mod

    def broken_project_root():  # noqa: ANN202
        raise RuntimeError("secret project root at /private/allowlist-root")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils_mod, "get_project_root", broken_project_root)

    messages: list[str] = []
    sink_id = workflow_common.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        resolved = workflow_common.resolve_workflows_file_allowlist_paths(["relative-allowlist"])
    finally:
        workflow_common.logger.remove(sink_id)

    assert resolved == [(tmp_path / "relative-allowlist").resolve(strict=False)]
    joined = "\n".join(messages)
    assert "Workflow file allowlist: failed to resolve project root" in joined
    assert "allowlist-root" not in joined
    assert "private/allowlist-root" not in joined


def test_workflow_file_base_dir_sanitizes_relative_override_logs(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Utils import Utils as utils_mod

    def broken_project_root():  # noqa: ANN202
        raise RuntimeError("secret project root at /private/workflows-root")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", "relative-secret-base")
    monkeypatch.setattr(utils_mod, "get_project_root", broken_project_root)

    messages: list[str] = []
    sink_id = workflow_common.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        assert workflow_common.workflow_file_base_dir({}, None) == (tmp_path / "relative-secret-base").resolve()
    finally:
        workflow_common.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Workflow file base dir: failed to resolve relative override" in joined
    assert "relative-secret-base" not in joined
    assert "private/workflows-root" not in joined


def test_workflow_file_base_dir_sanitizes_invalid_user_id_logs(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.DB_Management import db_path_utils

    monkeypatch.delenv("WORKFLOWS_FILE_BASE_DIR", raising=False)
    monkeypatch.setattr(db_path_utils.DatabasePaths, "get_single_user_id", lambda: 7)
    monkeypatch.setattr(
        db_path_utils.DatabasePaths,
        "get_user_base_directory",
        lambda user_id: tmp_path / f"user-{user_id}",
    )

    messages: list[str] = []
    sink_id = workflow_common.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        assert workflow_common.workflow_file_base_dir({"user_id": "secret-user-token"}, None) == tmp_path / "user-7"
    finally:
        workflow_common.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Workflow file base dir: invalid user id; using single-user fallback" in joined
    assert "secret-user-token" not in joined


def test_workflow_file_base_dir_sanitizes_per_user_failure_logs(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.DB_Management import db_path_utils

    def broken_user_base(user_id):  # noqa: ANN001, ANN202
        raise RuntimeError("secret per-user base at /private/user-base.db")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WORKFLOWS_FILE_BASE_DIR", raising=False)
    monkeypatch.setattr(db_path_utils.DatabasePaths, "get_single_user_id", lambda: 7)
    monkeypatch.setattr(db_path_utils.DatabasePaths, "get_user_base_directory", broken_user_base)

    messages: list[str] = []
    sink_id = workflow_common.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        assert workflow_common.workflow_file_base_dir({}, None) == (tmp_path / "Databases").resolve()
    finally:
        workflow_common.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Workflow file base dir: failed to resolve per-user base dir; using Databases fallback" in joined
    assert "private/user-base.db" not in joined


def test_resolve_artifacts_dir_sanitizes_base_resolve_logs(monkeypatch, tmp_path):
    base_dir = tmp_path / "artifacts-base-token"
    base_dir.mkdir()
    original_resolve = workflow_common.Path.resolve

    def broken_resolve(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if self == base_dir:
            raise OSError("secret artifacts backend at /private/artifacts.db")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(workflow_common, "artifacts_base_dir", lambda: base_dir)
    monkeypatch.setattr(workflow_common.Path, "resolve", broken_resolve)

    messages: list[str] = []
    sink_id = workflow_common.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        assert workflow_common.resolve_artifacts_dir("run-1") == (base_dir / "run-1").resolve(strict=False)
    finally:
        workflow_common.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Artifacts base dir resolve failed. Using unresolved base dir." in joined
    assert "artifacts-base-token" not in joined
    assert "private/artifacts.db" not in joined


def test_resolve_workflow_file_path_sanitizes_base_resolve_logs(monkeypatch, tmp_path):
    base_dir = tmp_path / "base-dir-token"
    base_dir.mkdir()
    original_resolve = workflow_common.Path.resolve

    def broken_resolve(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if self == base_dir:
            raise OSError("secret base resolve at /private/base-dir.db")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(workflow_common, "workflow_file_base_dir", lambda context, config: base_dir)
    monkeypatch.setattr(workflow_common.Path, "resolve", broken_resolve)

    messages: list[str] = []
    sink_id = workflow_common.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        assert workflow_common.resolve_workflow_file_path("child.txt", {}, None) == (
            base_dir / "child.txt"
        ).resolve(strict=False)
    finally:
        workflow_common.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Failed to resolve workflow file base directory" in joined
    assert "base-dir-token" not in joined
    assert "private/base-dir.db" not in joined


def test_resolve_workflow_file_path_sanitizes_allowlist_failure_logs(monkeypatch, tmp_path):
    base_dir = tmp_path / "base"
    target = base_dir / "allowed.txt"
    base_dir.mkdir()

    def broken_allowlist(context):  # noqa: ANN001, ANN202
        raise RuntimeError("secret allowlist policy at /private/allowlist-policy.db")

    monkeypatch.setenv("WORKFLOWS_ALLOW_UNSAFE_FILE_ACCESS", "true")
    monkeypatch.setattr(workflow_common, "workflow_file_base_dir", lambda context, config: base_dir)
    monkeypatch.setattr(workflow_common, "workflow_file_allowlist", broken_allowlist)

    messages: list[str] = []
    sink_id = workflow_common.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        assert workflow_common.resolve_workflow_file_path(str(target), {}, None) == target.resolve(strict=False)
    finally:
        workflow_common.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Workflow file allowlist: failed to resolve allowlist" in joined
    assert "private/allowlist-policy.db" not in joined
