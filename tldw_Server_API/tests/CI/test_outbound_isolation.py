"""Test the Docker runner's boundary and report contract without a daemon."""

from __future__ import annotations

import io
import json
import subprocess  # nosec B404
import tarfile
from pathlib import Path

import pytest

IMAGE_ID = "sha256:" + "a" * 64
CONTAINER_ID = "b" * 64


def test_project_uses_current_tracked_sources_and_explicit_harness_files(tmp_path: Path) -> None:
    from Helper_Scripts.ci.outbound_isolation import prepare_project

    source = tmp_path / "source"
    source.mkdir()
    tracked = {
        "pyproject.toml": "[project]",
        "tldw_Server_API/app/main.py": "old",
        "apps/mcp-unified/src/mcp_unified/__init__.py": "",
        "apps/extension/wxt.config.ts": "",
        "apps/extension/entrypoints/background.ts": "",
        "apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx": "",
        "packages/tldw_profile_core/src/tldw_profile_core/__init__.py": "",
        "tldw_Server_API/Config_Files/.env": "synthetic-secret",
        "tldw_Server_API/Config_Files/session_encryption.key": "synthetic-key",
        "tldw_Server_API/Databases/runtime.db": "synthetic-database",
    }
    for relative, contents in tracked.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    (source / "tldw_Server_API/app/writable-link").symlink_to(source / "pyproject.toml")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)  # nosec B603 B607
    subprocess.run(["git", "add", "."], cwd=source, check=True)  # nosec B603 B607
    (source / "tldw_Server_API/app/main.py").write_text("edited", encoding="utf-8")
    harness = source / "Helper_Scripts/ci/outbound_isolation.py"
    harness.parent.mkdir(parents=True)
    harness.write_text("untracked test harness", encoding="utf-8")
    (harness.parent / "scratch.py").write_text("untracked local file", encoding="utf-8")

    project = prepare_project(source, tmp_path / "build")

    assert sorted(path.relative_to(project).as_posix() for path in project.rglob("*") if path.is_file()) == [
        "Helper_Scripts/ci/outbound_isolation.py",
        "apps/extension/entrypoints/background.ts",
        "apps/extension/wxt.config.ts",
        "apps/mcp-unified/src/mcp_unified/__init__.py",
        "apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx",
        "packages/tldw_profile_core/src/tldw_profile_core/__init__.py",
        "pyproject.toml",
        "tldw_Server_API/app/main.py",
    ]
    assert (project / "tldw_Server_API/app/main.py").read_text(encoding="utf-8") == "edited"


def test_prepared_image_rejects_changed_worktree_sources(monkeypatch, tmp_path: Path) -> None:
    from Helper_Scripts.ci import outbound_isolation as isolation

    root = tmp_path / "source"
    root.mkdir()
    source = root / "pyproject.toml"
    source.write_text("current", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # nosec B603 B607
    subprocess.run(["git", "add", "."], cwd=root, check=True)  # nosec B603 B607
    dockerfile = root / "Dockerfiles/Dockerfile.outbound-check"
    dockerfile.parent.mkdir()
    dockerfile.write_text("fixture", encoding="utf-8")
    reports = tmp_path / "reports"

    def docker(args, **kwargs):
        assert args[0] == "build"
        assert args[args.index("--target") + 1] == "backend"
        Path(args[args.index("--iidfile") + 1]).write_text(IMAGE_ID, encoding="ascii")
        return subprocess.CompletedProcess(args, 0, b"")

    monkeypatch.setattr(isolation, "_docker", docker)
    assert isolation.prepare_image(root, "backend", reports) == IMAGE_ID
    assert isolation.load_prepared_image(root, "backend", reports) == IMAGE_ID

    source.write_text("drifted", encoding="utf-8")
    with pytest.raises(isolation.IsolationError, match="stale") as raised:
        isolation.load_prepared_image(root, "backend", reports)
    assert raised.value.code == "prepared_image_stale"


def test_missing_preparation_record_has_a_specific_code(tmp_path: Path) -> None:
    from Helper_Scripts.ci import outbound_isolation as isolation

    with pytest.raises(isolation.IsolationError) as raised:
        isolation.load_prepared_image(tmp_path, "backend", tmp_path)
    assert raised.value.code == "prepared_image_missing"


def _container_inspect() -> dict:
    return {
        "Image": IMAGE_ID,
        "Mounts": [],
        "Config": {"User": "10001:10001", "OpenStdin": False, "Tty": False},
        "HostConfig": {
            "NetworkMode": "none",
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "Privileged": False,
            "AutoRemove": False,
        },
        "NetworkSettings": {"Ports": {}, "Networks": {"none": {}}},
    }


@pytest.mark.parametrize(
    ("section", "change"),
    [
        ("HostConfig", {"NetworkMode": "host"}),
        ("HostConfig", {"Privileged": True}),
        ("HostConfig", {"CapAdd": ["NET_RAW"]}),
        ("HostConfig", {"CapDrop": []}),
        ("HostConfig", {"SecurityOpt": []}),
        ("HostConfig", {"Binds": ["/example:/example"]}),
        ("HostConfig", {"PortBindings": {"80/tcp": [{"HostPort": "8000"}]}}),
        ("HostConfig", {"IpcMode": "host"}),
        ("Config", {"User": "0:0"}),
        ("Config", {"OpenStdin": True}),
        ("NetworkSettings", {"Networks": {"none": {}, "bridge": {}}}),
    ],
)
def test_rejected_inspection_never_starts_container(monkeypatch, tmp_path: Path, section, change) -> None:
    from Helper_Scripts.ci import outbound_isolation as isolation

    report = tmp_path / "result.json"
    report.write_text('{"status":"pass"}', encoding="utf-8")
    inspected = _container_inspect()
    inspected[section].update(change)
    calls = []

    def docker(args, **kwargs):
        calls.append(args[0])
        output = {
            "create": CONTAINER_ID.encode(),
            "inspect": json.dumps([inspected]).encode(),
            "rm": b"",
        }[args[0]]
        return subprocess.CompletedProcess(args, 0, output)

    monkeypatch.setattr(isolation, "_docker", docker)
    with pytest.raises(isolation.IsolationError, match="container isolation"):
        isolation.run_isolated(IMAGE_ID, ["-m", "example"], tmp_path, timeout=2)

    assert calls == ["create", "inspect", "rm"]
    assert not report.exists()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_run_extracts_report_after_exit_and_cleans_up(monkeypatch, tmp_path: Path, exit_code: int) -> None:
    from Helper_Scripts.ci import outbound_isolation as isolation

    calls = []

    def docker(args, **kwargs):
        calls.append(args[0])
        if args[0] == "create":
            assert args[args.index("--network") + 1] == "none"
            assert args[args.index("--user") + 1] == "10001:10001"
            assert all(option not in args for option in ("-v", "--mount", "-p", "--publish"))
            socket_index = args.index("_inside") + 1
            assert Path(args[socket_index]).is_socket()
            output = CONTAINER_ID.encode()
        elif args[0] == "inspect":
            output = json.dumps([_container_inspect()]).encode()
        elif args[0] == "wait":
            output = str(exit_code).encode()
        else:
            output = b""
        return subprocess.CompletedProcess(args, 0, output)

    def copy_report(container, target, **kwargs):
        assert container == CONTAINER_ID
        calls.append("cp")
        target.write_text('{"sample":"result"}', encoding="utf-8")

    monkeypatch.setattr(isolation, "_docker", docker)
    monkeypatch.setattr(isolation, "_copy_report", copy_report)
    assert isolation.run_isolated(IMAGE_ID, ["-m", "example"], tmp_path, timeout=2) == exit_code
    assert calls == ["create", "inspect", "start", "wait", "cp", "rm"]
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {"sample": "result"}


def test_timeout_removes_container_and_clears_stale_result(monkeypatch, tmp_path: Path) -> None:
    from Helper_Scripts.ci import outbound_isolation as isolation

    stale = tmp_path / "result.json"
    stale.write_text('{"old":"report"}', encoding="utf-8")
    calls = []

    def docker(args, **kwargs):
        calls.append(args[0])
        if args[0] == "wait":
            raise subprocess.TimeoutExpired(args, 2)
        output = {
            "create": CONTAINER_ID.encode(),
            "inspect": json.dumps([_container_inspect()]).encode(),
            "start": b"",
            "rm": b"",
        }[args[0]]
        return subprocess.CompletedProcess(args, 0, output)

    monkeypatch.setattr(isolation, "_docker", docker)
    with pytest.raises(isolation.IsolationError, match="timed out") as raised:
        isolation.run_isolated(IMAGE_ID, ["-m", "example"], tmp_path, timeout=2)

    assert raised.value.code == "container_timeout"
    assert not stale.exists()
    assert calls == ["create", "inspect", "start", "wait", "rm"]


def _archive(name: str, body: bytes, *, link: bool = False) -> bytes:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(body)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "/somewhere/else"
        archive.addfile(member, io.BytesIO(body))
    return data.getvalue()


def test_report_accepts_one_regular_json_object() -> None:
    from Helper_Scripts.ci import outbound_isolation as isolation

    assert isolation._report_bytes(_archive("result.json", b'{"ok": true}')) == b'{"ok": true}'


@pytest.mark.parametrize(
    "archive",
    [
        _archive("unexpected.json", b"{}"),
        _archive("result.json", b"{}", link=True),
        _archive("result.json", b"[]"),
        _archive("result.json", b"invalid JSON"),
        _archive("result.json", b" " * (256 * 1024 + 1)),
    ],
)
def test_report_rejects_unsafe_or_oversized_entries(archive: bytes) -> None:
    from Helper_Scripts.ci import outbound_isolation as isolation

    with pytest.raises(isolation.IsolationError, match="report"):
        isolation._report_bytes(archive)


def test_preflight_rejects_reachable_host_socket(monkeypatch, tmp_path: Path) -> None:
    from Helper_Scripts.ci import outbound_isolation as isolation

    monkeypatch.setattr(isolation.sys, "platform", "linux")
    monkeypatch.setattr(isolation.os, "getuid", lambda: 10001)
    monkeypatch.setattr(isolation.os, "getgid", lambda: 10001)
    monkeypatch.setattr(isolation, "_active_interfaces", lambda: {"lo"})
    # Keep the socket name short enough on hosts with long pytest tmp paths.
    with isolation._host_socket() as (path, _):
        with pytest.raises(isolation.IsolationError, match="host socket"):
            isolation._preflight(path)
    isolation._preflight(path)


def test_only_operational_interfaces_count_toward_preflight(monkeypatch, tmp_path: Path) -> None:
    from Helper_Scripts.ci import outbound_isolation as isolation

    monkeypatch.setattr(isolation.socket, "if_nameindex", lambda: [(1, "lo"), (2, "tunl0")])
    for name, flags in (("lo", "0x9\n"), ("tunl0", "0x80\n")):
        interface = tmp_path / name
        interface.mkdir()
        (interface / "flags").write_text(flags, encoding="ascii")

    assert isolation._active_interfaces(tmp_path) == {"lo"}
    (tmp_path / "tunl0/flags").write_text("0x81\n", encoding="ascii")
    assert isolation._active_interfaces(tmp_path) == {"lo", "tunl0"}


def test_local_config_edit_never_enters_build_context(tmp_path: Path) -> None:
    from Helper_Scripts.ci.outbound_isolation import IsolationError, prepare_project

    root = tmp_path / "source"
    root.mkdir()
    config_path = Path("tldw_Server_API/Config_Files/config.txt")
    config = root / config_path
    config.parent.mkdir(parents=True)
    committed = "[API]\nopenai_api_key =\n"
    config.write_text(committed, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # nosec B603 B607
    subprocess.run(["git", "add", "."], cwd=root, check=True)  # nosec B603 B607
    subprocess.run(  # nosec B603 B607
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--no-gpg-sign",
            "-qm",
            "fixture defaults",
        ],
        cwd=root,
        check=True,
    )

    project = prepare_project(root, tmp_path / "clean-build")
    assert (project / config_path).read_text(encoding="utf-8") == committed

    local_key = "local-synthetic-credential"
    config.write_text(committed.replace(" =\n", " = " + local_key + "\n"), encoding="utf-8")
    with pytest.raises(IsolationError, match="committed defaults") as failed:
        prepare_project(root, tmp_path / "dirty-build")
    assert local_key not in str(failed.value)
    assert not (tmp_path / "dirty-build").exists()
