"""Failure controls for the exact-artifact frontend experiment, not native proof."""

import importlib.util
import json
import re
import subprocess  # nosec B404
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/frontend/qualify.py"
WORKFLOW = ROOT / ".github/workflows/frontend-runtime-candidate.yml"
BASELINE = "sha256:" + "a" * 64
CANDIDATE = "sha256:" + "b" * 64
HEALTH = "fetch('http://localhost:3000').then((r)=>process.exit(r.ok ? 0 : 1)).catch(()=>process.exit(1))"
FORBIDDEN_WORKFLOW_COMMAND = re.compile(r"(?:^|[\n;]|&&?|\|\|?)\s*(?:rm(?:\s|$)|docker\s+push(?:\s|$))")


def _has_forbidden_workflow_command(script):
    """Detect only executable deletion/publication commands at shell boundaries."""
    return FORBIDDEN_WORKFLOW_COMMAND.search(script) is not None


def qualifier():
    assert SCRIPT.is_file(), "native frontend qualifier is not implemented"
    spec = importlib.util.spec_from_file_location("frontend_qualify", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_config(application):
    admin = application == "admin-ui"
    port = 3001 if admin else 3000
    health = (
        "fetch('http://localhost:3001/api/health/ready').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
        if admin
        else HEALTH
    )
    env = ["NODE_ENV=production", "NEXT_TELEMETRY_DISABLED=1", "HOSTNAME=0.0.0.0", f"PORT={port}"]
    env += (
        ["APP_VERSION=0.0.0"]
        if admin
        else ["NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart", "TLDW_INTERNAL_API_ORIGIN=http://app:8000"]
    )
    return {
        "Cmd": ["node", "server.js"],
        "Entrypoint": ["docker-entrypoint.sh"],
        "User": "adminui" if admin else "webui",
        "WorkingDir": "/app/admin-ui" if admin else "/app/apps/tldw-frontend",
        "Healthcheck": {
            "Test": ["CMD-SHELL", f'node -e "{health}"'],
            "Interval": 30000000000,
            "Timeout": 5000000000,
            "Retries": 5,
        },
        "ExposedPorts": {f"{port}/tcp": {}},
        "Env": env,
    }


class DockerBoundary:
    """Emulate only external command results; validation/reporting are real code."""

    def __init__(self, application="webui", fault=None):
        self.application = application
        self.fault = fault
        self.calls = []
        self.containers = {}
        self.backend_running = True

    def __call__(self, argv, *, capture_output, text, timeout, check):
        assert isinstance(argv, list) and capture_output and text and timeout <= 60 and check is False
        self.calls.append(argv)
        out, code = "", 0
        if argv[0] == "git" and argv[-2:] == ["rev-parse", "HEAD"]:
            out = "c" * 40
        elif argv[:2] == ["docker", "info"]:
            out = json.dumps({"OSType": "linux", "Architecture": "arm64" if self.fault == "daemon" else "x86_64"})
        elif argv[:3] == ["docker", "image", "inspect"]:
            subject = argv[3]
            config = image_config(self.application)
            if subject == BASELINE:
                config["Env"] += ["YARN_VERSION=1.22.22"]
            if subject == CANDIDATE and self.fault == "config":
                config["Env"][0] = "NODE_ENV=development"
            if subject == CANDIDATE and self.fault == "health_config":
                config["Healthcheck"]["Test"] = ["NONE"]
            out = json.dumps(
                [
                    {
                        "Id": "sha256:" + "d" * 64 if self.fault == "identity" else subject,
                        "Os": "linux",
                        "Architecture": "arm64" if self.fault == "image_platform" else "amd64",
                        "Config": config,
                    }
                ]
            )
        elif argv[:3] == ["docker", "network", "create"]:
            out = "e" * 64
        elif argv[:3] == ["docker", "network", "inspect"]:
            out = json.dumps([{"Id": "e" * 64, "Internal": True}])
        elif argv[:2] == ["docker", "create"]:
            cid = f"{len(self.containers) + 1:064x}"
            subject_index = next(i for i, item in enumerate(argv) if item in (BASELINE, CANDIDATE))
            self.containers[cid] = {"argv": argv, "subject": argv[subject_index], "command": argv[subject_index + 1 :]}
            out = cid
        elif argv[:3] == ["docker", "container", "inspect"]:
            container = self.containers[argv[3]]
            args = container["argv"]
            uid = 10003 if self.application == "admin-ui" else 10002
            workdir = image_config(self.application)["WorkingDir"]
            out = json.dumps(
                [
                    {
                        "Id": argv[3],
                        "Image": container["subject"],
                        "Config": image_config(self.application),
                        "State": {
                            "Running": False,
                            "ExitCode": (
                                1
                                if self.fault == "sharp" and container["command"] == ["-e", qualifier().SHARP_JS]
                                else 0
                            ),
                            "OOMKilled": False,
                        },
                        "HostConfig": {
                            "ReadonlyRootfs": self.fault != "writable",
                            "CapDrop": ["ALL"],
                            "SecurityOpt": ["no-new-privileges:true"],
                            "PidsLimit": 128,
                            "Memory": 1073741824,
                            "NanoCpus": 2000000000,
                            "PortBindings": {},
                            "NetworkMode": args[args.index("--network") + 1],
                            "Tmpfs": {
                                "/tmp": f"rw,nosuid,nodev,noexec,size=256m,uid={uid},gid={uid}",
                                f"{workdir}/.next/cache": f"rw,nosuid,nodev,noexec,size=256m,uid={uid},gid={uid}",
                            },
                        },
                        "Mounts": [],
                    }
                ]
            )
        elif argv[:3] == ["docker", "start", "--attach"]:
            container = self.containers[argv[3]]
            if "dpkg-query" in container["argv"]:
                out = "libc6:amd64\t2.39-0ubuntu8.8\nzlib1g:amd64\t1:1.3.dfsg-3.1ubuntu2.2\nlibstdc++6:amd64\t14.2.0-4ubuntu2~24.04\n"
                if self.fault == "vendor":
                    out = out.replace("2.39-0ubuntu8.8", "2.39-0ubuntu8.7")
            elif container["command"] == ["-e", qualifier().SHARP_JS]:
                out, code = ("", 1) if self.fault == "sharp" else ('{"width":3,"height":3}', 0)
            else:
                uid = 10003 if self.application == "admin-ui" else 10002
                workdir = image_config(self.application)["WorkingDir"]
                observation = {
                    "node": {"version": "v24.20.0", "modules": "137", "napi": "10", "sha256": "f" * 64},
                    "uid": uid,
                    "gid": uid,
                    "home": "/home/adminui" if self.application == "admin-ui" else "/home/webui",
                    "cwd": workdir,
                    "ownership": {"cwd": [uid, uid], "server": [uid, uid], "cache": [uid, uid]},
                    "roots": {"count": 118, "sha256": "1" * 64},
                    "systemCA": None,
                }
                if self.fault in ("uid", "gid"):
                    observation[self.fault] = 0
                if self.fault == "node":
                    observation["node"]["version"] = "v24.20.1"
                if self.fault == "workdir":
                    observation["cwd"] = "/tmp"
                if self.fault == "trust" and container["subject"] == CANDIDATE:
                    observation["roots"]["sha256"] = "2" * 64
                out = json.dumps(observation)
        elif argv[:2] == ["docker", "exec"]:
            if argv[-1].startswith("http://"):
                url = argv[-1]
                status = 503 if not self.backend_running and url.endswith("/ready") else 200
                if self.fault == "health":
                    status = 500
                out = json.dumps({"status": status})
            else:
                code = 1 if not self.backend_running and self.application == "admin-ui" else 0
        elif argv[:2] == ["docker", "kill"]:
            if "--signal=TERM" not in argv:
                pytest.fail("qualifier may not send SIGKILL")
            container = self.containers[argv[-1]]
            if "--network-alias" in container["argv"]:
                self.backend_running = False
        elif argv[:2] == ["docker", "wait"]:
            if self.fault == "sigterm":
                raise subprocess.TimeoutExpired(argv, timeout, output="still running")
            out = "0"
        elif argv[:2] == ["docker", "start"]:
            if "--network-alias" in self.containers[argv[-1]]["argv"]:
                self.backend_running = True
        elif argv[:2] == ["docker", "logs"] or argv[:2] == ["docker", "rm"] or argv[:3] == ["docker", "network", "rm"]:
            pass
        else:
            pytest.fail(f"unexpected command boundary: {argv}")
        return subprocess.CompletedProcess(argv, code, out, "fixture failure" if code else "")


def run_probe(tmp_path, application="webui", fault=None, host=None, baseline=BASELINE):
    boundary = DockerBoundary(application, fault)
    result = qualifier().qualify(
        application,
        baseline,
        CANDIDATE,
        tmp_path,
        execute=boundary,
        host=host or {"system": "Linux", "machine": "x86_64"},
        sleep=lambda _: None,
    )
    assert json.loads((tmp_path / "qualification.json").read_text()) == result
    return result, boundary


@pytest.mark.parametrize("application", ["webui", "admin-ui"])
def test_success_retains_exact_subjects_controls_and_scoped_report(tmp_path, application):
    report, boundary = run_probe(tmp_path, application)
    assert report["passed"] is True
    assert report["scope"] == "native-frontend-compatibility-not-release-admission"
    assert report["baseline"]["subject"] == BASELINE
    assert report["candidate"]["subject"] == CANDIDATE
    assert report["candidate"]["sharp"] == {"width": 3, "height": 3}
    assert report["source"]["commit"] == "c" * 40
    assert report["candidate"]["termination"]["exitCode"] in (0, 143)
    assert any("--internal" in call for call in boundary.calls)
    assert all("--publish" not in call and "-p" not in call for call in boundary.calls)
    assert report["baseline"]["backendTermination"]["exitCode"] in (0, 143)
    assert report["candidate"]["backendTermination"]["exitCode"] in (0, 143)
    if application == "admin-ui":
        assert report["candidate"]["backendDown"] == {"ready": 503, "live": 200, "healthcheckExit": 1}


@pytest.mark.parametrize("host", [{"system": "Darwin", "machine": "arm64"}, {"system": "Linux", "machine": "aarch64"}])
def test_non_native_host_fails_before_any_container_creation(tmp_path, host):
    report, boundary = run_probe(tmp_path, host=host)
    assert report["passed"] is False
    assert "native" in report["error"]
    assert not any(call[:2] == ["docker", "create"] for call in boundary.calls)


@pytest.mark.parametrize("subject", ["node:24", "sha256:ABC", "repo@" + BASELINE, BASELINE + ";echo wrong"])
def test_malformed_subject_is_rejected_without_docker(tmp_path, subject):
    report, boundary = run_probe(tmp_path, baseline=subject)
    assert report["passed"] is False
    assert "subject" in report["error"]
    assert not any(call[0] == "docker" for call in boundary.calls)


@pytest.mark.parametrize(
    "fault",
    [
        "daemon",
        "identity",
        "image_platform",
        "config",
        "health_config",
        "node",
        "uid",
        "gid",
        "workdir",
        "sharp",
        "health",
        "sigterm",
        "vendor",
        "trust",
        "writable",
    ],
)
def test_failed_control_cannot_be_admitted_and_retains_raw_commands(tmp_path, fault):
    report, boundary = run_probe(tmp_path, fault=fault)
    assert report["passed"] is False
    assert report["error"]
    records = [json.loads(path.read_text()) for path in sorted((tmp_path / "commands").glob("*.json"))]
    assert records and all("returncode" in item and "stdout" in item and "stderr" in item for item in records)
    created = {call[-1] for call in boundary.calls if call[:2] == ["docker", "rm"]}
    assert created == set(boundary.containers)
    if fault == "sigterm":
        assert any(item["timedOut"] for item in records)


def test_workflow_uses_native_local_exact_oci_artifacts_and_always_retains_evidence():
    assert WORKFLOW.is_file(), "candidate workflow is not implemented"
    workflow = yaml.safe_load(WORKFLOW.read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_dispatch", "push"}
    assert triggers["push"]["branches"] == ["codex/task-13013-7-supply-chain-design"]
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["qualify"]
    assert job["runs-on"] == "ubuntu-24.04" and job["timeout-minutes"] == 90
    assert {row["application"] for row in job["strategy"]["matrix"]["include"]} == {"webui", "admin-ui"}
    steps = job["steps"]
    actions = {step.get("uses", "").split("@")[0]: step for step in steps if "uses" in step}
    assert not any("qemu" in name or "login" in name for name in actions)
    assert json.loads(actions["docker/setup-docker-action"]["with"]["daemon-config"])["features"][
        "containerd-snapshotter"
    ]
    builds = [step for step in steps if step.get("uses", "").startswith("docker/build-push-action@")]
    assert len(builds) == 2
    for build in builds:
        settings = build["with"]
        assert settings["platforms"] == "linux/amd64" and settings["push"] is False
        assert settings["provenance"] == "mode=max" and settings["sbom"] is True
        assert settings["outputs"].startswith("type=oci,dest=")
    assert builds[0]["with"]["build-args"] == builds[1]["with"]["build-args"]
    upload = actions["actions/upload-artifact"]
    assert "always()" in upload["if"] and upload["with"]["retention-days"] == 14
    assert "*.oci.tar" in upload["with"]["path"]
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "runtime_probe.py config" in runs and "docker load" in runs and "qualify.py" in runs
    assert "docker run --rm" in runs and "--platform linux/amd64" in runs
    assert not _has_forbidden_workflow_command(runs)


def test_diagnostic_failure_still_retains_bound_container_exit_state(tmp_path):
    report, boundary = run_probe(tmp_path, fault="sharp")
    assert not report["passed"]
    assert "diagnostics" in report["baseline"]
    assert report["baseline"]["diagnostics"]["sharp"]["exitCode"] == 1


def test_command_timeout_preserves_partial_bytes_and_bounded_call(tmp_path):
    def timeout(argv, **kwargs):
        assert kwargs["timeout"] == 5
        raise subprocess.TimeoutExpired(argv, 5, output=b"partial", stderr=b"stalled")

    commands = qualifier().Commands(tmp_path, timeout)
    result = commands.run(["docker", "info"], timeout=5, checked=False)
    assert result["stdout"] == "partial" and result["stderr"] == "stalled"
    assert result["timedOut"] and result["returncode"] is None
    assert json.loads((tmp_path / "commands/0001.json").read_text()) == result


@pytest.mark.parametrize(
    "script",
    [
        "docker run --rm --network none scanner:fixed",
        "scanner image --platform linux/amd64 --output report.json",
        "docker run --rm scanner:fixed image --platform linux/amd64",
    ],
)
def test_workflow_command_guard_allows_rm_and_platform_options(script):
    assert not _has_forbidden_workflow_command(script)


@pytest.mark.parametrize(
    "script",
    [
        "rm evidence/baseline.oci.tar",
        "docker push ghcr.io/example/frontend:latest",
        'echo "built" && rm -f evidence/candidate.oci.tar',
        "docker inspect frontend; docker push ghcr.io/example/frontend:latest",
        "test -s report.json || rm report.json",
    ],
)
def test_workflow_command_guard_rejects_deletion_and_publication_commands(script):
    assert _has_forbidden_workflow_command(script)


def test_source_identity_is_bound_to_script_repository_from_another_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report, boundary = run_probe(tmp_path)
    assert report["passed"]
    assert ["git", "-C", str(ROOT), "rev-parse", "HEAD"] in boundary.calls


def test_workflow_actions_match_existing_trusted_pins_and_health_build_args():
    assert WORKFLOW.is_file(), "candidate workflow is not implemented"
    workflow = yaml.safe_load(WORKFLOW.read_text())
    existing = yaml.safe_load((ROOT / ".github/workflows/container-build-check.yml").read_text())
    expected = {step["uses"] for step in existing["jobs"]["build-and-scan"]["steps"] if "uses" in step}
    for step in workflow["jobs"]["qualify"]["steps"]:
        if "uses" in step:
            assert step["uses"] in expected
    rows = {row["application"]: row for row in workflow["jobs"]["qualify"]["strategy"]["matrix"]["include"]}
    assert "NEXT_PUBLIC_API_URL=http://backend:8000" in rows["admin-ui"]["build_args"].splitlines()
    assert "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart" in rows["webui"]["build_args"].splitlines()
    assert "TLDW_INTERNAL_API_ORIGIN=http://backend:8000" in rows["webui"]["build_args"].splitlines()
