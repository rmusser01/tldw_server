#!/usr/bin/env python3
"""Compare locally loaded exact frontend artifacts on native Linux amd64 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import secrets

# Fixed diagnostic argv, no shell execution.
import subprocess  # nosec B404
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Container-private memory mount, never a host temporary file/directory.
TMP_MOUNT = "/tmp"  # nosec B108
APPS = {
    "webui": (10002, "webui", "/app/apps/tldw-frontend", 3000),
    "admin-ui": (10003, "adminui", "/app/admin-ui", 3001),
}
HEALTH_JS = {
    "webui": "fetch('http://localhost:3000').then((r)=>process.exit(r.ok ? 0 : 1)).catch(()=>process.exit(1))",
    "admin-ui": "fetch('http://localhost:3001/api/health/ready').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))",
}
OBSERVE_JS = r"""
const fs = require('node:fs'), crypto = require('node:crypto');
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const owner = path => { const s = fs.statSync(path); return [s.uid, s.gid]; };
const bundle = '/etc/ssl/certs/ca-certificates.crt';
const roots = require('node:tls').rootCertificates;
console.log(JSON.stringify({
  node: {version: process.version, modules: process.versions.modules, napi: process.versions.napi,
         sha256: hash(fs.readFileSync(process.execPath))},
  uid: process.getuid(), gid: process.getgid(), home: require('node:os').homedir(), cwd: process.cwd(),
  ownership: {cwd: owner('.'), server: owner('server.js'), cache: owner('.next/cache')},
  roots: {count: roots.length, sha256: hash(roots.join('\n'))},
  systemCA: fs.existsSync(bundle) ? {path: bundle, sha256: hash(fs.readFileSync(bundle))} : null
}));
"""
SHARP_JS = r"""
const assert = require('node:assert/strict');
const { createRequire } = require('node:module');
const nextRequire = createRequire(require.resolve('next/dist/server/image-optimizer'));
const sharp = nextRequire('sharp');
(async () => {
const input = await sharp({create:{width:2,height:2,channels:3,background:'#123456'}}).png().toBuffer();
const output = await sharp(input).resize(3,3).png().toBuffer();
const metadata = await sharp(output).metadata();
assert.equal(metadata.width,3);
assert.equal(metadata.height,3);
console.log(JSON.stringify({width: metadata.width, height: metadata.height}));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
BACKEND_JS = r"""
require('node:http').createServer((req, res) => {
  res.writeHead(req.url === '/api/v1/health' ? 200 : 404, {'content-type': 'application/json'});
  res.end(JSON.stringify({status: req.url === '/api/v1/health' ? 'ok' : 'not_found'}));
}).listen(8000, '0.0.0.0');
"""
HTTP_JS = r"""
fetch(process.argv[1], {signal: AbortSignal.timeout(3000)}).then(async response => {
  console.log(JSON.stringify({status: response.status, body: (await response.text()).slice(0,4096)}));
}).catch(error => { console.error(error); process.exitCode = 1; });
"""


def require(condition: bool, message: str) -> None:
    """Fail the experiment when an observed contract does not hold."""
    if not condition:
        raise ValueError(message)


class Commands:
    """Bound every subprocess and retain its result before validating it."""

    def __init__(self, evidence: Path, execute: Callable, signing_key: str = ""):
        self.directory = evidence / "commands"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.execute = execute
        self.count = 0
        self.signing_key = signing_key

    def redact(self, text: str) -> str:
        """Remove only this invocation's ephemeral fixture from persisted text."""
        return text.replace(self.signing_key, "[REDACTED]") if self.signing_key else text

    def run(self, argv: list[str], *, timeout: int = 30, checked: bool = True) -> dict:
        """Execute fixed argv with no shell; persist failures and partial output."""
        started = time.monotonic()
        result = {"argv": argv, "timeoutSeconds": timeout, "timedOut": False}
        try:
            completed = self.execute(argv, capture_output=True, text=True, timeout=timeout, check=False)
            result.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        except subprocess.TimeoutExpired as exc:
            result.update(returncode=None, stdout=exc.stdout or "", stderr=exc.stderr or "", timedOut=True)
        except OSError as exc:
            result.update(returncode=None, stdout="", stderr=str(exc))
        for stream in ("stdout", "stderr"):
            if isinstance(result[stream], bytes):
                result[stream] = result[stream].decode("utf-8", errors="replace")
        result["elapsedSeconds"] = time.monotonic() - started
        self.count += 1
        (self.directory / f"{self.count:04d}.json").write_text(self.redact(json.dumps(result, indent=2)) + "\n")
        if checked:
            require(result["returncode"] == 0, f"command failed: {argv[:3]} (see commands/{self.count:04d}.json)")
        return result

    def data(self, argv: list[str]) -> object:
        """Decode a successful command after its raw output has been retained."""
        return json.loads(self.run(argv)["stdout"])


def application_config(image: dict, application: str) -> dict:
    """Validate canonical startup and select only application-owned ENV keys."""
    _, user, workdir, port = APPS[application]
    config = image["Config"]
    expected = {
        "Cmd": ["node", "server.js"],
        "Entrypoint": ["docker-entrypoint.sh"],
        "User": user,
        "WorkingDir": workdir,
        "ExposedPorts": {f"{port}/tcp": {}},
        "Healthcheck": {
            "Test": ["CMD-SHELL", f'node -e "{HEALTH_JS[application]}"'],
            "Interval": 30_000_000_000,
            "Timeout": 5_000_000_000,
            "Retries": 5,
        },
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"wrong application {key}")
    keys = ["NODE_ENV", "NEXT_TELEMETRY_DISABLED", "HOSTNAME", "PORT"]
    keys += (
        ["APP_VERSION"]
        if application == "admin-ui"
        else ["NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "TLDW_INTERNAL_API_ORIGIN"]
    )
    env = {}
    for entry in config.get("Env", []):
        key, _, value = entry.partition("=")
        if key in keys:
            require(key not in env, f"ambiguous ENV key {key}")
            env[key] = value
    require(set(env) == set(keys), "missing application ENV keys")
    return {**expected, "Env": env}


def validate_observation(observation: dict, application: str) -> None:
    """Require the unchanged Node executable and non-root application identity."""
    uid, user, workdir, _ = APPS[application]
    require(observation["node"]["version"] == "v24.20.0", "wrong Node version")
    require(observation["uid"] == uid and observation["gid"] == uid, "wrong UID/GID")
    require(observation["home"] == f"/home/{user}" and observation["cwd"] == workdir, "wrong home/workdir")
    require(all(value == [uid, uid] for value in observation["ownership"].values()), "wrong file ownership")
    require(set(observation["ownership"]) == {"cwd", "server", "cache"}, "missing ownership observation")
    require(observation["roots"]["count"] > 0, "empty Node root store")
    for value in (observation["node"]["sha256"], observation["roots"]["sha256"]):
        require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid observation hash")
    require(bool(observation["node"]["modules"]) and bool(observation["node"]["napi"]), "missing Node ABI")


class Containers:
    """Own a private network and exact container IDs for this invocation only."""

    def __init__(self, commands: Commands, application: str, token: str):
        self.commands, self.application, self.token = commands, application, token
        self.ids: list[str] = []
        self.network: str | None = None

    def create(
        self,
        subject: str,
        role: str,
        command: list[str] | None = None,
        *,
        backend: bool = False,
        entrypoint: str = "node",
    ) -> str:
        """Create an isolated non-root container, then verify its effective settings."""
        uid, _, workdir, _ = APPS[self.application]
        tmpfs = f"rw,nosuid,nodev,noexec,size=256m,uid={uid},gid={uid}"
        argv = [
            "docker",
            "create",
            "--name",
            f"{self.token}-{role}",
            "--label",
            f"tldw.frontend-candidate={self.token}",
            "--pull",
            "never",
            "--platform",
            "linux/amd64",
            "--network",
            self.network,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "128",
            "--memory",
            "1g",
            "--cpus",
            "2",
            "--tmpfs",
            f"{TMP_MOUNT}:{tmpfs}",
            "--tmpfs",
            f"{workdir}/.next/cache:{tmpfs}",
        ]
        if self.application == "webui":
            argv += ["--env", "TLDW_INTERNAL_API_ORIGIN=http://backend:8000"]
        admin_app = self.application == "admin-ui" and command is None and not backend
        if admin_app:
            require(bool(self.commands.signing_key), "missing Admin signing fixture")
            argv += ["--env", f"JWT_SECRET_KEY={self.commands.signing_key}"]
        if backend:
            argv += ["--network-alias", "backend"]
        if command is not None:
            argv += ["--entrypoint", entrypoint]
        argv += [subject] + (command or [])
        cid = self.commands.run(argv)["stdout"].strip()
        require(re.fullmatch(r"[0-9a-f]{64}", cid) is not None, "invalid created container identity")
        self.ids.append(cid)
        observed = self.commands.data(["docker", "container", "inspect", cid])[0]
        require(observed["Id"] == cid and observed["Image"] == subject, "created image identity mismatch")
        settings = observed["HostConfig"]
        require(settings["ReadonlyRootfs"] and settings["CapDrop"] == ["ALL"], "container isolation mismatch")
        require("no-new-privileges:true" in settings["SecurityOpt"], "missing no-new-privileges")
        require(
            settings["PidsLimit"] == 128 and settings["Memory"] == 1073741824 and settings["NanoCpus"] == 2000000000,
            "container resource limits mismatch",
        )
        require(settings["NetworkMode"] == self.network and not settings["PortBindings"], "container network mismatch")
        expected_tmpfs = {TMP_MOUNT: tmpfs, f"{workdir}/.next/cache": tmpfs}
        require(settings["Tmpfs"] == expected_tmpfs, "container tmpfs mismatch")
        require(
            not any(mount.get("Type") in {"bind", "volume"} for mount in observed["Mounts"]),
            "unexpected writable mount",
        )
        require(observed["Config"]["User"] == APPS[self.application][1], "container must use image non-root USER")
        if admin_app:
            signing_env = [value for value in observed["Config"].get("Env", []) if value.startswith("JWT_SECRET_KEY=")]
            require(signing_env == [f"JWT_SECRET_KEY={self.commands.signing_key}"], "Admin signing fixture mismatch")
        return cid

    def diagnostic(self, subject: str, role: str, command: list[str], record: dict, *, entrypoint: str = "node") -> str:
        """Run a bounded diagnostic at the image's default identity and workdir."""
        cid = self.create(subject, role, command, entrypoint=entrypoint)
        result = self.commands.run(["docker", "start", "--attach", cid], timeout=60, checked=False)
        state = self.commands.data(["docker", "container", "inspect", cid])[0]["State"]
        record.update(container=cid, exitCode=state["ExitCode"], timedOut=result["timedOut"])
        require(
            result["returncode"] == 0 and state["ExitCode"] == 0 and not state["Running"] and not state["OOMKilled"],
            f"diagnostic failed: {role}",
        )
        return result["stdout"]

    def terminate(self, cid: str) -> dict:
        """Send TERM and require exit within ten seconds; never escalate this check."""
        started = time.monotonic()
        self.commands.run(["docker", "kill", "--signal=TERM", cid], timeout=3)
        remaining = max(1, 10 - int(time.monotonic() - started))
        wait = self.commands.run(["docker", "wait", cid], timeout=remaining, checked=False)
        elapsed = time.monotonic() - started
        self.commands.run(["docker", "logs", cid], checked=False)
        state = self.commands.data(["docker", "container", "inspect", cid])[0]["State"]
        require(wait["returncode"] == 0 and elapsed <= 10, "SIGTERM timeout")
        exit_code = int(wait["stdout"].strip())
        require(
            exit_code in (0, 143)
            and state["ExitCode"] == exit_code
            and not state["Running"]
            and not state["OOMKilled"],
            "SIGTERM did not exit cleanly without SIGKILL",
        )
        return {"signal": "SIGTERM", "elapsedSeconds": elapsed, "exitCode": exit_code}

    def cleanup(self) -> list[str]:
        """Retain final logs/inspections, then remove only this invocation's IDs."""
        errors = []
        for cid in reversed(self.ids):
            self.commands.run(["docker", "logs", cid], checked=False)
            self.commands.run(["docker", "container", "inspect", cid], checked=False)
            if self.commands.run(["docker", "rm", "--force", cid], checked=False)["returncode"] != 0:
                errors.append(f"container cleanup failed: {cid}")
        if (
            self.network
            and self.commands.run(["docker", "network", "rm", self.network], checked=False)["returncode"] != 0
        ):
            errors.append(f"network cleanup failed: {self.network}")
        return errors


def http_status(commands: Commands, cid: str, url: str, sleep: Callable, *, expected: int) -> int:
    """Allow bounded startup/recovery retries while preserving every HTTP attempt."""
    for attempt in range(30):
        result = commands.run(["docker", "exec", cid, "node", "-e", HTTP_JS, url], timeout=5, checked=False)
        if result["returncode"] == 0 and json.loads(result["stdout"])["status"] == expected:
            return expected
        if attempt < 29:
            sleep(1)
    raise ValueError(f"health mismatch: expected {expected} for {url}")


def qualify(
    application: str,
    baseline: str,
    candidate: str,
    evidence: Path,
    *,
    execute: Callable = subprocess.run,
    host: dict | None = None,
    sleep: Callable = time.sleep,
) -> dict:
    """Retain a failure-closed compatibility report; never grant release admission."""
    evidence.mkdir(parents=True, exist_ok=True)
    commands = Commands(evidence, execute, secrets.token_hex(32) if application == "admin-ui" else "")
    containers = Containers(commands, application, "frontend-" + uuid.uuid4().hex)
    report = {
        "schemaVersion": 1,
        "application": application,
        "scope": "native-frontend-compatibility-not-release-admission",
        "passed": False,
        "limits": [
            "Controlled health stub only; not real-backend certification.",
            "Compatibility only; CVE-2026-85091 remains unresolved.",
        ],
        "host": host if host is not None else {"system": platform.system(), "machine": platform.machine()},
        "baseline": {"subject": baseline},
        "candidate": {"subject": candidate},
    }
    try:
        require(application in APPS, "unknown application")
        for subject in (baseline, candidate):
            require(re.fullmatch(r"sha256:[0-9a-f]{64}", subject) is not None, "invalid exact OCI subject")
        require(baseline != candidate, "baseline and candidate subjects must differ")
        commit = commands.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])["stdout"].strip()
        require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid source commit")
        paths = [
            f"Dockerfiles/Dockerfile.{application}",
            "Dockerfiles/candidates/frontend/render.py",
            "Dockerfiles/candidates/frontend/qualify.py",
            "admin-ui/bun.lock" if application == "admin-ui" else "apps/bun.lock",
        ]
        report["source"] = {
            "commit": commit,
            "sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
        }
        if (evidence / "inputs.json").exists():
            report["inputs"] = json.loads((evidence / "inputs.json").read_text())
            require(report["inputs"]["sourceCommit"] == commit, "workflow/source commit mismatch")
            for role, subject in (("baseline", baseline), ("candidate", candidate)):
                require(report["inputs"][role]["subject"] == subject, "workflow/loaded subject mismatch")
        require(report["host"] == {"system": "Linux", "machine": "x86_64"}, "native Linux/x86_64 host required")
        report["daemon"] = commands.data(["docker", "info", "--format", "{{json .}}"])
        require(
            report["daemon"]["OSType"] == "linux" and report["daemon"]["Architecture"] in ("amd64", "x86_64"),
            "native linux/amd64 Docker daemon required",
        )
        for role, subject in (("baseline", baseline), ("candidate", candidate)):
            images = commands.data(["docker", "image", "inspect", subject])
            require(len(images) == 1, "ambiguous loaded image")
            image = images[0]
            require(
                image["Id"] == subject and (image["Os"], image["Architecture"]) == ("linux", "amd64"),
                "loaded exact image identity/platform mismatch",
            )
            report[role]["config"] = application_config(image, application)
        require(report["baseline"]["config"] == report["candidate"]["config"], "application configuration mismatch")
        network_id = commands.run(
            [
                "docker",
                "network",
                "create",
                "--internal",
                "--label",
                f"tldw.frontend-candidate={containers.token}",
                containers.token,
            ]
        )["stdout"].strip()
        require(re.fullmatch(r"[0-9a-f]{64}", network_id) is not None, "invalid network identity")
        containers.network = network_id
        network = commands.data(["docker", "network", "inspect", containers.network])[0]
        require(network["Id"] == containers.network and network["Internal"], "private network mismatch")
        backend = containers.create(candidate, "backend", ["-e", BACKEND_JS], backend=True)
        for role, subject in (("baseline", baseline), ("candidate", candidate)):
            diagnostics = report[role]["diagnostics"] = {"facts": {}, "sharp": {}}
            observation = json.loads(
                containers.diagnostic(subject, f"{role}-facts", ["-e", OBSERVE_JS], diagnostics["facts"])
            )
            report[role]["observation"] = observation
            validate_observation(observation, application)
            report[role]["sharp"] = json.loads(
                containers.diagnostic(subject, f"{role}-sharp", ["-e", SHARP_JS], diagnostics["sharp"])
            )
            require(report[role]["sharp"] == {"width": 3, "height": 3}, "sharp operation failed")
            if role == "candidate":
                diagnostics["vendor"] = {}
                vendor = containers.diagnostic(
                    subject,
                    "candidate-vendor",
                    ["-W", "-f=${binary:Package}\t${Version}\n", "libc6", "zlib1g", "libstdc++6"],
                    diagnostics["vendor"],
                    entrypoint="dpkg-query",
                )
                packages = dict(line.split("\t", 1) for line in vendor.splitlines())
                packages = {key.split(":")[0]: value for key, value in packages.items()}
                report[role]["vendorPackages"] = packages
                require(
                    packages["libc6"] == "2.39-0ubuntu8.8" and packages["zlib1g"] == "1:1.3.dfsg-3.1ubuntu2.2",
                    "candidate vendor package mismatch",
                )
                require(bool(packages["libstdc++6"]), "missing vendor C++ runtime")
            commands.run(["docker", "start", backend])
            app = containers.create(subject, f"{role}-app")
            commands.run(["docker", "start", app])
            port = APPS[application][3]
            url = f"http://127.0.0.1:{port}"
            live_path = "/api/health" if application == "admin-ui" else "/"
            report[role]["live"] = http_status(commands, app, url + live_path, sleep, expected=200)
            if application == "admin-ui":
                report[role]["ready"] = http_status(commands, app, url + "/api/health/ready", sleep, expected=200)
            # Execute the exact configured Node health body directly, without a shell.
            report[role]["healthcheckExit"] = commands.run(
                ["docker", "exec", app, "node", "-e", HEALTH_JS[application]], timeout=5
            )["returncode"]
            report[role]["backendTermination"] = containers.terminate(backend)
            if application == "admin-ui":
                ready = http_status(commands, app, url + "/api/health/ready", sleep, expected=503)
                live = http_status(commands, app, url + "/api/health", sleep, expected=200)
                health = commands.run(
                    ["docker", "exec", app, "node", "-e", HEALTH_JS[application]], timeout=5, checked=False
                )
                require(health["returncode"] == 1, "configured healthcheck must fail with backend down")
                report[role]["backendDown"] = {"ready": ready, "live": live, "healthcheckExit": health["returncode"]}
            report[role]["termination"] = containers.terminate(app)
        require(
            report["baseline"]["observation"] == report["candidate"]["observation"],
            "Node/ABI/identity/trust/ownership mismatch",
        )
        report["passed"] = True
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        report["error"] = commands.redact(str(exc))
    finally:
        report["cleanupErrors"] = containers.cleanup()
        if report["cleanupErrors"]:
            report["passed"] = False
        (evidence / "qualification.json").write_text(
            commands.redact(json.dumps(report, indent=2, sort_keys=True)) + "\n"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run exact-image qualification and return a nonzero exit on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application", choices=tuple(APPS), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    report = qualify(args.application, args.baseline, args.candidate, args.evidence)
    print(json.dumps({"passed": report["passed"], "report": str(args.evidence / "qualification.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
