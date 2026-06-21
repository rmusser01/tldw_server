# ACP Sandbox Host Runtime Verification - 2026-06-19

This records release-host evidence for
[#2400](https://github.com/rmusser01/tldw_server/issues/2400) under parent
[#2398](https://github.com/rmusser01/tldw_server/issues/2398).

## Release Decision

Docker is the only sandbox runtime selected for this release-host evidence. The
Docker-backed sandbox service lifecycle has explicit pass evidence on the host
below. Lima and Apple Virtualization Framework runtimes remain untested for this
release and must stay caveated in setup, compatibility, and release surfaces.

This evidence is a core sandbox runtime check, not a named downstream-agent ACP
certification. Do not mark a named agent as sandbox-supported unless that agent
also has an agent-specific `workspace-live-e2e` run with sandbox expectations
enabled, such as `ACP_E2E_EXPECT_SANDBOX=1`.

Allowed release wording:

- Docker-backed ACP sandbox runtime lifecycle was verified on the recorded
  macOS/Docker Desktop host.
- Named downstream-agent rows can keep `sandbox=skip` until their own sandbox
  run passes.

Disallowed release wording:

- ACP sandbox support is verified for Lima, VZ, or all runtimes.
- A named downstream agent is sandbox-supported because the generic Docker
  lifecycle test passed.

## Host And Commit

| Field | Value |
| --- | --- |
| Date | 2026-06-19 |
| Host OS | macOS 15.6, build 24G84 |
| Host architecture | arm64 |
| Repo commit | `7d7fc4f3908c92fcd7b7ffde10b0a8bff3221c3a` |
| Worktree | `.worktrees/acp-sandbox-host-runtime-verification` |

## Runtime Inventory

| Runtime | Result | Evidence |
| --- | --- | --- |
| Docker | Pass for core sandbox lifecycle | Docker Desktop 4.59.1, Docker Engine 29.2.0, API 1.53, server OS/arch `linux/arm64`, client OS/arch `darwin/arm64`, context `desktop-linux`. |
| Lima | Not selected; untested | `limactl --version` exited 127 with `zsh:1: command not found: limactl`. |
| Apple Virtualization Framework / VZ Linux or macOS | Not selected; untested | No VZ helper/template lifecycle was run for this release evidence. Existing unit/fail-closed coverage remains useful, but it is not release-host pass evidence. |

The Docker image `python:3.11-slim` was already available locally
(`sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0`),
so the live Docker run did not require an image pull.

## Commands And Results

Host/runtime probes:

```bash
sw_vers
# ProductName: macOS
# ProductVersion: 15.6
# BuildVersion: 24G84

uname -m
# arm64

docker version
# Client: Docker 29.2.0, darwin/arm64, context desktop-linux
# Server: Docker Desktop 4.59.1, Engine 29.2.0, linux/arm64

limactl --version
# zsh:1: command not found: limactl
```

Live Docker sandbox lifecycle:

```bash
source .venv/bin/activate
SANDBOX_ENABLE_EXECUTION=1 python -m pytest \
  tldw_Server_API/tests/sandbox/test_docker_runner_integration.py \
  -q -m sandbox_real_docker
```

Result: `1 passed, 3 warnings in 13.82s`.

The passing test created a Docker sandbox session, uploaded an inline Python
script into the workspace, started a run in `python:3.11-slim`, observed exit
code 0, checked artifact listing, and asserted no run-owned container or
network remained after completion.

Workspace allowlist, ACP sandbox policy, and runner coverage:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Agent_Orchestration/test_workspace_api_helpers.py \
  -q

python -m pytest \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_runtime_policy_service.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py \
  -q
```

Workspace allowlist result: `17 passed, 6 warnings in 2.23s`.

Result: `33 passed, 8 warnings in 2.75s`.

The workspace helper coverage explicitly checked allowed base path enforcement,
empty allowlist rejection, relative cwd containment inside a workspace root, and
absolute cwd override rejection. The ACP sandbox coverage checked
workspace/runtime policy fail-closed behavior, default sandbox network policy,
runtime availability rejection for unselected runtimes, configured plus
per-session env merging through `ACP_AGENT_ENV_JSON`, session control metadata,
and stream-backed sandbox session paths.

Post-run cleanup probes:

```bash
docker ps -a --filter label=tldw_run_id -q
# no output

docker network ls --filter name=tldw_sbx -q
# no output
```

Docker socket access required running Docker commands outside the managed Codex
command sandbox. The runtime evidence therefore depends on host Docker Desktop
access, not only on the repository test harness.

## Remaining Caveats

- Lima remains `host_runtime_missing` / `sandbox_unverified` until `limactl` is
  installed and a Lima sandbox lifecycle run passes.
- VZ Linux/macOS remains `sandbox_unverified` until the helper, image/template,
  session lifecycle, and cleanup path pass on the target macOS host.
- Named agents remain `sandbox=skip` unless their compatibility row links
  agent-specific sandbox evidence. Use `workspace-live-e2e` with
  `ACP_E2E_EXPECT_SANDBOX=1` when promoting a named agent to `sandbox_tested`.
- The Docker lifecycle run verifies runtime/session/artifact/cleanup behavior.
  It does not prove provider auth, downstream ACP protocol behavior, MCP
  injection, reviewer-loop behavior, or artifact-producing agent workflows.
