# macOS VZ Helper Protocol

This document freezes the first real control-plane contract between the Python sandbox
service and the native `vz_linux` macOS helper daemon.

## Transport

- Unix domain socket
- request/response JSON messages
- one operation per request

## Required Response Metadata

Every successful response should include:

- `protocol_version`
- `helper_version`

Every failure response should include:

- `protocol_version`
- `helper_version`
- `error_code`
- `message`

Successful `details` dictionaries may include helper-owned generation metadata:

- `helper_instance_id`: per-helper-process UUID, changes after helper restart
- `helper_started_at`: helper process start timestamp in ISO-8601 format

## Required Operations

- `ping`
- `validate_host`
- `register_template`
- `validate_template`
- `create_vm`
- `exec_guest`
- `get_vm_status`
- `list_vms`
- `terminate_vm`

## Canonical Reply Shapes

### `ping`

```json
{
  "protocol_version": "1",
  "helper_version": "0.1.0",
  "status": "ok",
  "details": {
    "helper_instance_id": "77777777-7777-7777-7777-777777777777",
    "helper_started_at": "2026-05-09T00:00:00Z",
    "transport": "unix"
  }
}
```

### `validate_host`

```json
{
  "protocol_version": "1",
  "helper_version": "0.1.0",
  "available": true,
  "execution_mode": "real",
  "transport": "vsock",
  "reasons": [],
  "details": {
    "runtime": "vz_linux",
    "network_policy": "deny_all"
  }
}
```

### `get_vm_status`

```json
{
  "protocol_version": "1",
  "helper_version": "0.1.0",
  "vm_id": "vm-123",
  "state": "running",
  "healthy": true,
  "metadata": {
    "owner": "tldw",
    "runtime": "vz_linux",
    "run_id": "run-123",
    "session_id": "session-456",
    "session_mode": true,
    "template_id": "vz_linux:bundle",
    "template_path": "/var/lib/tldw/vz-linux/debian-arm64",
    "run_manifest_path": "/var/lib/tldw/image-store/runs/run-123/manifest.json",
    "planning_source": "image_store",
    "workspace_path": "/tmp/tldw-vz-linux-workspace",
    "network_policy": "deny_all",
    "created_at": "2026-04-30T18:00:00Z"
  },
  "details": {
    "helper_instance_id": "77777777-7777-7777-7777-777777777777",
    "helper_started_at": "2026-05-09T00:00:00Z",
    "transport": "vsock",
    "network_policy": "deny_all",
    "guest_version": "1.0.0",
    "guest_workspace_root": "/workspace",
    "guest_capabilities_known": "true",
    "guest_capabilities": "exec,output_cap_v1"
  }
}
```

### `create_vm`

Request:

```json
{
  "operation": "create_vm",
  "protocol_version": "1",
  "request": {
    "owner": "tldw",
    "runtime": "vz_linux",
    "vm_name": "run-123",
    "run_id": "run-123",
    "session_id": "session-456",
    "session_mode": true,
    "template_id": "vz_linux:bundle",
    "template": "/var/lib/tldw/vz-linux/debian-arm64",
    "template_path": "/var/lib/tldw/vz-linux/debian-arm64",
    "run_manifest_path": "/var/lib/tldw/image-store/runs/run-123/manifest.json",
    "planning_source": "image_store",
    "workspace_path": "/tmp/tldw-vz-linux-workspace",
    "network_policy": "deny_all",
    "timeout_sec": 300
  }
}
```

`template_id`, `template_path`, `run_manifest_path`, and `planning_source` are
optional provenance fields. `template_id` is the logical image-store/template
identifier to persist in VM metadata, `template_path` mirrors the concrete bundle
path when callers want it explicit, `run_manifest_path` points at the persisted
per-run clone manifest when image-store planning is used, and `planning_source`
identifies the planner, for example `image_store`.
`network_policy` defaults to `deny_all`; helper-side VM creation rejects any
other value and returns `strict_allowlist_not_supported` for `allowlist`.

Response:

```json
{
  "protocol_version": "1",
  "helper_version": "0.1.0",
  "vm_id": "run-123",
  "state": "running",
  "metadata": {
    "owner": "tldw",
    "runtime": "vz_linux",
    "run_id": "run-123",
    "session_id": "session-456",
    "session_mode": true,
    "template_id": "vz_linux:bundle",
    "template_path": "/var/lib/tldw/vz-linux/debian-arm64",
    "run_manifest_path": "/var/lib/tldw/image-store/runs/run-123/manifest.json",
    "planning_source": "image_store",
    "workspace_path": "/tmp/tldw-vz-linux-workspace",
    "network_policy": "deny_all",
    "created_at": "2026-04-30T18:00:00Z"
  },
  "details": {
    "helper_instance_id": "77777777-7777-7777-7777-777777777777",
    "helper_started_at": "2026-05-09T00:00:00Z",
    "transport": "vsock",
    "network_policy": "deny_all",
    "guest_version": "1.0.0",
    "guest_workspace_root": "/workspace",
    "guest_capabilities_known": "true",
    "guest_capabilities": "exec,output_cap_v1"
  }
}
```

### `exec_guest`

Request:

```json
{
  "operation": "exec_guest",
  "protocol_version": "1",
  "request": {
    "vm_id": "run-123",
    "argv": ["/bin/echo", "ok"],
    "cwd": "/workspace",
    "env": {
      "PATH": "/usr/local/bin:/usr/bin:/bin"
    },
    "timeout_sec": 30,
    "max_output_bytes": 10485760
  }
}
```

`vm_id` and `argv` are required. `cwd` defaults to `/workspace`, `env` defaults
to an empty object, `timeout_sec` defaults to `30`, and `max_output_bytes` is
optional.

The helper enforces this request contract before forwarding execution to the
guest agent:

- `argv` must be a non-empty string array, cannot contain empty or NUL-bearing
  arguments, and is capped at 128 arguments / 32 KiB total argument text.
- `cwd` must lexically remain under `/workspace` and cannot contain `..` path
  components. This is a helper protocol boundary, not a full guest filesystem
  authorization model.
- `env` must be a string-to-string object with at most 128 entries / 32 KiB
  total text. Keys must be non-empty and cannot contain `=`, NUL, or control
  characters. Values cannot contain NUL.
- `timeout_sec` must be finite, positive, and no greater than 3600 seconds.
- `max_output_bytes`, when present, must be a JSON integer in the range
  `1...268435456`. It is forwarded to guest agents that support guest-side
  output cap enforcement and also caps the combined stdout/stderr bytes returned
  in the helper response. When both streams exceed the helper response cap and
  the cap is at least 2 bytes, each stream receives a non-empty fair share and
  unused stream budget may be reused by the other stream.

Malformed JSON shape or missing required fields returns `invalid_request`.
Semantic contract denials return one of `exec_argv_invalid`, `exec_cwd_invalid`,
`exec_env_invalid`, `exec_timeout_invalid`, or `exec_output_limit_invalid`; the
`message` field contains the stable reason.

Guest agents rebuilt with output-cap support terminate the command when the
combined stdout/stderr observation exceeds `max_output_bytes` and return
guest-prefixed detail metadata. The helper still applies host-side response
capping as defense in depth and as fallback for older guest agents.

Response:

```json
{
  "protocol_version": "1",
  "helper_version": "0.1.0",
  "exit_code": 0,
  "stdout": "ok\n",
  "stderr": "",
  "details": {
    "transport": "vsock",
    "vm_id": "run-123",
    "output_limit_bytes": "10485760",
    "guest_output_limit_bytes": "10485760",
    "guest_output_limit_exceeded": "false",
    "guest_stdout_bytes_observed": "3",
    "guest_stderr_bytes_observed": "0",
    "guest_stdout_bytes_returned": "3",
    "guest_stderr_bytes_returned": "0",
    "stdout_bytes_original": "3",
    "stderr_bytes_original": "0",
    "stdout_bytes_returned": "3",
    "stderr_bytes_returned": "0",
    "stdout_truncated": "false",
    "stderr_truncated": "false"
  }
}
```

Output limit detail values are encoded as strings to preserve the existing
`details` shape. Guest-side counters use `guest_` prefixes and host-side
response-cap counters keep the existing unprefixed keys.
Guest-prefixed counters are conditional: older guest agents may omit `guest_*`
metadata and rely only on host-side fallback counters. Clients must fall back to
the unprefixed keys when `guest_*` keys are absent during mixed-version rollouts.

### `validate_template`

```json
{
  "protocol_version": "1",
  "helper_version": "0.1.0",
  "template_id": "vz_linux:bundle",
  "source": "/tmp/vz-linux-bundle",
  "ready": true,
  "boot_mode": "bundle",
  "validation_strength": "strong",
  "reasons": []
}
```

### `list_vms`

```json
{
  "protocol_version": "1",
  "helper_version": "0.1.0",
  "vms": [
    {
      "protocol_version": "1",
      "helper_version": "0.1.0",
      "vm_id": "vm-123",
      "state": "running",
      "healthy": true,
      "metadata": {
        "owner": "tldw",
        "runtime": "vz_linux",
        "run_id": "run-123",
        "session_id": "session-456",
        "session_mode": true,
        "template_id": "vz_linux:bundle",
        "template_path": "/var/lib/tldw/vz-linux/debian-arm64",
        "run_manifest_path": "/var/lib/tldw/image-store/runs/run-123/manifest.json",
        "planning_source": "image_store",
        "workspace_path": "/tmp/tldw-vz-linux-workspace",
        "network_policy": "deny_all",
        "created_at": "2026-04-30T18:00:00Z"
      },
      "details": {
        "transport": "vsock",
        "network_policy": "deny_all",
        "guest_version": "1.0.0",
        "guest_workspace_root": "/workspace",
        "guest_capabilities_known": "true",
        "guest_capabilities": "exec,output_cap_v1"
      }
    }
  ]
}
```

## Stability Rules

- Python must reject incompatible `protocol_version` values.
- Helper runtime truth wins over env-only scaffolding.
- VM ownership metadata is additive in protocol version `1`. Missing or malformed
  metadata must parse as unknown ownership on the Python side.
- The helper assigns `created_at` when VM creation metadata omits it. Python uses
  `owner=tldw`, `runtime=vz_linux`, non-empty `run_id`, and session metadata to
  decide whether orphan repair may terminate a live VM.
- `vz_linux` helper VM creation accepts only `network_policy=deny_all` until a
  separately verified allowlist implementation exists. The accepted policy is
  echoed in VM metadata and status details.
- `vz_linux` guest-agent readiness metadata is additive in protocol version `1`.
  `guest_version`, `guest_workspace_root`, `guest_capabilities_known`, and
  `guest_capabilities` are diagnostic details only. Older guests may omit
  capabilities, in which case helpers report `guest_capabilities_known=false`
  instead of failing VM readiness.
- Template validation must report `boot_mode` and `validation_strength` when the
  helper can resolve the template successfully.
- Python remains the source of truth for sandbox sessions; the helper only reports
  runtime VM state.
