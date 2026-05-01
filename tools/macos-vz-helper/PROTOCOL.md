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
    "runtime": "vz_linux"
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
    "created_at": "2026-04-30T18:00:00Z"
  },
  "details": {
    "runtime": "vz_linux"
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
    "created_at": "2026-04-30T18:00:00Z"
  },
  "details": {
    "transport": "vsock"
  }
}
```

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
        "created_at": "2026-04-30T18:00:00Z"
      },
      "details": {
        "runtime": "vz_linux"
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
- Template validation must report `boot_mode` and `validation_strength` when the
  helper can resolve the template successfully.
- Python remains the source of truth for sandbox sessions; the helper only reports
  runtime VM state.
