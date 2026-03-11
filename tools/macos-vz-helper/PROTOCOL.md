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
  "details": {
    "runtime": "vz_linux"
  }
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
- Python remains the source of truth for sandbox sessions; the helper only reports
  runtime VM state.
