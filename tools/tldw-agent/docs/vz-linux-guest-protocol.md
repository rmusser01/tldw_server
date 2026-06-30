# VZ Linux Guest Protocol

This document defines the first guest protocol used between the Swift
`macos-vz-helper` daemon and guest-mode `tldw-agent`.

## Transport

- vsock
- JSON request/response messages
- one request per response

## Versioning

- guest protocol version starts at `1`
- this protocol is separate from the host helper protocol

## Core Messages

### Handshake request

```json
{
  "protocol_version": "1",
  "request_id": "req-handshake",
  "type": "handshake",
  "vm_id": "vm-1",
  "connection_token": "token-1",
  "guest_version": "1.0.0",
  "workspace_root": "/workspace",
  "capabilities": ["exec", "output_cap_v1"]
}
```

`guest_version`, `workspace_root`, and `capabilities` are optional readiness
metadata. New guests should send a stable, sorted capability list; hosts must
accept older guests that omit it and treat capabilities as unknown rather than
failing the handshake.

### Handshake ack

```json
{
  "protocol_version": "1",
  "request_id": "req-handshake",
  "type": "handshake_ack",
  "status": "accepted",
  "vm_id": "vm-1"
}
```

### Readiness request

```json
{
  "protocol_version": "1",
  "request_id": "req-ready",
  "type": "ready"
}
```

### Readiness response

```json
{
  "protocol_version": "1",
  "request_id": "req-ready",
  "status": "ready",
  "workspace_root": "/workspace"
}
```

### Heartbeat request

```json
{
  "protocol_version": "1",
  "request_id": "req-heartbeat",
  "type": "heartbeat",
  "vm_id": "vm-1"
}
```

### Heartbeat response

```json
{
  "protocol_version": "1",
  "request_id": "req-heartbeat",
  "type": "heartbeat",
  "status": "alive",
  "vm_id": "vm-1"
}
```

### Reconnect request

```json
{
  "protocol_version": "1",
  "request_id": "req-reconnect",
  "type": "reconnect",
  "vm_id": "vm-1",
  "connection_token": "token-1"
}
```

### Reconnect ack

```json
{
  "protocol_version": "1",
  "request_id": "req-reconnect",
  "type": "reconnect_ack",
  "status": "accepted",
  "vm_id": "vm-1"
}
```

### Exec request

```json
{
  "protocol_version": "1",
  "request_id": "req-1",
  "type": "exec",
  "argv": ["/bin/echo", "ok"],
  "cwd": "/workspace",
  "env": {
    "EXAMPLE": "1"
  },
  "timeout_sec": 30,
  "max_output_bytes": 1048576
}
```

`max_output_bytes` is optional. When present, it is a combined stdout/stderr
byte cap. The guest agent terminates the process when observed output exceeds
the cap, returns only a bounded UTF-8-safe prefix, and reports guest-prefixed
detail metadata.

### Exec response

```json
{
  "protocol_version": "1",
  "request_id": "req-1",
  "exit_code": 0,
  "stdout": "ok\n",
  "stderr": "",
  "details": {
    "guest_output_limit_bytes": "1048576",
    "guest_output_limit_exceeded": "false",
    "guest_stdout_bytes_observed": "3",
    "guest_stderr_bytes_observed": "0",
    "guest_stdout_bytes_returned": "3",
    "guest_stderr_bytes_returned": "0"
  }
}
```

When output cap enforcement terminates the command, the response uses exit code
`137` and sets `guest_output_limit_exceeded` to `"true"` with
`guest_output_kill_reason` set to `"output_limit"`.

### Error response

```json
{
  "protocol_version": "1",
  "request_id": "req-1",
  "error_code": "invalid_request",
  "message": "argv is required"
}
```
