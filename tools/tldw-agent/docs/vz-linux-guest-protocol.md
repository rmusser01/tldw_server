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
  "workspace_root": "/workspace"
}
```

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
  "timeout_sec": 30
}
```

### Exec response

```json
{
  "protocol_version": "1",
  "request_id": "req-1",
  "exit_code": 0,
  "stdout": "ok\n",
  "stderr": ""
}
```

### Error response

```json
{
  "protocol_version": "1",
  "request_id": "req-1",
  "error_code": "invalid_request",
  "message": "argv is required"
}
```
