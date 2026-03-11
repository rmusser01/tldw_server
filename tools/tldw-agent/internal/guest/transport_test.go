package guest

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestGuestServerServeStreamHandlesReadyAndExecRequests(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	var input bytes.Buffer
	input.WriteString(`{"protocol_version":"1","request_id":"req-ready","type":"ready"}` + "\n")
	input.WriteString(`{"protocol_version":"1","request_id":"req-exec","type":"exec","argv":["/bin/echo","ok"],"cwd":"."}` + "\n")

	var output bytes.Buffer
	if err := server.ServeStream(&input, &output); err != nil {
		t.Fatalf("ServeStream() error = %v", err)
	}

	lines := bytes.Split(bytes.TrimSpace(output.Bytes()), []byte("\n"))
	if len(lines) != 2 {
		t.Fatalf("expected 2 response lines, got %d", len(lines))
	}

	var ready ReadyResponse
	if err := json.Unmarshal(lines[0], &ready); err != nil {
		t.Fatalf("unmarshal ready response: %v", err)
	}
	if ready.Status != "ready" {
		t.Fatalf("expected ready status, got %q", ready.Status)
	}
	if ready.WorkspaceRoot != root {
		t.Fatalf("expected workspace root %q, got %q", root, ready.WorkspaceRoot)
	}

	var execResp ExecResponse
	if err := json.Unmarshal(lines[1], &execResp); err != nil {
		t.Fatalf("unmarshal exec response: %v", err)
	}
	if execResp.ExitCode != 0 {
		t.Fatalf("expected exit code 0, got %d", execResp.ExitCode)
	}
	if execResp.Stdout != "ok\n" {
		t.Fatalf("expected stdout %q, got %q", "ok\n", execResp.Stdout)
	}
}

func TestGuestServerServeStreamRejectsUnsupportedRequests(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	var input bytes.Buffer
	input.WriteString(`{"protocol_version":"1","request_id":"req-unsupported","type":"noop"}` + "\n")

	var output bytes.Buffer
	if err := server.ServeStream(&input, &output); err != nil {
		t.Fatalf("ServeStream() error = %v", err)
	}

	var resp ErrorResponse
	if err := json.Unmarshal(bytes.TrimSpace(output.Bytes()), &resp); err != nil {
		t.Fatalf("unmarshal error response: %v", err)
	}
	if resp.ErrorCode != "invalid_request" {
		t.Fatalf("expected invalid_request, got %q", resp.ErrorCode)
	}
	if resp.RequestID != "req-unsupported" {
		t.Fatalf("expected request ID %q, got %q", "req-unsupported", resp.RequestID)
	}
}

func TestGuestTransportRejectsWrongProtocolVersion(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	var input bytes.Buffer
	input.WriteString(`{"protocol_version":"999","request_id":"req-wrong-version","type":"handshake","vm_id":"vm-1","connection_token":"token-1"}` + "\n")

	var output bytes.Buffer
	if err := server.ServeStream(&input, &output); err != nil {
		t.Fatalf("ServeStream() error = %v", err)
	}

	var resp ErrorResponse
	if err := json.Unmarshal(bytes.TrimSpace(output.Bytes()), &resp); err != nil {
		t.Fatalf("unmarshal error response: %v", err)
	}
	if resp.ErrorCode != "protocol_mismatch" {
		t.Fatalf("expected protocol_mismatch, got %q", resp.ErrorCode)
	}
	if resp.RequestID != "req-wrong-version" {
		t.Fatalf("expected request ID %q, got %q", "req-wrong-version", resp.RequestID)
	}
}

func TestGuestTransportAcceptsHeartbeatWithoutExec(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	var input bytes.Buffer
	input.WriteString(`{"protocol_version":"1","request_id":"req-heartbeat","type":"heartbeat","vm_id":"vm-heartbeat"}` + "\n")

	var output bytes.Buffer
	if err := server.ServeStream(&input, &output); err != nil {
		t.Fatalf("ServeStream() error = %v", err)
	}

	var resp HeartbeatResponse
	if err := json.Unmarshal(bytes.TrimSpace(output.Bytes()), &resp); err != nil {
		t.Fatalf("unmarshal heartbeat response: %v", err)
	}
	if resp.Status != "alive" {
		t.Fatalf("expected heartbeat status alive, got %q", resp.Status)
	}
	if resp.VMID != "vm-heartbeat" {
		t.Fatalf("expected vm_id %q, got %q", "vm-heartbeat", resp.VMID)
	}
}
