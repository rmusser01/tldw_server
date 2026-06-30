package guest

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestGuestServerServeStreamHandlesHandshakeAndReadyMessages(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	var input bytes.Buffer
	input.WriteString(
		`{"protocol_version":"1","request_id":"req-handshake","type":"handshake","vm_id":"vm-1","connection_token":"token-1","guest_version":"1.0.0","workspace_root":"` +
			root +
			`"}` + "\n",
	)
	input.WriteString(`{"protocol_version":"1","request_id":"req-ready","type":"ready"}` + "\n")

	var output bytes.Buffer
	if err := server.ServeStream(&input, &output); err != nil {
		t.Fatalf("ServeStream() error = %v", err)
	}

	lines := bytes.Split(bytes.TrimSpace(output.Bytes()), []byte("\n"))
	if len(lines) != 2 {
		t.Fatalf("expected 2 response lines, got %d", len(lines))
	}

	var handshake HandshakeAck
	if err := json.Unmarshal(lines[0], &handshake); err != nil {
		t.Fatalf("unmarshal handshake response: %v", err)
	}
	if handshake.Status != "accepted" {
		t.Fatalf("expected handshake status accepted, got %q", handshake.Status)
	}
	if handshake.VMID != "vm-1" {
		t.Fatalf("expected vm_id vm-1, got %q", handshake.VMID)
	}

	var ready ReadyResponse
	if err := json.Unmarshal(lines[1], &ready); err != nil {
		t.Fatalf("unmarshal ready response: %v", err)
	}
	if ready.Status != "ready" {
		t.Fatalf("expected ready status, got %q", ready.Status)
	}
	if ready.WorkspaceRoot != root {
		t.Fatalf("expected workspace root %q, got %q", root, ready.WorkspaceRoot)
	}
}
