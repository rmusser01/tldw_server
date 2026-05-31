package guest

import (
	"encoding/json"
	"testing"
)

func TestParseExecRequest(t *testing.T) {
	req := ExecRequest{
		ProtocolVersion: "1",
		RequestID:       "req-1",
		Argv:            []string{"/bin/echo", "ok"},
		Cwd:             "/workspace",
	}

	if req.ProtocolVersion != "1" {
		t.Fatalf("expected protocol version 1, got %q", req.ProtocolVersion)
	}
	if req.RequestID != "req-1" {
		t.Fatalf("expected request id req-1, got %q", req.RequestID)
	}
	if len(req.Argv) != 2 || req.Argv[0] != "/bin/echo" || req.Argv[1] != "ok" {
		t.Fatalf("unexpected argv: %#v", req.Argv)
	}
	if req.Cwd != "/workspace" {
		t.Fatalf("expected cwd /workspace, got %q", req.Cwd)
	}
}

func TestExecRequestMaxOutputBytesOptional(t *testing.T) {
	var req ExecRequest
	if err := json.Unmarshal([]byte(`{"protocol_version":"1","request_id":"req-1","type":"exec","argv":["/bin/echo","ok"]}`), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.MaxOutputBytes != nil {
		t.Fatalf("expected nil MaxOutputBytes, got %v", *req.MaxOutputBytes)
	}
}

func TestExecRequestMaxOutputBytesPreservesExplicitZero(t *testing.T) {
	var req ExecRequest
	if err := json.Unmarshal([]byte(`{"protocol_version":"1","request_id":"req-1","type":"exec","argv":["/bin/echo","ok"],"max_output_bytes":0}`), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.MaxOutputBytes == nil || *req.MaxOutputBytes != 0 {
		t.Fatalf("expected explicit zero cap, got %#v", req.MaxOutputBytes)
	}
}
