package guest

import "testing"

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
