package guest

import "testing"

func TestGuestServerRejectsEmptyArgv(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	resp, execErr := server.Exec(ExecRequest{
		ProtocolVersion: ProtocolVersion,
		RequestID:       "req-empty",
		Type:            "exec",
	})
	if execErr == nil {
		t.Fatalf("expected error response, got nil and response %#v", resp)
	}
	if execErr.ErrorCode != "invalid_request" {
		t.Fatalf("expected invalid_request, got %q", execErr.ErrorCode)
	}
	if execErr.Message == "" {
		t.Fatal("expected non-empty error message")
	}
}
