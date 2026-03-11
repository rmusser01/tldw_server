package guest

import "testing"

func TestGuestServerReportsReady(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	resp := server.Ready(ReadyRequest{
		ProtocolVersion: ProtocolVersion,
		RequestID:       "req-ready",
		Type:            "ready",
	})

	if resp.ProtocolVersion != ProtocolVersion {
		t.Fatalf("expected protocol version %q, got %q", ProtocolVersion, resp.ProtocolVersion)
	}
	if resp.Status != "ready" {
		t.Fatalf("expected status ready, got %q", resp.Status)
	}
	if resp.WorkspaceRoot != root {
		t.Fatalf("expected workspace root %q, got %q", root, resp.WorkspaceRoot)
	}
}

func TestGuestServerExecutesArgvWithoutShell(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	resp, execErr := server.Exec(ExecRequest{
		ProtocolVersion: ProtocolVersion,
		RequestID:       "req-exec",
		Type:            "exec",
		Argv:            []string{"/bin/echo", "ok"},
		Cwd:             ".",
	})
	if execErr != nil {
		t.Fatalf("Exec() unexpected error = %#v", execErr)
	}
	if resp.ExitCode != 0 {
		t.Fatalf("expected exit code 0, got %d", resp.ExitCode)
	}
	if resp.Stdout != "ok\n" {
		t.Fatalf("expected stdout %q, got %q", "ok\n", resp.Stdout)
	}
}
