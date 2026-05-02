package guest

import (
	"strings"
	"testing"
	"unicode/utf8"
)

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

func TestGuestServerRejectsInvalidMaxOutputBytes(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	for _, capBytes := range []int{0, 256*1024*1024 + 1} {
		resp, execErr := server.Exec(ExecRequest{
			ProtocolVersion: ProtocolVersion,
			RequestID:       "req-invalid-cap",
			Type:            "exec",
			Argv:            []string{"/bin/echo", "ok"},
			MaxOutputBytes:  &capBytes,
		})
		if execErr == nil {
			t.Fatalf("expected error response for cap %d, got nil and response %#v", capBytes, resp)
		}
		if execErr.ErrorCode != "invalid_request" {
			t.Fatalf("expected invalid_request for cap %d, got %q", capBytes, execErr.ErrorCode)
		}
		if execErr.Message != "max_output_bytes out of range" {
			t.Fatalf("expected max_output_bytes out of range, got %q", execErr.Message)
		}
	}
}

func TestGuestServerExecStopsAtMaxOutputBytes(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	capBytes := 16
	resp, execErr := server.Exec(ExecRequest{
		ProtocolVersion: ProtocolVersion,
		RequestID:       "req-cap",
		Type:            "exec",
		Argv:            []string{"/bin/sh", "-c", "i=0; while [ $i -lt 4096 ]; do printf x; i=$((i+1)); done"},
		MaxOutputBytes:  &capBytes,
	})
	if execErr != nil {
		t.Fatalf("Exec() unexpected error = %#v", execErr)
	}
	if got := len([]byte(resp.Stdout)) + len([]byte(resp.Stderr)); got > capBytes {
		t.Fatalf("returned output exceeds cap: got %d, cap %d", got, capBytes)
	}
	if resp.ExitCode != 137 {
		t.Fatalf("expected output-limit exit 137, got %d", resp.ExitCode)
	}
	if resp.Details["guest_output_limit_exceeded"] != "true" {
		t.Fatalf("expected guest output limit metadata, got %#v", resp.Details)
	}
	if resp.Details["guest_output_limit_bytes"] != "16" {
		t.Fatalf("expected limit detail 16, got %#v", resp.Details)
	}
}

func TestGuestServerExecOutputCapKeepsUTF8Valid(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	capBytes := 5
	resp, execErr := server.Exec(ExecRequest{
		ProtocolVersion: ProtocolVersion,
		RequestID:       "req-utf8",
		Type:            "exec",
		Argv:            []string{"/bin/sh", "-c", "printf 'éééé'"},
		MaxOutputBytes:  &capBytes,
	})
	if execErr != nil {
		t.Fatalf("Exec() unexpected error = %#v", execErr)
	}
	output := resp.Stdout + resp.Stderr
	if !utf8.ValidString(output) {
		t.Fatalf("expected valid UTF-8 output, got %q", output)
	}
	if len([]byte(output)) > capBytes {
		t.Fatalf("returned output exceeds cap: got %d, cap %d", len([]byte(output)), capBytes)
	}
	if strings.Contains(output, "\uFFFD") {
		t.Fatalf("expected truncation without replacement characters, got %q", output)
	}
}
