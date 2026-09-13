//go:build linux

package guest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestGuestServerExecTimeoutDrainsDescendantPipes(t *testing.T) {
	for _, command := range []string{"sleep 30 & wait", "sleep 30 &"} {
		t.Run(command, func(t *testing.T) {
			server, err := NewServer(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			capBytes := 1024
			started := time.Now()
			resp, execErr := server.Exec(ExecRequest{
				ProtocolVersion: ProtocolVersion,
				RequestID:       "req-timeout-drain",
				Type:            "exec",
				Argv:            []string{"/bin/sh", "-c", command},
				TimeoutSec:      1,
				MaxOutputBytes:  &capBytes,
			})
			if execErr == nil || execErr.ErrorCode != "timeout_exceeded" {
				t.Fatalf("expected timeout while draining descendant pipes, got %#v, %#v", resp, execErr)
			}
			if elapsed := time.Since(started); elapsed > 5*time.Second {
				t.Fatalf("pipe draining exceeded timeout budget: %v", elapsed)
			}
		})
	}
}

func TestGuestServerExecTimeoutBoundsEscapedDescendantPipeDrain(t *testing.T) {
	server, err := NewServer(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	argv := escapedDescendantArgv(t, "wait")
	capBytes := 1024
	started := time.Now()
	resp, execErr := server.Exec(ExecRequest{
		ProtocolVersion: ProtocolVersion,
		RequestID:       "req-timeout-escaped-drain",
		Type:            "exec",
		Argv:            argv,
		TimeoutSec:      2,
		MaxOutputBytes:  &capBytes,
	})
	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Fatalf("escaped descendant pipe drain exceeded timeout budget: %v", elapsed)
	}
	if execErr == nil || execErr.ErrorCode != "timeout_exceeded" {
		t.Fatalf("expected timeout while draining escaped descendant pipes, got %#v, %#v", resp, execErr)
	}
}

func TestGuestServerExecOutputLimitBoundsEscapedDescendantPipeDrain(t *testing.T) {
	server, err := NewServer(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	argv := escapedDescendantArgv(t, "while :; do printf 0123456789abcdef; done")
	capBytes := 16
	started := time.Now()
	resp, execErr := server.Exec(ExecRequest{
		ProtocolVersion: ProtocolVersion,
		RequestID:       "req-output-limit-escaped-drain",
		Type:            "exec",
		Argv:            argv,
		TimeoutSec:      15,
		MaxOutputBytes:  &capBytes,
	})
	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Fatalf("escaped descendant pipe drain exceeded output-limit budget: %v", elapsed)
	}
	if execErr != nil {
		t.Fatalf("Exec() unexpected error = %#v", execErr)
	}
	if resp.ExitCode != 137 || resp.Details["guest_output_kill_reason"] != "output_limit" {
		t.Fatalf("expected output-limit cancellation, got %#v", resp)
	}
	if got := len(resp.Stdout) + len(resp.Stderr); got > capBytes {
		t.Fatalf("returned output exceeds cap: got %d, cap %d", got, capBytes)
	}
}

func escapedDescendantArgv(t *testing.T, afterReady string) []string {
	t.Helper()
	setsid, err := exec.LookPath("setsid")
	if err != nil {
		t.Fatalf("setsid is required for escaped-descendant regression: %v", err)
	}
	dir := t.TempDir()
	pidPath := filepath.Join(dir, "child.pid")
	readyPath := filepath.Join(dir, "child.ready")
	t.Cleanup(func() {
		if _, err := os.Stat(readyPath + ".observed"); err != nil {
			t.Errorf("parent never observed detached child readiness: %v", err)
		}
		data, err := os.ReadFile(pidPath)
		if err != nil {
			t.Errorf("read detached child PID for cleanup: %v", err)
			return
		}
		pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
		if err != nil || pid <= 1 {
			t.Errorf("invalid detached child PID: %q", data)
			return
		}
		if err := syscall.Kill(pid, syscall.SIGKILL); err != nil && err != syscall.ESRCH {
			t.Errorf("kill detached child %d: %v", pid, err)
		}
	})
	// Only the setsid child publishes readiness. Exec preserves its PID for cleanup.
	command := `"$1" /bin/sh -ec '
printf "%s\n" "$$" > "$1"
: > "$2"
exec sleep 8
' escaped-child "$2" "$3" &
child=$!
while [ ! -f "$3" ]; do
    kill -0 "$child" 2>/dev/null || exit 99
    sleep 0.01
done
: > "$3.observed"
` + afterReady
	return []string{"/bin/sh", "-ec", command, "escaped-pipe-parent", setsid, pidPath, readyPath}
}
