//go:build linux

package guest

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

const pipeDrainTestWatchdog = time.Minute

func TestGuestServerExecTimeoutDrainsDescendantPipes(t *testing.T) {
	for _, afterReady := range []string{"wait", "exit 0"} {
		t.Run(afterReady, func(t *testing.T) {
			server, err := NewServer(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			capBytes := 1024
			resp, execErr := execWithPipeHoldingDescendant(t, server, ExecRequest{
				ProtocolVersion: ProtocolVersion,
				RequestID:       "req-timeout-drain",
				Type:            "exec",
				TimeoutSec:      1,
				MaxOutputBytes:  &capBytes,
			}, false, afterReady)
			if execErr == nil || execErr.ErrorCode != "timeout_exceeded" {
				t.Fatalf("expected timeout while draining descendant pipes, got %#v, %#v", resp, execErr)
			}
		})
	}
}

func TestGuestServerExecTimeoutBoundsEscapedDescendantPipeDrain(t *testing.T) {
	for _, afterReady := range []string{"wait", "exit 0"} {
		t.Run(afterReady, func(t *testing.T) {
			server, err := NewServer(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			capBytes := 1024
			resp, execErr := execWithPipeHoldingDescendant(t, server, ExecRequest{
				ProtocolVersion: ProtocolVersion,
				RequestID:       "req-timeout-escaped-drain",
				Type:            "exec",
				TimeoutSec:      2,
				MaxOutputBytes:  &capBytes,
			}, true, afterReady)
			if execErr == nil || execErr.ErrorCode != "timeout_exceeded" {
				t.Fatalf("expected timeout while draining escaped descendant pipes, got %#v, %#v", resp, execErr)
			}
		})
	}
}

func TestGuestServerExecOutputLimitBoundsEscapedDescendantPipeDrain(t *testing.T) {
	server, err := NewServer(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	capBytes := 16
	resp, execErr := execWithPipeHoldingDescendant(t, server, ExecRequest{
		ProtocolVersion: ProtocolVersion,
		RequestID:       "req-output-limit-escaped-drain",
		Type:            "exec",
		TimeoutSec:      120, // The output-limit path must finish before the deadlock watchdog.
		MaxOutputBytes:  &capBytes,
	}, true, "while :; do printf 0123456789abcdef; done")
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

func execWithPipeHoldingDescendant(t *testing.T, server *Server, req ExecRequest, escaped bool, afterReady string) (*ExecResponse, *ErrorResponse) {
	t.Helper()
	var setsid, launchPrefix string
	if escaped {
		var err error
		setsid, err = exec.LookPath("setsid")
		if err != nil {
			t.Fatalf("setsid is required for escaped-descendant regression: %v", err)
		}
		launchPrefix = `"$2" `
	}
	dir := t.TempDir()
	// No natural exit can make an unbounded drain pass. Only test cleanup releases
	// the pipes; removing the directory also stops a child that starts during cleanup.
	command := launchPrefix + `/bin/sh -ec '
: > "$1/ready"
while [ -d "$1" ] && [ ! -f "$1/release" ]; do
    if [ -f "$1/check-live" ]; then
        : > "$1/live"
    fi
    sleep 0.01
done
exec 1>&- 2>&-
: > "$1/stopped"
' pipe-holder "$1" &
while [ ! -f "$1/ready" ]; do
    [ -d "$1" ] && [ ! -f "$1/release" ] || exit 99
    sleep 0.01
done
: > "$1/observed"
` + afterReady
	req.Argv = []string{"/bin/sh", "-ec", command, "pipe-parent", dir, setsid}
	var resp *ExecResponse
	var execErr *ErrorResponse
	done := make(chan struct{})
	t.Cleanup(func() {
		if err := os.WriteFile(filepath.Join(dir, "release"), nil, 0600); err != nil {
			t.Errorf("release pipe-holding child: %v", err)
		}
		watchdog := time.NewTimer(pipeDrainTestWatchdog)
		defer watchdog.Stop()
		select {
		case <-done:
		case <-watchdog.C:
			t.Error("Exec did not finish after releasing descendant pipes")
		}
		if escaped {
			if _, err := os.Stat(filepath.Join(dir, "ready")); err == nil {
				if err := waitForPipeTestFile(filepath.Join(dir, "stopped")); err != nil {
					t.Errorf("detached child did not close its pipes after release: %v", err)
				}
			}
		}
	})
	go func() {
		resp, execErr = server.Exec(req)
		close(done)
	}()
	watchdog := time.NewTimer(pipeDrainTestWatchdog)
	defer watchdog.Stop()
	select {
	case <-done:
	case <-watchdog.C:
		t.Fatal("Exec deadlocked while a descendant retained stdout/stderr")
	}
	if _, err := os.Stat(filepath.Join(dir, "observed")); err != nil {
		t.Fatalf("parent never observed child readiness: %v", err)
	}
	if escaped {
		// A fresh acknowledgement after Exec returns proves the escaped child is
		// still alive with both pipes open, rather than merely having been ready once.
		if err := os.WriteFile(filepath.Join(dir, "check-live"), nil, 0600); err != nil {
			t.Fatal(err)
		}
		if err := waitForPipeTestFile(filepath.Join(dir, "live")); err != nil {
			t.Fatalf("detached child did not retain its pipes until cleanup: %v", err)
		}
	}
	return resp, execErr
}

func waitForPipeTestFile(path string) error {
	watchdog := time.NewTimer(pipeDrainTestWatchdog)
	defer watchdog.Stop()
	poll := time.NewTicker(10 * time.Millisecond)
	defer poll.Stop()
	for {
		if _, err := os.Stat(path); err == nil {
			return nil
		} else if !os.IsNotExist(err) {
			return err
		}
		select {
		case <-poll.C:
		case <-watchdog.C:
			return fmt.Errorf("deadlock watchdog waiting for %s", path)
		}
	}
}
