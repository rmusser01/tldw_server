//go:build linux

package guest

import (
	"os/exec"
	"syscall"
)

func configureCommandProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func terminateCommandProcess(cmd *exec.Cmd) bool {
	if cmd == nil || cmd.Process == nil {
		return false
	}
	terminated := false
	pid := cmd.Process.Pid
	if pid > 0 {
		terminated = syscall.Kill(-pid, syscall.SIGKILL) == nil
	}
	if cmd.Process.Kill() == nil {
		terminated = true
	}
	return terminated
}
